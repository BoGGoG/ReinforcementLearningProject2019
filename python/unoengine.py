import numpy as np
from arena import GameEnv


class UnoEngine(GameEnv):
    def __init__(self):

        # cards are coded by color (rygb) + number/special.
        # special cards: d: take 2, s: skip round

        colors = ['R', 'Y', 'G', 'B']
        n_numbers = 10
        numbers = [str(i) for i in range(n_numbers)]
        specials = ['d', 's']

        self.max_hand_cards = 20  # loss when more cards on hand

        # generate an array containing the string codes for the cards
        card_names = []
        for c in colors:
            for n in numbers:
                card_names.append(c + n)
            for s in specials:
                card_names.append(c + s)
        card_names = np.array(card_names)
        n_different_cards = len(card_names)

        self.state_dim = n_different_cards + 1 + 1  # hand cards + open card + opponent card count
        self.action_dim = n_different_cards + 1  # play one of n_cards cards or draw a card

        # generate a matrix encoding the legal moves
        legality_matrix = np.zeros((n_different_cards, n_different_cards), dtype=bool)
        for i, ni in enumerate(card_names):
            for j, nj in enumerate(card_names):
                if ni[0] == nj[0] or ni[1] == nj[1]:
                    legality_matrix[i, j] = True

        game_state = {'open_card': 0,
                      'p_cards': (np.zeros(self.state_dim), np.zeros(self.state_dim)),
                      'turn': 0
                      }

        self.n_different_cards = n_different_cards
        self.card_names = card_names
        self.game_state = game_state
        self.leg_matrix = legality_matrix

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def reset(self):
        game_state = self.game_state
        n_cards = self.n_different_cards

        # draw open card
        game_state['open_card'] = np.random.randint(n_cards)

        # draw p1 and p2 cards
        game_state['p_cards'] = (np.zeros(n_cards), np.zeros(n_cards))
        for i in range(7):
            for player in [0, 1]:
                card = np.random.randint(n_cards)
                game_state['p_cards'][player][card] += 1

        # player 0 gets to start
        game_state['turn'] = 0

        return self.make_game_info(reward=0, game_over=False)

    def make_p_state(self, player):
        opponent = 1 if player == 0 else 0
        open_card = self.game_state['open_card']
        hand_cards = self.game_state['p_cards'][player]
        n_opponent_cards = int(self.game_state['p_cards'][opponent].sum())
        return np.append(hand_cards, [open_card, n_opponent_cards])

    def make_game_info(self, reward, game_over):
        return {'turn': self.game_state['turn'],
                'p_state': self.make_p_state(self.game_state['turn']),
                'legal_actions': self.get_legal_actions(),
                'reward': reward,
                'game_over': game_over
                }

    def get_legal_actions(self):
        player = self.game_state['turn']
        open_card = self.game_state['open_card']
        legal_cards = self.leg_matrix[open_card] * self.game_state['p_cards'][player] > 0
        return np.append(legal_cards, True)  # append the draw action

    def play(self, action):
        player = self.game_state['turn']
        opponent = 1 if player == 0 else 0

        # if the opponent already lost or won, the game needs to end here
        player_lost = self.game_state['p_cards'][opponent].sum() < 1
        player_won = self.game_state['p_cards'][opponent].sum() > self.max_hand_cards
        # NOTE: the following is received by the opponent!
        if player_lost:
            reward = 100
            self.game_state['turn'] = opponent
            return self.make_game_info(reward, game_over=True)
        elif player_won:
            reward = -100
            self.game_state['turn'] = opponent
            return self.make_game_info(reward, game_over=True)

        # if the player draws a card:
        if action >= self.n_different_cards:
            new_card = np.random.randint(self.n_different_cards)
            self.game_state['p_cards'][player][new_card] += 1
            self.game_state['turn'] = opponent
            reward = 0 # 0 # TODO
            # print('player {} draws a card'.format(player))

        # all other cases are the player playing a card
        else:
            card_name = self.card_names[action]

            # if the player plays a skip card
            if card_name[1] == 's':
                self.game_state['p_cards'][player][action] -= 1
                self.game_state['open_card'] = action
                # we need to consider the case when the player wins with a skip card separately
                # for the reward passing to be consistent
                self.game_state['turn'] = player
                reward = 0

            # if the player plays a draw2 card:
            if card_name[1] == 'd':
                self.game_state['p_cards'][player][action] -= 1
                self.game_state['open_card'] = action
                new_cards = np.random.randint(self.n_different_cards, size=2)
                for new_card in new_cards:
                    self.game_state['p_cards'][opponent][new_card] += 1
                self.game_state['turn'] = opponent
                reward = 0

            # if the player plays a number card
            if card_name[1:].isnumeric():
                self.game_state['p_cards'][player][action] -= 1
                self.game_state['open_card'] = action
                self.game_state['turn'] = opponent
                reward = 0

        # if the player lost or won, the opponent needs to know this in the next step
        player_won = self.game_state['p_cards'][player].sum() < 1
        player_lost = self.game_state['p_cards'][player].sum() > self.max_hand_cards
        # NOTE: the following is received by the opponent!
        if player_won:
            reward = -100
            self.game_state['turn'] = opponent  # in case the player wins with a skip card
            game_over = True
        elif player_lost:
            reward = 100
            self.game_state['turn'] = opponent
            game_over = True
        else:
            game_over = False

        return self.make_game_info(reward, game_over)

    def get_card_names(self, card_array):
        card_names = []
        for c, n in enumerate(card_array):
            if n > 0:
                card_names.append(str(int(n)) + ' ' + self.card_names[c])
        return ', '.join(card_names)


import numpy as np


class UnoEngine:
    def __init__(self):

        # cards are coded by color (rygb) + number/special.
        # special cards: d: take 2, s: skip round

        colors = ['r', 'y', 'g', 'b']
        n_numbers = 8
        numbers = [str(i) for i in range(n_numbers)]
        specials = ['d', 's']

        # generate an array containing the string codes for the cards
        card_names = []
        for c in colors:
            for n in numbers:
                card_names.append(c + n)
            for s in specials:
                card_names.append(c + s)
        card_names = np.array(card_names)
        n_cards = len(card_names)

        # generate a matrix encoding the legal moves
        legality_matrix = np.zeros((n_cards, n_cards), dtype=bool)
        for i, ni in enumerate(card_names):
            for j, nj in enumerate(card_names):
                if ni[0] == nj[0] or ni[1] == nj[1]:
                    legality_matrix[i, j] = True

        game_state = {'open_card': '',
                      'p_cards': (np.zeros(n_cards), np.zeros(n_cards)),
                      'turn': 0
                      }

        self.n_cards = n_cards
        self.card_names = card_names
        self.game_state = game_state
        self.leg_matrix = legality_matrix

    def reset(self):
        game_state = self.game_state
        n_cards = self.n_cards

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

        return {'turn': game_state['turn'],
                'p_state': self.p_state(game_state['turn']),
                'reward': 0,
                'done': False
                }

    def p_state(self, player):
        opponent = 1 if player == 0 else 0
        return {'open_card': self.game_state['open_card'],
                'hand_cards': self.game_state['p_cards'][player],
                'n_opponent_cards': self.game_state['p_cards'][opponent].sum()
                }

    def legal_cards(self):
        return self.leg_matrix[self.game_state['open_card']]

    def step(self, card):
        player = self.game_state['turn']
        opponent = 1 if player == 0 else 0

        reward = 0
        done = False

        if self.game_state['p_cards'][opponent].sum() < 1:
            reward = -100
            done = True

        # card = -1 indicates that the player draws a card
        elif card == -1:
            new_card = np.random.randint(self.n_cards)
            self.game_state['p_cards'][player][new_card] += 1
            self.game_state['turn'] = opponent
        elif not self.legal_cards()[card]:
            print('illegal card!')
        elif self.game_state['p_cards'][player][card] < 1:
            print('you do not have this card!')
        else:
            self.game_state['p_cards'][player][card] -= 1
            self.game_state['open_card'] = card

            card_name = self.card_names[card]
            if card_name[1] == 'd':
                new_cards = np.random.randint(self.n_cards, size=2)
                for c in new_cards:
                    self.game_state['p_cards'][opponent][c] += 1

            if card_name[1] != 's':
                self.game_state['turn'] = opponent

        if self.game_state['p_cards'][player].sum() < 1:
            reward = 100
            done = True

        return {'turn': self.game_state['turn'],
                'p_state': self.p_state(self.game_state['turn']),
                'reward': reward,
                'done': done
                }

    def text_step(self, card_name):
        if card_name == '':
            card = -1
        else:
            card = np.argwhere(self.card_names == card_name)[0, 0]
        dic = self.step(card)
        hand_cards = []
        for c, n in enumerate(dic['p_state']['hand_cards']):
            if n > 0:
                hand_cards.append(str(int(n)) + ' ' + self.card_names[c])
        hand_cards = ', '.join(hand_cards)

        print()
        print('Player {}:'.format(dic['turn']))
        print('open card: ' + self.card_names[dic['p_state']['open_card']])
        print('Hand Cards: ' + hand_cards)
        print('Opponent has {} cards'.format(dic['p_state']['n_opponent_cards']))

        return done

    def text_reset(self):
        dic = self.reset()
        hand_cards = []
        for c, n in enumerate(dic['p_state']['hand_cards']):
            if n > 0:
                hand_cards.append(str(int(n)) + ' ' + self.card_names[c])
        hand_cards = ', '.join(hand_cards)
        print()
        print('Player {}:'.format(dic['turn']))
        print('open card: ' + self.card_names[dic['p_state']['open_card']])
        print('Hand Cards: ' + hand_cards)
        print('Opponent has {} cards'.format(dic['p_state']['n_opponent_cards']))


if __name__ == '__main__':
    uno = UnoEngine()
    uno.text_reset()
    done = False
    while not done:
        card_name = input('Play card (leave empty to draw): ')
        done = uno.text_step(card_name)

class Arena:
    def __init__(self, agent_0, agent_1, game_env):
        """
        Framework to standardize game playing for two agents.
        The Goal is that independent teams can create their own Agents and Games independently and still be able
        to use the other team's game. Also, duelling of different agents should be possible.
        :param agent_0: should subclass the Agent class
        :param agent_1: " "
        :param game_env: should subclass the GameEnv class
        """
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.game_env = game_env

        self.game_info = self.game_env.reset()

        # when one player wins, the other still has to receive the game_info in the next step to receive the reward.
        # we can reset the game only after both players received their rewards, so we need to kep track of for whom
        # the game ended separately
        self.agent_0_done = False
        self.agent_1_done = False

        self.cum_agent_0_reward = 0.0
        self.cum_agent_1_reward = 0.0

    def step(self):
        """
        Execute one step, i.e. wichever agent's turn it is receives his game info (see Agent class) and plays one card
        :return: nothing
        """

        if self.game_info['turn'] == 0:
            agent = self.agent_0
            self.cum_agent_0_reward += self.game_info['reward']
        else:
            agent = self.agent_1
            self.cum_agent_1_reward += self.game_info['reward']

        if self.agent_0_done and self.agent_1_done:
            self.game_info = self.game_env.reset()
            self.agent_0_done = False
            self.agent_1_done = False
            return None

        action = agent.digest(self.game_info)

        if not action:
            self.game_info = self.game_env.step(action)
        elif self.game_info['legal_actions'][action]:
            self.game_info = self.game_env.step(action)
        else:
            raise ValueError('agent {} attempted an illegal move'.format(self.game_info['turn']))

        if self.game_info['game_over']:
            if self.game_info['turn'] == 0:
                self.agent_0_done = True
            else:
                self.agent_1_done = True

        return action


class Agent(object):
    """Agent that plays the game. May be human, random or a reinforcement learner.
    """
    def digest(self, game_info):
        """
        :param game_info: dictionary with the following entries:

                        'turn':             0 or 1, whose turn it is
                        'p_state':          float array encoding the state ofthe game
                                            as viewed by the player who's 'turn' it is
                        'legal_actions':    boolean array encoding wich actions are legal
                        'reward':           float reward that the player receives as a result from HIS last turn
                        'game_over':        boolean weather the game is over or not

                        NOTE: History should be encoded within the p_state array
        :return: integer encoding the action the player takes or None if the game is over.
                Integers must be in [0, ..., GameEnv.get_state_dim() - 1]
        """
        raise NotImplementedError


class GameEnv(object):
    """Game environment that simulates the card game"""
    def reset(self):
        """
        reset the environment to begin a new game
        :return: game_info for the first turn. (see Agent class)
        """
        raise NotImplementedError

    def step(self, action):
        """
        Execute one step, i.e. let the player who's turn it is play one card
        :param action: integer encoding the action, or None if the game is over and there is not action to take
        :return: game_info for the next turn. (see Agent class)
        """
        raise NotImplementedError

    def get_state_dim(self):
        """
        This function exists so that one does not need to know all details about the game environment in order
        to construct an agent
        :return: dimension of the array that encodes p_sate (see Agent class)
        """
        raise NotImplementedError

    def get_action_dim(self):
        """
        This function exists so that one does not need to know all details about the game environment in order
        to construct an agent
        :return: number of possible actions (see Agent class)
        """
        raise NotImplementedError


if __name__ == '__main__':
    """to random agents play one round against each other. The game is printed in textform
    """
    from unoengine import UnoEngine
    from agents import RandomAgent

    unoengine = UnoEngine()
    agent_0 = RandomAgent(unoengine.get_action_dim())
    agent_1 = RandomAgent(unoengine.get_action_dim())

    arena = Arena(agent_0, agent_1, unoengine)

    # main loop
    i = 0
    finished = False
    while not finished:
        finished = (arena.agent_0_done and arena.agent_1_done)

        info = arena.game_info
        player = info['turn']
        open_card = unoengine.card_names[int(info['p_state'][-2])]
        n_opponent_cards = info['p_state'][-1]
        hand_cards = unoengine.get_card_names(info['p_state'][:-2])
        game_over = info['game_over']
        reward = info['reward']

        action = arena.step()

        if game_over:
            output = 'Player {} finishes with a reward of {}'.format(player, reward)
        else:
            if action >= unoengine.n_different_cards:
                a_name = 'draws '
            else:
                a_name = 'plays ' + unoengine.card_names[action] + ' '

            output = 'Step {:3}: '.format(i)
            output += 'player {} '.format(player)
            output += a_name
            output += 'on ' + open_card + '. '
            output += 'Hand cards before that: {' + hand_cards + '}. '
            output += 'Number of opponents cards: {}'.format(n_opponent_cards) + '.'

        print(output)
        i += 1





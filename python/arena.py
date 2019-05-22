from abc import ABC


class Arena:
    def __init__(self, agent_0, agent_1, game_env):
        """
        Framework to standardize game playing for two agents
        :param agent_0: should subclass the Agent class
        :param agent_1:
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
        if self.agent_0_done and self.agent_1_done:
            self.game_info = self.game_env.reset()

        if self.game_info['turn'] == 0:
            agent = self.agent_0
            self.cum_agent_0_reward += self.game_info['reward']
        else:
            agent = self.agent_1
            self.cum_agent_1_reward += self.game_info['reward']

        action = agent.digest(self.game_info)

        if self.game_info['game_over']:
            if self.game_info['turn'] == 0:
                self.agent_0_done = True
            else:
                self.agent_1_done = True
            self.game_info = self.game_env.step(None)
        elif self.game_info['legal_actions'][action]:
            self.game_info = self.game_env.step(action)
        else:
            raise ValueError('agent {} attempted an illegal move'.format(self.game_info['turn']))


class Agent(ABC):
    """Agent that plays the game. May be human, random or a reinforcement learner.
    """
    def digest(self, game_info):
        """
        :param game_info: dictionary with the following entries:
                        'turn': 0 or 1, whose turn it is
                        'p_state': float array encoding the state of the game as viewed by the player who's 'turn' it is
                        'legal_actions': boolean array encoding wich actions are legal
                        'reward': float reward that the player receives as a result from HIS last turn
                        'game_over': boolean weather the game is over or not
                        NOTE: History should be encoded within the p_state array
        :return: integer encoding the action the player takes or None if the game is over
        """
        pass


class GameEnv(ABC):
    """Game environment that simulates the card game"""
    def reset(self):
        """
        reset the environment to begin a new game
        :return: game_info for the first turn. (see Agent class)
        """
        pass

    def step(self, action):
        """
        Execute one step, i.e. let the player who's turn it is play one card
        :param action: integer encoding the action, or None if the game is over and there is not action to take
        :return: game_info for the next turn. (see Agent class)
        """
        pass



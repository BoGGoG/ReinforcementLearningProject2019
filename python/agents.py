import numpy as np
from arena import Agent
from qnetwork import Policy


class RandomAgent(Agent):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def digest(self, game_info):
        if game_info['game_over']:
            return None
        print("game info:")
        print(game_info)
        print("p_state size: {}", len(game_info['p_state']))
        print("legal_actions size: {}", len(game_info['legal_actions']))

        legal_actions = game_info['legal_actions']
        if legal_actions.sum() <= 1:  # can only draw
            return self.action_dim-1
        else:
            p = game_info['legal_actions'][:-1]
            p = p / p.sum()
            return np.random.choice(self.action_dim-1, p=p)

class ReinforcementAgent(Agent):
    def __init__(self, action_dim):
        """
        UNTIL NOW ONLY RANDOM MOVES
        Agent that learns by Reinforcement Learning.
        Method: Qlearning with Neural Network
        inputLength is action_dim + 2 (last played card, number of opponent's hand cards)
        :param action_dim: Size of action space
        """
        self.action_dim = action_dim
        self.qnetwork = Policy(inputLength = action_dim + 2, outputLength = action_dim)

    def digest(self, game_info):
        if game_info['game_over']:
            print("GG easy")
            return None

        legal_actions = game_info['legal_actions']
        if legal_actions.sum() <= 1:  # can only draw
            return self.action_dim-1
        else:
            p = game_info['legal_actions'][:-1]
            p = p / p.sum()
            return np.random.choice(self.action_dim-1, p=p)

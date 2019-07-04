import numpy as np
from arena import Agent
from qnetwork import Policy

DEBUG = False

class RandomAgent(Agent):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def digest(self, game_info):
        if game_info['game_over']:
            return None

        if DEBUG:
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
        Decides on action based on Neural Network.
        Method: Qlearning with Neural Network.
        :param action_dim: Size of action space: number of cards + 1 (draw)
        """
        self.action_dim = action_dim
        self.policy = Policy(inputLength = action_dim + 1, outputLength = action_dim)

    def digest(self, game_info):
        """
            take game info, throw into neural network, return action
            until now: return sample from output of neural network (categorical distribution)
            ToDo: Implement epsilon greedy
        """
        if game_info['game_over']:
            "should now read reward and evaluate, backpropagate"
            return None

        # if game_info['legal_actions'].sum() <= 1:  # can only draw
            # return self.action_dim-1

        return(self.greedyAction(game_info))
        # return(self.sampleAction(game_info))
        # return(self.randomAction(game_info))

    def sampleAction(self, game_info):
        "random sample from categorical distribution (output of neural network)"
        action = self.policy.sampleAction(game_info)
        return action

    def greedyAction(self, game_info):
        action = self.policy.greedyAction(game_info)
        return action

    def randomAction(self, game_info):
        legal_actions = game_info['legal_actions']
        if legal_actions.sum() <= 1:  # can only draw
            return self.action_dim-1
        else:
            p = game_info['legal_actions'][:-1]
            p = p / p.sum()
            return np.random.choice(self.action_dim-1, p=p)


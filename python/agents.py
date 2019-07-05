import numpy as np
from arena import Agent
from qnetwork import Policy
import torch

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
    def __init__(self, action_dim, epsilon = 0.1, prevGameInfo = 0, prevAction = -1, gamesPlayed = 0):
        """
        Decides on action based on Neural Network.
        Method: Qlearning with Neural Network.
        :param action_dim: Size of action space: number of cards + 1 (draw)
        """
        self.action_dim = action_dim
        self.policy = Policy(inputLength = action_dim + 1, outputLength = action_dim)
        self.epsilon = epsilon
        self.prevGameInfo = prevGameInfo
        self.gamesPlayed = gamesPlayed

    def digest(self, gameInfo):
        """
            take game info, throw into neural network, return action
            epsilon greedy. After every step learn. After every match decrease epsilon
            param: gameInfo: dict from unoengine
        """
        self.policy.rewards.append(gameInfo['reward'])
        reward = gameInfo['reward']
        if gameInfo['game_over']:
            "game ended, make epsilon smaller"
            self.gamesPlayed += 1
            self.epsilon = 1. / (self.gamesPlayed / 50. + 10.)
            action = None
            finalState = True
            self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]),
                self.prevAction, reward, torch.Tensor(gameInfo["p_state"]), finalState)
            self.prevAction = -1
            self.prevGameInfo = 0
            return(0)
        else:
            action = self.epsilonGreedyAction(gameInfo)
            finalState = False

        if self.prevGameInfo != 0 and self.prevAction != -1:
            self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]),
                self.prevAction, reward, torch.Tensor(gameInfo["p_state"]), finalState)


        self.prevAction = action
        self.prevGameInfo = gameInfo
        return(action)

    def sampleAction(self, game_info):
        """random legal sample from categorical distribution (output of neural network).
        Always legal action"""
        action = self.policy.sampleAction(game_info)
        return action

    def greedyAction(self, game_info):
        "legal action with highest log prob"
        action = self.policy.greedyAction(game_info['p_state'], game_info['legal_actions'])
        return action

    def epsilonGreedyAction(self, game_info):
        "with chance epsilon take random action, else greedy action"
        if np.random.rand(1) < self.epsilon:
            return self.randomAction(game_info)
        else:
            return self.greedyAction(game_info)

    def randomAction(self, game_info):
        "random action, always legal"
        legal_actions = game_info['legal_actions']
        if legal_actions.sum() <= 1:  # can only draw
            return self.action_dim-1
        else:
            p = game_info['legal_actions'][:-1]
            p = p / p.sum()
            return np.random.choice(self.action_dim-1, p=p)


import numpy as np
from arena import Agent
import torch
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
        :epsilon: epsilon greedy, but not really implemented, this param will only change very first iteration
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Active device: ", self.device)

        self.action_dim = action_dim
        self.policy = Policy(inputLength=action_dim + 1, outputLength = action_dim).to(self.device)
        self.epsilon = epsilon
        self.prevGameInfo = prevGameInfo
        self.gamesPlayed = gamesPlayed

    def digest(self, gameInfo):
        """
            ToDo: Write better, this is ugly code.
            Take game info, throw into neural network, return action
            epsilon greedy. After every step learn. After every match decrease epsilon
            param: gameInfo: dict from unoengine
            return: action between 0 and action_dim -1. -1 if finalState
        """
        # self.policy.rewards.append(gameInfo['reward']) # why?
        reward = gameInfo['reward']
        if gameInfo['game_over']:
            "game ended, make epsilon smaller"
            self.epsilon = 1. / (self.gamesPlayed / 50. + 8.)
            action = None
            finalState = True
            self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]).to(self.device),
                              self.prevAction, reward, torch.Tensor(gameInfo["p_state"]).to(self.device), finalState)
            self.prevAction = -1
            self.prevGameInfo = 0
            self.gamesPlayed += 1
            # print('RFL agent finished with reward {}'.format(gameInfo['reward']))
            return action
        else:
            action = self.get_action(gameInfo)
            finalState = False

        if self.prevGameInfo != 0 and self.prevAction != -1:

            self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]).to(self.device),
                self.prevAction, reward, torch.Tensor(gameInfo["p_state"]).to(self.device), finalState)


        self.prevAction = action
        self.prevGameInfo = gameInfo
        return action

    def get_action(self, game_info, random_Q=10):
        p_state = torch.Tensor(game_info['p_state']).to(self.device)
        with torch.no_grad():
            Qs = self.policy(p_state).cpu().numpy()
        Qs += random_Q * np.random.rand(len(Qs))
        legal_inds = np.argwhere(game_info['legal_actions']).flatten()
        legal_Qs = Qs[legal_inds]
        best_ind = legal_inds[legal_Qs.argmax()]
        return best_ind

    def sampleAction(self, game_info):
        """random legal sample from categorical distribution (output of neural network).
        Always legal action"""
        pState = torch.Tensor(game_info['p_state'])
        legalActions = torch.Tensor(game_info['legal_actions'])
        action = self.policy.sampleAction(pState, legalActions)
        return action

    def greedyAction(self, game_info):
        "legal action with highest log prob"
        p_state = torch.Tensor(game_info['p_state']).to(self.device)
        legal_act = torch.Tensor(game_info['legal_actions']).to(self.device)
        action = self.policy.greedyAction(p_state, legal_act)
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

    def saveModel(self, path):
        modelParams = { 
                'gamesPlayed':self.gamesPlayed,
                'stateDict': self.policy.state_dict(),
                'epsilon': self.epsilon,
                'optimizerStateDict': self.policy.optimizer.state_dict()
                }
        torch.save(modelParams, path)
        print('Model Saved to', path)
        return(0) # 0 no error

    def loadModel(self, path):
        modelParams = torch.load(path)
        self.gamesPlayed = modelParams['gamesPlayed']
        self.epsilon = modelParams['epsilon']
        self.policy.load_state_dict(modelParams['stateDict'])
        self.policy.optimizer.load_state_dict(modelParams['optimizerStateDict'])
        print('Model Loaded from', path)
        print('previously played {} games'.format(self.gamesPlayed))
        return(0) # 0 no error



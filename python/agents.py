import numpy as np
from arena import Agent
import torch
from qnetwork import Policy
import torch
from collections import deque
import random

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
    def __init__(self, action_dim, epsilon=1., prevGameInfo = 0, prevAction = -1, gamesPlayed = 0,
                 memory_size=50000):
        """
        Decides on action based on Neural Network.
        Method: Qlearning with Neural Network.
        :param action_dim: Size of action space: number of cards + 1 (draw)
        :epsilon: epsilon greedy, but not really implemented, this param will only change very first iteration
        """
        self.memory = deque(maxlen=memory_size)

        self.Q_std = 10000
        self.Q_mean = 0
        self.Q_stats_decay = .01

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Active device: ", self.device)

        self.action_dim = action_dim
        self.policy = Policy(inputLength=action_dim + 1, outputLength = action_dim).to(self.device)
        self.epsilon = epsilon
        self.prevGameInfo = prevGameInfo
        self.prevAction = None
        self.gamesPlayed = gamesPlayed
        self.discount = 0.99

        self.action_list = np.identity(self.action_dim)


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
            if self.gamesPlayed < 1000:
                self.epsilon = 1
            else:
                self.epsilon = 1. / (self.gamesPlayed / 50. + 1.)
            action = None
            final_state = True
            #self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]).to(self.device),
            #                  self.prevAction, reward, torch.Tensor(gameInfo["p_state"]).to(self.device), finalState)
            self.gamesPlayed += 1
            # print('RFL agent finished with reward {}'.format(gameInfo['reward']))
            # return action
        else:
            action = self.get_action(gameInfo, epsilon=self.epsilon, random_Q=self.Q_std)
            final_state = False

        # if self.prevGameInfo != 0 and self.prevAction != None:
            #self.policy.learn(torch.Tensor(self.prevGameInfo["p_state"]).to(self.device),
            #    self.prevAction, reward, torch.Tensor(gameInfo["p_state"]).to(self.device), finalState)

        if self.prevAction:
            self.remember(self.prevGameInfo['p_state'], self.prevAction, reward, gameInfo['p_state'], final_state)

        if final_state:
            self.fit_from_memory()
            self.openingState = True

        self.prevAction = action
        self.prevGameInfo = gameInfo
        return action

    def remember(self, st1, ac, rew, st2, don):
        """Commit round to memory.
        """
        self.memory.append([st1, ac, rew, st2, don])

    def fit_from_memory(self, epochs=3, batch_size=16):
        """Fit the Q model on random data from the memory
        """
        sample_size = min(batch_size, len(self.memory))
        for i in range(epochs):
            sample = random.sample(self.memory, sample_size)
            st1, ac, rew, st2, don = map(np.array, zip(*sample))
            q_next = self.policy(torch.Tensor(st2).to(self.device))
            don = torch.Tensor(don).to(self.device)
            ac = self.action_list[ac]
            ac = torch.Tensor(ac).to(self.device)
            q_next, _ = q_next.max(dim=1)
            q_next = q_next * (1 - don)
            rew = torch.Tensor(rew).to(self.device)
            q_target = rew + self.discount * q_next

            q_old = self.policy(torch.Tensor(st1).to(self.device)) * ac
            q_old = q_old.sum(dim=1)
            # print(q_old)

            loss = (q_target - q_old)**2
            loss = loss.mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

    def get_action(self, game_info, random_Q=10, epsilon=0.):
        if epsilon > 0 and np.random.rand() < epsilon:
            legal_inds = np.argwhere(game_info['legal_actions']).flatten()
            action = np.random.choice(legal_inds)
        else:
            p_state = torch.Tensor(game_info['p_state']).to(self.device)
            with torch.no_grad():
                Qs = self.policy(p_state).cpu().numpy()

            if len(self.memory) >= self.memory.maxlen:
                self.Q_std = self.Q_stats_decay * Qs.std() \
                             + (1 - self.Q_stats_decay) * self.Q_std
            self.Q_mean = self.Q_stats_decay * Qs.mean() \
                          + (1 - self.Q_stats_decay) * self.Q_mean
            Qs += random_Q * np.random.rand(len(Qs))
            legal_inds = np.argwhere(game_info['legal_actions']).flatten()
            legal_Qs = Qs[legal_inds]
            action = legal_inds[legal_Qs.argmax()]
        return action

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



# inspiration:
# https://github.com/pytorch/examples/tree/master/reinforcement_learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

dropoutRate = 0.5
hiddenLayerSizes = [564, 200, 4]

class Policy(nn.Module):
    def __init__(self, inputLength, outputLength):
        """Qnetwork for uno. Takes pState [0,1,0,0,2,...] as input and outputs probabilities.
        :param inputLength: state dimension: all cards + 'draw'
        :param outpuLength: action dimension: all cards, open card, opponents hand cards =usually  inputLength + 1
        ToDo:
        - save network
        - load network
        """
        super(Policy, self).__init__()
        self.inputLength = inputLength
        self.outputLength = outputLength

        self.affine1 = nn.Linear(self.inputLength, hiddenLayerSizes[0])
        self.affine2 = nn.Linear(hiddenLayerSizes[0], hiddenLayerSizes[1])
        self.affine3 = nn.Linear(hiddenLayerSizes[1], hiddenLayerSizes[2])
        self.last_affine = nn.Linear(hiddenLayerSizes[2], self.outputLength)

        self.dropout = nn.Dropout(p=dropoutRate)
        # self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = 0.99

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, pState):
        """Feed game_info into network, get values for actions
        param: game_info: from unoengine
        """
        pState = self.affine1(pState)
        pState = self.dropout(pState)
        pState = F.relu(pState)
        pState = self.affine2(pState)
        pState = self.dropout(pState)
        pState = F.relu(pState)
        pState = self.affine3(pState)
        pState = F.relu(pState)
        Qs = self.last_affine(pState)
        return Qs

    def sampleAction(self, pState, legalActions):
        """Returns sample action from categorical distribution of NN output
        Before evaluation checks if there is a legal action except drawing.
        param: game_info: from unoengine
        """

        if hasToDraw(legalActions):
            return self.outputLength - 1
        probs = self.forward(pState)
        m = Categorical(probs)
        legalAction = False
        while not(legalAction):
            "possible endless loop?"
            action = m.sample()
            legalAction = isLegalAction(action, legalActions)

        # self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def greedyAction(self, pState, legalActions):
        """returns action with largest log probability
        Before evaluation checks if there is a legal action except drawing
        """

        if hasToDraw(legalActions):
            return self.outputLength - 1
        probs = self.forward(pState)
        m = Categorical(probs)
        logProbsArray = [[i, m.log_prob(torch.Tensor([i])).item()] for i in range(self.outputLength)]
        # sort by decresing log probability
        logProbsArray = sorted(logProbsArray, key = lambda tup: tup[1], reverse = True) 
        legalAction = False
        i = 0
        while not(legalAction):
            action = logProbsArray[i][0]
            legalAction = isLegalAction(action, legalActions)
            i += 1

        # self.saved_log_probs.append(m.log_prob(torch.scalar_tensor(action)))
        return action

    def learn(self, oldpState, action, reward, newpState, finalState = False):
        """Minimize loss by backpropagation
        targetQs = reward + gamma * newStateQs
        loss = smoothL1Loss(oldStateQs, targetQs)
        param: oldpState: torch.Tensor of old pState
        param: action
        param: reward: after action
        param: newpState: torch.Tensor, after action
        """
        oldStateQ = self.forward(oldpState)[action]
        newStateQ = self.forward(newpState).max()

        with torch.no_grad():
            if finalState:
                targetQ = torch.Tensor([reward])
            else:
                targetQ = torch.scalar_tensor(reward) + self.gamma * newStateQ
        targetQ = reward + self.gamma * newStateQ
        loss = self.criterion(oldStateQ, targetQ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def isLegalAction(action, legalActions):
    """Check if the action is legal.
    :param action: Int
    :param legalityMatrix
    :return: true/false
    """
    return legalActions[action]

def hasToDraw(legalActions):
    """Check if legal action except drawing exists
    :param legalityMatrix
    :return: true/false
    """
    if legalActions.sum() <= 1:  # can only draw
        return True
    else:
        return False






# inspiration:
# https://github.com/pytorch/examples/tree/master/reinforcement_learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

dropoutRate = 0.6
hiddenLayerSize = 128

class Policy(nn.Module):
    def __init__(self, inputLength, outputLength):
        """
        NO LEARNING IMPLEMENTED
        ToDO:
        - check if legal move
        - only output legal move
        - reinforcement learning
        Qnetwork for uno. Takes pState [0,1,0,0,2,...] as input and outputs probabilities (?).
        :param inputLength: Size of the state, every element of the vector means how many of this card the agent has
        :param outpuLength: Number of actions, i.e. number of different cards in the game
        """
        super(Policy, self).__init__()
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.affine1 = nn.Linear(self.inputLength, hiddenLayerSize)
        self.dropout = nn.Dropout(p = dropoutRate)
        self.affine2 = nn.Linear(128, self.outputLength)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, game_info):
        pState = game_info['p_state']
        pState = torch.Tensor(pState)
        pState = self.affine1(pState)
        pState = self.dropout(pState)
        pState = F.relu(pState)
        action_scores = self.affine2(pState)
        return F.softmax(action_scores, dim=0)

    def sampleAction(self, game_info):
        probs = self.forward(game_info)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def greedyAction(self, game_info):
        "returns action with largest log probability"
        probs = self.forward(game_info)
        m = Categorical(probs)
        logProbsArray = [m.log_prob(torch.Tensor([i])).item() for i in range(self.outputLength)]
        return np.argmax(logProbsArray)

def isLegalAction(action, legalActions):
    """
    Check if the action is legal.
    :param action: Int
    :param legalityMatrix
    :return: true/false
    """
    return legalActions[action]





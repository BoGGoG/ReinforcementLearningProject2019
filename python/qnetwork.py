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
    def __init__(self, inputLength, outputLength, discount = 0.95, explorationRate = 1.0, iterations = 10000):
        """NO LEARNING IMPLEMENTED
        ToDO:
        - check if legal move
        - only output legal move
        - reinforcement learning
        Qnetwork for uno. Takes pState [0,1,0,0,2,...] as input and outputs probabilities (?).
        :param inputLength: state dimension: all cards + 'draw'
        :param outpuLength: action dimension: all cards, open card, opponents hand cards =usually  inputLength + 1
        """
        super(Policy, self).__init__()
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.discount = discount; self.explorationRate = explorationRate; self.iterations = iterations
        self.affine1 = nn.Linear(self.inputLength, hiddenLayerSize)
        self.dropout = nn.Dropout(p = dropoutRate)
        self.affine2 = nn.Linear(hiddenLayerSize, self.outputLength)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, game_info):
        pState = game_info['p_state']
        pState = torch.Tensor(pState)
        pState = self.affine1(pState)
        pState = self.dropout(pState)
        pState = F.relu(pState)
        action_scores = self.affine2(pState)
        return F.softmax(action_scores, dim=-1)

    def sampleAction(self, game_info):
        """Returns sample action from categorical distribution of NN output
        Before evaluation checks if there is a legal action except drawing
        """

        if hasToDraw(game_info['legal_actions']):
            return self.outputLength - 1
        probs = self.forward(game_info)
        m = Categorical(probs)
        legalAction = False
        while not(legalAction):
            "possible endless loop?"
            action = m.sample()
            legalAction = isLegalAction(action, game_info['legal_actions'])

        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def greedyAction(self, game_info):
        """returns action with largest log probability
        Before evaluation checks if there is a legal action except drawing
        """

        if hasToDraw(game_info['legal_actions']):
            return self.outputLength - 1
        probs = self.forward(game_info)
        m = Categorical(probs)
        logProbsArray = [[i, m.log_prob(torch.Tensor([i])).item()] for i in range(self.outputLength)]
        # sort by decresing log probability
        logProbsArray = sorted(logProbsArray, key = lambda tup: tup[1], reverse = True) 
        legalAction = False
        i = 0
        while not(legalAction):
            action = logProbsArray[i][0]
            legalAction = isLegalAction(action, game_info['legal_actions'])
            i += 1

        return action

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






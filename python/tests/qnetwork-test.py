import torch
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from qnetwork import Policy, isLegalAction, hasToDraw
from arena import Arena, Agent
from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent

TEST_QNETWORK = True
TEST_AGENT = True
TEST_ARENA = True
TEST_GREEDY = True


"""
Those tests are not full unit tests, but just some production tests that should work
after every change.
We build a neural network with input size 50 and output size 49,
since there is also the open card and number of opponents hand cards in the input,
but only all possible cards (48) and 'draw' in the output.
The network gives us an output ('probabilities') and also the function
selectAction, which automatically selects the best action for a given input.
"""
if TEST_QNETWORK:
    print("---------------------")
    print("TEST Q_NETWORK")
    print("---------------------")
    unoengine = UnoEngine()
    reinforcementAgent = ReinforcementAgent(unoengine.get_action_dim())
    randomAgent = RandomAgent(unoengine.get_action_dim())
    arena = Arena(reinforcementAgent, randomAgent, unoengine)
    game_info = arena.get_game_info()
    action_dim = unoengine.get_action_dim() # all cards + 'draw'
    state_dim = action_dim + 1 # all cards, open card, opponents hand cards

    policy = Policy(state_dim, action_dim)
    output = policy(game_info)
    action = policy.sampleAction(game_info)
    print("input: {}".format(game_info))
    print("output: {}".format(output))
    print("sample action: {}".format(action))

    # test if no legal card works
    legalActions = game_info['legal_actions']
    onlyIllegalActions = np.zeros(action_dim, dtype = bool)
    onlyIllegalActions[-1] = True
    print(onlyIllegalActions)
    print(legalActions)
    print('Has to draw (True): ', hasToDraw(onlyIllegalActions))
    onlyIllegalActions[0] = True
    print('Has to draw (False): ', hasToDraw(onlyIllegalActions))



if TEST_ARENA:
    print("---------------------")
    print("TEST ARENA")
    print("---------------------")
    unoengine = UnoEngine()
    agent_0 = ReinforcementAgent(unoengine.get_action_dim())
    agent_1 = RandomAgent(unoengine.get_action_dim())
    arena = Arena(reinforcementAgent, randomAgent, unoengine)
    game_info = arena.get_game_info()
    for _ in range(5):
        lastAction = arena.step()
        print('Last action: ', lastAction)

if TEST_AGENT:
    print("---------------------")
    print("TEST AGENT")
    print("---------------------")
    unoengine = UnoEngine()
    reinforcementAgent = ReinforcementAgent(unoengine.get_action_dim())
    randomAgent = RandomAgent(unoengine.get_action_dim())
    arena = Arena(reinforcementAgent, randomAgent, unoengine)
    game_info = arena.get_game_info()
    digestAction = reinforcementAgent.digest(game_info)
    sampleAction = reinforcementAgent.sampleAction(game_info)
    greedyAction = reinforcementAgent.greedyAction(game_info)
    randomAction = reinforcementAgent.randomAction(game_info)
    print("ReinforcementAgent suggests action (digest): {}".format(digestAction))
    print("is legal: ", isLegalAction(digestAction, game_info['legal_actions']))
    print("ReinforcementAgent stochastic action: {}".format(sampleAction))
    print("is legal: ", isLegalAction(sampleAction, game_info['legal_actions']))
    print("ReinforcementAgent greedy action: {}".format(greedyAction))
    print("is legal:", isLegalAction(greedyAction, game_info['legal_actions']))
    print("ReinforcementAgent random action: {}".format(randomAction))
    print("is legal:", isLegalAction(randomAction, game_info['legal_actions']))



    





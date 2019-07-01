import torch
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from qnetwork import Policy
from arena import Arena, Agent
from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent

TEST_QNETWORK = True
TEST_AGENT = True
TEST_ARENA = True

"""
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
    policy = Policy(50, 49)
    inpt = np.random.choice([0,1,2], size = 50, p = [0.7, 0.2, 0.1])
    output = policy(inpt)
    action = policy.sampleAction(inpt)
    print("input: {}".format(inpt))
    print("output: {}".format(output))
    print("sample action: {}".format(action))

    print(policy.greedyAction(inpt))

if TEST_ARENA:
    print("---------------------")
    print("TEST ARENA")
    print("---------------------")
    unoengine = UnoEngine()
    agent_0 = ReinforcementAgent(unoengine.get_action_dim())
    agent_1 = RandomAgent(unoengine.get_action_dim())
    print(agent_0)
    print(agent_1)

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
    stochasticAction = reinforcementAgent.stochasticAction(game_info['p_state'])
    greedyAction = reinforcementAgent.greedyAction(game_info['p_state'])
    print("reinforcementAgent suggests action (digest): {}".format(digestAction))
    print("reinforcementAgent stochastic action: {}".format(stochasticAction))
    print("reinforcementAgent greedy action: {}".format(greedyAction))




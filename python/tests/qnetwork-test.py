import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from qnetwork import Policy, isLegalAction, hasToDraw
from arena import Arena, Agent
from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent

TEST_QNETWORK = False
TEST_AGENT = False
TEST_ARENA = False
TEST_GREEDY = False
TEST_TRAINING = False
TEST_SAVING = True


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

    pState = torch.Tensor(game_info['p_state'])
    legalActions = game_info['legal_actions']
    policy = Policy(state_dim, action_dim)
    output = policy(pState)
    action = policy.sampleAction(pState, legalActions)
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
    reinforcementAgent = ReinforcementAgent(unoengine.get_action_dim())
    randomAgent = RandomAgent(unoengine.get_action_dim())
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



if TEST_TRAINING:    
    print("---------------------")
    print("TEST TRAINING")
    print("---------------------")
    unoengine = UnoEngine()
    reinforcementAgent = ReinforcementAgent(unoengine.get_action_dim())
    randomAgent = RandomAgent(unoengine.get_action_dim())
    arena = Arena(reinforcementAgent, randomAgent, unoengine)
    gameInfo = arena.get_game_info()
    action_dim = unoengine.get_action_dim() # all cards + 'draw'
    state_dim = action_dim + 1 # all cards, open card, opponents hand cards
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    policy = Policy(state_dim, action_dim)
    # policy.apply(init_weights)
    oldGameInfo = gameInfo
    action = arena.step()
    gameInfo = arena.get_game_info()
    reward = gameInfo['reward']
    policy.learn(torch.Tensor(oldGameInfo["p_state"]), action,
            reward, torch.Tensor(gameInfo['p_state']))

if TEST_SAVING:
    print("---------------------")
    print("TEST SAVING")
    print("---------------------")
    unoengine = UnoEngine()
    reinforcementAgent = ReinforcementAgent(unoengine.get_action_dim())
    randomAgent = RandomAgent(unoengine.get_action_dim())
    arena = Arena(reinforcementAgent, randomAgent, unoengine)
    for _ in range(100):
        arena.step()
    print('first agent (some) parameters')
    modelParams = reinforcementAgent.policy.state_dict()
    print(modelParams['affine1.weight'][0])
    error = reinforcementAgent.saveModel('save/model.pwf')
    if not(error):
        print('model saved')
    reinforcementAgent2 = ReinforcementAgent(unoengine.get_action_dim())
    print('new agent (some) parameters')
    print(reinforcementAgent2.policy.state_dict()['affine1.weight'][0])
    print('games played before loading', reinforcementAgent2.gamesPlayed)
    reinforcementAgent2.loadModel('save/model.pwf')
    print('loading model')
    print('parameters after loading')
    print(reinforcementAgent2.policy.state_dict()['affine1.weight'][0])
    print('games played after loading', reinforcementAgent2.gamesPlayed)



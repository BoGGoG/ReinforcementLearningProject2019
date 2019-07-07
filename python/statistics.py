from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


numberOfGames = 10000
rollingMeanWindow = 200
modelSavePath = 'save/savedModel.pwf'
loadModel = False

# -------------------------------------
# SETUP
# -------------------------------------

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim())
if loadModel:
    agent_0.loadModel(modelSavePath)
# agent_0 = RandomAgent(unoengine.get_action_dim())
agent_1 = RandomAgent(unoengine.get_action_dim())
arena = Arena(agent_0, agent_1, unoengine)

gamesHistory = np.array([0, 0])
stepsPerGame = 0

def rollingMean(gamesHistory, windowSize = 100):
    # get only 0th element (RFL)
    gamesHistory = np.array(list(map(lambda row: row[0], gamesHistory)))
    rollingMeanHistory = [0.]
    for i in range(windowSize, gamesHistory.shape[0] + 1):
        windowMean = gamesHistory[i - windowSize:i].mean()
        rollingMeanHistory.append(windowMean)
    return(rollingMeanHistory)

# ----------------------------------------------------------------
# Play many games, train, collect data, plot
# ----------------------------------------------------------------
for i in tqdm(range(1, numberOfGames)):
    finished = False
    stepNumber = 0

    while not(finished):
        currentGame = np.empty_like([0,0])
        stepNumber += 1
        finished = (arena.agent_0_done and arena.agent_1_done)
        game_info = arena.get_game_info()
        game_over = game_info['game_over']
        player = game_info['turn']
        reward = game_info['reward']
        action = arena.step()
        if game_over:
            if reward == 100:
                currentGame[player] += 1

    gamesHistory = np.vstack((gamesHistory, currentGame))
    stepsPerGame += stepNumber
    if i+1 % 1000 == 0:
        print("{} episodes finished", i)
        print("epsilon: ", agent_0.epsilon)
                

agent_0.saveModel(modelSavePath)
stepsPerGame = stepsPerGame / numberOfGames
rollingMeanHistory = rollingMean(gamesHistory, rollingMeanWindow)
plt.plot(rollingMeanHistory)
plt.title("Games Won rolling mean ({})\ngamesPlayed = {}".format(rollingMeanWindow, agent_0.gamesPlayed))
plt.xlabel("games played")
plt.ylabel("win rate")
plt.show()

from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


# numberOfGames = 100000
numberOfGames = 10000
# numberOfGames = 15000
rollingMeanWindow = 4000
modelSavePath = 'save/savedModel.pwf'
oldModelSavePath = 'save/oldSavedModel.pwf'
# loadModel = False
# learning = True
loadModel = True
learning = False

# -------------------------------------
# SETUP
# -------------------------------------

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim())
if loadModel:
    agent_0.loadModel(modelSavePath)
# agent_0 = RandomAgent(unoengine.get_action_dim())
    agent_0.saveModel(oldModelSavePath)
agent_0.learning = learning
agent_1 = RandomAgent(unoengine.get_action_dim(), allowDraw = False)
arena = Arena(agent_0, agent_1, unoengine)

# gamesHistory = np.array([0, 0])
gamesHistory = np.zeros((numberOfGames, 2))
stepsPerGame = 0

print("Running statistics.py with loadModel = {} and learning = {}".format(loadModel, learning))

def rollingMean(gamesHistory, windowSize = 100):
    """calculate rolling mean of player 0
    :param gamesHistory: numpy array [[1,0],[0,1],...] of wins
    :param windowSize
    """
    gamesHistory = np.array(list(map(lambda row: row[0], gamesHistory)))
    rollingMeanHistory = np.empty(gamesHistory.shape[0] - windowSize + 1)
    for i in range(windowSize, gamesHistory.shape[0] + 1):
        windowMean = gamesHistory[i - windowSize:i].mean()
        rollingMeanHistory[i - windowSize] = windowMean
    return(rollingMeanHistory)

def totalMean(gamesHistory):
    gamesHistory = np.array(list(map(lambda row: row[0], gamesHistory)))
    return gamesHistory.mean()


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
                gamesHistory[i][player] += 1

    stepsPerGame += stepNumber
    if (i+1) % 1000 == 0:
        print("{} episodes finished".format(i+1))
        print("Q prediction std: {}; mean: {}".format(agent_0.Q_std, agent_0.Q_mean))
        print("avg win rate: {}".format(agent_0.avg_win_rate))
        print("avg Qs: {}".format(agent_0.Q_avgs))


agent_0.saveModel(modelSavePath)
stepsPerGame = stepsPerGame / numberOfGames
rollingMeanHistory = rollingMean(gamesHistory, rollingMeanWindow)
axes = plt.gca()
axes.set_ylim([0.3, 0.65])
plt.plot(rollingMeanHistory)
plt.axhline(y = 0.5, color = 'gray', linestyle = 'dashed')
plt.title("Games Won rolling mean ({}) while learning = {}\ngames played total: {} \nwin rate of last {} games: {}\nTotal win rate: {}:".format(rollingMeanWindow,
    learning, agent_0.gamesPlayed, rollingMeanWindow, rollingMeanHistory[-1], totalMean(gamesHistory)))
plt.xlabel("games played (+{})".format(rollingMeanWindow))
plt.ylabel("win rate")
plt.show()
plt.savefig('save/lastPlot.png')

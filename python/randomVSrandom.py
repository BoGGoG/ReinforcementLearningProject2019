"""Play random agent vs random agent in order to find out how much
influence going first has
"""
from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
from statisticsHelpers import rollingMean, totalMean
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


numberOfGames = 5000
rollingMeanWindow = 2000

# -------------------------------------
# SETUP
# -------------------------------------

unoengine = UnoEngine()
agent_0 = RandomAgent(unoengine.get_action_dim(), allowDraw = False)
agent_1 = RandomAgent(unoengine.get_action_dim(), allowDraw = True)
arena = Arena(agent_0, agent_1, unoengine)

gamesHistory = np.zeros((numberOfGames, 2))
stepsPerGame = 0

print("Running statistics.py with random agent vs random agent")

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


stepsPerGame = stepsPerGame / numberOfGames
rollingMeanHistory = rollingMean(gamesHistory, rollingMeanWindow)
axes = plt.gca()
axes.set_ylim([0.3, 1.0])
plt.plot(rollingMeanHistory)
plt.axhline(y = 0.5, color = 'gray', linestyle = 'dashed')
plt.title("Random vs Random, going first: rolling mean window: {} , Total win rate: {}:".format(rollingMeanWindow, totalMean(gamesHistory)))
plt.xlabel("games played (+{})".format(rollingMeanWindow))
plt.ylabel("win rate")
plt.show()
plt.savefig('save/lastPlot.png')

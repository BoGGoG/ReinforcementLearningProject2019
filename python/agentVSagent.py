"""
let two different reinforcement agents play against each other.
Let each one go first for a number of games.
"""

from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
from statisticsHelpers import rollingMean, totalMean
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

numberOfGames = 15000
rollingMeanWindow = 2000

model0SavePath = 'save/savedModel2.pwf'
model1SavePath = 'save/savedModel2.pwf'
# model1SavePath = 'save/savedModel2.pwf'
agent0Learning = False
agent1Learning = False

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim(), learning = agent0Learning)
agent_1 = ReinforcementAgent(unoengine.get_action_dim(), learning = agent1Learning)
agent_0.learning = agent0Learning
agent_1.learning = agent1Learning

arena = Arena(agent_0, agent_1, unoengine)

gamesHistory = np.zeros((numberOfGames, 2))

print("Running agentVSagent.py with agent0Learning = {}, agent1Learning = {}".format(agent0Learning, agent1Learning))

for i in tqdm(range(1, numberOfGames)):
    finished = False

    while not(finished):
        currentGame = np.empty_like([0,0])
        finished = (arena.agent_0_done and arena.agent_1_done)
        game_info = arena.get_game_info()
        game_over = game_info['game_over']
        player = game_info['turn']
        reward = game_info['reward']
        action = arena.step()
        if game_over:
            if reward == 100:
                gamesHistory[i][player] += 1

    # if (i+1) % 1000 == 0:
        # print("{} episodes finished".format(i+1))
        # print("Q prediction std: {}; mean: {}".format(agent_0.Q_std, agent_0.Q_mean))
        # print("avg win rate: {}".format(agent_0.avg_win_rate))
        # print("avg Qs: {}".format(agent_0.Q_avgs))

rollingMeanHistory = rollingMean(gamesHistory, rollingMeanWindow)
axes = plt.gca()
axes.set_ylim([0.0, 1.0])
plt.plot(rollingMeanHistory)
plt.axhline(y = 0.5, color = 'gray', linestyle = 'dashed')
plt.title("Rolling mean window = {}\nReinforcement Agent 0 (learning = {}) vs Reinforcement Agent 1 (learning = {})\nWin rate of agent 0: {}".format(
    rollingMeanWindow, agent0Learning, agent1Learning, totalMean(gamesHistory)))
plt.xlabel("games played (+{})".format(rollingMeanWindow))
plt.ylabel("win rate")
plt.show()
# plt.savefig('save/lastPlot.png')

from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


numberOfGames = 5000
# numberOfGames = 10000

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim())
# agent_0 = RandomAgent(unoengine.get_action_dim())
agent_1 = RandomAgent(unoengine.get_action_dim())
arena = Arena(agent_0, agent_1, unoengine)

gamesHistory = np.array([0,0])
stepsPerGame = 0

def rollingMean(gamesHistory, windowSize = 100):
    # get only 0th element (RFL)
    gamesHistory = np.array(list(map(lambda row: row[0], gamesHistory)))
    rollingMeanHistory = [0.]
    for i in range(windowSize, gamesHistory.shape[0] + 1):
        windowMean = gamesHistory[i - windowSize:i].mean()
        rollingMeanHistory.append(windowMean)
    return(rollingMeanHistory)

for i in tqdm(range(1, numberOfGames)):
    finished = False
    stepNumber = 0

    while not(finished):
        currentGame = np.array([0,0])
        action = arena.step()
        stepNumber += 1
        finished = (arena.agent_0_done and arena.agent_1_done)
        game_info = arena.get_game_info()
        game_over = game_info['game_over']
        player = game_info['turn']
        reward = game_info['reward']
        if game_over:
            if reward == 100:
                currentGame[player] += 1


    gamesHistory = np.vstack((gamesHistory, currentGame))
    stepsPerGame += stepNumber
    if i % 1001 == 0:
        print("{} episodes finished", i)
        print("epsilon: ", agent_0.epsilon)
                

stepsPerGame = stepsPerGame / numberOfGames
rollingMeanHistory = rollingMean(gamesHistory, 200)
plt.plot(rollingMeanHistory)
plt.title("Games Won rolling mean (200)")
plt.xlabel("games played")
plt.ylabel("win rate")
plt.show()

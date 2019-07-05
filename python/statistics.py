from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

numberOfGames = 2000
# numberOfGames = 10000

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim())
# agent_0 = RandomAgent(unoengine.get_action_dim())
agent_1 = RandomAgent(unoengine.get_action_dim())
arena = Arena(agent_0, agent_1, unoengine)

gamesWon = np.array(([0,0], [0,0]))
stepsPerGame = 0

for i in tqdm(range(1, numberOfGames)):
    finished = False
    stepNumber = 0
    gamesWonTemp = copy.copy(gamesWon[-1])

    while not(finished):
        action = arena.step()
        stepNumber += 1
        finished = (arena.agent_0_done and arena.agent_1_done)
        game_info = arena.get_game_info()
        game_over = game_info['game_over']
        player = game_info['turn']
        reward = game_info['reward']
        if game_over:
            if reward == 100:
                gamesWonTemp[player] += 1

    prev = gamesWon[-1]

    gamesWon = np.vstack((gamesWon, gamesWonTemp))
    stepsPerGame += stepNumber
    if i % 1000 == 0:
        print("games won RFA vs RA: ", gamesWonTemp)
        print("epsilon: ", agent_0.epsilon)
                

stepsPerGame = stepsPerGame / numberOfGames
gamesWon = list(map(lambda row: row / sum(row), gamesWon))
print("games won: ", gamesWon[-1])
print("average steps per game ", stepsPerGame)
plt.plot(gamesWon)
plt.title("Games Won")
plt.xlabel("games played")
plt.ylabel("win rate")
plt.legend(["Reinforcement Agent", "Random Agent"])
plt.show()

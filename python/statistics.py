from unoengine import UnoEngine
from agents import RandomAgent, ReinforcementAgent
from arena import Arena

numberOfGames = 100

unoengine = UnoEngine()
agent_0 = ReinforcementAgent(unoengine.get_action_dim())
agent_1 = RandomAgent(unoengine.get_action_dim())
arena = Arena(agent_0, agent_1, unoengine)

gamesWon = [0,0]
for i in range(numberOfGames):
    finished = False
    stepNumber = 0

    while not(finished):
        action = arena.step()
        stepNumber += 1
        finished = (arena.agent_0_done and arena.agent_1_done)
        game_info = arena.get_game_info()
        game_over = game_info['game_over']
        player = game_info['turn']
        reward = game_info['reward']
        if game_over:
            print("game over")
            print("ended after {} steps".format(stepNumber))
            output = 'Player {} finishes with a reward of {}'.format(player, reward)
            print(output)
            if reward == 100:
                gamesWon[player] += 1
                

print(gamesWon)

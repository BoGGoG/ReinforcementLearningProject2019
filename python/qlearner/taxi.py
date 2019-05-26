# https://gym.openai.com/envs/Taxi-v2/
# MK: Working, but no good results.
# Wrong dropoffs every time, I guess taxi is not
# well suited for qlearning

import gym
import numpy as np 
import time

# PARAMETERS
eta = .628
gamma = .9
epis = 10000
revList = [] # rewards per episode

env = gym.make('Taxi-v2')
Q = np.zeros([env.observation_space.n,env.action_space.n])


observation = env.reset()
# env.render()


# Q-learning
learning = True

print("Training")
if learning:
    for i in range(epis):
        s = env.reset()
        rewardAll = 0
        done = False
        j = 0
        for j in range(500):
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
            s1, reward, done, _ = env.step(a)
            Q[s, a] = Q[s, a] + eta * (reward + gamma * np.max(Q[s1, :]) - Q[s, a])
            rewardAll += reward
            s = s1
            if done == True:
                break
        revList.append(rewardAll)
        # env.render()
print("Training finished")

print("Q = ")
print(Q)
print("")
print("Testing:")
state = env.reset()


def removeLastPrint(n = 1):
    for _ in range(n):
        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

def testStep(state, n = 1):
    for i in range(n):
        if i > 0:
            removeLastPrint(9)
        env.render()
        bestAction = np.argmax(Q[state,:])
        print("State {}, should use action {}".format(state, bestAction))
        state, reward, done, _ = env.step(bestAction)
        time.sleep(0.5)

testStep(state, 22)




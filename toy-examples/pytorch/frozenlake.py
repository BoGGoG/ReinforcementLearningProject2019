import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# PARAMETERS
e = 0.1
lr = .03
y = .999
# numEpisodes = 2000
# numStepsPerEp = 99
numEpisodes = 2
numStepsPerEp = 1

env = gym.make('FrozenLake-v0')
env.reset()
env.render()
actionSpace = env.action_space.n
stateSpace = env.observation_space.n


# NOTES:
# {
    # newState, reward, done, info = env.step(action)
# }

def print_policy():
    policy = [agent(s).argmax(1)[0].detach().item() for s in range(stateSpace)]
    policy = np.asarray([actions[action] for action in policy])
    policy = policy.reshape((game.max_row, game.max_col))
    print("\n\n".join('\t'.join(line) for line in policy) + "\n")

class QNetwork(nn.Module):
    def __init__(self, stateSpace, actionSpace):
        super(QNetwork, self).__init__()
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace
        self.hiddenSize = stateSpace

        self.l1 = nn.Linear(in_features = self.stateSpace, out_features = self.hiddenSize)
        self.l2 = nn.Linear(in_features = self.hiddenSize, out_features = self.actionSpace)

    def forward(self,x):
        x = self.one_hot_encoding(x)
        out1 = torch.sigmoid(self.l1(x))
        return self.l2(out1)

    def one_hot_encoding(self, x):
        outTensor = torch.zeros([1, stateSpace])
        outTensor[0][x] = 1
        return outTensor
    
agent = QNetwork(stateSpace, actionSpace)

jList = []
rList = []

optimizer = optim.Adam(params = agent.parameters())
citerion = nn.SmoothL1Loss()
actions = [a for a in range(actionSpace)]

for episode in tqdm(range(numEpisodes)):
    state = env.reset()
    rAll = 0
    for step in range(numStepsPerEp):
        with torch.no_grad():
            action = agent(state).max(1)[1].view(1,1)

        if np.random.rand(1) < e:
            action[0][0] = np.random.randint(0,actionSpace - 1)
        env.step(actions[action])
        print(actions[action])

        

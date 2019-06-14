"""
Tutorial:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
input = torch.randn(1, 1, 32, 32)
print('flat features: {}'.format(net.num_flat_features(input)))
out = net(input)
net.zero_grad()
out.backward(torch.randn(1, 10))
print('output for random input: {}'.format(out))
print(net.forward(input))
print('gradient net.grad: {}'.format(out.grad))

# LOSS FUNCTION
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print('loss: {}'.format(loss))
print('grad(loss): {}'.format(loss.grad_fn))

# BACKPROPAGATION
net.zero_grad()
print('conv1.bias.grad before backward: {}'.format(net.conv1.bias.grad))

loss.backward()

print('conv1.bias.grad after backward: {}'.format(net.conv1.bias.grad))

# UPDATE THE WEIGHTS
# for param in net.parameters():
    # param.data.sub_(param.grad.data * learning_rate)
import torch.optim as optim

learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr = learning_rate)
optimizer.zero_grad()
optimizer.step()




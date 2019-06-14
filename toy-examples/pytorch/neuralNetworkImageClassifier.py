"""
Tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Using CIFAR10 training and test dataset as images
"""

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,
        transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
        shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,
        transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
        shuffle = True, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# SHOW SOME IMAGES
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import Net
from Net import Net
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# TRAIN THE NETWORK
epochs = 2
for epoch in range(epochs):
    print('epoch 1 / {}'.format(epochs))
    running_loss = 0.0
    for datanum, data in tqdm(enumerate(trainloader)):
        inputs, labels = data
        optimizer.zero_grad
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if datanum % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, datanum + 1, running_loss / 2000.))
            running_loss = 0.0
print("Finished Training")

import torch
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from qnetwork import Policy

"""
We build a neural network with input size 50 and output size 49,
since there is also the open card and number of opponents hand cards in the input,
but only all possible cards (48) and 'draw' in the output.
The network gives us an output ('probabilities') and also the function
selectAction, which automatically selects the best action for a given input.
"""
policy = Policy(50, 49)
inpt = np.random.choice([0,1,2], size = 50, p = [0.7, 0.2, 0.1])
output = policy(inpt)
print("input: {}".format(inpt))
print("output {}".format(output))

print(policy.selectAction(inpt))

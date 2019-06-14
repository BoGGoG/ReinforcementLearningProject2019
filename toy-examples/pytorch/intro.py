# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

from __future__ import print_function
import torch
x = torch.empty(5,3)
y = torch.rand(5,3)
z = torch.zeros(5,3, dtype = torch.long)
w = torch.tensor([5.3, 3])
v = torch.rand_like(x, dtype = torch.float)
print(v)
print(v.size())

a = torch.rand(5,4)
b = torch.rand(5,4)
c = a + b
print(c)

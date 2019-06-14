"""
Automatic Differentiation
========================================
Following [tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)  
compile this to pdf with pandoc if you want
"""


from __future__ import print_function
import torch

x = torch.ones(2, 2, requires_grad = True, dtype = torch.float)
y = x + 2
z = y * y * 3

v = torch.tensor([1,2], dtype = torch.float)
# out = torch.mm(v.T, z)
out = v@z@v
out2 = torch.matmul(v,z)
out2 = torch.matmul(out2, v)
out = z.mean()
out.backward()

# $out = \sum_{i,j} v_i v_j ((x_{i,j} + 2)(x_{i,j} + 2) * 3)$
# $d(out) / dx_{1,1} = ... = 3 * v_1 * v_1 * 2 * (x_{1,1} + 2) = 18$

print(x)
print(y)
print(z)
print(v)
print(out)
print(out2)
print(x.grad)

# z.backward(torch.tensor([[2,2],[2,2]], dtype = torch.float))
# equivalent to grad(z) * [[2,2],[2,2]]
# which I think should be \sum{k,l} grad(z)_{i,j,k,l} w_k w_l
# with w = [2,2], though I am not sure about the indices





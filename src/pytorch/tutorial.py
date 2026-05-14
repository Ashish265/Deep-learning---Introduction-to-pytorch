import torch

print("PyTorch version:",torch.__version__)

a = torch.tensor([7,4,3,2,6])

print(a)
print("a.dtype:", a.dtype)
print("a.type():", a.type())

b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0],dtype =torch.int32)
print(b)
print("b.dtype:", b.dtype)
print("b.type():", b.type())

c =torch.FloatTensor([1,2,3,4,5])
print(c)
print("c.dtype:", c.dtype)
print("c.type():", c.type())


print("size:",a.size())
print("shape:",a.shape)
print("ndim:",a.ndim)
print("element at index 0:",a[0])
print("elements from index 1 to 3:",a[1:4])

# view
d = a.view(5,1)
print(d)
print(d.ndim)

#numpy to tensor
import numpy as np
e = np.array([1,2,3,4,5])
f = torch.from_numpy(e)
print("tensor from numpy:",f)
print(f.dtype)

#tensor to numpy
g = f.numpy()
print("numpy array from tensor:",g)

#pandas to tensor
import pandas as pd
h = pd.Series([1,2,3,4,5])
i = torch.from_numpy(h.values)
print("tensor from pandas:",i)

#vector operations
j = torch.tensor([1,2,3])
k = torch.tensor([4,5,6])

print("addition:",j+k)
print("subtraction:",j-k)
print("element-wise multiplication:",j*k)
print("dot product:",torch.dot(j,k))
print("cross product:",torch.cross(j,k))
print("scalar multiplication:",2*j)
print("scalar addition:",j+2)
print("scalar subtraction:",j-2)
print("scalar division:",j/2)







import torch
a=torch.tensor([[0,1,1],[1,0,1]])

print(a)
print("a.dtype:", a.dtype)
print("a.type():", a.type())
print("a.shape:", a.shape)
print("a.ndim:", a.ndim)


X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]]) 
X_times_Y = X * Y
print(X_times_Y)


A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)
print(A_times_B)
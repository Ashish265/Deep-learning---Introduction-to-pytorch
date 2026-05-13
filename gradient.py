
import torch


x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)


x = torch.tensor(2.0, requires_grad=True)
y =x**2 + 2*x + 1
y.backward()
print(x.grad)


#complex function gradient
x = torch.tensor(1.0, requires_grad=True)
y = x**3 + 2*x**2 + 3*x + 4
y.backward()
print(x.grad)



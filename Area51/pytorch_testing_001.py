# Convolutions Neural Network with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Model
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=0.01)

# Training
nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # H(x)
    hypothesis = x_train * W + b

    # Cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Gradient Descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Print
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# Testing
print('H(x) = ', x_train * W + b)

# Gradient Descent with pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    # Define the data
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    losses = []

    # Train the model
    for epoch in range(100):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(f"Epoch: {epoch} Loss: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Test the model
    x_test = torch.tensor([[5.0]])
    y_test = model(x_test)
    print(f"y = {y_test.item()}")

    # Plot the loss and data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(losses)
    ax[0].set_title("Loss")
    ax[1].scatter(x, y)
    ax[1].plot(x, y_pred.detach().numpy(), color="red")
    ax[1].set_title("Data")
    plt.show()
    
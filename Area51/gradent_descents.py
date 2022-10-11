# Gradient descent computation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Create random beautiful dataset for linear regression with 500 samples and noise outliers
x = np.random.randn(100, 1)
y = 2 * x + 3 + np.random.randn(100, 1)
y[::10] = 10 * np.random.randn(10, 1)  # Add noise outliers
w, b, = 0, 0  # Initial weights and bias

# Hyper parameters
learning_rate = 0.01
epochs = 100

# Initialize weights
w = np.random.randn(1, 1)

# Initialize bias
b = np.random.randn(1, 1)

# Initialize loss
loss = np.mean((w * x + b - y) ** 2)

# Initialize loss stock
loss_stock = []


# Gradient descent
def gradient_descent(x, y, w, b, learning_rate):
    # Compute loss
    loss = np.mean((w * x + b - y) ** 2)  # MSE

    # Compute gradients
    dw = 2 * np.mean((w * x + b - y) * x)  # This is the derivative of MSE with respect to weights
    db = 2 * np.mean(w * x + b - y)  # This is the derivative of MSE with respect to bias

    # Update weights
    w = w - learning_rate * dw  # Update weights
    b = b - learning_rate * db  # Update bias

    return w, b, loss  # Return updated weights and loss


# Gradient descent iterations
for epoch in range(epochs):
    w, b, loss = gradient_descent(x, y, w, b, learning_rate)  # Update weights and loss
    print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss:.4f}')  # Print loss
    loss_stock.append(loss)  # Append loss to loss stock

# Plot loss
plt.plot(range(epochs), loss_stock)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot data and fitted line
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, w * x + b, label='Fitted line')
plt.legend()
plt.show()

# Create the test data
x_test = np.random.randn(100, 1)
y_test = 2 * x_test + 3 + np.random.randn(100, 1)

# Compute loss on test data
loss = np.mean((w * x_test + b - y_test) ** 2)
print(f'Loss on test data: {loss:.4f}')

# Compute accuracy on test data
accuracy = 1 - loss / np.mean(y_test ** 2)
print(f'Accuracy on test data: {accuracy:.4f}')

# Plot data and fitted line
plt.plot(x_test, y_test, 'ro', label='Original data')
plt.plot(x_test, w * x_test + b, label='Fitted line')
plt.legend()
plt.show()

# Maximum Likelihood Estimation

import numpy as np
import matplotlib.pyplot as plt

# Create the dataset (1000 samples)
X = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

# Plot the dataset
def plot_dataset(X, y):
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

plot_dataset(X, y)

# Calculate the mean and variance of the dataset
def mean(X):
    return sum(X) / len(X)

def variance(X):
    mu = mean(X)
    return sum((x - mu) ** 2 for x in X) / len(X)

# Calculate the covariance between X and y
def covariance(X, y):
    mu_x, mu_y = mean(X), mean(y)
    return sum((x - mu_x) * (y - mu_y) for x, y in zip(X, y)) / len(X)

# Calculate the coefficients
def coefficients(X, y):
    b1 = covariance(X, y) / variance(X)
    b0 = mean(y) - b1 * mean(X)
    return [b0, b1]

# Make predictions
def simple_linear_regression(X, y, x):
    b0, b1 = coefficients(X, y)
    return b0 + b1 * x

# Plot the regression line
def plot_regression_line(X, y, b):
    plt.scatter(X, y, color = "m", marker = "o", s = 30)
    y_pred = b[0] + b[1] * X
    plt.plot(X, y_pred, color = "g")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

b = coefficients(X, y)
print('Coefficients: B0 = {}, B1 = {}'.format(b[0], b[1]))
plot_regression_line(X, y, b)

# Calculate the root mean squared error
def rmse(y, y_pred):
    return np.sqrt(sum((y - y_pred) ** 2) / len(y))

# Calculate the coefficient of determination
def r2_score(y, y_pred):
    y_mean = np.mean(y)
    ss_tot = sum((y - y_mean) ** 2)
    ss_res = sum((y - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

# Make predictions
y_pred = [simple_linear_regression(X, y, x) for x in X]
rmse = rmse(y, y_pred)
r2 = r2_score(y, y_pred)
print('RMSE: ', rmse)
print('R2: ', r2)

# Plot the regression line
plot_regression_line(X, y, b)

# Plot the residuals
def plot_residuals(X, y, y_pred):
    plt.scatter(X, y - y_pred)
    plt.xlabel('X')
    plt.ylabel('Residuals')
    plt.show()

plot_residuals(X, y, y_pred)


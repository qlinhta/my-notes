# Logistic Regression from scratch

import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 10)
w = np.random.randn(10, 1)
b = np.random.randn(1, 1)


# K-fold cross validation
def k_fold_cross_validation(X, y, k, epochs, learning_rate):
    # Split the data into k folds
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)

    # Initialize the accuracy list
    accuracy = []

    # Loop through the folds
    for i in range(k):
        # Get the test data
        X_test = X_folds[i]
        y_test = y_folds[i]

        # Get the training data
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])

        # Train the model
        w, b = train(X_train, y_train, epochs, learning_rate)

        # Get the accuracy
        accuracy.append(get_accuracy(X_test, y_test, w, b))

    # Return the average accuracy
    return np.mean(accuracy)


# Train the model
def train(X, y, epochs, learning_rate):
    # Initialize the weights and bias
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1, 1)

    # Loop through the epochs
    for i in range(epochs):
        # Get the predictions
        y_pred = sigmoid(np.dot(X, w) + b)

        # Calculate the gradient
        w_grad = np.dot(X.T, (y_pred - y))
        b_grad = np.sum(y_pred - y)

        # Update the weights and bias
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad

    # Return the weights and bias
    return w, b


# Get the accuracy
def get_accuracy(X, y, w, b):
    # Get the predictions
    y_pred = sigmoid(np.dot(X, w) + b)

    # Get the predictions in the form of 0s and 1s
    y_pred = np.round(y_pred)

    # Get the accuracy
    accuracy = np.sum(y_pred == y) / len(y)

    # Return the accuracy
    return accuracy


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Evaluate the model
accuracy = k_fold_cross_validation(X, y, 5, 100, 0.1)
print("Accuracy:", accuracy)

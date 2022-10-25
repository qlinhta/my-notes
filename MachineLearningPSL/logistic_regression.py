# Logistric Regression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class LogisticRegression:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def main():
    X1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 500)
    X2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], 500)

    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(500), np.ones(500)))

    # Plot dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
    plt.show()

    # Train model
    regressor = LogisticRegression(lr=0.01, n_iters=1000)
    regressor.fit(X, y)
    predictions = regressor.predict(X)

    # print accuracy
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print("Logistic Regression classification accuracy", accuracy(y, predictions))

    # Create test dataset
    X1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 100)
    X2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], 100)

    X_test = np.vstack((X1, X2))
    y_test = np.hstack((np.zeros(50), np.ones(50)))

    # Test model
    predictions = regressor.predict(X_test)

    # Plot predictions and test dataset and decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap=plt.cm.Accent)
    x1 = np.amin(X_test[:, 0])
    x2 = np.amax(X_test[:, 0])
    x1, x2 = -4, 4
    w1, w2 = regressor.weights[0], regressor.weights[1]
    b = regressor.bias
    x = np.linspace(x1, x2, 100)
    y = - (w1 * x + b) / w2
    plt.plot(x, y, color='blue', linewidth=2)
    plt.show()


if __name__ == "__main__":
    main()

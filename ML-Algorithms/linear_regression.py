# Linear Regression Algorithm

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

        def __init__(self, learning_rate=0.01, n_iters=1000):
            self.lr = learning_rate
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
                y_predicted = np.dot(X, self.weights) + self.bias
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

        def predict(self, X):
            y_approximated = np.dot(X, self.weights) + self.bias
            return y_approximated

        def mean_squared_error(self, y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def r2_score(self, y_true, y_pred):
            u = ((y_true - y_pred) ** 2).sum()
            v = ((y_true - y_true.mean()) ** 2).sum()
            return 1 - u / v

        def plot(self, X, y):
            fig, ax = plt.subplots()
            ax.scatter(X, y, marker='o', s=30, color='r')
            ax.plot(X, self.predict(X), color='b')
            plt.show()


if __name__ == "__main__":
    # Testing the algorithm
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    regressor = LinearRegression()
    regressor.fit(X, y)
    predictions = regressor.predict(X)

    print("Mean Squared Error:", regressor.mean_squared_error(y, predictions))
    print("R2 Score:", regressor.r2_score(y, predictions))

    regressor.plot(X, y)
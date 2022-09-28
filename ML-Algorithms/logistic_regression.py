# Logistic Regression Algorithm

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    # Add intercept term to X
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    # Sigmoid Function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Loss Function
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # Fit the model
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if (self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    # Predict the output
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


if __name__ == '__main__':
    # Create the dataset (1000 samples)
    X = np.random.randn(1000, 3)
    y = np.random.randint(0, 2, 1000)

    # Fit the model
    model = LogisticRegression(learning_rate=0.01, num_iterations=300000)
    model.fit(X, y)

    # Predict the output
    preds = model.predict(X)
    print(preds)

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
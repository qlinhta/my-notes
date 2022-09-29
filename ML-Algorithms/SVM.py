# SVM algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def plot_data_classification(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])
        x1_1 = -(self.w[0] * x0_1 + self.b) / self.w[1]
        x1_2 = -(self.w[0] * x0_2 + self.b) / self.w[1]
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])
        plt.show()

    def plot_data_classification_margin(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])
        x1_1 = -(self.w[0] * x0_1 + self.b) / self.w[1]
        x1_2 = -(self.w[0] * x0_2 + self.b) / self.w[1]
        x1_1_m = -(self.w[0] * x0_1 + self.b - 1) / self.w[1]
        x1_2_m = -(self.w[0] * x0_2 + self.b - 1) / self.w[1]
        x1_1_p = -(self.w[0] * x0_1 + self.b + 1) / self.w[1]
        x1_2_p = -(self.w[0] * x0_2 + self.b + 1) / self.w[1]
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])
        plt.show()

    def plot_data_classification_margin_hyperplane(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])
        x1_1 = -(self.w[0] * x0_1 + self.b) / self.w[1]
        x1_2 = -(self.w[0] * x0_2 + self.b) / self.w[1]
        x1_1_m = -(self.w[0] * x0_1 + self.b - 1) / self.w[1]
        x1_2_m = -(self.w[0] * x0_2 + self.b - 1) / self.w[1]
        x1_1_p = -(self.w[0] * x0_1 + self.b + 1) / self.w[1]
        x1_2_p = -(self.w[0] * x0_2 + self.b + 1) / self.w[1]
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])
        plt.show()

    def plot_data_classification_margin_hyperplane_support_vectors(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])
        x1_1 = -(self.w[0] * x0_1 + self.b) / self.w[1]
        x1_2 = -(self.w[0] * x0_2 + self.b) / self.w[1]
        x1_1_m = -(self.w[0] * x0_1 + self.b - 1) / self.w[1]
        x1_2_m = -(self.w[0] * x0_2 + self.b - 1) / self.w[1]
        x1_1_p = -(self.w[0] * x0_1 + self.b + 1) / self.w[1]
        x1_2_p = -(self.w[0] * x0_2 + self.b + 1) / self.w[1]
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])
        plt.show()


if __name__ == '__main__':
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)
    clf = SVM()
    clf.fit(X, y)
    clf.plot_data_classification(X)
    clf.plot_data_classification_margin(X)
    clf.plot_data_classification_margin_hyperplane(X)
    clf.plot_data_classification_margin_hyperplane_support_vectors(X)

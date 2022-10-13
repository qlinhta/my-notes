# Ridge Regression and Lasso Regression Visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def main():
    # Generate data
    np.random.seed(1)
    n_samples = 30
    degrees = [1, 4, 15]

    X = np.sort(np.random.rand(n_samples))
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1
    X = X[:, np.newaxis]
    y = y[:, np.newaxis]

    # Ridge Regression
    plt.figure(figsize=(14, 5))
    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, len(degrees), i + 1)
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-2))
        model.fit(X, y)
        y_plot = model.predict(X)
        plt.scatter(X, y, s=10)
        plt.plot(X, y_plot, label="degree %d" % degree)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
    plt.show()

    # Lasso Regression
    plt.figure(figsize=(14, 5))
    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, len(degrees), i + 1)
        model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=1e-2))
        model.fit(X, y)
        y_plot = model.predict(X)
        plt.scatter(X, y, s=10)
        plt.plot(X, y_plot, label="degree %d" % degree)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.savefig("./images/ridge_and_lasso.png")
    plt.show()

if __name__ == '__main__':
    main()
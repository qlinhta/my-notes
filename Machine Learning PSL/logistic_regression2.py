# Estimate the maximum likelihood parameters of a logistic regression model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# Newton-Raphson algorithm
def newton_raphson(x_tr, y_tr, x_te, y_te):
    # Initialize the parameters
    w = np.zeros(x_tr.shape[1])

    # Compute the gradient and the Hessian
    grad = gradient(x_tr, y_tr, w)
    hess = hessian(x_tr, y_tr, w)

    # Newton-Rapson algorithm
    while np.linalg.norm(grad) > 1e-5:
        w = w - np.dot(np.linalg.inv(hess), grad)
        grad = gradient(x_tr, y_tr, w)
        hess = hessian(x_tr, y_tr, w)

    # Compute the accuracy
    y_pred = np.dot(x_te, w)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    acc = np.mean(y_pred == y_te)

    return w, acc


# Compute the gradient descent
def gradient(x, y, w):
    return np.dot(x.T, sigmoid(np.dot(x, w)) - y)


# Compute the Hessian matrix
def hessian(x, y, w):
    hess = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        hess += np.outer(x[i], x[i]) * sigmoid(np.dot(x[i], w)) * (1 - sigmoid(np.dot(x[i], w)))
    return hess


# Compute the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    # Create the dataset centered at (0, 0)
    n, d = 1000, 2

    y_tr = 2 * np.random.binomial(1, 0.5, n) - 1
    x_tr = np.zeros((n, d))
    x1_tr = np.random.randn((y_tr == -1).sum(), d) + np.array([100, 0])
    x2_tr = np.random.randn((y_tr == 1).sum(), d) + np.array([103, 0])
    x_tr[y_tr == -1, :2] = x1_tr
    x_tr[y_tr == 1, :2] = x2_tr

    y_te = 2 * np.random.binomial(1, 0.5, n) - 1
    x_te = np.zeros((n, d))
    x1_te = np.random.randn((y_te == -1).sum(), d) + np.array([100, 0])
    x2_te = np.random.randn((y_te == 1).sum(), d) + np.array([103, 0])
    x_te[y_te == -1, :2] = x1_te
    x_te[y_te == 1, :2] = x2_te

    # Adding a dummy dimension for dealing with the shifts.
    ones_tr = np.ones((x_tr.shape[0], 1))
    ones_te = np.ones((x_te.shape[0], 1))

    x_tr_d = np.concatenate([x_tr, ones_tr], axis=1)
    x_te_d = np.concatenate([x_te, ones_te], axis=1)


    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr, cmap=plt.cm.Accent)
    plt.show()

    '''
    # Estimate the parameters
    w, acc = newton_raphson(x_tr_d, y_tr, x_te_d, y_te)
    print("Accuracy: {:.2f}".format(acc))

    # predict the test set with the estimated parameters
    y_pred = np.dot(x_te_d, w)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(x_te[:, 0], x_te[:, 1], c=y_pred, cmap=plt.cm.Accent)
    plt.show()
    '''
    # Estimate a:
    a_hat = np.linalg.solve(x_tr_d.T @ x_tr_d,
                            x_tr_d.T @ y_tr)

    # Predict y_te:
    y_te_hat = np.sign(x_te_d @ a_hat)

    # Plot the predictions:
    plt.figure(figsize=(8, 6))
    plt.scatter(x_te[:, 0], x_te[:, 1], c=y_te_hat, cmap=plt.cm.Accent)
    plt.show()


if __name__ == "__main__":
    main()

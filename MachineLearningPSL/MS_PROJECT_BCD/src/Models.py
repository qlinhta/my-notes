import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scipy.stats import multivariate_normal
from numpy.linalg import det, inv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from src import metrics


def fit_and_predict_all(X_train, X_test, y_train, y_test, verbose=False):
    lr = LogisticRegression(learning_rate=5, max_iter=1000, verbose=verbose)
    # lda = LinearDiscriminantAnalysis()
    lda = LDA()
    nn = NeuralNet(hidden_size=100, num_classes=2, num_epochs=100, batch_size=16, learning_rate=0.1)
    lsvm = LinearSVC(C=1, max_iter=100, random_state=42, verbose=verbose)
    ridge = RidgeClassifier(alpha=0.1, random_state=42)
    xgboost = XGBClassifier(learning_rate=0.1, max_depth=13, n_estimators=500, verbose=verbose)
    models = {
        'Logistic Regression': lr,
        'LDA': lda,
        'Neural Network': nn,
        'Linear SVM': lsvm,
        'Ridge': ridge,
        'XGBoost': xgboost,
    }
    y_preds = {}
    for model in models.values():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_preds[model.__class__.__name__] = y_pred
        accuracy = metrics.accuracy(y_test, y_pred)
        if verbose:
            print(f'{model.__class__.__name__} accuracy: {accuracy:.4f}')
        if not os.path.exists('models'):
            os.makedirs('models')
        if model.__class__.__name__ == 'NeuralNet':
            torch.save(model.state_dict(), f'models/{model.__class__.__name__}.pth')
        else:
            from joblib import dump
            dump(model, f'models/{model.__class__.__name__}.joblib')
    print('Done')
    return models, y_preds


def _tuning_lr(X, y, learning_rates, max_iters, k=10, verbose=True):
    assert len(X) == len(y), "Need to have same number of samples for X and y"
    assert k > 0, "k needs to be positive"
    assert k < len(X), "k needs to be less than number of samples"
    assert k == int(k), "k needs to be an integer"
    assert len(learning_rates) > 0, "Need to have at least one learning rate"
    assert len(max_iters) > 0, "Need to have at least one max iter"
    best_accuracy = 0
    best_learning_rate = None
    best_max_iter = None
    for learning_rate in learning_rates:
        for max_iter in max_iters:
            model = LogisticRegression(learning_rate, max_iter, verbose=False, random_state=42)
            accuracy_list = model.cross_validation(X, y, k)
            accuracy = np.mean(accuracy_list)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = learning_rate
                best_max_iter = max_iter
            if verbose:
                print(
                    f'Learning rate: {learning_rate}, max iter: {max_iter}, accuracy: {accuracy:.4f}')
    return best_learning_rate, best_max_iter, best_accuracy


class LogisticRegression:
    def __init__(self, learning_rate, max_iter, verbose=False, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.weights = None
        self.bias = None

        self.losses = []
        self.accuracies = []
        self.weights_list = []
        self.bias_list = []
        self.coef_ = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _losses(y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        np.random.seed(self.random_state)  # Set the random seed

        if self.weights is None:
            self.weights = np.random.randn(X.shape[1])
        if self.bias is None:
            self.bias = np.random.randn()

        # Start training
        for i in range(self.max_iter):
            # Forward propagation
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)

            # Compute the loss
            self.losses.append(self._losses(y, y_pred))
            # Compute the accuracy
            self.accuracies.append(metrics.accuracy(y, self.predict(X)))

            # Backward propagation
            dw = np.dot(X.T, (y_pred - y)) / y.shape[0]
            db = np.sum(y_pred - y) / y.shape[0]

            # Update the weights and the bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Update coef_
            self.coef_ = np.append(self.bias, self.weights)

            # Print the loss and the accuracy
            if self.verbose:
                print(
                    f'Iteration: {i + 1}/{self.max_iter}, loss: {self.losses[-1]:.4f}, accuracy: {self.accuracies[-1]:.4f}')

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model has not been trained yet")
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(y_pred)

    def predict_proba(self, X):
        if self.weights is None:
            raise Exception("Model has not been trained yet")
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(np.array([1 - y_pred, y_pred]).T, 2)

    def cross_validation(self, X, y, n_splits=10):
        # Split the dataset into n_splits
        X_split = np.array_split(X, n_splits)
        y_split = np.array_split(y, n_splits)
        accuracy_list = []
        # Start the cross validation
        for i in range(n_splits):
            # Get the test set
            X_test = X_split[i]
            y_test = y_split[i]
            # Get the train set
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            # Train the model
            self.fit(X_train, y_train)
            # Get the accuracy
            accuracy = metrics.accuracy(y_test, self.predict(X_test))
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)


class LinearDiscriminantAnalysis(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None

    def fit(self, X_train, y_train):
        self.mu = np.zeros((X_train.shape[1], len(np.unique(y_train))))
        self.sigma = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i, label in enumerate(np.unique(y_train)):
            X_train_label = X_train[y_train == label]
            self.mu[:, i] = np.mean(X_train_label, axis=0)
            self.sigma += np.cov(X_train_label.T)
        self.sigma /= len(np.unique(y_train))
        self.w = np.dot(np.linalg.inv(self.sigma), self.mu)
        self.b = -0.5 * np.sum(np.dot(self.mu.T, np.dot(np.linalg.inv(self.sigma), self.mu)), axis=1)
        if self.verbose:
            print('Weight Vector: {}'.format(self.w))
            print('Bias: {}'.format(self.b))

    def predict(self, X_test):
        return np.argmax(np.dot(X_test, self.w) + self.b, axis=1)

    def score(self, X_test, y_test):
        return np.sum(self.predict(X_test) == y_test) / len(y_test)

    def predict_proba(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def predict_log_proba(self, X_test):
        return np.log(self.predict_proba(X_test))

    def cross_validation(self, X_train, y_train, k=10):
        cross_validation = []
        avg = 0
        # Split the data into k folds
        X_train = np.array_split(X_train, k)
        y_train = np.array_split(y_train, k)
        # Cross validation
        for i in range(k):
            X_train_ = np.concatenate(X_train[:i] + X_train[i + 1:])
            y_train_ = np.concatenate(y_train[:i] + y_train[i + 1:])
            X_test_ = X_train[i]
            y_test_ = y_train[i]
            self.fit(X_train_, y_train_)
            print('Fold {}: {}'.format(i, self.score(X_test_, y_test_)))
            cross_validation.append(self.score(X_test_, y_test_))
            avg += self.score(X_test_, y_test_)
        print('Average: {}'.format(avg / k))
        return cross_validation


class RidgeRegressionClassifier(object):
    def __init__(self, alpha=1.0, verbose=False):
        self.alpha = alpha
        self.verbose = verbose
        self.model = RidgeClassifier(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def cross_validation(self, X, y, n_splits=10):
        # Split the dataset into n_splits
        X_split = np.array_split(X, n_splits)
        y_split = np.array_split(y, n_splits)
        accuracy_list = []
        # Start the cross validation
        for i in range(n_splits):
            # Get the test set
            X_test = X_split[i]
            y_test = y_split[i]
            # Get the train set
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            # Train the model
            self.fit(X_train, y_train)
            # Get the accuracy
            accuracy = metrics.accuracy(y_test, self.predict(X_test))
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)


class XGBoostClassifier(object):
    def __init__(self, learning_rate=0.1, max_depth=7, n_estimators=100, verbose=False):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                   verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def cross_validation(self, X, y, n_splits=10):
        X_split = np.array_split(X, n_splits)
        y_split = np.array_split(y, n_splits)
        accuracy_list = []
        for i in range(n_splits):
            X_test = X_split[i]
            y_test = y_split[i]
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            self.fit(X_train, y_train)
            accuracy = metrics.accuracy(y_test, self.predict(X_test))
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)


class LinearSVM(object):
    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter
        self.model = LinearSVC(C=C, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.decision_function(X)

    def cross_validation(self, X, y, n_splits=10):
        X_split = np.array_split(X, n_splits)
        y_split = np.array_split(y, n_splits)
        accuracy_list = []
        for i in range(n_splits):
            X_test = X_split[i]
            y_test = y_split[i]
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            self.fit(X_train, y_train)
            accuracy = metrics.accuracy(y_test, self.predict(X_test))
            accuracy_list.append(accuracy)
        return np.mean(accuracy_list)


class NeuralNet(nn.Module):
    def __init__(self, hidden_size, num_classes, num_epochs, batch_size, learning_rate, verbose=False):
        super(NeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.fc1 = nn.Linear(30, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, X_train, y_train):
        # Convert the data to tensor
        X_train = torch.from_numpy(X_train.values).float()
        y_train = torch.from_numpy(y_train.values).long()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for i in range(0, X_train.shape[0], self.batch_size):
                # Forward pass
                outputs = self(X_train[i:i + self.batch_size])
                loss = criterion(outputs, y_train[i:i + self.batch_size])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose and (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))

    def predict(self, X_test):
        # Convert the data to tensor
        X_test = torch.from_numpy(X_test.values).float()
        # Test the model
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X_test):
        # Convert the data to tensor
        X_test = torch.from_numpy(X_test.values).float()
        # Test the model
        with torch.no_grad():
            outputs = self(X_test)
        return outputs.numpy()


class LDA:

    def __init__(self):
        self.n_classes = 2
        self.coefficient = np.zeros((self.n_classes,))
        self.intercept = 0.0
        self.priors = []

    def mean(self, X, y):
        classes = np.unique(y)
        means = np.zeros((classes.shape[0], X.shape[1]))
        for i in range(classes.shape[0]):
            means[i, :] = np.mean(X[y == i], axis=0)
        return means

    def prob_k(self, X, y):
        classes = np.unique(y)
        pi_k = np.zeros((len(classes),))
        for c in classes:
            pi_k[c] = len(y[y == c]) / len(y)
        return pi_k

    def general_cov(self, X, y, ):
        classes = np.unique(y)
        sigma = np.zeros((X.shape[1], X.shape[1]))
        for c in classes:
            sigma = sigma + len(X[y == c]) * np.cov(X[y == c].T)
        sigma = sigma / X.shape[0]
        return sigma

    def fit(self, X, y):
        means_overall = self.mean(X, y)
        pi_overall = self.prob_k(X, y)
        sigma_inv = np.linalg.inv(self.general_cov(X, y))
        self.coefficient = sigma_inv @ (means_overall[1] - means_overall[0])
        p = (means_overall[1] - means_overall[0]) @ sigma_inv @ (means_overall[0] + means_overall[1])
        self.intercept = (-0.5 * p - np.log(pi_overall[0] / pi_overall[1]))

    def decision_boundary(self, X):  # x.Tw + b = 0
        return X @ self.coefficient.T + self.intercept

    def predict(self, X):
        scores = self.decision_boundary(X)
        y_predicted = [1 if i > 0 else 0 for i in scores]
        return np.array(y_predicted)

    def cross_validation_lda(self, X, y, k):
        X_folds = np.array_split(X, k)
        y_folds = np.array_split(y, k)
        model = LDA()
        accuracies = []
        for i in range(k):
            # Get the training data
            X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
            y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
            X_val = X_folds[i]
            y_val = y_folds[i]
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_val)
            accuracies.append(metrics.accuracy(y_val, y_predicted))
        return np.mean(accuracies)

    def predict_proba_to_plot(self, X):
        y_pred = self.predict(X)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = (y_pred == 1).astype(int)
        proba[:, 0] = (y_pred == 0).astype(int)
        return np.round(proba)

    def set_priors(self, X, y):
        for i in range(2):
            self.priors.append(len(y[y == i]) / len(y))

    def normal_multivariate(self, X, mean, cov):
        n = mean.shape[0]
        cov_inv = inv(cov)
        cov_det = det(cov)
        f = -0.5 * ((X - mean) @ cov_inv) * (X - mean)
        nominator = np.exp(f.sum(axis=1))
        denominator = (2 * np.pi) ** (n / 2) * cov_det ** 0.5
        probas = nominator / denominator
        return probas

    def predict_proba(self, X, mean, cov0, cov1, priors):
        probability = np.zeros((X.shape[0], self.n_classes))
        for k in range(self.n_classes):
            mean_k = mean[k]
            if (k == 0):
                cov_k = cov0
            else:
                cov_k = cov1
            p_k = priors[k]
            probability[:, k] = p_k * multivariate_normal.pdf(X, mean_k, cov_k)
        y_predicted = [1 if probability[i][1] > probability[i][0] else 0 for i, row in enumerate(probability)]
        return np.array(y_predicted)

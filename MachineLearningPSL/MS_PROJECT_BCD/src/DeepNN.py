from scipy.linalg import svd
import dalex as dx
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lime.lime_tabular
import lime.lime_image
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import shap

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)

warnings.filterwarnings('ignore')
from src import metrics
from sklearn.model_selection import train_test_split, KFold

"""
Deep neural network classifies
Create a deep neural network to classify the data
"""


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


if __name__ == '__main__':
    df = pd.read_csv('../dataset/breast-cancer-wisconsin-processed.csv')
    label = pd.read_csv('../dataset/breast-cancer-wisconsin-processed-label.csv')
    data = pd.concat([df, label], axis=1)

    # Split the data with stratified sampling
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Find the best hyperparameters
    """
    best_model = None
    best_acc = 0
    for hidden_size in [10, 50, 100, 200]:
        for num_epochs in [100, 500, 1000]:
            for batch_size in [16, 32, 64]:
                for learning_rate in [0.1, 0.5, 1, 5]:
                    model = NeuralNet(hidden_size=hidden_size, num_classes=2, num_epochs=num_epochs,
                                      batch_size=batch_size, learning_rate=learning_rate, verbose=True)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = metrics.accuracy(y_test, y_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                        print('hidden_size: {}, num_epochs: {}, batch_size: {}, learning_rate: {}, acc: {}'.format(
                            hidden_size, num_epochs, batch_size, learning_rate, acc))
    print('Best accuracy: {:.2f}'.format(best_acc))
    # Train the best model
    model = best_model
    """
    model = NeuralNet(hidden_size=100, num_classes=2, num_epochs=100, batch_size=16, learning_rate=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # metrics
    print('Accuracy: {:.2f}'.format(metrics.accuracy(y_test, y_pred)))
    print('Precision: {:.2f}'.format(metrics.precision(y_test, y_pred)))
    print('Recall: {:.2f}'.format(metrics.recall(y_test, y_pred)))
    print('F1: {:.2f}'.format(metrics.f1_score(y_test, y_pred)))

    metrics.classification_summary(y_test, y_pred)
    metrics.roc_curve(y_test, y_pred)
    metrics.precision_recall_curve(y_test, y_pred)

    plt.subplots(figsize=(8, 8))
    plt.title('Predicted Labels')
    plt.scatter(X_test[y_pred == 0]['smoothness_mean_log'], X_test[y_pred == 0]['texture_mean_log'], marker='o',
                label='Benign', s=100, edgecolors='blue', facecolors='white')
    plt.scatter(X_test[y_pred == 1]['smoothness_mean_log'], X_test[y_pred == 1]['texture_mean_log'], marker='v',
                label='Malignant', s=100, edgecolors='red', facecolors='red')
    plt.scatter(X_test[y_pred != y_test]['smoothness_mean_log'], X_test[y_pred != y_test]['texture_mean_log'],
                marker='x',
                label='Misclassified', s=100, edgecolors='black', facecolors='black')
    plt.xlabel('Log Scale of Smoothness Mean')
    plt.ylabel('Log Scale of Texture Mean')
    plt.legend()
    plt.show()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = NeuralNet(hidden_size=100, num_classes=2, num_epochs=100, batch_size=16, learning_rate=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(metrics.accuracy(y_test, y_pred))
    print('Cross validation accuracy: {:.2f}'.format(np.mean(accuracies)))



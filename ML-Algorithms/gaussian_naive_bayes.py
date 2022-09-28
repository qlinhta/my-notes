# Gaussian Naive Bayes Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Create a random dataset
def create_dataset(n_samples, n_features, n_classes):
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(n_classes, size=n_samples)
    return X, y


# Create a dataset
X, y = create_dataset(1000, 10, 2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = gnb.predict(X_test)

# Calculate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Output:
# Accuracy: 0.52

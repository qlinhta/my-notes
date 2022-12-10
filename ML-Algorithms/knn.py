# K Nearest Neighbors Algorithm

import numpy as np
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute distances
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def plot(self):
        plt.figure()
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, s=15, cmap='viridis')
        plt.show()


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from collections import Counter

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", clf.accuracy(y_test, predictions))

    clf.plot()

    # KNN classification accuracy 0.9473684210526315
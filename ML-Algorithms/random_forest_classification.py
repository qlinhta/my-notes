# Random Forest Classification Algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=5, min_size=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_size=self.min_size)
            sample = self._get_bootstrap(X, y)
            tree.fit(sample[0], sample[1])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        y_pred = [max(set(row), key=list(row).count) for row in predictions]
        return y_pred

    def _get_bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_size=2):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (n_labels == 1) or (n_samples < self.min_size) or (depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return Node(leaf_value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)
        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature_idx, threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = self._gini(y)

        left_idxs, right_idxs = self._split(X_column, split_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy

        return ig

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        _, counts = np.unique(y, return_counts=True)
        return np.argmax(counts)

    def _predict(self, inputs):
        node = self.tree

        while node.leaf_value is None:
            if inputs[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.leaf_value


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, leaf_value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value = leaf_value

    def __repr__(self):
        if self.leaf_value is not None:
            return f"Node(leaf_value={self.leaf_value})"
        return f"Node(feature_idx={self.feature_idx}, threshold={self.threshold})"


# Create dataset and split into train and test sets
def create_dataset(n_samples=1000, n_features=10, n_informative=5, n_classes=2):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=1234,
    )
    return X, y


X, y = create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train and evaluate model
model = RandomForestClassifier(n_trees=3, max_depth=5, min_size=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.3f}")


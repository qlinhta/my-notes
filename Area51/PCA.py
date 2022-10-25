'''
PCA Algorithm for Dimensionality Reduction
'''

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        # Compute the covariance matrix
        X = X - X.mean(axis=0)
        cov = np.cov(X.T)
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Sort the eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Store the first n eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project the data onto the principal components
        X = X - X.mean(axis=0)
        return np.dot(X, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # Project the data back onto the original space
        return np.dot(X, self.components.T) + X.mean(axis=0)

    def plot(self, X, y):
        # Plot the projected data
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()


if __name__ == '__main__':
    # Create dataset for PCA test
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)

    # Create PCA object
    pca = PCA(n_components=2)

    # Fit and transform the data
    X_pca = pca.fit_transform(X)

    # Plot the projected data
    pca.plot(X_pca, y)

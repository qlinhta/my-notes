import matplotlib.pyplot as plt
import numpy as np

class PCA():
    def __init__(self, data, n_components):
        self.data = data
        self.n_components = n_components
        self.mean = np.mean(self.data, axis=0)
        self.cov = np.cov(self.data.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov)
        self.eigenvectors = self.eigenvectors.T
        self.idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[self.idx]
        self.eigenvectors = self.eigenvectors[self.idx]

    def transform(self):
        self.data = self.data - self.mean
        self.components = self.eigenvectors[0:self.n_components]
        return np.dot(self.data, self.components.T)

    def inverse_transform(self, components):
        return np.dot(components, self.components) + self.mean

    def plot(self):
        plt.plot(self.data[:,0], self.data[:,1], 'bo')
        plt.plot(self.mean[0], self.mean[1], 'ro')
        for vector in self.eigenvectors:
            start, end = self.mean, self.mean + vector
            plt.arrow(*start, *(end-start), color='r', width=0.01, head_width=0.1)
        plt.show()

if __name__ == '__main__':
    # Generate data for PCA
    data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)
    data = np.vstack((data, np.random.multivariate_normal([5,5], [[1,0],[0,1]], 100)))
    data = np.vstack((data, np.random.multivariate_normal([10,10], [[1,0],[0,1]], 100)))
    data = np.vstack((data, np.random.multivariate_normal([15,15], [[1,0],[0,1]], 100)))

    # Plot data
    plt.plot(data[:,0], data[:,1], 'bo')
    plt.show()

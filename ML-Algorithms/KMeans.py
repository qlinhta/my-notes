# KMeans Algorithm

import numpy as np
import matplotlib.pyplot as plt

# Generate random data
def generate_data():
    data = np.random.rand(100, 2)
    return data

# Calculate Euclidean distance
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

# Calculate the mean of the data
def mean(data):
    return np.mean(data, axis=0)

# Calculate the distance between each point and the mean
def distance_from_mean(data, mean):
    return np.array([euclidean_distance(x, mean) for x in data])


# KMeans Algorithm
def kmeans(data, k):
    # Randomly select k points as the initial centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    # Calculate the distance between each point and the mean
    distances = np.array([distance_from_mean(data, mean) for mean in centroids])
    # Find the minimum distance and assign the point to the cluster
    clusters = np.argmin(distances, axis=0)
    # Calculate the mean of each cluster
    means = np.array([mean(data[clusters == i]) for i in range(k)])
    # Repeat until convergence
    while not np.array_equal(centroids, means):
        centroids = means
        distances = np.array([distance_from_mean(data, mean) for mean in centroids])
        clusters = np.argmin(distances, axis=0)
        means = np.array([mean(data[clusters == i]) for i in range(k)])
    return clusters, means

# Plot the data
def plot_data(data, clusters, means):
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(means[:, 0], means[:, 1], c='red')
    plt.show()

if __name__ == '__main__':
    data = generate_data()
    clusters, means = kmeans(data, 3)
    plot_data(data, clusters, means)


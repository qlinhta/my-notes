import numpy as np
import matplotlib.pyplot as plt

# Create data for Gaussian Mixture Clustering
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# Plot the data
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Create the Gaussian Mixture Clustering model step by step without sklearn
# Step 1: Initialize the parameters
# Step 2: Calculate the distance between the data points and the mean of the clusters
# Step 3: Assign the data points to the clusters
# Step 4: Update the mean of the clusters
# Step 5: Repeat steps 2-4 until convergence

# Step 1: Initialize the parameters
# Initialize the mean of the clusters
mean_clusters = np.array([[0, 0], [1, 1], [2, 2]])
# Initialize the covariance matrix of the clusters
cov_clusters = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
# Initialize the probability of the clusters
prob_clusters = np.array([1 / 3, 1 / 3, 1 / 3])


# Step 2: Calculate the distance between the data points and the mean of the clusters
# Calculate the distance between the data points and the mean of the clusters
def distance(X, mean_clusters):
    # Initialize the distance matrix
    distance_matrix = np.zeros((X.shape[0], mean_clusters.shape[0]))
    # Calculate the distance between the data points and the mean of the clusters
    for i in range(X.shape[0]):
        for j in range(mean_clusters.shape[0]):
            distance_matrix[i][j] = np.linalg.norm(X[i] - mean_clusters[j])
    return distance_matrix


# Step 3: Assign the data points to the clusters
# Assign the data points to the clusters
def assign_clusters(distance_matrix):
    # Initialize the cluster matrix
    cluster_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
    # Assign the data points to the clusters
    for i in range(distance_matrix.shape[0]):
        cluster_matrix[i][np.argmin(distance_matrix[i])] = 1
    return cluster_matrix


# Step 4: Update the mean of the clusters
# Update the mean of the clusters
def update_mean(X, cluster_matrix):
    # Initialize the mean of the clusters
    mean_clusters = np.zeros((cluster_matrix.shape[1], X.shape[1]))
    # Update the mean of the clusters
    for i in range(cluster_matrix.shape[1]):
        mean_clusters[i] = np.dot(cluster_matrix[:, i], X) / np.sum(cluster_matrix[:, i])
    return mean_clusters


# Step 5: Repeat steps 2-4 until convergence
# Repeat steps 2-4 until convergence
def repeat_steps(X, mean_clusters, cov_clusters, prob_clusters):
    # Initialize the distance matrix
    distance_matrix = distance(X, mean_clusters)
    # Initialize the cluster matrix
    cluster_matrix = assign_clusters(distance_matrix)
    # Initialize the mean of the clusters
    mean_clusters = update_mean(X, cluster_matrix)
    return mean_clusters, cluster_matrix


# Repeat steps 2-4 until convergence
mean_clusters, cluster_matrix = repeat_steps(X, mean_clusters, cov_clusters, prob_clusters)

# Plot the data
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=cluster_matrix[:, 0], cmap='viridis')
ax.scatter(X[:, 0], X[:, 1], c=cluster_matrix[:, 1], cmap='viridis')
ax.scatter(X[:, 0], X[:, 1], c=cluster_matrix[:, 2], cmap='viridis')
ax.scatter(mean_clusters[:, 0], mean_clusters[:, 1], c='blue', s=200, alpha=0.5)
plt.show()

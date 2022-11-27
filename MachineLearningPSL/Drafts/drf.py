import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA


def make_gaussian_blobs(means, covs, Ns=100, seed=42):
    ''' Create different blobs, each blob being a gaussian
        from given mean and covariance
    '''
    rs = RandomState(MT19937(SeedSequence(seed)))
    M = min(len(means), len(covs))

    if not isinstance(Ns, list):
        aux = [Ns for _ in range(M)]
        Ns = aux
    X = np.concatenate([rs.multivariate_normal(m,
                                               cov=S,
                                               size=N)
                        for m, S, N in zip(means, covs, Ns)
                        ],
                       axis=0)
    Y = []

    for i, N in enumerate(Ns):
        Y += [i] * N

    Y = np.array(Y).ravel()

    return X, Y


def rotate_cov(A, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ A @ R.T


means = np.array([[0, 4], [1, -1], [2, 3]])
covs = [rotate_cov(np.array([[1, 0.9],
                             [0.9, 1]]), np.pi / 180 * 30),
        rotate_cov(np.array([[1, 0.2],
                             [0.2, 1]]), np.pi / 180 * 0),
        rotate_cov(np.array([[1, 0.1],
                             [0.1, 1]]), np.pi / 180 * 90)
        ]
X, Y = make_gaussian_blobs(means, covs, Ns=[100, 100, 200], seed=42)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.scatter(X[:, 0], X[:, 1], c=Y)


# For all distances we assume the clusters are given as numpy arrays
# The dimensions of the arrays are NxD where N is the number of points
# and D is the dimension of the vectors

def nearest_neighbor(A, B):
    dmin = np.inf
    for a in A:
        for b in B:
            d = np.linalg.norm(a - b)
            if d < dmin:
                dmin = d
    return dmin


def maximum_diameter(A, B):
    dmax = 0.0
    for a in A:
        for b in B:
            d = np.linalg.norm(a - b)
            if d > dmax:
                dmax = d
    return dmax


def average_distance(A, B):
    na, nb = A.shape[0], B.shape[0]
    dtot = 0.0
    for a in A:
        for b in B:
            dtot += np.linalg.norm(a - b)
    return dtot


def ward_distance(A, B):
    na, nb = A.shape[0], B.shape[0]
    return np.sqrt((na * nb) / (na + nb)) * np.linalg.norm(A.mean(0) - B.mean(0))


def CHA(X, dist_func):
    N, d = X.shape
    Y_historic = np.zeros((N, N))
    Y = np.arange(N)
    Y_historic[:, 0] = Y.copy()
    for c in range(N, 1, -1):
        # For current clustering Y, compute distances between clusters
        D = np.zeros((c, c))
        for i in range(c):
            for j in range(i + 1, c):
                ind_i = (Y == i)
                ind_j = (Y == j)
                D[i, j] = dist_func(X[ind_i], X[ind_j])
                D[j, i] = D[i, j]

        # Find the two clusters ca, cb that are the most similar
        ca, cb = np.unravel_index(np.argmin(D), D.shape)
        # Join the clusters ca and cb
        Y[Y == cb] = ca
        # Update the historic of clusterings
        Y_historic[:, N - c] = Y.copy()

    return Y_historic


def clusters_to_1N_range(Y):
    # Auxiliary function just for plotting
    if len(Y.shape) == 1:
        aux = Y.reshape(-1, 1)
    else:
        aux = Y
    d = aux.shape[1]
    res = np.zeros_like(aux)
    for j in range(d):
        u = np.unique(aux[:, j])
        for i, k in enumerate(u):
            ind = (aux[:, j] == k)
            res[ind, j] = i
    return res

# Metrics

def inertia_intra(X, Y):
    inertia = 0.0
    for i in range(Y.max() + 1):
        ind = (Y == i)
        inertia += np.sum(np.linalg.norm(X[ind] - X[ind].mean(0), axis=1) ** 2)
    return inertia

def inertia_inter(X, Y):
    mu = X.mean(0)
    inertia = 0.0
    for i in range(Y.max() + 1):
        ind = (Y == i)
        inertia += np.sum(np.linalg.norm(X[ind] - mu, axis=1) ** 2)
    return inertia


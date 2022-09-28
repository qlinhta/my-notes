# Markov chain

import numpy as np

def markov_chain(P, x0, n):
    """Markov chain"""
    x = x0
    for i in range(n):
        x = np.dot(x, P)
    return x

if __name__ == "__main__":
    P = np.array([[0.5, 0.5], [0.3, 0.7]])
    x0 = np.array([0.5, 0.5])
    n = 10
    print(markov_chain(P, x0, n))
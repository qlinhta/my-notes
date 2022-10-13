# Visualization quadratic form

import numpy as np
import matplotlib.pyplot as plt

# Create function to plot
def f(x):
    return x ** 2 + 3 * x + 19

def main():
    # Plot the function f(x) in 3D axes
    x = np.linspace(-10, 10, 100)
    y = f(x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, zs=0, zdir='z', label='f(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

# Polynomial Regression Algorithm

import numpy as np
import matplotlib.pyplot as plt


def polynomial_regression(x, y, degree):
    # x: independent variable
    # y: dependent variable
    # degree: degree of polynomial

    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    # printing regression coefficients
    print("Estimated coefficients: b_0 = {} \t b_1 = {}".format(b_0, b_1))

    # plotting regression line
    max_x = np.max(x) + 100
    min_x = np.min(x) - 100

    # calculating line values x and y
    x = np.linspace(min_x, max_x, 1000)
    y = b_0 + b_1*x

    # plotting line
    plt.plot(x, y, color='#58b970', label='Regression Line')

    # plotting scatter points
    plt.scatter(x, y, c='#ef5423', label='Scatter Plot')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # observations
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 5, 4, 5, 7, 9, 10, 12, 12])

    # estimating coefficients
    polynomial_regression(x, y, 1)

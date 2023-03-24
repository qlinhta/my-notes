import matplotlib.pyplot as plt
import numpy as np


# plot a normal distribution of x (sample = 1000)
x = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots()
ax.hist(x, bins=100)
plt.show()

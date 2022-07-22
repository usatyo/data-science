from random import randrange
import matplotlib.pyplot as plt
import numpy as np


def sample(x):
    x = np.arange(0.0, 1.0, 0.01)
    y = np.sin(2 * np.pi * x)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(title="sample")
    ax.grid()

    plt.show()


def rand_array(n):
    ret = np.array([np.random.rand() for i in range(n)])
    return ret


def sin_noise(x):
  y = np.sin(2 *np.pi*x)+np.random.normal(0,0.1)

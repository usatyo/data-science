import matplotlib.pyplot as plt
import numpy as np


def sample():
    x = np.arange(0.0, 1.0, 0.01)
    y = np.sin(2 * np.pi * x)
    plot_data(x, y, "sample")


def plot_data(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, "o")

    ax.set(title=title)
    ax.grid()

    plt.show()


def rand_array(n):
    ret = np.array(sorted([np.random.rand() for i in range(n)]))
    return ret


def sin_noise(x):
    noise = np.array([np.random.normal(0, 0.1) for i in range(len(x))])
    y = np.sin(2 * np.pi * x) + noise
    # plot_data(x, y, "noised sin")
    ret = np.array([x,y]).T
    return ret


def save_data():
    for n in [10, 20, 50, 100]:
        data = sin_noise(rand_array(n))
        np.savetxt("data/sample_" + str(n) + ".txt", data)


# save_data()

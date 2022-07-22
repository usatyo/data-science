from cv2 import moveWindow
import matplotlib.pyplot as plt
import numpy as np


def calc_A(m, x):
    ret = np.zeros((m + 1, m + 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)])
        ret += np.dot(geo, geo.T)
    return ret


def calc_T(m, x, t):
    ret = np.zeros((m + 1, 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)]) * t[i]
        ret += geo
    return ret


def calc_w(m, x, t):
    A = calc_A(m, x)
    T = calc_T(m, x, t)
    w = np.dot(np.linalg.inv(A), T)
    return w


def calc_y(m,w,x):
    plot_y = np.zeros(len(x))
    for i in range(m + 1):
        plot_y += w[i] * x**i
    return plot_y


def move_w(n):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    x = data[0]
    y = data[1]

    fig, ax = plt.subplots()
    ax.plot(x, y, "o")

    plot_x = np.arange(0.0, 1.0, 0.01)

    for m in range(6, 9):
        w = calc_w(m, x, y)
        plot_y = calc_y(m,w,plot_x)
        ax.plot(plot_x, plot_y)

    plt.axis([0, 1, -1, 1])
    plt.show()




# move_w(10)

import numpy as np


def A(m, x):
    ret = np.zeros((m + 1, m + 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)])
        ret += np.dot(geo, geo.T)
    return ret


def T(m, x, t):
    ret = np.zeros((m + 1, 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)]) * t[i]
        ret += geo
    return ret




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

print(calc_w(2, np.array([-3,0,1]), np.array([9,0,1])))
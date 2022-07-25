import numpy as np
from day_2 import calc_A, calc_T


def calc_reg_w(m, x, t, l):
    A = calc_A(m, x) + l * np.eye(m + 1)
    T = calc_T(m, x, t)
    w = np.dot(np.linalg.inv(A), T)
    return w



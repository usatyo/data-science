import numpy as np
from day_2 import calc_w, calc_y, plot_all
from day_3 import E_rms


def rand_int(n):
    ret = np.random.randint(1, 6, (n))
    for i in range(5):
        ret[i] = i + 1
    np.random.shuffle(ret)
    np.savetxt("data/randint_" + str(n) + ".txt", ret)
    return ret


def cross(n, m, idx):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    groups = np.loadtxt("data/randint_" + str(n) + ".txt")
    train_x = data[0, groups != idx]
    train_y = data[1, groups != idx]
    test_x = data[0, groups == idx]
    test_y = data[1, groups == idx]

    w = calc_w(m, train_x, train_y)
    # print(E_rms(calc_y(m, w, test_x), test_y))
    return E_rms(calc_y(m, w, test_x), test_y)


def best_mean_cross(n):
    min_m = 1
    min_mean = 10**10
    for m in range(1, 10):
        sum_cross = 0
        for i in range(5):
            sum_cross += cross(n, m, i + 1)
        if sum_cross / 5 < min_mean:
            min_mean = sum_cross / 5
            min_m = m
        print(sum_cross / 5)
    return min_m, min_mean


def plot_best_m(n):
    m, min_mean = best_mean_cross(n)
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    train_x = data[0]
    train_y = data[1]
    w = calc_w(m, train_x, train_y)
    plot_all(m, w)


plot_best_m(20)

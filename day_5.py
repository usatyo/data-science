import numpy as np
from day_1 import plot_data
from day_2 import calc_A, calc_T, calc_y, plot_all
from day_3 import E_rms


def calc_reg_w(m, x, t, l):
    A = calc_A(m, x) + l * np.eye(m + 1)
    T = calc_T(m, x, t)
    w = np.dot(np.linalg.inv(A), T)
    return w


def vec_norm(w):
    return np.sum(w**2) ** 0.5


def cross_reg(n, m, l):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    train_x = data[0, : n // 2]
    train_y = data[1, : n // 2]
    test_x = data[0, n // 2 :]
    test_y = data[1, n // 2 :]

    w = calc_reg_w(m, train_x, train_y, l)
    E_train = E_rms(calc_y(m, w, train_x), train_y)
    E_test = E_rms(calc_y(m, w, test_x), test_y)
    nor = vec_norm(w)
    return E_train, E_test, nor


def cross_reg_five(n, m, idx, l):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    groups = np.loadtxt("data/randint_" + str(n) + ".txt")
    train_x = data[0, groups != idx]
    train_y = data[1, groups != idx]
    test_x = data[0, groups == idx]
    test_y = data[1, groups == idx]

    w = calc_reg_w(m, train_x, train_y, l)
    E_train = E_rms(calc_y(m, w, train_x), train_y)
    E_test = E_rms(calc_y(m, w, test_x), test_y)
    nor = vec_norm(w)
    return E_train, E_test, nor


def best_lambda_cross(n):
    m = 9
    l_list = [10**i for i in range(-8, 5)]
    l_exp_list = [i for i in range(-8, 5)]
    train_list = []
    test_list = []
    size_list = []
    for l in l_list:
        sum_train = 0
        sum_test = 0
        sum_size = 0
        for i in range(5):
            E_train, E_test, size = cross_reg(n, m, l)
            # E_train, E_test, size = cross_reg_five(n, m, i + 1, l)
            sum_train += E_train
            sum_test += E_test
            sum_size += size
        train_list.append(sum_train / 5)
        test_list.append(sum_test / 5)
        size_list.append(sum_size / 5)
    plot_data(l_exp_list, train_list, "train")
    plot_data(l_exp_list, test_list, "test")
    plot_data(l_exp_list, size_list, "size")
    return l_list[np.argmin(test_list)]


def plot_best(m, l, n):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    w = calc_reg_w(m, data[0], data[1], l)
    print(w)
    plot_all(m, w, n)
    return 0


print('best lambda:', '{:.2e}'.format(best_lambda_cross(20)))
# plot_best(9, 10 ** (-8), 20)



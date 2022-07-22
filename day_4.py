import numpy as np
from day_2 import calc_w, calc_y
from day_3 import E_rms


def rand_int(n):
    ret = np.random.randint(1, 6, (n))
    for i in range(5):
        ret[i] = i + 1
    np.random.shuffle(ret)
    np.savetxt("data/randint_" + str(n) + ".txt", ret)
    return ret


def cross(n, idx):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    groups = np.loadtxt("data/randint_" + str(n) + ".txt")
    train_x = data[0, groups != idx]
    train_y = data[1, groups != idx]
    test_x = data[0, groups == idx]
    test_y = data[1, groups == idx]

    m = 3
    w = calc_w(m, train_x, train_y)
    print(E_rms(calc_y(m, w, test_x), test_y))
    return E_rms(calc_y(m, w, test_x), test_y)


def mean_cross(n):
    sum_cross = 0
    for i in range(5):
        sum_cross += cross(n, i+1)
    return sum_cross / 5

print(mean_cross(20))
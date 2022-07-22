import numpy as np
from day_2 import calc_w, calc_y


def E_rms(est, ans):
    ret = 0
    for i in range(len(ans)):
        ret += (est[i] - ans[i]) ** 2
    ret = (ret / len(ans)) ** 0.5
    return ret


def check_train(n):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    train_x = data[0, : n // 2]
    train_y = data[1, : n // 2]
    test_x = data[0, n // 2 :]
    test_y = data[1, n // 2 :]

    for m in range(1, 10):
        w = calc_w(m, train_x, train_y)
        print(E_rms(calc_y(m, w, train_x), train_y))
        print(E_rms(calc_y(m, w, test_x), test_y))


check_train(20)

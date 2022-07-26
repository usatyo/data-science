import numpy as np
from day_2 import calc_w, calc_y, plot_all


# 平均二乗平方根誤差を計算
def E_rms(est, ans):
    ret = 0
    for i in range(len(ans)):
        ret += (est[i] - ans[i]) ** 2
    ret = (ret / len(ans)) ** 0.5
    return ret


# サイズ n のデータを訓練データ、テストデータに2分割
# 訓練データを用いて
def check_score(n, m_list):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    train_x = data[0, : n // 2]
    train_y = data[1, : n // 2]
    test_x = data[0, n // 2 :]
    test_y = data[1, n // 2 :]

    for m in m_list:
        w = calc_w(m, train_x, train_y)
        print(E_rms(calc_y(m, w, train_x), train_y))
        print(E_rms(calc_y(m, w, test_x), test_y))
        print()
        plot_all(m, w, n)


# 7.12 ~ 7.14
# check_score(20, [3])

# 7.15
# check_score(20, range(1, 10))

# 7.16
# check_score(50, range(1, 10))
# check_score(100, range(1, 10))

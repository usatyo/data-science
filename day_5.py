import numpy as np
from day_1 import plot_data
from day_2 import calc_A, calc_T, calc_y, plot_all
from day_3 import E_rms


# calc_w を正則化の影響を考慮して修正した関数
# A が A + λI に置き換わった場合の w を計算
def calc_reg_w(m, x, t, l):
    A = calc_A(m, x) + l * np.eye(m + 1)
    T = calc_T(m, x, t)
    w = np.dot(np.linalg.inv(A), T)
    return w


# ベクトルの大きさを返す
def vec_norm(w):
    return np.sum(w**2) ** 0.5


# 正則化を施した場合についてデータを訓練データと
# テストデータに2分割し、学習を行なったのち
# 訓練誤差、テスト誤差、重みベクトルの大きさを計算
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


# 正則化を施した場合について5分割交差検定を用いて学習を行なったのち、
# 訓練誤差、テスト誤差、係数ベクトルの大きさを計算
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


# 交差検定を用いて λ を 10^-8 ~ 10^4 まで動かしたときの
# 訓練誤差、テスト誤差、係数ベクトルの大きさを計算し、
# それらの λ に対する変化をプロット（λ は対数目盛）
# 返り値はテスト誤差が最も小さかったときの λ とした
def best_lambda_cross(n, cut):
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
            # 分割数によって処理を分岐
            if cut == 2:
                E_train, E_test, size = cross_reg(n, m, l)
            else:
                E_train, E_test, size = cross_reg_five(n, m, i + 1, l)
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


# 交差検定でえられた最良の λ について訓練データを全て用いて
# w を再計算し、多項式関数のグラフと訓練データをプロット
def plot_best(m, n):
    l = best_lambda_cross(n)
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    w = calc_reg_w(m, data[0], data[1], l)
    print(w)
    plot_all(m, w, n)
    return 0


# 7.24, 7.25
# best_lambda_cross(20, 2)

# 7.26
# best_lambda_cross(20, 5)

# 7.27
# print("best lambda:", "{:.2e}".format(best_lambda_cross(20)))

# 7.28
# plot_best(9, 20)

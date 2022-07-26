import numpy as np
from day_1 import plot_data
from day_2 import calc_w, calc_y, plot_all
from day_3 import E_rms


# 1から5までの整数値をとる一様乱数を n 個発生させ配列に格納
# ただし、1番目から5番目までは対応する添字を代入し、のちにランダムに
# シャッフルすることでどの数字も少なくとも1つは出現するようにした
def rand_int(n):
    ret = np.random.randint(1, 6, (n))
    for i in range(5):
        ret[i] = i + 1
    np.random.shuffle(ret)
    np.savetxt("data/randint_" + str(n) + ".txt", ret)
    return ret


# サイズ n の m 次多項式関数の係数を決定し、テスト誤差を計算
# ただし、対応する乱数の値が idx と異なるものは訓練データ、
# 同じものはテストデータとして用いる
def cross(n, m, idx):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    groups = np.loadtxt("data/randint_" + str(n) + ".txt")
    train_x = data[0, groups != idx]
    train_y = data[1, groups != idx]
    test_x = data[0, groups == idx]
    test_y = data[1, groups == idx]

    w = calc_w(m, train_x, train_y)
    return E_rms(calc_y(m, w, test_x), test_y)


# idx を1から5まで変化させて求めたテスト誤差の平均値を計算
def five_cross(n, m):
    ret = 0
    for i in range(5):
        ret += cross(n, m, i + 1)
    return ret / 5


# m を1から9まで変化させて5分割交差検定を行い、
# m と求めたテスト誤差の平均値の関係をプロット
def plot_mean_cross(n):
    mean_cross = []
    m_list = range(1, 10)
    for m in m_list:
        mean_cross.append(five_cross(n, m))
    plot_data(m_list, mean_cross, "mean-cross")


# m を1から9まで変化させて交差検定を行い、
# 最良の m とそのときの平均値を返す
def best_mean_cross(n):
    min_m = 1
    min_mean = 10**10
    for m in range(1, 10):
        mean_cross = five_cross(n, m)
        if mean_cross < min_mean:
            min_mean = mean_cross
            min_m = m
    return min_m, min_mean


# 交差検定の平均値が最も小さかった m について
# 訓練データを全て使って学習を行い、
# m 次多項式関数のグラフと訓練データを描画
def plot_best_m(n):
    m, min_mean = best_mean_cross(n)
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    train_x = data[0]
    train_y = data[1]
    w = calc_w(m, train_x, train_y)
    plot_all(m, w, n)


# 7.17
# print(rand_int(20))

# 7.18
# print(cross(20, 3, 5))

# 7.19
# print(five_cross(20, 3))

# 7.20
# plot_mean_cross(20)

# 7.21
# print(best_mean_cross(20))

# 7.22
# plot_best_m(20)

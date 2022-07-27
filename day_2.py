import matplotlib.pyplot as plt
import numpy as np


# 線形方程式における行列 A を計算
def calc_A(m, x):
    ret = np.zeros((m + 1, m + 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)])
        ret += np.dot(geo, geo.T)
    return ret


# 線型方程式における行列 T を計算
def calc_T(m, x, t):
    ret = np.zeros((m + 1, 1))
    for i in range(len(x)):
        geo = np.array([[x[i] ** j] for j in range(m + 1)]) * t[i]
        ret += geo
    return ret


# 与えられたデータから A、T を作成し、w を算出
def calc_w(m, x, t):
    A = calc_A(m, x)
    # 逆行列が存在しない場合の処理
    if np.linalg.det(A) == 0:
        print('error: make another random file')
    T = calc_T(m, x, t)
    w = np.dot(np.linalg.inv(A), T)
    return w


# 重み w の m 次多項式関数に点列 x を代入した際の
# 結果出力される点列 y を計算
def calc_y(m, w, x):
    plot_y = np.zeros(len(x))
    for i in range(m + 1):
        plot_y += w[i] * x**i
    return plot_y


# サイズ n の訓練データをプロットした上で、
# 重み w の m 次多項式関数のグラフを描画する
def plot_all(m, w, n):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    fig, ax = plt.subplots()

    x = np.arange(0.0, 1.0, 0.01)
    y = calc_y(m, w, x)

    ax.plot(data[0], data[1], "o")
    ax.plot(x, y)
    plt.axis([0, 1, -1, 1])
    plt.show()


# m を変化させたときの多項式関数のグラフを重ねて描画する
# さらに訓練データも同じ場所にプロットする
def move_m(n):
    data = np.loadtxt("data/sample_" + str(n) + ".txt")
    x = data[0]
    y = data[1]

    fig, ax = plt.subplots()
    ax.plot(x, y, "o")

    plot_x = np.arange(0.0, 1.0, 0.01)

    for m in range(7, 10):
        w = calc_w(m, x, y)
        plot_y = calc_y(m, w, plot_x)
        ax.plot(plot_x, plot_y)

    plt.axis([0, 1, -1, 1])
    plt.show()


# 7.7 ~ 7.9
# move_m(10)

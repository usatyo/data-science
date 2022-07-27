import matplotlib.pyplot as plt
import numpy as np


# y = sin(2πx) のグラフを描画
def sample():
    x = np.arange(0.0, 1.0, 0.01)
    y = np.sin(2 * np.pi * x)
    plot_data(x, y, "")


# x と y に格納されたデータをプロット
def plot_data(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, "o")

    ax.set(title="")
    ax.grid()

    plt.show()


# 長さ n のランダムな要素を持つ配列を作成
def rand_array(n):
    ret = np.array([np.random.rand() for i in range(n)])
    return ret


# x に格納された点に対して sin(2πx) の変換を行い、
# 平均0、分散0.1のノイズを加えた配列を作成
def sin_noise(x):
    noise = np.array([np.random.normal(0, 0.1) for i in range(len(x))])
    y = np.sin(2 * np.pi * x) + noise
    plot_data(x, y, "noised sin")
    ret = np.array([x, y])
    return ret


# n_list のそれぞれの大きさでノイズを加えた
# データセットを作成し、dataディレクトリに保管
def save_data(n_list):
    for n in n_list:
        data = sin_noise(rand_array(n))
        np.savetxt("data/sample_" + str(n) + ".txt", data)


# 7.2
# sample()

# 7.3
# print(rand_array(10))

# 7.4
# x = rand_array(10)
# print(sin_noise(x))

# 7.5
# save_data([10, 20, 50, 100])

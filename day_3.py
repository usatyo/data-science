import numpy as np
from day_2 import calc_w, calc_y


def E_rms(est, ans):
    ret = 0
    for i in range(len(ans)):
        ret += (est[i] - ans[i]) ** 2
    ret = (ret / len(ans)) ** 0.5
    return ret


data = np.loadtxt("data/sample_" + str(10) + ".txt")
x = data[0]
y = data[1]
w = calc_w(8, x, y)
print(E_rms(calc_y(8,w,x), y))

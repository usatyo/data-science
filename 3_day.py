import numpy as np


def E_rms(est, ans):
    ret = 0
    for i in range(len(ans)):
        ret += (est - ans) ** 2
    ret = (ret / len(ans)) ** 0.5
    return ret


print(E_rms())
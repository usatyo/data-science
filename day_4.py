import numpy as np
from torch import randint


def rand_int(n):
    ret = np.random.randint(1, 6, (n))
    for i in range(5):
        ret[i] = i + 1
    np.random.shuffle(ret)
    np.savetxt("data/randint_" + str(n) + ".txt", ret)
    return ret

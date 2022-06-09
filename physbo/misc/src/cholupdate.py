import numpy as np


def cholupdate64(L, x):
    N = x.shape[0]
    x2 = x

    for k in range(0, N):
        r = np.hypot(L[k, k], x2[k])
        c = r / L[k, k]
        s = x2[k] / L[k, k]
        L[k, k] = r

        for i in range(k + 1, N):
            L[k, i] = (L[k, i] + s * x2[i]) / c
            x2[i] = c * x2[i] - s * L[k, i]

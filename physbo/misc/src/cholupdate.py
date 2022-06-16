import copy
import numpy as np


def cholupdate64(L, x):
    N = x.shape[0]
    x2 = copy.copy(x)

    for k in range(0, N):
        r = np.hypot(L[k, k], x2[k])
        c = r / L[k, k]
        s = x2[k] / L[k, k]
        L[k, k] = r
        ic = 1.0/c

        L[k, k+1:N] = ic * (L[k, k+1:N] + s * x2[k+1:N])
        x2[k+1:N] = c * x2[k+1:N] - s * L[k, k+1:N]

        # for i in range(k + 1, N):
        #     L[k, i] = (L[k, i] + s * x2[i]) / c
        #     x2[i] = c * x2[i] - s * L[k, i]

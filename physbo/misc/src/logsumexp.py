import numpy as np


def logsumexp64(x):
    """ Calculate log(sum(exp(x)))

    Parameters
    ==========
    x: np.ndarray
    """
    N = x.shape[0]
    tmp = 0.0

    xmax = np.max(x)

    for i in range(0, N):
        tmp += np.exp(x[i] - xmax)

    return np.log(tmp) + xmax

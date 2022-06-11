import numpy as np


def logsumexp64(x):
    """ Calculate log(sum(exp(x)))

    Parameters
    ==========
    x: np.ndarray
    """

    xmax = np.max(x)
    return np.log(np.sum(np.exp(x-xmax))) + xmax

    # N = x.shape[0]
    # tmp = 0.0
    #
    # for i in range(0, N):
    #     tmp += np.exp(x[i] - xmax)
    #
    # return np.log(tmp) + xmax

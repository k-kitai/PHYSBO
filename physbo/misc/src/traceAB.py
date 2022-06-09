import numpy as np


def traceAB3_64(A, B):
    """ Calculates vector of trace of AB[i], where i is the first axis of 3-rank tensor B

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        dxMxN tensor

    Returns
    =======
    traceAB: np.ndarray
    """
    N = A.shape[0]
    M = A.shape[1]
    D = B.shape[0]

    traceAB = np.zeros(D, dtype=np.float64)

    for d in range(D):
        traceAB[d] = 0
        for i in range(N):
            for j in range(M):
                traceAB[d] += A[i, j] * B[d, j, i]
    return traceAB


def traceAB2_64(A, B):
    """ Calculates trace of AB

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        MxN matrix

    Returns
    =======
    traceAB: float
        trace of the matrix AB
    """
    N = A.shape[0]
    M = A.shape[1]

    traceAB = 0.0

    for i in range(N):
        for j in range(M):
            traceAB += A[i, j] * B[j, i]
    return traceAB

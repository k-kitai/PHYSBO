import numpy as np


def diagAB_64(A, B):
    """ Return diagonal part of AB

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        MxN matrix

    Returns
    =======
    d: np.ndarray
        Diagonal part of the matrix AB
    """

    return np.einsum("ij,ji->i", A, B)

    # N = A.shape[0]
    # M = A.shape[1]
    #
    # diagAB = np.zeros(N, dtype=np.float64)
    #
    # for i in range(N):
    #     for j in range(M):
    #         diagAB[i] += A[i, j] * B[j, i]
    #
    # return diagAB

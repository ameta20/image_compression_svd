import numpy as np

def gram_schmidt_general(A):
    """
    Modified Gram-Schmidt algorithm for any matrix A (m×n).
    Returns orthonormal Q and upper-triangular R such that A = Q @ R.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            # skip dependent column
            continue
        Q[:, j] = v / R[j, j]

    return Q, R

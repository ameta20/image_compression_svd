import numpy as np

def householder_vector(x):
    x = x.astype(float)
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return np.zeros_like(x)

    sign = -1.0 if x[0] < 0 else 1.0
    u1 = x[0] + sign * norm_x
    v = x.copy()
    v[0] = u1
    v /= np.linalg.norm(v)
    return v


def bidiagonalize(A):
    A = A.astype(float).copy()
    m, n = A.shape
    U = np.eye(m)
    Vt = np.eye(n)

    for k in range(min(m, n)):
        # ---- Left Householder ----
        x = A[k:, k]
        v = householder_vector(x)
        if np.linalg.norm(v) != 0:
            A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
            U[:, k:] -= 2 * (U[:, k:] @ np.outer(v, v))

        # ---- Right Householder ----
        if k < n - 1:
            x = A[k, k+1:]
            v = householder_vector(x)
            if np.linalg.norm(v) != 0:
                A[k:, k+1:] -= 2 * (A[k:, k+1:] @ np.outer(v, v))
                Vt[k+1:, :] -= 2 * np.outer(v, v @ Vt[k+1:, :])

    return U, A, Vt


def svd_golub_kahan(A, k=None):
    m, n = A.shape

    #bidiagonalization
    U0, B, V0t = bidiagonalize(A)

    #SVD of bidiagonal matrix B
    Ub, S, Vbt = np.linalg.svd(B, full_matrices=False)

    # Combine
    U = U0 @ Ub
    Vt = Vbt @ V0t

    # Apply truncation k
    r = len(S)

    if k is not None:
        k = min(k, r)
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]

    return U, S, Vt


def reconstruct(U, S, Vt):
    return U @ np.diag(S) @ Vt

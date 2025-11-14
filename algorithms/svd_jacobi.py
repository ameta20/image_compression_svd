import numpy as np

def svd_jacobi(M, eps=1.0e-15):
    """
    One-sided Jacobi SVD of matrix M (m x n).

    Returns:
        U  : (m x r) left singular vectors
        s  : (r,)    singular values (not explicitly sorted)
        Vh : (r x n) right singular vectors (Hermitian transpose)
    """
    A = M.astype(float).copy()
    m, n = A.shape

    # working right singular vectors (V)
    V = np.eye(n)
    # working singular values / error estimates
    t = np.zeros(n)

    # heuristics mimicking the original code
    DBL_EPSILON = eps
    tolerance = 10.0 * m * DBL_EPSILON
    sweep_max = max(5 * n, 12)

    # initial column error estimates
    for j in range(n):
        cj = A[:, j]
        t[j] = DBL_EPSILON * np.linalg.norm(cj)

    count = 1
    sweep = 0

    # main Jacobi sweeps
    while count > 0 and sweep <= sweep_max:
        count = n * (n - 1) // 2

        for j in range(n - 1):
            for k in range(j + 1, n):
                cj = A[:, j]
                ck = A[:, k]

                p = 2.0 * np.dot(cj, ck)
                a = np.linalg.norm(cj)
                b = np.linalg.norm(ck)

                q = a * a - b * b
                v = np.hypot(p, q)  # sqrt(p^2 + q^2) safely

                abserr_a = t[j]
                abserr_b = t[k]

                sorted_cols = (a >= b)
                orthog = (abs(p) <= tolerance * (a * b))
                noisya = (a < abserr_a)
                noisyb = (b < abserr_b)

                # skip rotation if already orthogonal or dominated by noise
                if sorted_cols and (orthog or noisya or noisyb):
                    count -= 1
                    continue

                # compute rotation (cosine, sine)
                if v == 0.0 or not sorted_cols:
                    c = 0.0
                    s_val = 1.0
                else:
                    c = np.sqrt((v + q) / (2.0 * v))
                    s_val = p / (2.0 * v * c)

                # apply rotation to columns j,k of A
                A_j = A[:, j].copy()
                A_k = A[:, k].copy()
                A[:, j] = c * A_j + s_val * A_k
                A[:, k] = -s_val * A_j + c * A_k

                # update error estimates
                t[j] = abs(c) * abserr_a + abs(s_val) * abserr_b
                t[k] = abs(s_val) * abserr_a + abs(c) * abserr_b

                # apply same rotation to V (right singular vectors)
                V_j = V[:, j].copy()
                V_k = V[:, k].copy()
                V[:, j] = c * V_j + s_val * V_k
                V[:, k] = -s_val * V_j + c * V_k

        sweep += 1

    # compute singular values and normalize columns of A → U
    s = np.zeros(n)
    prev_norm = -1.0

    for j in range(n):
        col = A[:, j]
        norm = np.linalg.norm(col)

        if norm == 0.0 or prev_norm == 0.0 or (j > 0 and norm <= tolerance * prev_norm):
            s[j] = 0.0
            A[:, j] = 0.0
            prev_norm = 0.0
        else:
            s[j] = norm
            A[:, j] /= norm
            prev_norm = norm

    if count > 0:
        print("Warning: Jacobi iterations did not fully converge")

    U = A
    Vh = V.T

    if m < n:
        U = U[:, :m]
        s = s[:m]
        Vh = Vh[:m, :]

    return U, s, Vh


def reconstruct_from_svd(U, s, Vh):
    return U @ np.diag(s) @ Vh

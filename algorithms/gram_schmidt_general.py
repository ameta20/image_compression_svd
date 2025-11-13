import numpy as np

def gram_schmidt_general(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-12:
            Q[:, j] = v / R[j, j]
    return Q, R

def power_iteration(A, num_vecs, tol=1e-10, max_iter=1000):
    n = A.shape[1]
    V = []
    for _ in range(num_vecs):
        b_k = np.random.rand(n)
        for _ in range(max_iter):
            b_k1 = A @ b_k
            b_k1_norm = np.linalg.norm(b_k1)
            if b_k1_norm < tol:
                break
            b_k = b_k1 / b_k1_norm
        # Deflate
        A = A - (A @ b_k).reshape(-1, 1) @ b_k.reshape(1, -1)
        V.append(b_k)
    return np.column_stack(V)

def svd_gram_schmidt_power(M, k=None):
    M = np.array(M, dtype=float)
    m, n = M.shape
    k = k or min(m, n)

    # Step 1: Compute M^T M
    MtM = M.T @ M

    # Step 2: Get top-k right singular vectors via power iteration
    V = power_iteration(MtM, k)

    # Step 3: Compute singular values and left singular vectors
    S = []
    U = []
    for i in range(k):
        v = V[:, i]
        Av = M @ v
        sigma = np.linalg.norm(Av)
        S.append(sigma)
        U.append(Av / sigma if sigma > 1e-12 else Av)

    U = np.column_stack(U)
    Vh = V.T
    S = np.array(S)

    return U, S, Vh

def reconstruct_image(U, S, Vh):
    return U @ np.diag(S) @ Vh
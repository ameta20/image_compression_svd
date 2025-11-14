import numpy as np

def arnoldi_iteration(A, k):

    n = A.shape[0]
    V = np.zeros((n, k+1))
    H = np.zeros((k+1, k))

    # start vector
    v0 = np.random.randn(n)
    v0 /= np.linalg.norm(v0)
    V[:, 0] = v0

    for j in range(k):
        w = A @ V[:, j]

        # Modified Gram–Schmidt
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        H[j+1, j] = np.linalg.norm(w)

        # breakdown (happens when Krylov subspace saturates)
        if H[j+1, j] < 1e-14:
            return V[:, :j+1], H[:j+1, :j]

        V[:, j+1] = w / H[j+1, j]

    return V, H


# ----------------------------------------------------------
# Arnoldi-based SVD
# ----------------------------------------------------------

def svd_arnoldi(M, k=20):
    A = M
    n = A.shape[1]

    B = A.T @ A

    # 1. Arnoldi on B
    V_big, H = arnoldi_iteration(B, k)

    # 2. Eigen-decompose the small matrix T = H^T H (Rayleigh-Ritz)
    H_small = V_big[:, :k].T @ (B @ V_big[:, :k])

    eigvals, eigvecs = np.linalg.eigh(H_small)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 3. Singular values
    sigma = np.sqrt(np.maximum(eigvals, 0))

    # 4. Right singular vectors (in original space)
    V = V_big[:, :k] @ eigvecs

    # 5. Left singular vectors
    U = np.zeros((A.shape[0], k))
    for i in range(k):
        if sigma[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / sigma[i]

    # Normalize U columns
    for i in range(k):
        nrm = np.linalg.norm(U[:, i])
        if nrm > 1e-14:
            U[:, i] /= nrm

    return U, sigma, V.T


def reconstruct(U, S, Vh):
    return U @ np.diag(S) @ Vh

import numpy as np

def lanczos_tridiagonalization(M, k=10, tol=1e-10):
    """
    Lanczos tridiagonalization of A = M.T @ M
    Returns tridiagonal matrix T and orthonormal basis V
    """
    n = M.shape[1]
    A = M.T @ M

    v1 = np.random.normal(size=n)
    v1 /= np.linalg.norm(v1)
    V = [v1]
    alpha = []
    beta = []

    for i in range(k):
        w = A @ V[i]
        if i > 0:
            w -= beta[i-1] * V[i-1]
        alpha_i = np.dot(V[i], w)
        alpha.append(alpha_i)
        w -= alpha_i * V[i]
        beta_i = np.linalg.norm(w)
        beta.append(beta_i)
        if beta_i < tol:
            break
        V.append(w / beta_i)

    m = len(alpha)
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]

    V_mat = np.column_stack(V[:m])
    return T, V_mat

def qr_eigen_tridiagonal(T, max_iter=1000, tol=1e-10):
    """
    QR algorithm for symmetric tridiagonal matrix T
    Returns eigenvalues and eigenvectors
    """
    n = T.shape[0]
    A = T.copy()
    Q_total = np.eye(n)

    for _ in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        Q_total = Q_total @ Q
        off_diag = np.sqrt(np.sum(np.tril(A, -1)**2))
        if off_diag < tol:
            break

    eigvals = np.diag(A)
    eigvecs = Q_total
    return eigvals, eigvecs

def svd_lanczos(M, k=10, tol=1e-10):
    """
    SVD approximation using Lanczos tridiagonalization and QR eigen solver
    """
    T, V_basis = lanczos_tridiagonalization(M, k, tol)
    eigvals, eigvecs = qr_eigen_tridiagonal(T, tol=tol)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    s = np.sqrt(np.maximum(eigvals[:k], 0))
    V = V_basis @ eigvecs[:, :k]

    U = np.zeros((M.shape[0], k))
    for i in range(k):
        if s[i] > tol:
            U[:, i] = (M @ V[:, i]) / s[i]

    Vh = V.T
    return U, s, Vh

def reconstruct_image(U, S, Vh):
    return U @ np.diag(S) @ Vh
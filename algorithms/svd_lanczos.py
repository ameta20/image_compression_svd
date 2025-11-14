import numpy as np

def lanczos_tridiagonalization(A, k=20, tol=1e-12, reorth=True):
    """
    Perform Lanczos tridiagonalization of a symmetric matrix A.
    Returns T (tridiagonal) and V (Lanczos basis).

    A must be symmetric: A = A.T
    """

    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    V = []
    alpha = []
    beta = []

    w = A @ v
    alpha_i = np.dot(v, w)
    w = w - alpha_i * v

    V.append(v)
    alpha.append(alpha_i)

    for i in range(1, k):

        # Compute beta_i
        beta_i = np.linalg.norm(w)
        beta.append(beta_i)

        # Stop early if Krylov subspace saturated
        if beta_i < tol:
            break

        # Next Lanczos vector
        v = w / beta_i

        # Re-orthogonalization (important!)
        if reorth:
            for j in range(len(V)):
                corr = np.dot(V[j], v)
                v -= corr * V[j]
            v /= np.linalg.norm(v)

        V.append(v)

        # Compute next w
        w = A @ v

        # Subtract orthogonal components
        w = w - beta_i * V[i-1]
        alpha_i = np.dot(v, w)
        alpha.append(alpha_i)
        w = w - alpha_i * v

    # Build the tridiagonal matrix T
    m = len(alpha)
    T = np.zeros((m, m))

    for i in range(m):
        T[i, i] = alpha[i]
        if i < m-1:
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]

    # Stack V into matrix
    V_mat = np.column_stack(V)

    return T, V_mat

def qr_eigen_tridiagonal(T, tol=1e-12, max_iter=5000):
    """
    Compute eigenvalues/vectors of symmetric tridiagonal T
    using QR with Wilkinson shift.
    """

    n = T.shape[0]
    A = T.copy()
    Q_total = np.eye(n)

    for _ in range(max_iter):
        # Wilkinson shift
        d = (A[n-2,n-2] - A[n-1,n-1]) / 2
        mu = A[n-1,n-1] - (np.sign(d) * A[n-1,n-2]**2) / (abs(d) + np.sqrt(d*d + A[n-1,n-2]**2))

        Q, R = np.linalg.qr(A - mu * np.eye(n))
        A = R @ Q + mu * np.eye(n)
        Q_total = Q_total @ Q

        off = np.sqrt(np.sum(np.tril(A, -1)**2))
        if off < tol:
            break

    return np.diag(A), Q_total


# ---------------------------------------------------------
# Full SVD from Lanczos
# ---------------------------------------------------------

def svd_lanczos(M, k=30, tol=1e-12):
    """
    Compute approximate SVD of M using Lanczos on A = M^T M.
    Returns U, S, Vh
    """

    A = M.T @ M  # symmetric positive semidefinite

    T, V_basis = lanczos_tridiagonalization(A, k=k, tol=tol)

    eigvals, eigvecs_T = qr_eigen_tridiagonal(T, tol=tol)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_T = eigvecs_T[:, idx]

    # Singular values
    s = np.sqrt(np.maximum(eigvals, 0))

    # Right singular vectors
    V = V_basis @ eigvecs_T

    # Compute left singular vectors
    r = len(s)
    U = np.zeros((M.shape[0], r))

    for i in range(r):
        if s[i] > tol:
            U[:, i] = (M @ V[:, i]) / s[i]

    # Normalize (safety)
    for i in range(r):
        if np.linalg.norm(U[:, i]) > 0:
            U[:, i] /= np.linalg.norm(U[:, i])

    return U, s, V.T

def reconstruct_image(U, S, Vh):
    return U @ np.diag(S) @ Vh
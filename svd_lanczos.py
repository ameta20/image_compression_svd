import numpy as np

def svd_lanczos(M, k=10):
    A = M.T @ M  # symmetric PSD matrix
    n = A.shape[0]

    v0 = np.zeros(n)
    v1 = np.random.normal(size=n)
    v1 = v1 / np.linalg.norm(v1)

    w = []
    beta = [0]
    alpha = []
    v = [v0, v1]

    # Lanczos iterations
    for i in range(k):
        w_aux = A @ v[i + 1] - beta[i] * v[i]
        alpha_i = np.dot(v[i + 1], w_aux)
        alpha.append(alpha_i)
        w_i = w_aux - alpha_i * v[i + 1]
        beta_i = np.linalg.norm(w_i)
        beta.append(beta_i)
        if beta_i < 1e-10:
            break
        v.append(w_i / beta_i)

    #tridiagonal matrix
    T = np.diag(alpha)
    for i in range(len(beta) - 2):
        T[i, i + 1] = beta[i + 1]
        T[i + 1, i] = beta[i + 1]

    # Eigen-decomposition of T
    eigvals, eigvecs = np.linalg.eigh(T)

    # Sort eigenvalues/vectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Singular values
    s = np.sqrt(np.abs(eigvals))

    V_basis = np.column_stack(v[1:len(eigvecs) + 1])
    V = V_basis @ eigvecs

    U = M @ V
    for i in range(U.shape[1]):
        norm = np.linalg.norm(U[:, i])
        if norm > 1e-12:
            U[:, i] /= norm

    return U[:, :k], s[:k], V[:, :k].T

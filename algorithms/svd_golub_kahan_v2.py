import numpy as np
import math


def _householder_vector(x: np.ndarray) -> np.ndarray:
    """
    Compute Householder vector v
    """
    x = x.astype(float)
    norm_x = np.linalg.norm(x)
    if norm_x == 0.0:
        return np.zeros_like(x)

    sign = -1.0 if x[0] < 0.0 else 1.0
    v = x.copy()
    v[0] += sign * norm_x
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return np.zeros_like(v)
    v /= v_norm
    return v



# Phase 1: Golub–Kahan bidiagonalization
def _bidiagonalize(A: np.ndarray):
    A = A.astype(float).copy()
    m, n = A.shape
    U = np.eye(m)
    Vt = np.eye(n)
    r = min(m, n)

    for k in range(r):
        # Left reflector on column k
        x = A[k:, k]
        v = _householder_vector(x)
        if np.linalg.norm(v) != 0.0:
            A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
            U[:, k:] -= 2.0 * (U[:, k:] @ np.outer(v, v))

        # Right reflector on row k
        if k < n - 1:
            x = A[k, k + 1:]
            v = _householder_vector(x)
            if np.linalg.norm(v) != 0.0:
                A[k:, k + 1:] -= 2.0 * (A[k:, k + 1:] @ np.outer(v, v))
                Vt[k + 1:, :] -= 2.0 * np.outer(v, v @ Vt[k + 1:, :])

    return U, A, Vt


# QR from scratch (Householder-based)
def qr_from_scratch(A: np.ndarray):
    A = A.astype(float).copy()
    m, n = A.shape
    Q = np.eye(m)
    R = A

    for k in range(min(m, n)):
        x = R[k:, k]
        v = _householder_vector(x)
        if np.linalg.norm(v) == 0.0:
            continue

        # Apply reflector to R
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

        # Apply reflector to Q
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R


# QR eigen-solver using qr_from_scratch
def qr_eigen_tridiagonal(T, tol=1e-12, max_iter=5000):
    n = T.shape[0]
    A = T.copy().astype(float)
    Q_total = np.eye(n)

    for _ in range(max_iter):
        if n == 1:
            break

        # Check convergence via off-diagonal Frobenius norm
        off = np.sqrt(np.sum(np.tril(A, -1)**2))
        if off < tol:
            break

        # Wilkinson shift from bottom-right 2x2
        d = (A[n - 2, n - 2] - A[n - 1, n - 1]) / 2.0
        b2 = A[n - 1, n - 2] ** 2
        if d == 0.0:
            mu = A[n - 1, n - 1] - abs(A[n - 1, n - 2])
        else:
            mu = A[n - 1, n - 1] - b2 / (d + np.sign(d) * math.sqrt(d*d + b2))

        # QR step
        Q, R = qr_from_scratch(A - mu * np.eye(n))
        A = R @ Q + mu * np.eye(n)
        Q_total = Q_total @ Q

    eigvals = np.diag(A)
    eigvecs = Q_total
    return eigvals, eigvecs



# Phase 2: SVD of bidiagonal block via QR eigen on T = B^T B

def _svd_bidiagonal_via_qr(Bn, tol=1e-12, max_iter=5000):
    T = Bn.T @ Bn

    eigvals, V = qr_eigen_tridiagonal(T, tol=tol, max_iter=max_iter)

    # Sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    S = np.sqrt(np.maximum(eigvals, 0.0))
    Vb = V
    Vbt = Vb.T

    r = Bn.shape[0]
    Ub = np.zeros((r, r))
    for i in range(r):
        if S[i] > tol:
            Ub[:, i] = (Bn @ Vb[:, i]) / S[i]
        else:
            Ub[:, i] = 0.0

    return Ub, S, Vbt



# Phase 3: Full Golub–Kahan SVD


def svd_golub_kahan_full(A: np.ndarray, tol=1e-12, max_iter=5000):
    A = np.array(A, dtype=float)
    m, n = A.shape
    r = min(m, n)

    if m >= n:
        U0, B, V0t = _bidiagonalize(A)
        Bn = B[:r, :r]
        Ub, S, Vbt = _svd_bidiagonal_via_qr(Bn, tol=tol, max_iter=max_iter)
        U = U0[:, :r] @ Ub
        Vt = Vbt @ V0t
        return U, S, Vt

    else:
        U2, S, Vt2 = svd_golub_kahan_full(A.T, tol=tol, max_iter=max_iter)
        V2 = U2
        U_tmp = Vt2.T
        U = U_tmp[:, :r]
        Vt = V2[:, :r].T
        return U, S[:r], Vt


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int = None) -> np.ndarray:
    if k is None:
        # If no k is given, use the full rank
        k = len(S)


    max_rank = len(S)
    k = min(k, max_rank)


    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]


    return U_k @ (S_k[:, None] * Vt_k)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

    for shape in [(4, 4), (5, 3), (3, 5), (8, 5), (50, 50)]:

        print("\n- Testing Shape:", shape, "---")
        A = np.random.randn(*shape)
        U, S, Vt = svd_golub_kahan_full(A)
        A_rec = reconstruct(U, S, Vt)
        rel_err = np.linalg.norm(A - A_rec) / np.linalg.norm(A)
        print(f"Relative reconstruction error: {rel_err:.2e}")

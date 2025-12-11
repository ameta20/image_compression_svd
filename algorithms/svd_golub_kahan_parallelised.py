import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numba import njit, prange, set_num_threads

set_num_threads(2)


@njit(parallel=True)
def apply_householder_left(A, v, w, k):

    m, n = A.shape
    r = m - k    # rows in trailing block
    c = n - k    # cols in trailing block

    for i in prange(r):
        for j in range(c):
            A[k + i, k + j] -= 2.0 * v[i] * w[j]


@njit(parallel=True)
def apply_householder_to_U(U, v, k):

    m, n = U.shape
    r = n - k    # width of trailing block in U

    for i in prange(m):
        # tmp = U[i, k:] · v
        tmp = 0.0
        for t in range(r):
            tmp += U[i, k + t] * v[t]

        # update row
        for j in range(r):
            U[i, k + j] -= 2.0 * tmp * v[j]


@njit(parallel=True)
def apply_householder_right_A(A, v, k):

    m, n = A.shape
    r = m - k            # rows in the block
    c = n - (k + 1)      # cols in the block (starting at k+1)

    for i in prange(r):
        # tmp = A[k+i, k+1:] · v
        tmp = 0.0
        for t in range(c):
            tmp += A[k + i, k + 1 + t] * v[t]

        # update row i of the block
        for j in range(c):
            A[k + i, k + 1 + j] -= 2.0 * tmp * v[j]


@njit(parallel=True)
def apply_householder_right_Vt(Vt, v, k):

    m, n = Vt.shape
    r = m - (k + 1)   # number of affected rows

    for j in prange(n):
        # s = vᵀ Vt[k+1:, j]
        s = 0.0
        for t in range(r):
            s += v[t] * Vt[k + 1 + t, j]

        # update column j in the block
        for i in range(r):
            Vt[k + 1 + i, j] -= 2.0 * v[i] * s

@njit(parallel=True)
def apply_reflector_R(R, v, k):
    m, n = R.shape
    r = m - k
    c = n - k

    # Compute w = vᵀ * R[k:, k:]
    w = np.zeros(c)
    for j in prange(c):
        s = 0.0
        for i in range(r):
            s += v[i] * R[k + i, k + j]
        w[j] = s

    # Apply: R[k:, k:] -= 2 * v * wᵀ
    for i in prange(r):
        for j in range(c):
            R[k + i, k + j] -= 2.0 * v[i] * w[j]

@njit(parallel=True)
def apply_reflector_Q(Q, v, k):
    m, n = Q.shape
    r = n - k

    # Compute u = Q[:, k:] @ v
    u = np.zeros(m)
    for i in prange(m):
        s = 0.0
        for t in range(r):
            s += Q[i, k + t] * v[t]
        u[i] = s

    # Apply: Q[:, k:] -= 2 * u * vᵀ
    for i in prange(m):
        for j in range(r):
            Q[i, k + j] -= 2.0 * u[i] * v[j]


def _householder_vector(x: np.ndarray) -> np.ndarray:

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



def _bidiagonalize(A: np.ndarray):

    A = A.astype(np.float64).copy()
    m, n = A.shape
    U = np.eye(m)
    Vt = np.eye(n)
    r = min(m, n)

    for k in range(r):

        x = A[k:, k]
        v = _householder_vector(x)

        if np.linalg.norm(v) != 0.0:
            # Compute w = v @ A[k:, k:] via explicit loops
            Ablock = A[k:, k:]
            rows, cols = Ablock.shape
            w = np.zeros(cols)
            for j in range(cols):
                s = 0.0
                for i in range(rows):
                    s += v[i] * Ablock[i, j]
                w[j] = s

            # Apply updates using Numba kernels
            apply_householder_left(A, v, w, k)
            apply_householder_to_U(U, v, k)

        if k < n - 1:
            x = A[k, k + 1:]
            v = _householder_vector(x)

            if np.linalg.norm(v) != 0.0:
                # Left multiplication on A's trailing block
                apply_householder_right_A(A, v, k)

                # Right multiplication on Vt's trailing block
                apply_householder_right_Vt(Vt, v, k)

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

        apply_reflector_R(R, v, k)
        
        apply_reflector_Q(Q, v, k)

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
        off = np.sqrt(np.sum(np.tril(A, -1) ** 2))
        if off < tol:
            break

        # Wilkinson shift from bottom-right 2x2
        d = (A[n - 2, n - 2] - A[n - 1, n - 1]) / 2.0
        b2 = A[n - 1, n - 2] ** 2
        if d == 0.0:
            mu = A[n - 1, n - 1] - abs(A[n - 1, n - 2])
        else:
            mu = A[n - 1, n - 1] - b2 / (d + np.sign(d) * math.sqrt(d * d + b2))

        # QR step
        Q, R = qr_from_scratch(A - mu * np.eye(n))
        A = R @ Q + mu * np.eye(n)
        Q_total = Q_total @ Q

    eigvals = np.diag(A)
    eigvecs = Q_total
    return eigvals, eigvecs


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


def compress_image_parallelised(A, k):
    # Run SVD
    U, S, Vt = svd_golub_kahan_full(A)

    # Reconstruct using built-in function
    A_rec = reconstruct(U, S, Vt, k)

    # Compute metrics
    err = np.linalg.norm(A - A_rec)
    rel = err / np.linalg.norm(A)
    psnr = peak_signal_noise_ratio(A, A_rec, data_range=1.0)
    ssim = structural_similarity(A, A_rec, data_range=1.0)

    return A_rec, err, rel, psnr, ssim


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

    for shape in [(4, 4), (5, 3), (3, 5), (8, 5), (50, 50)]:
        print("\n- Testing Shape:", shape, "---")
        A = np.random.randn(*shape)
        U, S, Vt = svd_golub_kahan_full(A)
        A_rec = reconstruct(U, S, Vt)
        rel_err = np.linalg.norm(A - A_rec) / np.linalg.norm(A)
        print(f"Relative reconstruction error: {rel_err:.2e}")

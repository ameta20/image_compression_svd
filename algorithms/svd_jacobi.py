import numpy as np


def svd_jacobi(M):
    DBL_EPSILON = 1.0e-15  # approximately
    A = np.copy(M)  # working copy U
    m = len(A)
    n = len(A[0])

    Q = np.eye(n)  # working copy V
    t = np.zeros(n)  # working copy s

    # init counters
    count = 1
    sweep = 0
    sweep_max = max(5 * n, 12)  # heuristic

    tolerance = 10 * m * DBL_EPSILON  # heuristic
    # store the column error estimates in t
    for j in range(n):
        cj = A[:, j]  # get col j
        sj = np.linalg.norm(cj)
        t[j] = DBL_EPSILON * sj

    # orthogonalize A by plane rotations
    while (count > 0 and sweep <= sweep_max):
        # initialize rotation counter
        count = n * (n - 1) / 2;
        for j in range(n - 1):
            for k in range(j + 1, n):
                cj = A[:, j]
                ck = A[:, k]
                p = 2 * np.dot(cj, ck)
                a = np.linalg.norm(cj)
                b = np.linalg.norm(ck)

                # test for columns j,k orthogonal,
                # or dominant errors
                abserr_a = t[j]
                abserr_b = t[k]

                q = (a * a) - (b * b)
                v = np.sqrt(p ** 2 + q ** 2)  # hypot()

                sorted = (a >= b)
                orthog = (abs(p) <= tolerance * (a * b))
                noisya = (a < abserr_a)
                noisyb = (b < abserr_b)

                if sorted and (orthog or \
                               noisya or noisyb):
                    count -= 1
                    continue

                # calculate rotation angles
                if v == 0 or sorted == False:
                    cosine = 0.0
                    sine = 1.0
                else:
                    cosine = np.sqrt((v + q) / (2.0 * v))
                    sine = p / (2.0 * v * cosine)

                # apply rotation to A (U)
                for i in range(m):
                    Aik = A[i][k]
                    Aij = A[i][j]
                    A[i][j] = Aij * cosine + Aik * sine
                    A[i][k] = -Aij * sine + Aik * cosine

                # update singular values
                t[j] = abs(cosine) * abserr_a + abs(sine) * abserr_b
                t[k] = abs(sine) * abserr_a + abs(cosine) * abserr_b

                # apply rotation to Q (V)
                for i in range(n):
                    Qij = Q[i][j]
                    Qik = Q[i][k]
                    Q[i][j] = Qij * cosine + Qik * sine
                    Q[i][k] = -Qij * sine + Qik * cosine

        sweep += 1
    # while

    # compute singular values
    prev_norm = -1.0
    for j in range(n):
        column = A[:, j]  # by ref
        norm = np.linalg.norm(column)
        # determine if singular value is zero
        if norm == 0.0 or prev_norm == 0.0 or \
                (j > 0 and norm < tolerance * prev_norm):
            t[j] = 0.0
            for i in range(len(column)):
                column[i] = 0.0  # updates A indirectly
            prev_norm = 0.0
        else:
            t[j] = norm
            for i in range(len(column)):
                column[i] = column[i] * (1.0 / norm)
            prev_norm = norm

    if count > 0:
        print("Jacobi iterations no converge")

    U = A  # mxn
    s = t
    Vh = np.transpose(Q)

    if m < n:
        U = U[:, 0:m]
        s = t[0:m]
        Vh = Vh[0:m, :]

    return U, s, Vh
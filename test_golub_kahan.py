from svd_golub_kahan import svd_golub_kahan, reconstruct
import numpy as np

np.set_printoptions(precision=4, suppress=True)

A = np.random.randn(5, 3)

U, S, Vt = svd_golub_kahan(A)
A_rec = reconstruct(U, S, Vt)

print("Original A:\n", A)
print("\nReconstructed A:\n", A_rec)
print("\nReconstruction error:", np.linalg.norm(A - A_rec))

#comparison with numpy
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
print("Difference in singular values:", np.linalg.norm(S - S_np))

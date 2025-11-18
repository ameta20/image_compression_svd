import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from algorithms.svd_jacobi import svd_jacobi, reconstruct_from_svd


def test_small_matrix():
    np.set_printoptions(precision=4, suppress=True)

    print("\n====================== SMALL MATRIX TEST (JACOBI) ======================")

    A = np.array([
        [1., 2., 4.],
        [5., 3., 8.],
        [7., 8., 10.],
        [9., 2., 7.]
    ])

    print("\nSource A:\n", A)

    # Compute SVD via Jacobi
    U, S, Vh = svd_jacobi(A)

    print("\nJacobi SVD U:\n", U)
    print("\nJacobi singular values S:\n", S)
    print("\nJacobi Vh:\n", Vh)

    A_rec = reconstruct_from_svd(U, S, Vh)
    print("\nReconstructed A:\n", A_rec)
    print("Reconstruction error:", np.linalg.norm(A - A_rec))

    # Compare with NumPy
    U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    A_np_rec = U_np @ np.diag(S_np) @ Vh_np

    print("\nNumPy SVD reconstruction error:", np.linalg.norm(A - A_np_rec))
    print("Difference in singular values:", np.linalg.norm(S - S_np[:len(S)]))


def test_image_compression():
    print("IMAGE COMPRESSION TEST (JACOBI)\n")

    img = Image.open("images/img_300.jpg").convert("L")
    A = np.array(img, dtype=float) / 255.0

    ranks = [5, 20, 50, 100, 200]

    plt.figure(figsize=(15, 6))
    plt.subplot(1, len(ranks) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, k in enumerate(ranks):
        print(f"Computing rank k={k} Jacobi SVD...")
        U, S, Vh = svd_jacobi(A)
        S_k = S[:k]        # keep first k singular values
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]
        A_k = reconstruct_from_svd(U_k, S_k, Vh_k)

        plt.subplot(1, len(ranks) + 1, i + 2)
        plt.imshow(A_k, cmap='gray')
        plt.title(f"k={k}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_small_matrix()
    test_image_compression()

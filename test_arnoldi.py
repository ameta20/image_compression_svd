import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from algorithms.svd_arnoldi import svd_arnoldi, reconstruct


def test_small_matrix():
    np.set_printoptions(precision=4, suppress=True)

    print("\n====================== SMALL MATRIX TEST ======================")

    A = np.array([
        [1., 2., 4.],
        [5., 3., 8.],
        [7., 8., 10.],
        [9., 2., 7.]
    ])

    print("\nSource A:\n", A)

    # Compute SVD via Arnoldi
    U, S, Vh = svd_arnoldi(A, k=3)

    print("\nArnoldi SVD U:\n", U)
    print("\nArnoldi singular values S:\n", S)
    print("\nArnoldi Vh:\n", Vh)

    A_rec = reconstruct(U, S, Vh)
    print("\nReconstructed A:\n", A_rec)
    print("Reconstruction error:", np.linalg.norm(A - A_rec))

    # Compare with NumPy
    U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    A_np_rec = U_np @ np.diag(S_np) @ Vh_np

    print("\nNumPy SVD reconstruction error:", np.linalg.norm(A - A_np_rec))
    print("Difference in singular values:", np.linalg.norm(S - S_np[:len(S)]))


def test_image_compression():
    print("\nIMAGE COMPRESSION TEST\n")
    img = Image.open("images/test_img.jpg").convert("L")


    A = np.array(img, dtype=float) / 255.0

    ranks = [5, 20, 50, 100, 200]

    plt.figure(figsize=(15, 6))
    plt.subplot(1, len(ranks) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, k in enumerate(ranks):
        print(f"Computing rank k={k} Arnoldi SVD...")
        U, S, Vh = svd_arnoldi(A, k=k)
        A_k = reconstruct(U, S, Vh)

        plt.subplot(1, len(ranks) + 1, i + 2)
        plt.imshow(A_k, cmap='gray')
        plt.title(f"k={k}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_small_matrix()
    test_image_compression()

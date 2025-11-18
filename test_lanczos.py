# test_svd_lanczos.py

from algorithms.svd_lanczos import svd_lanczos, reconstruct_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def test_small_matrix():
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    A = np.array([
        [1, 2, 4],
        [5, 3, 8],
        [7, 8, 10],
        [9, 2, 7]
    ], dtype=float)

    print("\n==================== Small Matrix Test ====================")
    print("Source matrix A:\n", A)

    U, S, Vh = svd_lanczos(A, k=3)

    print("\nLanczos SVD results:")
    print("U =\n", U)
    print("\nSingular values S =\n", S)
    print("\nVh =\n", Vh)

    A_rec = reconstruct_image(U, S, Vh)
    print("\nReconstructed A =\n", A_rec)

    err = np.linalg.norm(A - A_rec)
    print("Reconstruction error (Frobenius norm):", err)

    # Compare with NumPy SVD
    U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    A_np_rec = U_np @ np.diag(S_np) @ Vh_np

    print("\nNumPy SVD reconstruction error:",
          np.linalg.norm(A - A_np_rec))

    print("Difference in singular values:",
          np.linalg.norm(S - S_np[:len(S)]))


def test_image_compression():
    print("\n==================== Image Compression Test ====================")

    img_path = "images/img_300.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Load grayscale image
    A = np.array(Image.open(img_path).convert("L"), dtype=float)
    A = A / 255.0

    ranks = [5, 20, 50, 100, 300]

    plt.figure(figsize=(15, 6))
    plt.subplot(1, len(ranks) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, k in enumerate(ranks):
        print(f"Computing rank-k={k} Lanczos SVD...")
        U, S, Vh = svd_lanczos(A, k=k)
        A_k = reconstruct_image(U, S, Vh)

        plt.subplot(1, len(ranks) + 1, i + 2)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_small_matrix()
    test_image_compression()

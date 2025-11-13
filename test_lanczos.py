from algorithms.svd_lanczos import svd_lanczos, reconstruct_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def test_small_matrix():
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    A = np.array([
        [1, 2, 4],
        [5, 3, 8],
        [7, 8, 10],
        [9, 2, 7]
    ], dtype=float)

    print("\nSource matrix:\n", A)

    U, S, Vh = svd_lanczos(A, k=3)

    print("\nU =\n", U)
    print("\nSingular values S =\n", S)
    print("\nVh =\n", Vh)

    A_reconstructed = reconstruct_image(U, S, Vh)
    print("\nReconstructed A =\n", A_reconstructed)
    print("Reconstruction error (Frobenius norm):", np.linalg.norm(A - A_reconstructed))

    # Compare with NumPy
    U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    print("\nNumPy SVD reconstruction error:", np.linalg.norm(A - U_np @ np.diag(S_np) @ Vh_np))
    print("Difference in singular values:", np.linalg.norm(S - S_np[:len(S)]))

def test_image_compression():
    A = np.array(Image.open("images/test_img.jpg").convert("L"), dtype=float)
    A = A / 255.0  # normalize to [0,1]

    ks = [5, 20, 50, 100, 400]

    plt.figure(figsize=(15, 6))
    plt.subplot(1, len(ks) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, k in enumerate(ks):
        U, S, Vh = svd_lanczos(A, k)
        A_k = reconstruct_image(U, S, Vh)
        plt.subplot(1, len(ks) + 1, i + 2)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_small_matrix()
    test_image_compression()
from algorithms import gram_schmidt_general as gs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    # Example matrix
    A = np.array([
        [1, 2, 4],
        [5, 3, 8],
        [7, 8, 10],
        [9, 2, 7]], dtype=float)

    print("\nSource matrix:\n", A)

    # Run SVD from scratch
    U, S, Vh = gs.svd_gram_schmidt_power(A)

    print("\nU =\n", U)
    print("\nSingular values S =\n", S)
    print("\nVh =\n", Vh)

    # Reconstruct the matrix
    A_reconstructed = U @ np.diag(S) @ Vh
    print("\nReconstructed A =\n", A_reconstructed)
    print("Reconstruction error (Frobenius norm):", np.linalg.norm(A - A_reconstructed))

    # Compare with NumPy's SVD
    U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    print("\nNumPy SVD reconstruction error:", np.linalg.norm(A - U_np @ np.diag(S_np) @ Vh_np))
    print("Difference in singular values:", np.linalg.norm(S - S_np))



if __name__ == "__main__":
    A = np.array(Image.open("images/test_img.jpg").convert("L"), dtype=float)
    A = A / 255.0  # normalize to [0,1]

    ks = [5, 20, 50, 100, 250]

    plt.figure(figsize=(15, 6))
    plt.subplot(1, len(ks) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # -------------------------
    # Reconstruct for different k
    # -------------------------
    for i, k in enumerate(ks):
        U, S, Vh = gs.svd_gram_schmidt_power(A, k)
        A_k = gs.reconstruct_image(U, S, Vh)
        plt.subplot(1, len(ks) + 1, i + 2)
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
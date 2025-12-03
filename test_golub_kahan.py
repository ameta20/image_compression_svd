import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from algorithms.svd_golub_kahan import svd_golub_kahan_full, reconstruct


def test_svd_on_image(image_path, ranks_to_test):

    try:
        img = Image.open(image_path).convert('L')
        A = np.array(img, dtype=float) / 255.0
    except FileNotFoundError:
        print(f"Error: The image file was not found at '{image_path}'")
        return

    m, n = A.shape
    print(f"Image loaded successfully. Shape: {A.shape}")

    print("\nPerforming SVD with your svd_golub_kahan function...")
    start_time = time.time()
    U, S, Vt = svd_golub_kahan_full(A)
    end_time = time.time()
    print(f"SVD computation took {end_time - start_time:.4f} seconds.")

    # Reconstruct the image
    plt.figure(figsize=(16, 8))

    # Display the original image
    plt.subplot(1, len(ranks_to_test) + 1, 1)
    plt.imshow(A, cmap='gray')
    plt.title(f"Original\n({m}x{n})")
    plt.axis('off')

    for i, k in enumerate(ranks_to_test):
        if k > len(S):
            print(f"Warning: Rank {k} is greater than the max possible rank {len(S)}. Skipping.")
            continue

        print(f"\nReconstructing with rank k = {k}...")

        A_reconstructed = reconstruct(U, S, Vt, k)

        original_data_size = m * n
        compressed_data_size = (m * k) + k + (k * n)
        compression_ratio = original_data_size / compressed_data_size

        plt.subplot(1, len(ranks_to_test) + 1, i + 2)
        plt.imshow(A_reconstructed, cmap='gray')
        plt.title(f"Rank k = {k}\nRatio: {compression_ratio:.1f}x")
        plt.axis('off')

    plt.suptitle("SVD Image Compression", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMAGE_FILE_PATH = 'images/img_300.jpg'

    RANKS = [5, 15, 30, 50, 100]

    test_svd_on_image(IMAGE_FILE_PATH, RANKS)
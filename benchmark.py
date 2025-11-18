import numpy as np
import time
import os
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from algorithms.svd_jacobi import svd_jacobi
from algorithms.svd_lanczos import svd_lanczos
from algorithms.svd_golub_kahan import svd_golub_kahan


def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=float) / 255.0

def run_algorithm(alg_fn, A, k=None):
    t0 = time.time()

    if k is not None:
        U, s, Vh = alg_fn(A, k)
    else:
        U, s, Vh = alg_fn(A)

    dt = time.time() - t0
    A_rec = U @ np.diag(s) @ Vh

    # errors
    err = np.linalg.norm(A - A_rec)
    rel = err / np.linalg.norm(A)

    # quality metrics
    psnr = peak_signal_noise_ratio(A, A_rec, data_range=1.0)
    ssim = structural_similarity(A, A_rec, data_range=1.0)

    return dt, err, rel, psnr, ssim, A_rec


def save_image(array, path):
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def benchmark_images():

    print("\nIMAGE COMPRESSION BENCHMARK")

    os.makedirs("results/compressed_images", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # dataframe to store results
    df = pd.DataFrame(columns=[
        "Image", "Algorithm", "k",
        "Time (s)", "Abs Error", "Rel Error",
        "PSNR", "SSIM"
    ])

    images = [
        "images/img_300.jpg",
        "images/img_700.jpg",
        "images/img_1000.jpg"
    ]

    ks = [10, 50, 100, 200]

    algorithms = [
        ("Jacobi", svd_jacobi),
        ("Lanczos", svd_lanczos),
        ("Golub-Kahan", svd_golub_kahan)
    ]

    # process images
    for img_path in images:
        A = load_image(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nTesting {img_name}, shape={A.shape}")

        for alg_name, alg_fn in algorithms:
            print(f"\nAlgorithm: {alg_name}")

            outdir = f"results/compressed_images/{alg_name}"
            os.makedirs(outdir, exist_ok=True)

            for k in ks:
                print(f"  k={k}")

                dt, err, rel, psnr, ssim, A_rec = run_algorithm(alg_fn, A, k)

                print(f"   time={dt:.3f}s  rel.err={rel:.3e}  PSNR={psnr:.2f}  SSIM={ssim:.4f}")

                # save compressed image
                save_path = f"{outdir}/{img_name}_k{k}.jpg"
                save_image(A_rec, save_path)

                # add row to dataframe
                df.loc[len(df)] = [
                    img_name, alg_name, k,
                    dt, err, rel,
                    psnr, ssim
                ]

    df.to_csv("results/benchmark_results.csv", index=False)
    print("\nBenchmark completed!")


if __name__ == "__main__":
    benchmark_images()

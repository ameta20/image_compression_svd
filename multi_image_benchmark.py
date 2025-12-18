import numpy as np
import time
import os
import pandas as pd
from PIL import Image
from multiprocessing import Pool
import random
import threading
import psutil

from algorithms.svd_golub_kahan_v2 import compress_image
from algorithms.svd_golub_kahan_parallelised import compress_image_parallelised

# CONFIG
number_of_images = 50
no_procs = 3
image_seed = 42
image_size = 256

cpu_samples = []
monitoring = False


# CPU monitoring for parallel benchmarks
def monitor_cpu_usage(interval=0.2):
    global cpu_samples, monitoring
    monitoring = True
    while monitoring:
        cpu_samples.append(psutil.cpu_percent(interval=interval))


# Image helpers
def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=float) / 255.0


def save_image(array, path):
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def pick_random_images(folder="images", n=100, seed=42):
    random.seed(seed)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    all_images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

    if len(all_images) < n:
        raise ValueError(f"Not enough images in folder: found {len(all_images)}, need {n}")

    selected = random.sample(all_images, n)
    return selected


# Worker functions
def process_single_image(args):
    """
    Worker using the baseline Golub–Kahan implementation (compress_image).
    """
    img_path, k = args
    A = load_image(img_path)

    # Resize to fixed size for fair comparison
    A = np.array(Image.fromarray((A * 255).astype(np.uint8)).resize((image_size, image_size))) / 255.0

    t0 = time.time()
    A_rec, err, rel, psnr, ssim = compress_image(A, k)
    img_time = time.time() - t0
    return img_path, k, A_rec, err, rel, psnr, ssim, img_time


def process_single_image_parallelised(args):
    """
    Worker using the Numba-parallelised Golub–Kahan implementation
    (compress_image_parallelised).
    """
    img_path, k = args
    A = load_image(img_path)

    # Same resize as baseline to keep things comparable
    A = np.array(Image.fromarray((A * 255).astype(np.uint8)).resize((image_size, image_size))) / 255.0

    t0 = time.time()
    A_rec, err, rel, psnr, ssim = compress_image_parallelised(A, k)
    img_time = time.time() - t0
    return img_path, k, A_rec, err, rel, psnr, ssim, img_time


# SEQUENTIAL benchmark (baseline compress_image)
def benchmark_images_sequential(images, k, df):
    start = time.time()

    for img_path in images:
        A = load_image(img_path)
        A = np.array(Image.fromarray((A * 255).astype(np.uint8)).resize((image_size, image_size))) / 255.0
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        print(f"Sequential: Processing {img_name}")

        t_img = time.time()
        A_rec, err, rel, psnr, ssim = compress_image(A, k)
        img_time = time.time() - t_img

        save_path = f"results/compressed_images/parallel_benchmark/{img_name}_k{k}_seq.jpg"
        save_image(A_rec, save_path)

        df.loc[len(df)] = [
            img_name,
            "sequential",
            k,
            err, rel, psnr, ssim,
            img_time
        ]

    total_time = time.time() - start
    print(f"\nTotal sequential runtime: {total_time:.3f} seconds\n")

    return total_time, df


# SEQUENTIAL benchmark (Numba-parallelised compress_image_parallelised)
def benchmark_images_sequential_numba(images, k, df):
    start = time.time()

    for img_path in images:
        A = load_image(img_path)
        A = np.array(Image.fromarray((A * 255).astype(np.uint8)).resize((image_size, image_size))) / 255.0
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        print(f"Sequential (Numba): Processing {img_name}")

        t_img = time.time()
        A_rec, err, rel, psnr, ssim = compress_image_parallelised(A, k)
        img_time = time.time() - t_img

        save_path = f"results/compressed_images/parallel_benchmark/{img_name}_k{k}_seq_numba.jpg"
        save_image(A_rec, save_path)

        df.loc[len(df)] = [
            img_name,
            "sequential_numba",
            k,
            err, rel, psnr, ssim,
            img_time
        ]

    total_time = time.time() - start
    print(f"\nTotal sequential (Numba) runtime: {total_time:.3f} seconds\n")

    return total_time, df


# PARALLEL benchmark (multi-image, baseline compress_image)
def benchmark_images_parallel(images, k, df):
    print("\nIMAGE COMPRESSION BENCHMARK (MULTIPLE IMAGE IN PARALLEL, baseline Golub–Kahan)")

    jobs = [(img_path, k) for img_path in images]
    print(f"\nRunning {len(jobs)} parallel tasks (baseline)...\n")

    start = time.time()
    # with Pool(processes=no_procs) as p:
    with Pool() as p:
        results = p.map(process_single_image, jobs)
    total_time = time.time() - start

    print(f"\nTotal parallel runtime (baseline)   : {total_time:.3f} seconds\n")

    for img_path, k_val, A_rec, err, rel, psnr, ssim, img_time in results:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        save_path = f"results/compressed_images/parallel_benchmark/{img_name}_k{k_val}_parallel_baseline.jpg"
        save_image(A_rec, save_path)

        df.loc[len(df)] = [
            img_name,
            "parallel_baseline",
            k_val,
            err, rel, psnr, ssim,
            img_time
        ]

    return total_time, df


# PARALLEL benchmark (multi-image, Numba-parallelised SVD)
def benchmark_images_parallel_numba(images, k, df):
    print("\nIMAGE COMPRESSION BENCHMARK (MULTIPLE IMAGE IN PARALLEL, Numba-parallelised Golub–Kahan)")

    jobs = [(img_path, k) for img_path in images]
    print(f"\nRunning {len(jobs)} parallel tasks (Numba-parallelised)...\n")

    start = time.time()
    # with Pool(processes=no_procs) as p:
    with Pool() as p:
        results = p.map(process_single_image_parallelised, jobs)
    total_time = time.time() - start

    print(f"\nTotal parallel runtime (Numba)      : {total_time:.3f} seconds\n")

    for img_path, k_val, A_rec, err, rel, psnr, ssim, img_time in results:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        save_path = f"results/compressed_images/parallel_benchmark/{img_name}_k{k_val}_parallel_numba.jpg"
        save_image(A_rec, save_path)

        df.loc[len(df)] = [
            img_name,
            "parallel_numba",
            k_val,
            err, rel, psnr, ssim,
            img_time
        ]

    return total_time, df


def run_all_benchmarks():
    global monitoring

    os.makedirs("results/compressed_images/parallel_benchmark", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    images = pick_random_images(folder="images", n=number_of_images, seed=image_seed)
    print(f"Selected {len(images)} images for benchmarking.")

    k = 50

    df = pd.DataFrame(columns=[
        "Image", "Mode", "k",
        "Abs Error", "Rel Error",
        "PSNR", "SSIM", "Time (s)"
    ])

    # Sequential baseline
    print("\nRUNNING SEQUENTIAL BENCHMARK (baseline compress_image)")
    seq_time, df = benchmark_images_sequential(images, k, df)

    # Multi-image parallel (baseline SVD)
    cpu_samples.clear()
    monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
    monitor_thread.start()

    print("\nRUNNING MULTIPLE IMAGE AT A TIME BENCHMARK (baseline)")
    par_time, df = benchmark_images_parallel(images, k, df)

    monitoring = False
    time.sleep(0.3)

    num_cores = psutil.cpu_count(logical=True)
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    effective_cores = (avg_cpu / 100) * num_cores

    # Sequential (Numba-parallelised compress_image_parallelised)
    print("\nRUNNING SEQUENTIAL BENCHMARK (Numba-parallelised compress_image)")
    seq_numba_time, df = benchmark_images_sequential_numba(images, k, df)

    # Multi-image parallel (Numba-parallelised SVD)

    print("\nRUNNING MULTIPLE IMAGE AT A TIME BENCHMARK (Numba-parallelised)")
    par_numba_time, df = benchmark_images_parallel_numba(images, k, df)

    # Speedups
    speedup_parallel    = seq_time / par_time if par_time > 0 else None
    speedup_numba       = seq_time / par_numba_time if par_numba_time > 0 else None
    speedup_seq_numba   = seq_time / seq_numba_time if seq_numba_time > 0 else None

    print(f"\nAvailable CPU cores          : {num_cores}")
    print(f"Average CPU utilization (par): {avg_cpu:.1f}%")
    print(f"Estimated cores utilized     : {effective_cores:.2f}")

    print(f"\nSequential time              = {seq_time:.3f}s")
    print(f"Sequential Numba time        = {seq_numba_time:.3f}s")
    print(f"Parallel baseline time       = {par_time:.3f}s")
    print(f"Parallel Numba time          = {par_numba_time:.3f}s")
    print("---------------------------------------------------------")
    print(f"Speedup (Numba sequential)   = {speedup_seq_numba:.2f}x")
    print(f"Speedup (baseline parallel)  = {speedup_parallel:.2f}x")
    print(f"Speedup (Numba parallel)     = {speedup_numba:.2f}x")

    df.to_csv("results/benchmark_results_comparison.csv", index=False)


if __name__ == "__main__":
    run_all_benchmarks()

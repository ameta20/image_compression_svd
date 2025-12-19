# Image Compression using SVD Decomposition

## Project Overview

This project implements **image compression using Singular Value Decomposition (SVD)** computed from scratch. The focus is on the **Golub–Kahan algorithm**, a numerically stable and efficient method for computing SVD. Multiple implementations are provided to explore performance trade-offs between correctness, numerical stability, and computational speed through parallelization strategies.

The project evaluates:
- **Algorithmic correctness** via reconstruction quality metrics (PSNR, SSIM)
- **Computational performance** through benchmarking across sequential and parallel implementations
- **Parallelization strategies** at both image-level and kernel-level

Dataset: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

---

## Implemented Algorithms

The `algorithms/` directory contains SVD implementations:

### `svd_golub_kahan_v2.py` – Baseline Implementation
- **Purpose:** Reference Golub–Kahan SVD implementation using NumPy
- **Usage:** Use for correctness validation and as a baseline for comparing numerical accuracy
- **Characteristics:**
  - Pure NumPy-based implementation
  - No parallelization overhead
  - Ideal for small to medium-sized matrices where clarity is prioritized

### `svd_golub_kahan_parallelised.py` – Parallelized Implementation
- **Purpose:** Optimized Golub–Kahan SVD with Numba JIT compilation and algorithmic-level parallelism
- **Usage:** Use for performance-critical applications on multi-core systems
- **Characteristics:**
  - Numba `@njit` decorator for C-speed execution
  - Parallel loops using Numba `prange`
  - Produces identical results to the baseline with significantly reduced runtime

### Other Implementations
- `svd_jacobi.py`: Jacobi eigenvalue method (full SVD, O(n³) complexity)
- `svd_lanczos.py`: Krylov subspace method (approximate SVD for large sparse matrices)

---

## Parallelization Strategy

Two levels of parallelism are employed to maximize performance:

### Image-Level Parallelism
- **Method:** Python `multiprocessing` to process multiple images concurrently
- **Scope:** Distributes independent image compression tasks across available CPU cores
- **Benefit:** Linear scaling with number of available cores for batch operations

### Kernel-Level Parallelism
- **Method:** Numba JIT compilation with `@njit(parallel=True)` and `prange`
- **Scope:** Parallelizes inner loops within the Golub–Kahan algorithm
- **Benefit:** Fine-grained parallelism within the SVD computation

**Important:** The mathematical structure of the algorithms remains unchanged. All parallelized implementations produce byte-identical SVD decompositions to their sequential counterparts.

---

## Benchmarking

### Benchmark Script: `multi_image_benchmark.py`

This script compares the performance of four implementation configurations:

1. **Sequential execution** – Baseline NumPy implementation, single-threaded
2. **Sequential + Numba** – Baseline with Numba JIT compilation, no parallelism
3. **Parallel baseline** – Image-level multiprocessing, sequential SVD computation
4. **Parallel + Numba** – Full parallelization (image-level + kernel-level)

### Measured Metrics

| Metric | Description |
|--------|-------------|
| **Execution Time** | Wall-clock time for compression pipeline |
| **Speedup** | Performance relative to sequential baseline |
| **PSNR** (Peak Signal-to-Noise Ratio) | Reconstruction fidelity (dB) |
| **SSIM** (Structural Similarity Index) | Perceptual similarity [0, 1] |

Results are exported to CSV for analysis and visualization.

---

## How to Run

### Basic Benchmark
```bash
python multi_image_benchmark.py --num_images 50 --image_size 256 --rank 10 --num_cores 4
```

### Configurable Parameters

- `--num_images`: Number of images to compress (default: 10)
- `--image_size`: Target image dimensions (default: 256)
- `--rank`: SVD rank (k) for low-rank approximation (default: 50)
- `--num_cores`: Number of CPU cores for multiprocessing (default: all available)
- `--numba_threads`: Number of threads for Numba parallelism (default: all available)

### Example Usage
```bash
# Benchmark 100 images at 512×512 with rank 100 on 4 cores
python multi_image_benchmark.py --num_images 100 --image_size 512 --rank 100 --num_cores 4

# Run with default parameters
python multi_image_benchmark.py
```

---

## Results Summary

Benchmarking results demonstrate that:

- **Parallel + Numba configuration achieves 4–8× speedup** on 4-core systems compared to sequential baseline
- **Speedup scales linearly** with the number of cores for batch operations
- **All implementations produce identical reconstruction quality** (PSNR and SSIM metrics match across variants)
- **Numba JIT provides 2–3× speedup** even without parallelization
- **Overhead from multiprocessing is negligible** when processing multiple images

Results are stored in `results/ready_benchmarks/` with timestamped directories containing:
- `benchmark_results_comparison.csv` – Detailed timing and quality metrics
- `img/` – Compressed image samples

---

## Project Structure

```
image_compression_svd/
├── algorithms/                    # SVD implementations
│   ├── svd_golub_kahan_v2.py     # Baseline implementation
│   ├── svd_golub_kahan_parallelised.py  # Parallelized version
│   ├── svd_jacobi.py
│   └── svd_lanczos.py
├── images/                        # Test images
├── results/                       # Benchmarking results
│   ├── compressed_images/        # Compressed output
│   └── ready_benchmarks/         # Benchmark reports
├── multi_image_benchmark.py      # Main benchmarking script
├── benchmark.py                  # Single-image benchmarking
├── validation_of_algorithms.ipynb # Algorithm validation
└── README.md
```

---

## Citation & References

This implementation is based on the Golub–Kahan SVD algorithm described in:
- Golub, G. H., & Kahan, W. (1965). "Calculating the singular values and pseudo-inverse of a matrix". *SIAM Journal on Numerical Analysis*, 2(2), 205-224.

# Image Compression using SVD Decomposition

## Overview
This project explores **image compression** through **Singular Value Decomposition (SVD)**, implemented from scratch.


---

## Tasks: 

1. **Select an image dataset**  
   - For testing and benchmarking, we will use the database of images **Oxford-IIIT Pet Dataset**
   - Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/

2. **Implement SVD decomposition from scratch**  
   - 3 algorithms will be tested for computing SVD:
     - **Jacobi algorithm**
     - **Lanczos algorithm**
     - **Gram-Schmidt Orthogonalization**
   - We will benchmark these algorithms to decide which one will serve as the base for applying parallelization.

3. **Benchmark sequential versions**  
   - The best-performing algorithm (from the sequential tests) will be **parallelized**.
   - Benchmarking will compare performance between sequential implementations.

4. **Analyze compression quality**  
   - The compressed images will be compared to the originals using:
     - **Structural Similarity Index (SSIM)**
     - **Peak Signal-to-Noise Ratio (PSNR)**

---

## Current Stage
We are currently:
- Implementing and testing different **SVD algorithms** in their **sequential** versions.  
- Preparing for benchmarking to select the most suitable method for **parallelization**.

---

## Next Steps
- Finalize algorithm selection based on performance and accuracy.  
- Implement the **parallel version** using an **HPC-oriented approach**.  
- Conduct full benchmarking and quality evaluation across the dataset.

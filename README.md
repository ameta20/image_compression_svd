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
     - **Arnoldi iteration**
   - We will benchmark these algorithms to decide which one will serve as the base for applying parallelization.

3. **Benchmark sequential versions**  
   - The best-performing algorithm (from the sequential tests) will be **parallelized**.
   - Benchmarking will compare performance between sequential implementations.

4. **Analyze compression quality**  
   - The compressed images will be compared to the originals using:
     - **Structural Similarity Index (SSIM)**
     - **Peak Signal-to-Noise Ratio (PSNR)**

---

### Algorithms Characteristics

#### 1. Jacobi Method
- **Type:** Iterative diagonalization algorithm  
- **Matrix type:** Works on any real matrix  
- **Characteristics:**
  - It produces full SVD  
  - **Computationally expensive** O(n³) complexity

#### 2. Lanczos Algorithm
- **Type:** Iterative projection method (Krylov subspace)  
- **Matrix type:** Designed for symmetric matrices (applied to AᵀA for SVD), not directly to A  
- **Characteristics:** 
  - Produces good approximations of dominant singular values/vectors  
  - **Parallelizable**: matrix–vector multiplications

#### 3. Gram-Schmidt Orthogonalization
- **Type:** Orthogonalization process (basis generation)  
- **Matrix type:** Any full-rank matrix  
- **Characteristics:**
  - Simple, but **numerically unstable** for large matrices  
  - Not a complete SVD algorithm by itself, it is used in **QR decomposition**

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

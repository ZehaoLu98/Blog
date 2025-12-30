---
title: Scaling Performance of Cublas GEMMs
published: 2025-12-29
tags: [GPU, GEMM, Profiling]
category: Performance Analysis
draft: false
lang: 'en'
---

:::warning
Under Construction!
:::

# Introduction
Attention layer is among the most time-consuming part of any LLM. It consists of multiple matrix multiplications that are theoretically bottlenecked by tensor core throughput according to the roofline model from this post[link]. However, our result reveals that the memory throughput instead is dragging down the performance of the attention layer. This discrepancy between theory and reality leads to this post. 

GEMM(i.e. General Matrix Multiplication) is always one of the hottest topic in HPC. The transformer architecture made this even more stark: every attention layer, every feed-forward block, every projection is fundamentally a GEMM, therefore even small amount of improvement on GEMM can cause huge impact on overall system performance. With the increasing problem size, the performance usually starts to be limited by either the memory throughput or compute resources. Through this post, I will introduce the GEMM performance of various matrix size, shape and batch size, revealing the bottleneck in different conditions. The attention layer GEMM performance of the popular open-source model, llama3, will also be provided for practical analysis.   

I will test 4 ways of computing GEMM using cuBLAS. To compare apples to apples, it is ensured that every cuBLAS GEMM performs the same amount of FLOPs with FP32.

## Single Big GEMM
This is simply calling *cublasSgemm* with wider matrix compared to other methods. This is expected to perform at least no worse than other metrics because:
* More oppotunities to reuse the memory.
* More algorithms available for larger matrices.
* Less padding needed for tiling compared to smaller GEMMs.
* Less kernels invoked.

Therefore the single big GEMM is considered as a upper bound on Performance among all the methods.

## Naive GEMM
Given many small matrix multiplications, we have two ways of distributing the work to GPU: either launching one kernel for each matrix multiplications or fused the kernels into one or several mega-kernels. The so called Naive GEMM refers to the first approach, by continuously launching per-multiplication small kernels in a for-loop. Here is a piece of sample code: 

```cpp
for (int i = 0; i < batch_size; i++) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k_dim,
                            &alpha,
                            d_A_array[i], m,
                            d_B_array[i], k_dim,
                            &beta,
                            d_C_array[i], m));
}
```

## Batched GEMM
As mentioned before, the second approach for many small matrix multiplications is batching. The Batched GEMM allows you to perform multiple independent matrix multiplications in a single or small amount of kernel launches, which are supposed to be much more efficient than launching separate GEMM operations. To batch the GEMM, we simply prepare the input matrices as array of pointers and call the corresponding CUBLASS API:

```cpp
// Allocating DEVICE matrices and store the pointers in HOST vectors
std::vector<float*> d_A_array(batch_size), d_B_array(batch_size), d_C_array(batch_size);
for (int i = 0; i < batch_size; i++) {
    if (cudaMalloc(&d_A_array[i], size_A) != cudaSuccess ||
        cudaMalloc(&d_B_array[i], size_B) != cudaSuccess ||
        cudaMalloc(&d_C_array[i], size_C) != cudaSuccess) {
        printf("CUDA malloc failed for Batched Gemm");
        for (int j = 0; j <= i; j++) {
            if (d_A_array[j]) cudaFree(d_A_array[j]);
            if (d_B_array[j]) cudaFree(d_B_array[j]);
            if (d_C_array[j]) cudaFree(d_C_array[j]);
        }
        return false;
    }
}

// Allocating DEVICE vectors to store DEVICE pointers to arrays 
float **d_A_ptr, **d_B_ptr, **d_C_ptr;
if (cudaMalloc(&d_A_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
    cudaMalloc(&d_B_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
    cudaMalloc(&d_C_ptr, batch_size * sizeof(float*)) != cudaSuccess) {
    printf("CUDA malloc failed for pointer arrays in testBatchedGemm - skipping test\n");
    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_A_array[i]);
        cudaFree(d_B_array[i]);
        cudaFree(d_C_array[i]);
    }
    return false;
}

// Copy pointers from HOST to DEVICE
CHECK_CUDA(cudaMemcpy(d_A_ptr, d_A_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_B_ptr, d_B_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_C_ptr, d_C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));

CHECK_CUBLAS(cublasSgemmBatched(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, k_dim,
                                &alpha,
                                (const float**)d_A_ptr, m,
                                (const float**)d_B_ptr, k_dim,
                                &beta,
                                d_C_ptr, m,
                                batch_size));
```

## Strided Batched GEMM
Similar to Batched GEMM, cuBLAS provides another way to batch small GEMMs. The essential difference of the two batched GEMM are the memory layout of the input matrices. Batched GEMM stores the matrices separately, whereas Strided Batched GEMM utilizes a contiguous piece memory to store each matrix. Instead of accepting pointer arrays to matrices, Strided Batched GEMM receives the base device pointers of matrices along with the stride, which are used to compute the address of sub-matrices.

The benefit of this method is clear: for array-of-pointer solution, it needs to first read the pointer, and then read the data from the memory. However, only one read is needed for the pointer-with-stride option because the pointer is calculated through the base pointer, batch id and stride.

Here is the sample code:
```cpp

// Pointers to matrices on the DEVICE
float *d_A, *d_B, *d_C;

// Allocate DEVICE memory
if (cudaMalloc(&d_A, total_size_A) != cudaSuccess ||
    cudaMalloc(&d_B, total_size_B) != cudaSuccess ||
    cudaMalloc(&d_C, total_size_C) != cudaSuccess) {
    printf("CUDA malloc failed in testStridedBatchedGemm - skipping test\n");
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    return false;
}

CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n, k_dim,
                                        &alpha,
                                        d_A, m, stride_A,
                                        d_B, k_dim, stride_B,
                                        &beta,
                                        d_C, m, stride_C,
                                        batch_size));
```
# Performance Analysis

## k batches of (N x N) matmul (N x N)

## Pratical Workloads



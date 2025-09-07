---
title: Nvidia GPU Basics
published: 2025-09-01
description: ''
image: ''
tags: [GPU, Note]
category: 'Computer Architecture'
draft: false 
lang: 'en'
---
# Terminology List
- Kernel: function run on each thread
- Device: GPU
- Host: CPU

# Introduction
Artificial intelligence is now a part of everyday life, powering everything from search engines to chatbots. But behind the scenes, it takes enormous compute power to train and run large language models (LLMs). This is where GPUs step into the spotlight. Their architecture is uniquely designed to handle the scale of parallelism that modern AI demands. In this post, I’ll introduce some of the core ideas of CUDA programming, Nvidia GPU architecture, and highlight how Nvidia software and hardware work together.
## GPU vs CPU
CPUs are built for versatility. They have a small number of powerful cores, optimized for handling complex tasks one after another. GPUs, on the other hand, are built for raw parallelism. A single high-end GPU like the GB202 packs over 24,000 CUDA cores—orders of magnitude more than a typical CPU.

Of course, having thousands of cores only helps if you have thousands of tasks to run in parallel. That’s why GPU programming looks very different from CPU programming: instead of focusing on a handful of threads, you need to think in terms of thousands.

Additionally, the GPU and the host memory are usually disjoint, so an explicit memory transfer call is usually required to move the data between the CPU and GPU. It is totally different in a CPU program, where the memory load/store are usually implicit.

Finally, the GPU may adhere to a different floating point representation, which may cause the result of a GPU program differs from a CPU program.
:::tip
It's a good habbit to verify the result of the GPU program with an equivalent CPU result.
:::

# CUDA Programming Model
A CUDA program is essentially a C/C++ program with some CUDA extension. The CUDA part, i.e. device code, will be compiled separately and linked with the host code.

## CUDA Function Definition
The functions are devided into three categories:

- Host: called and run on the CPU
- Global: called on the CPU/GPU and run on the GPU
- Device: called and run on the GPU

They are marked with \_\_host\_\_(can be ommited) \_\_global\_\_ and \_\_device\_\_ in the function definition, for example:

```c++
__global__ void vecAdd(float *d_a, float *d_b, float *d_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_c[i] = d_a[i] + d_b[i];
    }
}
```
:::note
Host and Device address are in different spaces, therefore you cannot pass an host address space pointer to a function marked \_\_global\_\_ as a parameter.
:::
:::tip
Use prefix "h_", "d_" to differentiate pointers in host and device address space.
:::

To call a kernel, the grid size and block size need to be specified.
```c++
vecAdd<<<gridDim, blockDim, sharedMemBytes, stream>>>(d_a, d_b, d_c, N);
``` 
gridDim and block Dim will be detailed in the next section. You can also explicitly specify how many bytes of shared memory for each block, and the shared memory will be discussed in the memory architecture. By default, kernels will be launched sequentially, one by one. If more streams are defined, more kernels can be run simultaneously.

Another thing worth noticing is that the pointers like d_a, d_b and d_c need to be allocated on the device memory first before passing to the kernel. You need to explicily "malloc" on the device memory by calling
```c++
cudaMalloc((void**)&d_data, size);
cudaFree(d_data);
```
To initialize the device memory, you need to either memset or memcopy
```c++
cudaMemset(d_array, 0, N * sizeof(int));
cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
```

Here is a piece of sample code:
```c++
// vec_add.cu
#include <cstdio>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                          \
do {                                                                              \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                __FILE__, __LINE__);                                              \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

__global__ void vecAdd(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1 << 20;                // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Host data
    std::vector<float> hA(N), hB(N), hC(N);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
    }

    // Device pointers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocate device memory (cudaMalloc)
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Copy host -> device (cudaMemcpy)
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // Clear output buffer (cudaMemset) — safe because we set to 0.0f bytes
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Launch kernel
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy device -> host (cudaMemcpy)
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Quick verify
    double max_abs_err = 0.0;
    for (int i = 0; i < 10; ++i) { // spot check first few
        printf("C[%d] = %f\n", i, hC[i]);
    }
    for (int i = 0; i < N; ++i) {
        max_abs_err = fmax(max_abs_err, std::fabs(hC[i] - (hA[i] + hB[i])));
    }
    printf("Max abs error: %g\n", max_abs_err);

    // Free device memory (cudaFree)
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
```

## Thread Organization
In kernel invocation, you specified two 3d structure *gridSize* and *blockSize*, which actually defines the thread orgnization, and has a profound impact on the performance of the kernel.

The blocks are orgnized in a 3D structure, called grid. Each block is a group of threads in at most 3 dimensions. The threads are aware of its position within the block and its block's position within the grid through CUDA defined variables, which can be used for retrieving the data needed for its workload. Two 3D intrinsic coordinate, *threadIdx* and *blockIdx*, along with two 3D dimension, blockDim and gridDim, can be used to calculate the global coordinate of the thread within the grid.
```c++
 int i = blockIdx.x * blockDim.x + threadIdx.x;ß
```
```tip
You can think of the orgnization of threads like a 6D array:
```c++
Thread t[gridDim.z][gridDim.y][gridDim.x][blockDim.z][blockDim.y][blockDim.x];
```
Or a 3D array:
```c++
Thread t[gridDim.z*blockDim.z]
```
The dimension of the grid doesn't matter performance-wise. For example, (16,16,0) and (256, 0, 0) both consist of 256 blocks, and from GPU side, there is no real difference. What really matters is the number of blocks. Each block can only run on a single SM(Steamming Multiprocessor), so to fully use all the GPU compute resources, at least SM-size of blcoks should be defined if you don't want to waste resources. The number of SMs varies among different GPU devices. On A100, there are 108 SMs, so at least 108 blocks should be given, no matter how the grid is defined, e.g. there is no difference between (4, 27, 0) and (108, 0, 0). This also applies to block dimensions. The dimensions of the blcok doesn't affect the performance, but the total number of threads per block matters.

Addition to the Grid and Blocks, threads on a block can be further separate to different warps, which are the scheduling units of the threads on an SM that are executing the same instructions and are scheduled to run at the same time. This is the __SIMT__ part of the GPU. This will be discussed when the architeture is explained.

# Architecture
Writing a runnable CUDA program is easy, but writing the one that can unleash the full power of GPU is definetely more difficult. It requires a good understanding of GPU architecture and Memory Hierarchy.

## Compute Resources
### SP  
GPU cores are essentially vector processing units that can apply a single instruction on large amount of data. Each of the core is called *SP*(Stream Processor) or CUDA core and has its own pipeline. In each cycle, a warp scheduler issues one command to one warp, which usually consists of 32 threads. These 32 threads will execute the instruction on 32 SPs simultaneously. This is why this execution model is called SIMT(Single Instruction Multiple Threads), as one instruction can be broadcasted to multiple data and execute on different pipelines at the same time. SIMD is similar to the SIMT. The only difference between SIMT and SIMD is that the number of parallel execution is determined by software instead of hardware. 
### SM
Multiple SPs executes instructions under the control of a single *SM*(Streaming Muliprocessor).   the w One GPU can have lots of SMs. As mentioned above, A100 have 108 SMs. 

## Memory Architecture

### DRAM

### L2/GLOBAL

### L1/Shared

### Registers

# More on Threads


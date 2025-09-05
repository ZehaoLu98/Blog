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
Nowadays, AI becomes an essential tool in our lives. With the tremendous compute resources needed by both training and inference of LLMs, GPUs were brought under the spot light because of its unique competence. In this blog, I will go through the most important concepts of both CUDA programming and Nvidia GPU architecture and explore the relationship between GPU software and hardware. 
## GPU vs CPU
Unlike CPU, GPUs devote lots of its silicon real estate into compute resources. The most recent GB202, for example, have 24,576 CUDA cores. In contrary, most mordern CPUs have only less than 100 cores. 

However, to make use all the cores, we need at least 24,576 threads, and even more if considering the switch of threads because of the stall. This implies that GPU programming style should be totally different from CPU programming, requiring to spawn thousands of threads so that most of the compute resources on a GPU are utilized.

Additionally, the GPU and the host memory are usually disjoint, so an explicit memory transfer call is usually needed to move the data between the CPU and GPU. It is totally different in a CPU program, where the memory load/store are usually implicit.

Finally, the GPU may adhere to a different floating point representation, which may cause the result of a GPU program differs from a CPU program.
:::tip
It's a good habbit to verify the result of the GPU program with the result of an equivalent CPU program.
:::

# CUDA Programming Model
A CUDA program is essentially a C/C++ program with some CUDA extension. The CUDA part, i.e. device code, will be compiled separately and linked with the host part.

## CUDA Function Definition
The functions are devided into three categories:

- Host: called and run on the CPU
- Global: called on the CPU/GPU and run on the GPU
- Device: called and run on the GPU

They are marked with \_\_host\_\_(can be ommited) \_\_global\_\_ and \_\_device\_\_ in the function definition, for example:

```c++
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    if (i < N) {
        C[i] = A[i] + B[i];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    }
}
```
:::note
Host and Device address are in different spaces, therefore you cannot pass an host address space pointer to a function marked \_\_global\_\_ as a parameter.
:::
:::tip
Use prefix "h_", "d_" to differentiate pointers in host and device address space.
:::

## Thread Organization

How a kernel is executed
Threads, Blocks and Grid


# Architecture
big picture

## Compute Resources
### SP  
### SM

## Memory Architecture

### DRAM

### L2/GLOBAL

### L1/Shared

### Registers

# More on Threads


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


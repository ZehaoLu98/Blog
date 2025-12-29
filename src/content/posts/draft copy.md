---
title: Scaling Performance of Cublas GEMMs
published: 2025-07-01
tags: [Markdown, Blogging, Demo]
category: Examples
draft: true
---

# introduction
GEMM is always one of the hot topic in HPC, bluh bluh

We use SGEMM throughout all the anaylysis.

We will test 4 types of cublas GEMM. To compare apples to apples, we ensure that every cublas operations perform same amount of FLOPs.
## Single Big GEMM
This is simply calling *cublasSgemm* with wider matrix compared to other methods. This is expected to perform at least no worse than other metrics because:
* More oppotunities to reuse the memory.
* More algorithms available for larger matrices.
* Less padding needed for tiling compared to smaller GEMMs.
* Less kernels invoked.

# Naive GEMM
This is

## Batched GEMM
Batched GEMM performs large amount of GEMMs with a single call. 

## Strided Batched GEMM

# Performance Analysis

## k batches of (N x N) matmul (N x N)

## Pratical Workloads



---
title: Building a GPU Profiler From Scratch Using CUPTI
published: 2025-09-03
description: 'How to create a fake, customized Nsight Compute'
image: ''
tags: [GPU, Note]
category: 'Computer Architecture'
draft: false 
lang: 'en'
---
# CUPTI Introduction

CUPTI is a profiling and tracing api that exposes the hardware counters of NVIDIA GPUs, collects CUDA runtime information and enables developers to build custom profilers. If you used Nsight Compute or Nsight System before, you won't be unfamiliar with it, because they are built on top of CUPTI, which means you can built whatever Nsight Compute or Nsight System implemented, plus extra features you would like to have.


The functionality of CUPTI can be divided into multiple sections, including 

* the Activity API,
* the Callback API,
* the Host Profiling API,
* the Range Profiling API,
* the PC Sampling API,
* the SASS Metric API,
* the PM Sampling API,
* the Checkpoint API,
* the Profiling API,


The Activity API , Callback API are host-side tracing APIs, which can collect timing or other information of specific activities or events, whereas the Range and Range Profiling API are device-side, which can provide both raw counter values and upper-level metrics like throughput and ratios. In this post, The Activity and Range PRofiling API will be focused to build the profiler.

:::note

Profiling API will no longer be supported for architectures after Blackwell, and has been deprecated in CUDA 13.0 release. Use Range Profiling API instead, which was introduced in CUDA 12.6 release.

:::

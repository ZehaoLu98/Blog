---
title: Scaling Performance of LLM Training
published: 2026-01-13
description: ''
image: ''
tags: [GPU, GEMM, Profiling]
category: Performance Analysis
draft: true 
lang: 'en'
---

# Introduction
Some Context.

In this post, we will present how LLM performance scales with increasing batch size(B), context length(T) and Head Count(H). The potential bottleneck of each block will also be discussed, comparing against the theotical roofline from the previous post here. 

# Setup
We tested on one layer of the transformer, deviding it into multiple blocks for nuanced breakdown of the performance. For each block, we varied three parameters:

| Parameter | Description| Values |
|-----------|------------|--------|
| B | number of batches | 4, 16 |
| T | context length | from 64 to 1536 with interval of 64 |
| H | number of heads in attention block | 4, 8, 12, 16, 32 |



# GPU Time
![gpu_time_100%_stacked_chart](gpu_time_100%25_stacked_chart.png)

# Wall Clock Time
![wallclock_time_100%_stacked_chart](wallclock_time_100%25_stacked_chart.png)
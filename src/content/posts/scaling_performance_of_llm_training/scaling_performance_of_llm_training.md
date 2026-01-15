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
We test on the second layer of the transformer, deviding it into multiple blocks for nuanced breakdown of the performance. For each block, we vary three parameters:

| Parameter | Description| Values |
|-----------|------------|--------|
| B | number of batches | 4, 16 |
| T | context length | from 64 to 1536 with interval of 64 |
| H | number of heads in attention block | 4, 8, 12, 16, 32 |

Unless explicitly stated, we use H=12 by default.

Due to small dataset, only a subset of the whole combinations of B, T can be tested, with less T for higher B. If we keep increasing B, only handful T can be run, which is insufficient to compare against B=4 and B=16. Therefore, only 2 Bs are chosen to compare with each other.  

# GPU Time
## Varing T
![gpu_time_100%_stacked_chart](gpu_time_100%25_stacked_chart.png)
![gpu_time_abs_stacked_chart_b=4](gpu_time_abs_stacked_chart_b=4.png)
![gpu_time_abs_stacked_chart_b=16](gpu_time_abs_stacked_chart_b=16.png)
We first look at the GPU execution time. They are displayed in the distribution plot above, scaling with T. With $T=64$, feed_forward and attention(accumulated from all the sub-ranges) dominate the GPU usage, collectively takes 72.4% in B=4 and 78.8% in B=16. This dominance persist when T increases in both scanrioes. The distribution of attention and feed_forward raised to 92.4% and 93.5% respectively with T raised to ~1.5k. This behaviour suggests that when improving the performance of transformer, probably we should pinpoint on the attention and feed_forward to maxmimize the overall system performance improvement. Therefore we will only focus on these big ranges in this post for brevity.

Further focusing on the attention and feed_forward by deviding the attention into multiple stages, we find that attention_qk, attention_softmax and attention_v scale much faster than other attention ranges and feed_forward. Collectively they raised from 12.2% to 52.4% of the total GPU time with B=4 and from 12.1% to 54.3% of the total GPU time with B=16. Individually, they raised from 4.4%, 3.2% and 4.5% to 16.8%, 16.9%, 18.5% with B=4 and from 4.4%, 3.1% and 4.5% to 17.5%, 17.8%, 19% with B=16. In contrast, feed_forward shirnks significantly from 36.2% to 28% and 43.5% and 27.4% on B=4 and B=64.

The reason for the distribution shift lies on the characteristics of these ranges. Here is a summary table of roofline analysis from the previous post:

| SUB BLOCKS | NUM ELEMENTS ACTIVATION | Total OPs | Bound |
|------------|--------------|-----------|-------|
|Q, K, V | $B * T * 3C$ |$6 * B * T * C^2$| Compute |
|SoftMAX($QK^T$) | $B * NH * T^2$ | $2 * B * T^2 * C + 3 * B * NH * T^2$| QK Compute, Softmax Memory|
|V Matmul | x | $2 * B * T^2 * C$|  Compute |
|O | $B * T * C$ | $2 * B * T * C^2$| Compute |
|MLP1 | $B * T * 4C$ | $8 * B * T * C^2$| Compute |
|MLP2 | $B * T * C$ |  $8 * B * T * C^2$| Compute |

Most of the mentioned time-consuming ranges are compute-bound and the total FLOPS scales quadratically with T, indicating the GPU time should be $O(T^2)$. The only memory-bound range, attention_softmax, still has $T^2$ activations to write to, meaning that in theory, the GPU time of attention_softmax should still be $O(T^2)$. This is why we see similar growth on attention_qk, attention_softmax and attention_v. On the other hand, layer norm and residual block are memory-bound, and their activations and the input matrices only scales linearly with $T$, leading to $O(n)$ GPU time. 

## Varing NH
![nh_scaling_comparison](nh_scaling_comparison.png)
The above chart shows how the GPU time scales with NH. Other than the attention_qk with B=4 and T=64, other ranges scales linearly with NH. The reason why attention_qk doesn't scale with NH on small B and T is probably because it underutilizes the GPU due to the lack of grid and block size. After testing, we found the grid size and block size of the kernels are as below:

| NH | Grid Configuration | Block Configuration |
|----|-------------------|---------------------|
| 4  | <<<1, 1, 16>>>   | <<<128, 1, 1>>>    |
| 8  | <<<1, 1, 32>>>   | <<<128, 1, 1>>>    |
| 12 | <<<1, 1, 48>>>   | <<<128, 1, 1>>>    |
| 16 | <<<1, 1, 64>>>   | <<<128, 1, 1>>>    |
| 32 | <<<1, 1, 128>>>  | <<<128, 1, 1>>>    |

Given A100's 108 SM and 64 Warps per SM in maximum, obviously the GPU is heavily underutilized, so increasing NH only raises the amount of threads executed concurrently on GPU, but doesn't increase the GPU time.

# Wall Clock Time
![wall_clock_time_100%_stacked_chart](wall_clock_time_100%25_stacked_chart.png)
![wall_clock_time_abs_stacked_chart_b=4](wall_clock_time_abs_stacked_chart_b=4.png)
![wall_clock_time_abs_stacked_chart_b=16](wall_clock_time_abs_stacked_chart_b=16.png)
In terms of wall clock time, the distriubtion shows a similar trend compared to the GPU execution time: due to $O(T^2)$ FLOPS and $O(T^2)$ loads/stores, attention_qk, attention_softmax, attention_v and feed_forward still dominate the wall clock time. An interesting point is that the feed_forward doesn't consume much wall clock time with smaller B and T despite of more kernel calls than other ranges, which can increases tens of microseconds of launch overhead. With $B=4$, $T=64$, the feed_forward wall clock time drops from 36.2% to 24.3%, and with $B=16$, $T=64$ it drops from 43.5% to 31.8% comparing the gpu time and wall clock time. However, this discrepancy quickly shrinks with increasing T, and eventually there is only minor difference between the two time metrics.


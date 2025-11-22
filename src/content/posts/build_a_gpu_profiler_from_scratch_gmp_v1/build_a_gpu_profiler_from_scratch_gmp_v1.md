---
title: Build a GPU Profiler From Scratch - GMP_V1
published: 2025-11-03
description: 'A tutorial to build an auto-range kernel-replay GPU Profiler'
image: ''
tags: ['LLM', 'CUPTI', 'GPU', Profiling]
category: 'Profiling Tutorials'
draft: false 
lang: 'en'
---

# Prerequisite

[Introduction to CUPTI](/posts/introduction_to_cupti/introduction_to_cupti)

# Introduction
GMP_V1 is a light-weight GPU profiler built on top of CUPTI, which provided the data of this [LLM introduction blog](https://ml-memory-profiling-group.github.io/blog_v2/note/intro-to-llm/). We used the CUPTI Activity and Range Profiling API, correlating the outputs of the two APIs so that the metrics associated with a specific range can be collected.

This profiler leverages auto range and kernel replay because we faced issues with user range and user replay, which are actually the best options for our need. Since these problems are resolved now, we have provided GMP_V2, which should be a more performant and simpler profiler that can substitute the current GMP_V1. Therefore we will only provide some brief introduction to this profiler as an example implementation to those who wish to quickly implement a GPU profiler or are curious what techniques we used to collect data for the llm blog. 

# Main Idea
GMP_V1 only provides push range and pop range as external APIs, which defines a GMP range for the wrapped region.

:::note
The range of GMP is different from the range of CUPTI. For auto range mode, **CUPTI range** contains a single kernel. Whereas our **GMP range** can contain multiple kernels, which is similar to the user range in CUPTI. It is used for grouping the metrics. 
:::

In push and pop range, we call the corresponding push and pop range function of CUPTI Range Profiling API to collect per-kernel metrics and store it within the counter buffer. At the same time, we flush the activity record buffer at both push and pop to ensure the kernels produced are launched between push and pop. Whenever the buffer is full or the flush is called, the completion callback is triggered. In this callback, we iterate the traces within the buffer and add it into a "session", representing a GMP range. Sessions are stored in a linked list managed by SessionManager and within each node there are traces associated with this session. During the completion callback, activity records are pushed into the tail session of this linked list. When the range is popped, the tail session will be deactivated and the remaining traces will be pushed into this session. After a new push is called, a new session node will be added to the tail of the linked list.

Now we get all the data we need in two places: one is the activity records in the session nodes, and one is the per-kernel metrics data in the counter buffer. We need to correlate them and accumulate all the metrics data within the GMP range. We noticed that the activity records and the metrics data are both collected following the launch order. Therefore we simply need to iterate both containers in the same order to match the trace records with metrics data. In the session nodes, we can find how many kernels are launched during this GMP range. Then we can retrieve the same amount of per-kernel metrics within the counter buffer and associate it with the GMP range. Those metrics are accumulated and becomes per-range metrics.

# limitation
The above method will work if there are less than 2000 kernels. However, two llm.cpp far exceeds the limit. This problem stems from an implicit limit of the counter buffer size. It will report error if you specify a counter buffer size over 2000 ranges during initial setup. Since we are using auto range, each kernel belongs to one range. Obviously the total number of kernel launched exceeds 2000 if we run the full training, so only 1 layer can be profiled in each run because of the limit.

For some metrics that are ratio, the way to **accumulate** the metrics within a GMP range is not clear. For example, if we know the per-kernel throughput within a GMP range, how can we infer the throughput of the GMP range? Calculating the average of the throughput is not a good estimation because, for example, if one kernel has 0% throughput, another one has 100% throughput, it doesn't indicate that a GMP range with these two kernels will have a throughput of 50%. We don't have a solution in this version of GMP, and as a workaround, we are calculating the per-range throughput using accumulated DRAM sector counts and GPU execution time.

Finally, our method depends on the assumption that the order of metrics and traces are both in launch order. If NVIDIA breaks this assumption, this GMP won't work any more.

Based on these limitations, a new version of GMP is necessary.
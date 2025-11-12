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
GMP_V1 is a light-weight GPU profiler built on top of CUPTI, which provided the data of another blog here. We used the CUPTI Activity and Range Profiling API, correlating the outputs of the two APIs so that the metrics associated with a specific range can be collected.

This profiler leverages auto range and kernel replay because we faced issues with user range and user replay, which are actually the best options for our need. Since these problems are resolved now, we have provided GMP_V2, which should be a more performant and simpler profiler that can substitute the current GMP_V1. Therefore we will only provide some brief introduction to this profiler as an example implementation to those who wish to quickly implement a GPU profiler or are curious what techniques we used to collect data for the llm blog. 

# Main Idea
GMP_V1 only provides push range and pop range as external APIs, which defines a GMP range for the wrapped region.

:::note
The range of GMP is different from the range of CUPTI. For auto range mode, **CUPTI range** contains a single kernel. Whereas our **GMP range** can contain multiple kernels, which is similar to the user range in CUPTI. It is used for grouping the metrics. 
:::

In push and pop range, we call the corresponding push and pop range function of CUPTI Range Profiling API to collect per-kernel metrics. At the same time, we flush the activity record buffer at both push and pop so that the buffer will only store the traces within the range when it's flushed during popping. When we flush the activity buffer, the completion callback is triggered. We iterate the traces within the buffer and push it to a "session", which represents a range. There is a session manager managing a linked list of all the sessions. During the completion callback, activity records are pushed into the tail active session. When the range is popped, the tail session will be deactivated and the remaining traces will be pushed into this session.

Now we get all the data we need in two places: one with kernel launch traces in session nodes, and one with per-kernel metrics data in the counter buffer. We need to correlate them and accumulate all the metrics data within the range. We noticed that the sequence of kernel traces and the metrics data are the same: they both follow the order of launching. Therefore we simply need to iterate both containers in the same order to match the trace records with metrics data.

# limitation
The above method will work if there are less than 2000 kernels. However, two llm.cpp far exceeds the limit. This problem stems from an implicit limit of the counter buffer size. It will report error if you specify a counter buffer size over 2000 ranges during initial setup. Since we are using auto range, each kernel belongs to one range. Obviously the total number of kernel launched exceeds 2000 if we run the full training, so only 1 layer can be profiled in each run because of the limit.

Additionally, it collect metrics that needs multiple passes.
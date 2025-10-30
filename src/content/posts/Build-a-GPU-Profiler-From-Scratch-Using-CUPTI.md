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

CUPTI is a set of API that enables developers to both retrieve hardware counters from NVidia GPUs and trace the host-side activities on CUDA. It serves as the foundation of NSight Compute, the official GPU profiler provided by NVidia. With CUPTI, independent developers can develop customized profilers that leverage the same sets of metrics and derive their own specialized insights through custom data processing

In the big picture, CUPTI has two key functionalities:

* Tracing: collecting host-side activities, like kernel launches and memset, etc.  
* Profiling: collecting hardware counters and other derived metrics like throughput.

It can also be divided into multiple sets by the way it collects data, including

* the Activity API,  
* the Callback API,  
* the Host Profiling API,  
* the Range Profiling API,  
* the PC Sampling API,  
* the SASS Metric API,  
* the PM Sampling API,  
* the Checkpoint API,  
* the Profiling API,

:::note

Profiling API will no longer be supported for architectures after Blackwell, and has been deprecated in CUDA 13.0 release. Use Range Profiling API instead, which was introduced in CUDA 12.6 release.

:::

In this tutorial, we will focus on Activity, Callback and Range Profiling API and introduce our custom profiler GPU Memory Profiler(GMP) built on top of Activity and Range Profiling API.

# Activity API
refer to cupti tutorial
how the buffer works

# Callback API
refer to cupti tutorial
Subscriptions

# Range Profiling API
## Range
## Metrics
## Replay

# Design a custom GPU Profiler
## Structure
UML
## How it is working
How we manage the activity record
How we match the activity records with metrics from range profiling
Utilized backend from the examples in the SDK
## Python Wrapper
Dependency Graph
Pybind


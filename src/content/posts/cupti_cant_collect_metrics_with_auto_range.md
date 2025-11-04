---
title: CUPTI can't collect metrics with auto range
published: 2025-11-03
description: 'Explain Activity API and Range Profiling API with samples'
image: ''
tags: [GPU, CUPTI, Profiling]
category: 'Solutions'
draft: false 
lang: 'en'
---
# TL;DR
Only with Push/Pop API can you retrieve metrics if auto range is enabled. More specifically:

```cpp
cuptiRangeProfilerPushRange(&pushRangeParams);
cuptiRangeProfilerPopRange(&popRangeParams);
```

Each range is still a kernel. This push and pop only defines the boundry to collect metrics. Any kernels launched outside the region wrapped by push and pop won't be profiled. 

# Problem
According to the sample code in the [official document](https://docs.nvidia.com/cupti/main/main.html#sample-code), when the range mode is set to auto range, It is calling **pRangeProfilerTarget->PushRange("name")**, which essentially invokes **cuptiRangeProfilerPushRange(&pushRangeParams)**. However, [another part](https://docs.nvidia.com/cupti/main/main.html#range-profiling-usage) of the document indicates that, for the **user range**, **"users can explicitly define ranges using Push/Pop APIs"**, which implies that users don't need to push/pop ranges if they are using auto range. Come on... Which part of the same document should I trust?

# Another Potential Way
I checked another [unofficial CUPTI tutorial](https://eunomia.dev/others/cupti-tutorial/autorange_profiling/). The suggested solution is to use Enable/Disable API:

```cpp
// Enable profiling
CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
    CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

// Launch the first kernel (VecAdd)
VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();

// Launch the second kernel (VecSub)
VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
cudaDeviceSynchronize();

// Disable profiling
CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
    CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
```

However, the [Enable/Disable APIs](https://docs.nvidia.com/cupti/api/group__CUPTI__PROFILER__API.html#group__cupti__profiler__api_1gaa64dfbcde27a202af83e1795fdd4a484) are Profiling API, which will be deprecated as of CUDA 13.0 and  removed in the future. It is recommended to use the Range Profiling API from the header cupti_range_profiler.h.

# Experiment
I speculated 3 possible ways How CUPTI defines the boundry of collection: Push/Pop API, Enable/Disable API or just leave it there(whole program as the boundry). Let's try it one by one in a simple CUDA program.

```cpp
int main(int argc, char **argv)
{
    GmpProfiler::getInstance()->init();

    GmpProfiler::getInstance()->startRangeProfiling();

    // Can we get the metrics of this kernel?
    hello_kernel<<<1,1>>>();

    GmpProfiler::getInstance()->stopRangeProfiling();

    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->decodeCounterData();
    GmpProfiler::getInstance()->printProfilerRanges(outputOption);
    GmpProfiler::getInstance()->produceOutput(outputOption);

    // CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}
```

The GmpProfiler is the customized GPU profiler I built on top of CUPTI. You can think the functions of GmpProfiler as wrappers of CUPTI functions, which can be safely ignored here. After the kernel finishes, **GmpProfiler::getInstance()->printProfilerRanges(outputOption);** will print the number of ranges in the counter data buffer.

## If Nothing is Done
I kept the program above intact.

``` bash
collected range size: 0
```

No luck.

## Push/Pop API
```cpp
    GmpProfiler::getInstance()->pushRange("hello", GmpProfileType::CONCURRENT_KERNEL);
    hello_kernel<<<1,1>>>();
    GmpProfiler::getInstance()->popRange("hello", GmpProfileType::CONCURRENT_KERNEL);
```

I wrapped the kernel with push and pop range function, which is a wrapper of CUPTI's **cuptiRangeProfilerPushRange** and **cuptiRangeProfilerPopRange**.

The result shows:

```bash
collected range size: 1
```

Good. We successfully collected the kernel metrics.

## Enable/Disable API

Then I wrapped the kernel with **cuptiProfilerEnableProfiling** and **cuptiProfilerDisableProfiling**.

```cpp
    // Enable profiling
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    cuptiProfilerEnableProfiling(&enableProfilingParams);

    hello_kernel<<<1,1>>>();

    // Disable profiling
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    cuptiProfilerDisableProfiling(&disableProfilingParams);
```

```bash
Number of ranges: 0
```

Unfortunately this method doesn't work.

# Conclusion

Don't be confused by the official doc, go with 

```cpp
cuptiRangeProfilerPushRange(&pushRangeParams);
cuptiRangeProfilerPopRange(&popRangeParams);
```

As we already know, auto range automatically setup each kernel to be a separate range, so the push and pop range here actually means the boundry of data collection. Any kernel launched outside the push and pop calls will not be collected.
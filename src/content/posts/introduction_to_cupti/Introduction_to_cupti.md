---
title: Introduction to CUPTI
published: 2025-10-31
description: 'Explain Activity API and Range Profiling API with samples'
image: ''
tags: [GPU, Note, CUPTI, Profiling]
category: 'Profiling Tutorials'
draft: false 
lang: 'en'
---
# Introduction

CUPTI is a set of API embedded in CUDA toolkit that enables developers to both retrieve hardware counters from NVidia GPUs and trace the host-side activities on CUDA. It serves as the foundation of NSight Compute, the official GPU profiler provided by NVidia. With CUPTI, independent developers can develop customized profilers that leverage the same sets of metrics and derive their own specialized insights through custom data processing

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
Starting with **CUDA 13.0**, the legacy *Profiling API* is **deprecated** for future architectures (post-Blackwell).  
Use the **Range Profiling API**, introduced in **CUDA 12.6**, as the long-term replacement.
:::

This tutorial focuses on the introduction of **Activity** and **Range Profiling** APIs. Some ready-to-deploy samples are also provided to enable a quick start. In addition, I will provide some tutorials on how I leveraged these APIs to build our in-house **GPU Memory Profiler (GMP)**.

# Activity API
Activity API provides a simple and low-overhead option to collect traces of various events in CUDA runtime and driver API. Common use case includes timing the kernel launches and memory transfers. The overhead is low compared with other metrics collection APIs because it only execute extra host-side instruments, whereas other metrics APIs, for example, Range Profiling API, will read the hardware performance monitor units, which involves more memory activities and thus is more time-consuming. Therefore if we don't need any low-level device-related data, Activity API is usually the way to go.

At the high level, we need to enables specific types of activity trace collection, register the callbacks to fulfill buffer requests and handles buffer completion. We will detail them below. 

## Enable Activity Collection
To start with, we need to enable the activity types we would like to collect for. Here is the function:
```cpp
CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind);
```
There are quite a lot of choices for the *kind* argument listed in the official docs [here](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv418CUpti_ActivityKind). Here is the big picture what you can collect through Activity APIs:

* Kernel and Memory Operations — execution of kernels, memory copies (H↔D, D↔D), and memory sets.
* API Calls — CUDA Runtime and Driver API calls, including durations and parameters.
* Device and Memory Usage — device info, allocations, unified memory page migrations, and memory pool events.
* Streams, Graphs, and Synchronization — stream timelines, CUDA graph execution, barriers, and synchronization events.
* Markers and Correlations — NVTX markers and external correlation to align GPU traces with host or other systems.
* Interconnect and I/O — NVLink and PCIe transfer activities across devices or between CPU and GPU.
* JIT and Overhead Information — JIT compilation, CUPTI’s own overhead, and preemption events.
:::warning
Metric-related types will be disabled starting from CUDA 13.0.
:::
:::tip
To avoid unnecessary overheads, it is recommended to enable as less kinds of activities as possible since the overhead is proportional to the number of kernels.
:::

The most common activities we trace is kernel launches. There are two corresponding enums:

* CUPTI_ACTIVITY_KIND_KERNEL
* CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
  
Despite of little difference on naming, they cause huge impact on the kernel execution. CUPTI_ACTIVITY_KIND_KERNEL forces serializing the kernel launches, i.e. all kernels will be executed separately. Whereas CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL preserves the original concurrency. For single-stream applications, these behave identically. For multi-stream workloads, CUPTI_ACTIVITY_KIND_KERNEL will serialize the kernels, but CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL will keep the execution intact.

## Activity Record Buffer
CUPTI manages the buffer in an asynchronous way. At the beginning of the program, you need to define two callbacks that enable CUPTI to request new buffers and hand over the filled buffer. 
```cpp
// Callback Registration
CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested, CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

// Callback Signatures
typedef void (*CUpti_BuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
typedef void (*CUpti_BuffersCallbackCompleteFunc)(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
```

:::warning
Avoid putting time-consuming operations in these callbacks.
:::

To store the trace records, you need to provide host-side buffer when CUPTI runs out of buffer and ask for it through **funcBufferRequested**. Typically, the buffer should be 1~10MB  The **funcBufferCompleted** is triggered either when the buffer is full or per the user requests. It passes the buffer back to the user with the current available trace records. In this callback, user can iterate records through **cuptiActivityGetNextRecord**. The record you get is in a general record type. Each type of activities has its own record struct, so you can check the comment on the enum to confirm which type you should convert the general record type to. 

There are multiple ways to flush the buffer and retrieve the activity records:

* **cuptiActivityFlushAll(0)** will trigger the **funcBufferCompleted** with completed traces immediately. 
* **cuptiActivityFlushPeriod(uint32_t time)** will call **funcBufferCompleted** with a fixed interval.
* Wait the buffer to be full.

## Sample Implementation
```cpp
// Initialize the profiler
void init()
{
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));

    CUPTI_CALL(cuptiActivityRegisterCallbacks(&GmpProfiler::bufferRequestedThunk, &GmpProfiler::bufferCompletedThunk));

    // Remaining initialization
}

static void CUPTIAPI bufferRequestedThunk(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    if (instance) instance->bufferRequestedImpl(buffer, size, maxNumRecords);
}

static void CUPTIAPI bufferCompletedThunk(CUcontext ctx, uint32_t streamId,
                                            uint8_t *buffer, size_t size, size_t validSize)
{
    if (instance) instance->bufferCompletedImpl(ctx, streamId, buffer, size, validSize);
}

void GmpProfiler::bufferRequestedImpl(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    // Allocate a 4MB activity record buffer
    *size = 4 * 1024 * 1024;
    *buffer = (uint8_t *)malloc(*size);
    *maxNumRecords = 0;
}

void GmpProfiler::bufferCompletedImpl(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = nullptr;
    GMP_LOG_DEBUG("Buffer completion callback called");
    for (;;)
    {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS)
        {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            {
                auto *kernel = (CUpti_ActivityKernel8 *)record;

                // Handle kernel activity record.
            }
            else if (record->kind == CUPTI_ACTIVITY_KIND_MEMORY2)
            {
                auto *memRecord = (CUpti_ActivityMemory4 *)record;
                
                // Handle memory activity record.
            }
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        {
            break;
        }
        else
        {
            CUPTI_CALL(status);
        }
    }

    // Some records may be dropped 
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (dropped != 0)
    {
        printf("CUPTI: Dropped %zu activity records\n", dropped);
    }
    free(buffer);
}

// Sample Usage
int main(){
    // register the callbacks
    init();

    // Invoke some kernels for profiling.
    // The callbacks may have been called multiple times.
    some_cuda();

    // Ensure all the kernels have finished
    cudaDeviceSynchronize()

    // All the job done, dump all the remaining trace records
    cuptiActivityFlushAll(0)
}
```

# Range Profiling API
Range Profiling API provides fine-grained access to the context-level GPU metrics within user-defined code regions(ranges). It supersedes the older Profiling API and supports both auto and manual range definitions.

## Available Metrics
Metrics can originate directly from the GPU’s hardware performance monitor units (PMUs), for example:

* SASS instruments issued
* Memory footprints
* GPU time executed
* Active warps 
* Stalled Cycles

Other metrics are derived from these raw counters, like:

* Dram throughput
* Cache hit rate
* SM utilization

There are huge amounts of metrics. You can query the full list of available metrics through
```bash
ncu --query-metrics
```

## Range Definition and Range Mode
Each range represents a code region for which CUPTI accumulates metrics. There are two ways to define ranges:

* Auto range mode, each kernel is automatically treated as a range. This mode includes a context synchronization at the end of each kernel launch, therefore only one kernel can be executed at the same time. Despite the official document implies not to use **cuptiRangeProfilerPushRange** and **cuptiRangeProfilerPopRange**, I verified that only by wrapping the region with **cuptiRangeProfilerPushRange** and **cuptiRangeProfilerPopRange** can you collect the metrics in this mode. Refer to my [experiment post](/posts/cupti_cant_collect_metrics_with_auto_range/) for detail.
* User range mode, a range is defined by wrapping the code with a pair of **cuptiRangeProfilerPushRange**  **cuptiRangeProfilerPopRange**. Ranges can be nested, and metrics are accumulated across multiple kernel launches within the same range.

Here is the push and pop sample implementation. For range mode setting, refer to the replay section. They are setup together.

```cpp
// rangeProfilerObject: Profiler pointer returned by cuptiRangeProfilerEnable during initialization
CUptiResult PushRange(CUpti_RangeProfiler_Object* rangeProfilerObject,const char *rangeName)
{
    CUpti_RangeProfiler_PushRange_Params pushRangeParams{CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    pushRangeParams.pRangeProfilerObject = rangeProfilerObject;
    pushRangeParams.pRangeName = rangeName;
    CUPTI_API_CALL(cuptiRangeProfilerPushRange(&pushRangeParams));
    return CUPTI_SUCCESS;
}

// rangeProfilerObject: Profiler pointer returned by cuptiRangeProfilerEnable during initialization
CUptiResult PopRange(CUpti_RangeProfiler_Object* rangeProfilerObject)
{
    CUpti_RangeProfiler_PopRange_Params popRangeParams{CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
    popRangeParams.pRangeProfilerObject = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerPopRange(&popRangeParams));

    return CUPTI_SUCCESS;
}
```

## Replay
Each SM in NVIDIA GPUs only contains a small amount of performance registers for the metrics, but there might be thousands of data points needed to collect for all the metrics. Therefore sometimes it may not be possible to collect all the metrics with only single pass(execution) of the program. The read rate of these counters are also limited. Therefore CUPTI introduces the replay mechanism that can re-execute kernels with the same memory status. At the beginning, CUPTI saves the current context state and clear the caches. After the first pass, CUPTI quickly diffs the start and end status and only restores the changed memory. In the remaining passes, CUPTI will only restore the memory that was changed during the first pass. Read-only memory is also skipped during replays. These ensures the least memory footage caused by replaying. 

If the GPU memory cannot store the context status, it will spill to the host memory. If the host memory cannot hold it as well, it will spill to the disk, which can drastically increase overhead, therefore it’s crucial to keep the replayed regions small.

There are three types of replay: 

* Kernel replay will replay each kernel separately so that only when CUPTI has collected all the metrics of a kernel will it proceeds to the next kernel. Because of the granularity, it might avoid the unnecessary transfer between host and device, but it can cause frequent allocation if there are excessive kernels. 
* User replay lets user define how to replay through checkpoint API, which enables user to manually save and restore the memory status. It provides the most freedom, but with the cost of simplicity. 
* Application replay re-executes the whole application with the same configuration. The execution path should be deterministic.

:::important
According to my experiments, user range + kernel replay always fail with: 

Function cuptiRangeProfilerSetConfig(&setConfigParams) failed with error(7): CUPTI_ERROR_INVALID_OPERATION
:::

All the range and replay mode options can be sent to CUPTI through **cuptiRangeProfilerSetConfig** during initialization. 
```cpp
// range: CUPTI_UserRange or CUPTI_AutoRange
// replayMode: CUPTI_UserReplay or CUPTI_KernelReplay
// configImageBlob: host-side config image created through cuptiProfilerHostGetConfigImage during initialization
// counterDataImage: host-side counter data image created through cuptiRangeProfilerCounterDataImageInitialize during initialization
// rangeProfilerObject: Profiler pointer returned by cuptiRangeProfilerEnable during initialization
CUptiResult SetConfig(
    CUpti_ProfilerRange range,
    CUpti_ProfilerReplayMode replayMode,
    std::vector<uint8_t> &configImageBlob,
    std::vector<uint8_t> &counterDataImage,
    CUpti_RangeProfiler_Object* rangeProfilerObject)
{
    configImage.resize(configImageBlob.size());
    std::copy(configImageBlob.begin(), configImageBlob.end(), configImage.begin());

    CUpti_RangeProfiler_SetConfig_Params setConfigParams{CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pRangeProfilerObject = rangeProfilerObject;
    setConfigParams.pConfig = configImage.data();
    setConfigParams.configSize = configImage.size();
    setConfigParams.pCounterDataImage = counterDataImage.data();
    setConfigParams.counterDataImageSize = counterDataImage.size();
    setConfigParams.maxRangesPerPass = MAX_RANGE_NUM;
    setConfigParams.numNestingLevels = MAX_NESTING_LEVELS;
    setConfigParams.minNestingLevel = MIN_NESTING_LEVELS;
    setConfigParams.passIndex = 0;
    setConfigParams.targetNestingLevel = 1;
    setConfigParams.range = range;
    setConfigParams.replayMode = replayMode;
    CUPTI_API_CALL(cuptiRangeProfilerSetConfig(&setConfigParams));
    return CUPTI_SUCCESS;
}
```
## Counter Buffer 
All the data is stored within a counter buffer provided during setup using **cuptiRangeProfilerSetConfig**. Refer to the **SetConfig** function above.

:::important
According to my experiments, the max range of the counter buffer is around 2000~3000. Exceeding this number will cause error

Function cuptiRangeProfilerSetConfig(&setConfigParams) failed with error(7): CUPTI_ERROR_INVALID_OPERATION.
:::

After finishing the profiling, you should call **cuptiRangeProfilerDecodeData** to transfer the metrics data from hardware to the buffer provided in **cuptiRangeProfilerSetConfig** and decode it. After the data is decoded, you can start to process the data into your data format. Here is the sample and you can adjust it based on your requirements:

```cpp
// rangeProfilerObject: Profiler pointer returned by cuptiRangeProfilerEnable during initialization
CUptiResult DecodeCounterData(CUpti_RangeProfiler_Object* rangeProfilerObject)
{
    CUpti_RangeProfiler_DecodeData_Params decodeDataParams{CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decodeDataParams.pRangeProfilerObject = rangeProfilerObject;
    CUPTI_API_CALL(cuptiRangeProfilerDecodeData(&decodeDataParams));
    return CUPTI_SUCCESS;
}
```

```cpp
// Your struct to store metrics data of a range.
// This is an example struct of storing metrics data for one range. 
struct ProfilerRange
{
    size_t rangeIndex;
    std::string rangeName;
    std::vector<std::string> nameList;   // Metric names
    std::vector<double> valueList;  // Metric values, corresponding to the order of nameList
};

// metricNum: number of metrics requested.
// counterDataImage: the buffer passed to CUPTI in cuptiRangeProfilerSetConfig.
// Return all the metrics of a range.
ProfilerRange EvaluateCounterData(
    size_t rangeIndex,
    size_t metricNum,
    std::vector<uint8_t> &counterDataImage)
{
    ProfilerRange profilerRange{};

    // Get range name
    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params getRangeInfoParams = {CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
    getRangeInfoParams.counterDataImageSize = counterDataImage.size();
    getRangeInfoParams.pCounterDataImage = counterDataImage.data();
    getRangeInfoParams.rangeIndex = rangeIndex;
    getRangeInfoParams.rangeDelimiter = "/";
    CUPTI_API_CALL(cuptiRangeProfilerCounterDataGetRangeInfo(&getRangeInfoParams));

    // Setup the structure.
    profilerRange.rangeIndex = rangeIndex;
    profilerRange.rangeName = getRangeInfoParams.rangeName;
    profilerRange.nameList.resize(metricNum);
    profilerRange.valueList.resize(metricNum);

    // Use CUPTI API to retrieve data from counter data image.
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams{CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalauateToGpuValuesParams.pHostObject = m_pHostObject;
    evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
    evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
    evalauateToGpuValuesParams.ppMetricNames = profilerRange.nameList.data();
    evalauateToGpuValuesParams.numMetrics = profilerRange.nameList.size();
    evalauateToGpuValuesParams.rangeIndex = rangeIndex;
    evalauateToGpuValuesParams.pMetricValues =profilerRange.valueList.data();
    CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));

    return profilerRange;
}
```
# What's Next
If you are interested how to build a complete GPU profiler, here are the customized in-house GPU Memory Profilers(GMP) I developed: 

* [GMP_v2](/posts/build_a_gpu_profiler_from_scratch_gmp_v2/build_a_gpu_profiler_from_scratch_gmp_v2/): under construction
* [GMP_v1](/posts/build_a_gpu_profiler_from_scratch_gmp_v1/build_a_gpu_profiler_from_scratch_gmp_v1/): completed

It's an evolving project. The versions are implemented with fundamental structural difference, I will, therefore,  preserve the tutorials of these versions to provide different design choices.

# Resources
Here is some useful links to dive deep into CUPTI:

* [NVIDIA CUPTI Document](https://docs.nvidia.com/cupti/main/main.html#usage)
* [An Unofficial CUPTI Tutorial](https://github.com/eunomia-bpf/cupti-tutorial/tree/master)
* [Nsight Compute Document](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
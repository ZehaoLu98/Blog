---
title: Introduction to CUPTI
published: 2025-10-31
description: 'Explain Activity API and Range Profiling API with samples'
image: ''
tags: [GPU, Note, CUPTI, Profiling]
category: 'Computer Architecture'
draft: false 
lang: 'en'
---
# CUPTI Introduction

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

Profiling API will no longer be supported for architectures after Blackwell, and has been deprecated in CUDA 13.0 release. Use Range Profiling API instead, which was introduced in CUDA 12.6 release.

:::

In this tutorial, we will focus on Activity, Callback and Range Profiling API and introduce our custom profiler GPU Memory Profiler(GMP) built on top of Activity and Range Profiling API.

# Activity API
Activity API provides a simple and low-overhead option to collect traces of various events in CUDA runtime and driver API. Common use case includes timing the kernel launches and memory transfers. The overhead is low compared with other metrics collection APIs because it only execute extra host-side instruments, whereas other metrics APIs, for example, Range Profiling API, will read the hardware performance monitor units, which involves more memory activities and thus is more time-consuming. Therefore if we don't need any low-level device-related data, Activity API is usually the way to go.

In high level, we need to enables specific types of activity trace collection, register the callbacks to fulfill buffer requests and handles buffer completion. We will detail all of them in sections below. 

## Enable Activity Collection
To start with, we need to enable the activity types we would like to collect for. Here is the function:
```cpp
CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind);
```
There are quite a lot of choices for the *kind* argument listed in the official docs [here](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv418CUpti_ActivityKind). 

:::note
To avoid unnecessary overheads, it is recommended to enable as less kinds of activities as possible since the overhead is proportional to the number of kernels.
:::

The most common activities we trace is kernel launches. There are two corresponding enums:
* CUPTI_ACTIVITY_KIND_KERNEL
* CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
Despite of little difference on naming, they cause huge impact on the kernel execution. CUPTI_ACTIVITY_KIND_KERNEL forces serializing the kernel launches, i.e. all kernels will be executed separately. Whereas CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL won't affect the execution of kernels. If the program to trace is single-stream, this difference doesn't matter. However, for multi-stream program, you need to consider the effect caused by the choice of activity types.

## Activity Record Buffer
CUPTI manages the buffer in an asynchronous way. At the beginning of the program, user need to define two callbacks that enable CUPTI to request new buffers and hand over the filled buffer. 
```cpp
// Callback Registration
CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested, CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

// Callback Signatures
typedef void (*CUpti_BuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
typedef void (*CUpti_BuffersCallbackCompleteFunc)(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
```

:::note
Avoid putting time-consuming operations in callbacks.
:::

To store the trace records, you need to provide host-side buffer when CUPTI runs out of buffer and ask for it through **funcBufferRequested**. Typically, the buffer should be 1~10MB  The **funcBufferCompleted** is triggered either when the buffer is full or per the user requests. It passes the buffer back to the user with the current available trace records. In this callback, user can iterate records through **cuptiActivityGetNextRecord**. The record you get is in a general record type. Each type of activities has its own record struct, so you can check the comment on the enum to confirm which type you should convert the general record type to. 

There are multiple ways for users to flush the buffer and retrieve the activity records:

* **cuptiActivityFlushAll(0)** will trigger the **funcBufferCompleted** with completed traces. 
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
    *size = 16 * 1024;
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

    // All the job done, dump all the remaining trace records
    cuptiActivityFlushAll(0)
}
```

# Range Profiling API
Range Profiling API provides a way to collect context-level GPU metrics within specific ranges of the program.

## Available Metrics
Some of the metrics are directly from the performance monitor units of the GPU. For example:

* SASS instruments issued
* Memory footprints
* GPU time executed
* Active warps 
* Stalled Cycles with various reasons

Some of metrics are derived from other metrics, like:

* Dram throughput
* Cache hit rate
* SM utilization

There are huge amounts of metrics. You can query the full list through
```bash
ncu --query-metrics
```

## Ranges Definition
All the metrics are grouped with ranges. There are two ways to define range. 
* Auto range mode, each kernel is automatically defined as a kernel. 
* User range mode, a range is defined by wrapping the code a pair of **cuptiRangeProfilerPushRange**  **cuptiRangeProfilerPopRange**. The ranges can be nested. The metrics will be accumulated among multiple kernels.

## Replay
Each SM in NVIDIA GPUs only contains a small amount of performance registers for the metrics, but there might be thousands of data points needed to collect for all the metrics. Therefore sometimes it may not be possible to collect all the metrics with only single pass(execution) of the program. CUPTI introduces replay so that all the kernels can be re-executed with the same memory status. This ensures that all the metrics in each replay are collected under the same memory condition as if they are produced within a single pass. In the first pass, CUPTI saves a copy of the accessed memory content in this pass. Every time one pass finishes, CUPTI will restore the memory status using this copy. It is obvious that this kind of replays adds extra memory footprints to the memory. If the dram cannot store the backup, the host memory will store it. If even the host memory cannot hold it, the file system of the host will have to save it. The cost of the data transfer will be tremendous, so it's usually a good idea to keep the replayed area small.

There are three types of replay. 

* Kernel replay will replay kernels separately so that only when CUPTI has collected all the metrics of a kernel will it proceeds to the next kernel. Because of the granularity, it might avoid the unnecessary transfer between host and device, but it can cause frequent allocation if there are excessive kernels. 
* User replay lets user define how to replay through checkpoint API, which enables user to manually save and restore the memory status. It provides the most freedom, but with the cost of simplicity. 
* Application replay re-executes the whole application with the same configuration. The execution path should be deterministic.

:::note
According to my experiments, User range + kernel replay always fail with: 

Function cuptiRangeProfilerSetConfig(&setConfigParams) failed with error(7): CUPTI_ERROR_INVALID_OPERATION
:::

All the range and replay mode options can be sent to CUPTI through **cuptiRangeProfilerSetConfig** during initialization.

## Counter Buffer
All the data is stored within a counter buffer provided during setup using **cuptiRangeProfilerSetConfig**.

:::note
According to my experiments, the max range of the counter buffer is around 2000~3000. Exceeding this number will cause error

Function cuptiRangeProfilerSetConfig(&setConfigParams) failed with error(7): CUPTI_ERROR_INVALID_OPERATION.
:::

After finishing the profiling, you should call **cuptiRangeProfilerDecodeData** to transfer the metrics data from hardware to the buffer provided in **cuptiRangeProfilerSetConfig** and decode it. After the data is decoded, you can start to process the data into your data format. Here is the sample and you can adjust it based on your requirements:

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
ProfilerRange CuptiProfilerHost::EvaluateCounterData(
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


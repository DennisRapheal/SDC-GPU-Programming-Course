#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess){                \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

// GPU Kernels
__global__ void kernelA(double* arrayA, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] += 1.0;}
};
__global__ void kernelB(double* arrayA, double* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayB[x] += arrayA[x] + 3.0;}
};

int main()
{
    constexpr int numOfBlocks = 1 << 20;
    constexpr int threadsPerBlock = 1024;
    constexpr int numberOfIterations = 50;
    // The array size smaller to avoid the relatively short kernel launch compared to memory copies
    constexpr size_t arraySize = 1U << 25;
    double *d_dataA;
    double *d_dataB;
    double initValueA = 0.0;
    double initValueB = 2.0;

    // Allocate pinned memory
    double* vectorA;
    double* vectorB;
    HIP_CHECK(hipHostMalloc(&vectorA, arraySize * sizeof(double)));
    HIP_CHECK(hipHostMalloc(&vectorB, arraySize * sizeof(double)));

    // 初始化值
    std::fill_n(vectorA, arraySize, initValueA);
    std::fill_n(vectorB, arraySize, initValueB);

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_dataA, arraySize * sizeof(*d_dataA)));
    HIP_CHECK(hipMalloc(&d_dataB, arraySize * sizeof(*d_dataB)));
    // Create streams
    hipStream_t streamA, streamB;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));
    // Create events
    hipEvent_t event, eventA, eventB;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventCreate(&eventA));
    HIP_CHECK(hipEventCreate(&eventB));
    for(unsigned int iteration = 0; iteration < numberOfIterations; iteration++)
    {
        // Stream 1: Host to Device 1
        HIP_CHECK(hipMemcpyAsync(d_dataA, vectorA, arraySize * sizeof(double), hipMemcpyHostToDevice, streamA));
        // Stream 2: Host to Device 2
        HIP_CHECK(hipMemcpyAsync(d_dataB, vectorB, arraySize * sizeof(double), hipMemcpyHostToDevice, streamB));
        // Stream 1: Kernel 1
        hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, streamA, d_dataA, arraySize);
        // Record event after the GPU kernel in Stream 1
        HIP_CHECK(hipEventRecord(event, streamA));
        // Stream 2: Wait for event before starting Kernel 2
        HIP_CHECK(hipStreamWaitEvent(streamB, event, 0));
        // Stream 2: Kernel 2
        hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, streamB, d_dataA, d_dataB, arraySize);
        // Stream 1: Device to Host 2 (after Kernel 1)
        HIP_CHECK(hipMemcpyAsync(vectorA, d_dataA, arraySize * sizeof(double), hipMemcpyDeviceToHost, streamA));
        // Stream 2: Device to Host 2 (after Kernel 2)
        HIP_CHECK(hipMemcpyAsync(vectorB, d_dataB, arraySize * sizeof(double), hipMemcpyDeviceToHost, streamB));
        // Wait for all operations in both streams to complete
        HIP_CHECK(hipEventRecord(eventA, streamA));
        HIP_CHECK(hipEventRecord(eventB, streamB));
        HIP_CHECK(hipStreamWaitEvent(streamA, eventA, 0));
        HIP_CHECK(hipStreamWaitEvent(streamB, eventB, 0));
    }

    // 需加入否則報錯
    HIP_CHECK(hipDeviceSynchronize());
    // Verify results
    double expectedA = (double)numberOfIterations;
    double expectedB =
        initValueB + (3.0 * numberOfIterations) +
        (expectedA * (expectedA + 1.0)) / 2.0;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(vectorA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << vectorA[i] << std::endl;
            break;
        }
        if(vectorB[i] != expectedB){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedB << " got " <<  vectorB[i] << std::endl;
            break;
        }
    }
    if(passed){
        std::cout << "Asynchronous execution with events completed successfully." << std::endl;
    }else{
        std::cerr << "Asynchronous execution with events failed." << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipEventDestroy(event));
    HIP_CHECK(hipEventDestroy(eventA));
    HIP_CHECK(hipEventDestroy(eventB));
    HIP_CHECK(hipStreamDestroy(streamA));
    HIP_CHECK(hipStreamDestroy(streamB));
    HIP_CHECK(hipFree(d_dataA));
    HIP_CHECK(hipFree(d_dataB));

    // free HIP
    HIP_CHECK(hipHostFree(vectorA));
    HIP_CHECK(hipHostFree(vectorB));

    return 0;
}
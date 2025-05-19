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

    std::vector<double> vectorA(arraySize, initValueA);
    std::vector<double> vectorB(arraySize, initValueB);
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_dataA, arraySize * sizeof(*d_dataA)));
    HIP_CHECK(hipMalloc(&d_dataB, arraySize * sizeof(*d_dataB)));
    
    /*
        1. Create Stream
        2. 
    */
    
    hipStream_t streamA, streamB;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));
    for(int iteration = 0; iteration < numberOfIterations; iteration++)
    {
        // Host to Device copies
        HIP_CHECK(hipMemcpy(d_dataA, vectorA.data(), arraySize * sizeof(*d_dataA), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_dataB, vectorB.data(), arraySize * sizeof(*d_dataB), hipMemcpyHostToDevice));
        // Launch the GPU kernels
        hipLaunchKernelGGL(kernelA, dim3(numOfBlocks), dim3(threadsPerBlock), 0, 0, d_dataA, arraySize);
        hipLaunchKernelGGL(kernelB, dim3(numOfBlocks), dim3(threadsPerBlock), 0, 0, d_dataA, d_dataB, arraySize);
        // Device to Host copies
        HIP_CHECK(hipMemcpy(vectorA.data(), d_dataA, arraySize * sizeof(*vectorA.data()), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(vectorB.data(), d_dataB, arraySize * sizeof(*vectorB.data()), hipMemcpyDeviceToHost));
    }
    // Wait for all operations to complete
    HIP_CHECK(hipDeviceSynchronize());

    // Verify results
    const double expectedA = (double)numberOfIterations;
    const double expectedB =
        initValueB + (3.0 * numberOfIterations) +
        (expectedA * (expectedA + 1.0)) / 2.0;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(vectorA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << vectorA[i] << " at index: " << i << std::endl;
            break;
        }
        if(vectorB[i] != expectedB){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedB << " got " <<  vectorB[i] << " at index: " << i << std::endl;
            break;
        }
    }

    if(passed){
        std::cout << "Sequential execution completed successfully." << std::endl;
    }else{
        std::cerr << "Sequential execution failed." << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_dataA));
    HIP_CHECK(hipFree(d_dataB));

    return 0;
}
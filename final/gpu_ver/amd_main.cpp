#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <numeric>
#include <cassert>
#include <cmath>        // fabsl

#define USE_HIP             // ← 保留代表只用 AMD ROCm

#ifdef USE_HIP
  #include <hip/hip_runtime.h>
  #define GPU_LAUNCH(kernel,grid,block,...)  hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
  #define gpuMalloc(ptr,sz)      hipMalloc(ptr,sz)
  #define gpuMemcpyToDevice(d,s,z) hipMemcpy(d,s,z,hipMemcpyHostToDevice)
  #define gpuMemcpyToHost(d,s,z)   hipMemcpy(d,s,z,hipMemcpyDeviceToHost)
  #define gpuMemset(ptr,v,sz)   hipMemset(ptr,v,sz)
  #define gpuDeviceSynchronize  hipDeviceSynchronize
  #define gpuFree(ptr)          hipFree(ptr)
#else        // CUDA 分支（若日後需要）
  #include <cuda_runtime.h>
  #define GPU_LAUNCH(kernel,grid,block,...)  kernel<<<grid,block>>>(__VA_ARGS__)
  #define gpuMalloc(ptr,sz)      cudaMalloc(ptr,sz)
  #define gpuMemcpyToDevice(d,s,z) cudaMemcpy(d,s,z,cudaMemcpyHostToDevice)
  #define gpuMemcpyToHost(d,s,z)   cudaMemcpy(d,s,z,cudaMemcpyDeviceToHost)
  #define gpuMemset(ptr,v,sz)   cudaMemset(ptr,v,sz)
  #define gpuDeviceSynchronize  cudaDeviceSynchronize
  #define gpuFree(ptr)          cudaFree(ptr)
#endif

/*--------------------------------------------------*
 *                Shishua (scalar)                  *
 *--------------------------------------------------*/
#define SHISHUA_TARGET_SCALAR 0
#define SHISHUA_TARGET        SHISHUA_TARGET_SCALAR
#include "shishua.h"             // 請確保與本檔同目錄
// real euler number: 2.71828 18284 59045 23536 02874 71352 66249 77572 47093 69995..... 50 decimals

class ShishuaGenerator {
    public:
        using result_type = uint64_t;
        explicit ShishuaGenerator(uint64_t seed64 = 1234) {
            uint64_t seed[4] = { seed64,
                                 seed64 ^ 0x9e3779b97f4a7c15ULL,
                                 seed64 + 0x12345678ULL,
                                 seed64 ^ 0x6a09e667f3bcc908ULL };
            prng_init(&state_, seed);
            output_index_ = 16;
        }
        uint64_t operator()() {
            if (output_index_ == 16) {
                prng_gen(&state_, nullptr, 128);
                output_index_ = 0;
            }
            return state_.output[output_index_++];
        }
    private:
        prng_state state_;
        int        output_index_;
    };
    
    // uint64_t → double ∈ [0,1)
    static inline double u64_to_unit_double(uint64_t x) {
        return (x >> 11) * (1.0 / 9007199254740992.0); // 1/2^53
    }

// 每 thread 處理一個 trial，input 已排成 (num_per_trial) 個 double
__global__
void while_sum_kernel(const double* __restrict in,
                      uint32_t*    __restrict out,
                      int num_per_trial,
                      uint64_t total_trials)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_trials) return;

    const double* p = in + tid * num_per_trial;
    double   s = 0.0;
    uint32_t c = 0;

#pragma unroll
    for (int k = 0; k < num_per_trial; ++k) {
        s += p[k];
        ++c;
        if (s >= 1.0) break;
    }
    out[tid] = c;          // 若真不足，host 可檢查再補（極罕見）
}


long double estimate_e_gpu(uint64_t total_trials,
                           uint32_t  num_per_trial   = 16,
                           uint64_t  trials_per_batch= 1'000'000)
{
    const int BLOCK = 256;
    const int GRID  = (trials_per_batch + BLOCK - 1) / BLOCK;

    double*   d_in  = nullptr;
    uint32_t* d_out = nullptr;
    gpuMalloc(&d_in , sizeof(double) * trials_per_batch * num_per_trial);
    gpuMalloc(&d_out, sizeof(uint32_t)* trials_per_batch);

    ShishuaGenerator rng(uint64_t(std::chrono::high_resolution_clock::now()
                                  .time_since_epoch().count()));

    std::vector<double>  h_in (trials_per_batch * num_per_trial);
    std::vector<uint32_t>h_out(trials_per_batch);

    uint64_t processed = 0, total_cnt = 0;

    while (processed < total_trials) {
        uint64_t batch = std::min<uint64_t>(trials_per_batch,
                                            total_trials - processed);

        /* 1) CPU 產 batch×num_per_trial 個亂數 */
        for (uint64_t i = 0; i < batch * num_per_trial; ++i)
            h_in[i] = rng.uniform01();

        /* 2) 傳到 GPU & 清 output */
        gpuMemcpyToDevice(d_in, h_in.data(),
                          sizeof(double)*batch*num_per_trial);
        gpuMemset(d_out, 0, sizeof(uint32_t)*batch);

        /* 3) 呼叫 kernel */
        GPU_LAUNCH(while_sum_kernel,
                   dim3(GRID), dim3(BLOCK),
                   d_in, d_out, num_per_trial, batch);

        gpuDeviceSynchronize();

        /* 4) 拷回次數並累加 */
        gpuMemcpyToHost(h_out.data(), d_out, sizeof(uint32_t)*batch);
        total_cnt += std::accumulate(h_out.begin(),
                                     h_out.begin()+batch, uint64_t(0));
        processed += batch;
    }

    gpuFree(d_in); gpuFree(d_out);
    return static_cast<long double>(total_cnt) / total_trials;
}

/*--------------------------------------------------*
 *                     main                         *
 *--------------------------------------------------*/
int main(int argc,char**argv)
{
    if(argc < 2){
        std::cout << "Usage: " << argv[0] << " <total_trials>\n";
        return 0;
    }
    uint64_t trials = std::strtoull(argv[1], nullptr, 10);

    auto t0 = std::chrono::high_resolution_clock::now();
    long double est = estimate_e_gpu(trials);
    auto t1 = std::chrono::high_resolution_clock::now();

    constexpr long double TRUE_E = 2.71828182845904523536L;
    long double abs_err = fabsl(est - TRUE_E);

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Estimated e    : " << est << '\n';
    std::cout << "Absolute Error : " << abs_err << '\n';
    std::cout << "Relative Error : " << (abs_err/TRUE_E*100) << " %\n";
    std::cout << std::setprecision(4)
              << "Elapsed Time   : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    return 0;
}
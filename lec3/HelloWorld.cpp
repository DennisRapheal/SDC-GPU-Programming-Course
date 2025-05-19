#include <hip/hip_runtime.h>
#include <iostream>

#define SUCCESS 0
#define FAILURE 1

using namespace std;

/*
    表示這是 GPU kernel 函數，會由 host（CPU）發動並在 GPU 上執行。

    呼叫方式是 <<<gridDim, blockDim>>>，而不是一般函數呼叫。

    這個 GPU 核心函數會由多個執行緒同時執行，每個執行緒計算向量中一個索引的元素相加：

    global_idx：計算該執行緒對應的資料索引。

    a[global_idx] += b[global_idx]：將 b 的對應值加到 a 上。
*/
__global__ void VectorAddKernel(float* a, const float* b)
{
    /*
        threadIdx.x	:當前 thread 在 block 中的編號（0 ~ blockDim.x-1）
        blockIdx.x	:當前 block 在 grid 中的索引編號（0 ~ gridDim.x-1）
        blockDim.x	:每個 block 有多少個 thread（thread 的數量）
        global_idx	:全域索引（global index），代表這個 thread 應該處理哪一個向量元素
    */
	int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
	a[global_idx] += b[global_idx];
}

int main(int argc, char* argv[])
{
	cout << "<Program Start>" << endl;
	int n = 10;
	float host_a[n];
	const float host_b[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
	float *device_a, *device_b;
	hipError_t err;
	
	for(int i=0; i<n; i++) host_a[i] = i;

	cout << "vector a: ";
	for(int i=0; i<n; i++) cout << host_a[i] << " ";
	cout << endl;

	cout << "vector b: ";
	for(int i=0; i<n; i++) cout << host_b[i] << " ";
	cout << endl;

    // 在 GPU 上配置 device_a, device_b
	err = hipMalloc(&device_a, n*sizeof(float));
	err = hipMalloc(&device_b, n*sizeof(float));

    // global -> host to device: 
	err = hipMemcpy(device_a, host_a, n*sizeof(float), hipMemcpyHostToDevice);
	err = hipMemcpy(device_b, host_b, n*sizeof(float), hipMemcpyHostToDevice);

    // 啟動一個 block，有 n 個 threads（剛好一人處理一個元素）。
	VectorAddKernel<<<dim3(1), dim3(n)>>>(device_a, device_b);

    // 從裝置端把加總結果傳回 host_a。
	err = hipMemcpy(host_a, device_a, n*sizeof(float), hipMemcpyDeviceToHost);
	err = hipFree(device_a);
	err = hipFree(device_b);

	cout << "vector a: ";
	for(int i=0; i<n; i++) cout << host_a[i] << " ";
	cout << endl;

	return SUCCESS;
}
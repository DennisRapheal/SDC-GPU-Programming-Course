#include <hip/hip_runtime.h>
#include <iostream>
#include <string>

#define SUCCESS 0
#define FAILURE 1

using namespace std;

//TODO:
// Implement the GPU Kernel function
// 1. Calculate the global thread ID
// 2. Perform Character Transformation

__global__ void CharTransformKernel(const char* input, char* output, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        output[idx] = input[idx] + 1; // 字元轉換邏輯：+1
    }
}

int main(int argc, char* argv[])
{
	/* Initial input,output for the host and create memory objects for the kernel */
	cout << "<Program Start>" << endl;
	const char* input = "GdkknVnqkc";
	size_t strlength = strlen(input);
	cout << "input string: ";
	cout << input << endl;
	char output[strlength + 1];

	/* The memory would be in GPU.                        */
	/* Notice that those two char* variables are pointer! */
	char* inputBuffer;
	char* outputBuffer;

	//TODO: 重點五步驟
	// 1. Allocate Device Memory (GPU Memory)
	// 2. Copy Data from Host Memory (CPU Data) to Device Memory (GPU Data)
	// 3. Launch the GPU Kernel
	// 4. Copy the Computation Result from Device Memory back to Host Memory
	// 5. Free the Device Memory

    // 1.
    hipMalloc(&inputBuffer, strlength * sizeof(char));
    hipMalloc(&outputBuffer, strlength * sizeof(char));

    // 2. copy data from host to memory, input -> inputBuffer
    hipMemcpy(inputBuffer, input, strlength * sizeof(char), hipMemcpyHostToDevice);

    // 3.
    CharTransformKernel<<<dim3(1), dim3(strlength)>>>(inputBuffer, outputBuffer, strlength);
	
    // 4.
    hipMemcpy(output, outputBuffer, strlength * sizeof(char), hipMemcpyDeviceToHost);

    // 5.
    hipFree(inputBuffer);
    hipFree(outputBuffer);

	output[strlength] = '\0';	// Add the terminal character to the end of output.
    
	cout << "output string: ";
	cout << output << endl;
	
	if(strcmp(output, "HelloWorld") == 0) cout << "[Pass]" << endl;
	else cout << "[Wrong Answer]" << endl;
	
	return SUCCESS; // The process is successfully executed whether the answer is correct or wrong.
}
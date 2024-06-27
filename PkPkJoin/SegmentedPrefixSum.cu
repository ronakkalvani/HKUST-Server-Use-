#include <cuda_runtime.h>
#include <iostream>

__global__ void segmentedPrefixSum(int* input, int* output, int n) {
    // Shared memory for storing the prefix sums within a block
    __shared__ int shared[8];
    __shared__ int seg_start[8];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Load data into shared memory
    shared[tid] = input[idx];
    __syncthreads();

    // Initialize segment start markers
    if (tid == 0) {
        seg_start[tid] = 0;
    } else {
        seg_start[tid] = (shared[tid] != shared[tid - 1]) ? 1 : 0;
    }
    __syncthreads();

    // Perform prefix sum within segments
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int temp = (tid >= offset) ? shared[tid - offset] : 0;
        int seg_temp = (tid >= offset) ? seg_start[tid - offset] : 0;
        __syncthreads();
        
        if (seg_start[tid] == 0) {
            shared[tid] += temp;
        }
        seg_start[tid] += seg_temp;
        __syncthreads();
    }

    // Write the results to output
    output[idx] = shared[tid];
}

void printArray(int* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << std::endl;
    }
}

int main() {
    const int blockSize = 8;
    const int dataSize = 24; // Multiple of blockSize
    int h_input[dataSize] = {0, 0, 0, 1, 1, 1, 2, 2,
                             0, 0, 1, 1, 1, 1, 2, 2,
                             0, 0, 0, 0, 0, 0, 1, 1};
    int h_output[dataSize];

    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, dataSize * sizeof(int));
    cudaMalloc(&d_output, dataSize * sizeof(int));

    cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    segmentedPrefixSum<<<dataSize / blockSize, blockSize>>>(d_input, d_output, dataSize);

    cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Input:" << std::endl;
    printArray(h_input, dataSize);

    std::cout << "Output:" << std::endl;
    printArray(h_output, dataSize);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

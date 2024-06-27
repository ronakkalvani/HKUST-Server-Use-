#include <cuda_runtime.h>
#include <iostream>

__global__ void segmentedPrefixSum(int *input, int *output, int n, int blockSize) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    if (global_tid < n) {
        shared[tid] = input[global_tid];
    }
    __syncthreads();

    // Initialize prefix sum within segments
    if (global_tid < n) {
        if (tid == 0 || shared[tid] != shared[tid - 1]) {
            output[global_tid] = 0;
        } else {
            output[global_tid] = output[global_tid - 1] + 1;
        }
    }

    // Perform prefix sum within segments in shared memory
    for (int stride = 1; stride < blockSize; stride *= 2) {
        __syncthreads();
        if (tid >= stride && shared[tid] == shared[tid - stride]) {
            output[global_tid] += output[global_tid - stride];
        }
    }
}

int main() {
    int blockSize = 12;
    int h_input[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    int n = sizeof(h_input) / sizeof(h_input[0]);
    int *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + blockSize - 1) / blockSize;
    segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n, blockSize);

    int h_output[n];
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % blockSize == 0) std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

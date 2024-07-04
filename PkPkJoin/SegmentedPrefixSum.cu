#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

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
            output[global_tid] = 1;
        }
    }
    __syncthreads();

    // Perform prefix sum within segments in shared memory
    for (int stride = 1; stride < blockSize; stride *= 2) {
        if (global_tid < n && tid >= stride && shared[tid] == shared[tid - stride]) {
            output[global_tid] += output[global_tid - stride];
        }
        __syncthreads();
    }
}

int main() {
    int blockSize = 8;
    std::vector<int> h_input = {
        0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1
    };
    int n = h_input.size();

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + blockSize - 1) / blockSize;
    segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n, blockSize);

    std::vector<int> h_output(n);
    cudaMemcpy(h_output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input and output for verification
    std::cout << "Input:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << h_input[i] << " ";
        if ((i + 1) % blockSize == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % blockSize == 0) std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>

// __global__ void segmentedPrefixSum(int *input, int *output, int n, int blockSize) {
//     extern __shared__ int shared[];

//     int tid = threadIdx.x;
//     int global_tid = blockIdx.x * blockDim.x + tid;

//     // Load input into shared memory
//     if (global_tid < n) {
//         shared[tid] = input[global_tid];
//     }
//     __syncthreads();

//     // Initialize prefix sum within segments
//     if (global_tid < n) {
//         if (tid == 0 || shared[tid] != shared[tid - 1]) {
//             output[global_tid] = 0;
//         } else {
//             output[global_tid] = output[global_tid - 1] + 1;
//         }
//     }

//     // Perform prefix sum within segments in shared memory
//     for (int stride = 1; stride < blockSize; stride *= 2) {
//         __syncthreads();
//         if (tid >= stride && shared[tid] == shared[tid - stride]) {
//             output[global_tid] += output[global_tid - stride];
//         }
//     }
// }

// int main() {
//     int blockSize = 512;
//     int n = 1e4;  // Large dataset size
//     std::vector<int> h_input(n);

//     for (int i = 0; i < n; ++i) {
//         if (i<100) h_input[i] = i /10;
//         else h_input[i] = i /100;
//     }

//     int *d_input, *d_output;
//     cudaMalloc(&d_input, n * sizeof(int));
//     cudaMalloc(&d_output, n * sizeof(int));

//     cudaMemcpy(d_input, h_input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

//     int numBlocks = (n + blockSize - 1) / blockSize;
//     segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n, blockSize);

//     std::vector<int> h_output(n);
//     cudaMemcpy(h_output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print input and output for verification
//     std::cout << "Input:" << std::endl;
//     for (int i = 0; i < n; ++i) {
//         std::cout << h_input[i] << " ";
//         if ((i + 1) % blockSize == 0) std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     std::cout << "Output:" << std::endl;
//     for (int i = 0; i < n; ++i) {
//         std::cout << h_output[i] << " ";
//         if ((i + 1) % blockSize == 0) std::cout << std::endl;
//     }

//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }

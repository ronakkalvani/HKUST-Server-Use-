#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256  // Define the block size

// Kernel to initialize block with sorted data
__global__ void radixSortBlocks(int* d_data, int n) {
    int blockIdx = blockIdx.x;
    int threadId = threadIdx.x;
    int blockOffset = blockIdx * BLOCK_SIZE;
    int offset = blockOffset + threadId;

    extern __shared__ int sharedData[];

    // Load data into shared memory
    if (offset < n) {
        sharedData[threadId] = d_data[offset];
    } else {
        sharedData[threadId] = INT_MAX;
    }
    __syncthreads();

    // Perform radix sort on the data within this block
    for (int exp = 1; exp <= 1000000; exp *= 10) {
        int bucket[10] = { 0 };

        // Count occurrences in shared memory
        if (sharedData[threadId] != INT_MAX) {
            atomicAdd(&bucket[(sharedData[threadId] / exp) % 10], 1);
        }
        __syncthreads();

        // Prefix sum
        int prefixSum = 0;
        for (int i = 0; i <= (sharedData[threadId] / exp) % 10; ++i) {
            prefixSum += bucket[i];
        }
        __syncthreads();

        // Re-arrange data in shared memory
        if (sharedData[threadId] != INT_MAX) {
            sharedData[--prefixSum] = d_data[offset];
        }
        __syncthreads();
    }

    // Write sorted data back to global memory
    if (offset < n) {
        d_data[offset] = sharedData[threadId];
    }
}

int main() {
    // Initialize host data
    thrust::host_vector<int> h_data = { 34, 78, 12, 56, 89, 21, 90, 34, 23, 45, 67, 11, 23, 56, 78, 99, 123, 45, 67, 89, 23, 45, 67, 34, 78 };

    int n = h_data.size();

    // Copy data to device
    thrust::device_vector<int> d_data = h_data;

    int* raw_d_data = thrust::raw_pointer_cast(d_data.data());

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel to sort blocks
    radixSortBlocks<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(raw_d_data, n);

    // Copy sorted data back to host
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());

    // Print sorted blocks
    for (int i = 0; i < h_data.size(); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


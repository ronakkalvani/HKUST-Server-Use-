#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256

// Kernel to sort each block individually using CUB's radix sort
__global__ void sortBlocks(int* d_data, int n) {
    int offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Allocate shared memory for sorting within the block
    extern __shared__ int sharedData[];

    // Load data into shared memory
    if (offset < n) {
        sharedData[threadIdx.x] = d_data[offset];
    } 
    else {
        sharedData[threadIdx.x] = INT_MAX;
    }
    __syncthreads();

    // Sorting within the block using CUB
    typedef cub::BlockRadixSort<int, BLOCK_SIZE> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    BlockRadixSort(temp_storage).Sort(sharedData);

    __syncthreads();

    // Write sorted data back to global memory
    if (offset < n) {
        d_data[offset] = sharedData[threadIdx.x];
    }
}

int main() {
    // Initialize host data
    std::vector<int> h_data = { 34, 78, 12, 56, 89, 21, 90, 34, 23, 45, 67, 11, 23, 56, 78, 99, 123, 45, 67, 89, 23, 45, 67, 34, 78 };
    int n = h_data.size();

    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel to sort blocks
    sortBlocks<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, n);

    // Copy sorted data back to host
    cudaMemcpy(h_data.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted blocks
    for (int i = 0; i < h_data.size(); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_data);

    return 0;
}


// Data Initialization: We initialize the data on the host and copy it to the device.\

// Kernel Launch: We launch a kernel with a number of blocks that covers the entire dataset. Each block operates on a chunk of the data.

// Shared Memory and Synchronization: Each block loads its chunk of data into shared memory. This local shared memory is used to avoid race conditions.

// Radix Sort: We implement the radix sort within each block, ensuring thread synchronization using __syncthreads().

// Copy Back to Host: After sorting, the data is copied back to the host and printed.



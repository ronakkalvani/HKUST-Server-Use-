#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to merge sorted blocks into a single sorted array
__global__ void mergeSortedBlocks(int* sorted_data, int* block_offsets, int* global_splitters, int num_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = block_offsets[blockIdx.x];
    int block_end = (blockIdx.x < num_blocks - 1) ? block_offsets[blockIdx.x + 1] : gridDim.x * blockDim.x;

    // Each thread merges its assigned block into the global sorted array
    for (int i = block_start + tid; i < block_end; i += blockDim.x) {
        // Perform binary search to find the correct position in global sorted array
        int value = sorted_data[i];
        int low = 0, high = num_blocks;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (value < global_splitters[mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        int insert_position = low;  // Position in the global sorted array
        // Perform atomic insertion (using CUDA atomicCAS for simplicity)
        int old = atomicCAS(&sorted_data[insert_position], sorted_data[insert_position], value);
        while (old != sorted_data[insert_position]) {
            old = atomicCAS(&sorted_data[insert_position], sorted_data[insert_position], value);
        }
    }
}

int main() {
    const int num_blocks = 4;  // Number of sorted blocks
    const int block_size = 256;  // Size of each sorted block
    const int total_size = num_blocks * block_size;

    // Initialize CUDA variables
    int* d_sorted_data;
    int* d_block_offsets;
    int* d_global_splitters;
    
    // Allocate memory on device
    cudaMalloc((void**)&d_sorted_data, total_size * sizeof(int));
    cudaMalloc((void**)&d_block_offsets, num_blocks * sizeof(int));
    cudaMalloc((void**)&d_global_splitters, (num_blocks - 1) * sizeof(int));
    
    // Assume sorted_data, block_offsets, and global_splitters are initialized properly
    // Copy data to device memory
    cudaMemcpy(d_sorted_data, sorted_data, total_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, block_offsets, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_splitters, global_splitters, (num_blocks - 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel to merge sorted blocks
    int threads_per_block = 256;
    int blocks_per_grid = num_blocks;
    mergeSortedBlocks<<<blocks_per_grid, threads_per_block>>>(d_sorted_data, d_block_offsets, d_global_splitters, num_blocks);

    // Copy sorted data back to host
    cudaMemcpy(sorted_data, d_sorted_data, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted data (example)
    std::cout << "Sorted Data: ";
    for (int i = 0; i < total_size; ++i) {
        std::cout << sorted_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_sorted_data);
    cudaFree(d_block_offsets);
    cudaFree(d_global_splitters);

    return 0;
}

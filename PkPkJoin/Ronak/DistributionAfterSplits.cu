#include <cuda_runtime.h>
#include <iostream>

const int N = 1024;  // Total number of elements
const int num_blocks = 4;  // Number of blocks (assuming for example)
const int block_size = N / num_blocks;  // Elements per block

__global__ void distribute_data(int* d_sorted_data, int* d_splitters, int* d_block_offsets, int* d_block_data) {
    int block_id = blockIdx.x;
    int start_idx = block_id * block_size;
    int end_idx = start_idx + block_size;

    // Find the range in the sorted data that belongs to this block
    int start = 0, end = N;
    for (int i = 0; i < num_blocks - 1; ++i) {
        if (start_idx >= d_splitters[i] && start_idx < d_splitters[i + 1]) {
            start = i;
        }
        if (end_idx > d_splitters[i] && end_idx <= d_splitters[i + 1]) {
            end = i + 1;
        }
    }

    // Calculate the offset for this block
    int offset = d_block_offsets[block_id];

    // Copy data from sorted array to block's data
    for (int i = start_idx; i < end_idx; ++i) {
        d_block_data[offset + (i - start_idx)] = d_sorted_data[i];
    }
}

int main() {
    // Allocate and initialize data on the host
    int h_sorted_data[N];
    for (int i = 0; i < N; ++i) {
        h_sorted_data[i] = rand() % 100; // Initialize with random values (0-99)
    }

    // Allocate memory on the device for sorted data
    int* d_sorted_data;
    cudaMalloc((void**)&d_sorted_data, N * sizeof(int));
    cudaMemcpy(d_sorted_data, h_sorted_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate and initialize splitters and block offsets on the host
    int h_splitters[num_blocks - 1];
    for (int i = 0; i < num_blocks - 1; ++i) {
        h_splitters[i] = (i + 1) * (100 / num_blocks);  // Example of evenly spaced splitters
    }

    int* d_splitters;
    cudaMalloc((void**)&d_splitters, (num_blocks - 1) * sizeof(int));
    cudaMemcpy(d_splitters, h_splitters, (num_blocks - 1) * sizeof(int), cudaMemcpyHostToDevice);

    int h_block_offsets[num_blocks];
    for (int i = 0; i < num_blocks; ++i) {
        h_block_offsets[i] = i * block_size;
    }

    int* d_block_offsets;
    cudaMalloc((void**)&d_block_offsets, num_blocks * sizeof(int));
    cudaMemcpy(d_block_offsets, h_block_offsets, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory on the device for block-wise data
    int* d_block_data;
    cudaMalloc((void**)&d_block_data, N * sizeof(int));

    // Launch kernel to distribute data to blocks
    distribute_data<<<num_blocks, 1>>>(d_sorted_data, d_splitters, d_block_offsets, d_block_data);
    cudaDeviceSynchronize();

    // Example: Print block-wise data
    for (int i = 0; i < num_blocks; ++i) {
        int* block_data = new int[block_size];
        cudaMemcpy(block_data, d_block_data + i * block_size, block_size * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Block " << i << " data: ";
        for (int j = 0; j < block_size; ++j) {
            std::cout << block_data[j] << " ";
        }
        std::cout << std::endl;
        delete[] block_data;
    }

    // Free allocated memory
    cudaFree(d_sorted_data);
    cudaFree(d_splitters);
    cudaFree(d_block_offsets);
    cudaFree(d_block_data);

    return 0;
}

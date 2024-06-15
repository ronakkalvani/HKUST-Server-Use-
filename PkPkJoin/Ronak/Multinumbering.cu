#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>

__global__ void multi_numbering_kernel(int *keys, int *multi_numbers, int num_elements) {
    // Allocate shared memory for CUB prefix sum
    __shared__ int prefix_sum[1024]; // Assuming a block size of 1024

    // Define CUB types
    typedef cub::BlockScan<int, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Compute thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // Determine if current element is the first of its key
        int is_first = (idx == 0 || keys[idx] != keys[idx - 1]) ? 1 : 0;

        // Compute prefix sum within block
        int block_prefix_sum;
        BlockScan(temp_storage).ExclusiveSum(is_first, block_prefix_sum);

        // Compute global index with prefix sum
        int global_prefix_sum;
        if (threadIdx.x == 0) {
            global_prefix_sum = block_prefix_sum;
            prefix_sum[blockIdx.x] = global_prefix_sum;
        }
        __syncthreads();

        // Compute multi-numbering index
        multi_numbers[idx] = prefix_sum[blockIdx.x] + is_first;
    }
}

void multi_numbering_cuda(int *h_keys, int *h_multi_numbers, int num_elements) {
    int *d_keys, *d_multi_numbers;

    // Allocate device memory
    cudaMalloc((void **)&d_keys, num_elements * sizeof(int));
    cudaMalloc((void **)&d_multi_numbers, num_elements * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_keys, h_keys, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Determine block size and launch the kernel
    int block_size = 1024; // Assuming a block size of 1024
    int num_blocks = (num_elements + block_size - 1) / block_size;
    multi_numbering_kernel<<<num_blocks, block_size>>>(d_keys, d_multi_numbers, num_elements);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the result back to host
    cudaMemcpy(h_multi_numbers, d_multi_numbers, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_multi_numbers);
}

int main() {
    // Example input data
    int h_keys[] = {1, 2, 1, 3, 2, 1};
    int num_elements = sizeof(h_keys) / sizeof(int);
    int h_multi_numbers[num_elements];

    // Call CUDA function to compute multi-numbering indices
    multi_numbering_cuda(h_keys, h_multi_numbers, num_elements);

    // Print results
    std::cout << "Original keys: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_keys[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Multi-numbering indices: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_multi_numbers[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

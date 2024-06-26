#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256

// Kernel for segmented prefix sum
__global__ void segmentedPrefixSum(int *d_in, int *d_out, int num_elements) {
    // Allocate shared memory for the block's input and output
    __shared__ int shared_in[BLOCK_SIZE];
    __shared__ int shared_out[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load input data into shared memory
    if (index < num_elements) {
        shared_in[tid] = d_in[index];
    } else {
        shared_in[tid] = 0;
    }
    __syncthreads();

    // Prepare CUB temp storage
    typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Compute segmented prefix sum
    int aggregate;
    BlockScan(temp_storage).InclusiveSum(shared_in[tid], shared_out[tid], aggregate);

    // Write result to global memory
    if (index < num_elements) {
        d_out[index] = shared_out[tid];
    }
}

int main() {
    int num_elements = 1024;
    std::vector<int> h_in(num_elements);
    std::vector<int> h_out(num_elements);

    // Initialize input data
    for (int i = 0; i < num_elements; ++i) {
        h_in[i] = 1; // or any value you want to test
    }

    // Allocate device memory
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, num_elements * sizeof(int));
    cudaMalloc((void**)&d_out, num_elements * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_in, h_in.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    segmentedPrefixSum<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, num_elements);

    // Copy output data back to host
    cudaMemcpy(h_out.data(), d_out, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Display results
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

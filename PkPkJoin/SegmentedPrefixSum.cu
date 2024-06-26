#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Error handling macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(error);                                                       \
        }                                                                      \
    } while (0)

// Kernel to initialize flags based on new segment starts
__global__ void InitializeFlags(const int* input, int* flags, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        flags[idx] = (idx == 0 || input[idx] != input[idx - 1]) ? 1 : 0;
    }
}

void SegmentedPrefixSum(const std::vector<int>& input, std::vector<int>& output) {
    const int num_items = input.size();

    int* d_input = nullptr;
    int* d_output = nullptr;
    int* d_flags = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, num_items * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), num_items * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize flags
    const int block_size = 256;
    const int grid_size = (num_items + block_size - 1) / block_size;
    InitializeFlags<<<grid_size, block_size>>>(d_input, d_flags, num_items);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Determine temporary device storage requirements
    cub::DeviceSegmentedScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output,
        num_items,
        num_items,  // number of segments
        d_flags, d_flags + 1);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run segmented prefix sum
    cub::DeviceSegmentedScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output,
        num_items,
        num_items,  // number of segments
        d_flags, d_flags + 1);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, num_items * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_temp_storage));
}

int main() {
    std::vector<int> input = {1, 2, 2, 3, 1, 1, 2, 3, 3, 3};
    std::vector<int> output(input.size(), 0);

    SegmentedPrefixSum(input, output);

    // Print the result
    for (int val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

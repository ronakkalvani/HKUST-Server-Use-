#include <iostream>
#include <vector>
#include <cub/cub.cuh>

const int BLOCK_SIZE = 8;

__global__ void segmentedPrefixSumKernel(int* d_in, int* d_out, int num_elements) {
    __shared__ int shared_in[BLOCK_SIZE];
    __shared__ int shared_out[BLOCK_SIZE];
    __shared__ bool shared_flags[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (tid < num_elements) {
        shared_in[threadIdx.x] = d_in[tid];
        shared_flags[threadIdx.x] = (threadIdx.x == 0) ? false : (shared_in[threadIdx.x] != shared_in[threadIdx.x - 1]);
    }

    __syncthreads();

    // Initialize temporary storage
    void* temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Calculate the required temporary storage size
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, shared_flags, shared_out, BLOCK_SIZE);
    cudaMalloc(&temp_storage, temp_storage_bytes);

    // Perform exclusive prefix sum on the flags
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, shared_flags, shared_out, BLOCK_SIZE);

    __syncthreads();

    if (tid < num_elements) {
        // Write the result to the output
        if (shared_flags[threadIdx.x]) {
            shared_out[threadIdx.x] = 0;
        } else {
            shared_out[threadIdx.x]++;
        }
        d_out[tid] = shared_out[threadIdx.x];
    }

    cudaFree(temp_storage);
}

void segmentedPrefixSum(const std::vector<int>& input, std::vector<int>& output) {
    int num_elements = input.size();
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_in;
    int* d_out;
    cudaMalloc(&d_in, num_elements * sizeof(int));
    cudaMalloc(&d_out, num_elements * sizeof(int));

    cudaMemcpy(d_in, input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);

    segmentedPrefixSumKernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, num_elements);

    cudaMemcpy(output.data(), d_out, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    std::vector<int> input = {0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2};
    std::vector<int> output(input.size());

    segmentedPrefixSum(input, output);

    for (int i = 0; i < output.size(); i++) {
        std::cout << output[i] << " ";
        if ((i + 1) % BLOCK_SIZE == 0) std::cout << std::endl;
    }

    return 0;
}

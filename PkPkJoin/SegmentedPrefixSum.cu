#include <iostream>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 8;

__global__ void segmentedPrefixSumKernel(int* d_in, int* d_out, int num_elements) {
    __shared__ int block_sum;
    __shared__ int previous_value;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements) {
        if (threadIdx.x == 0) {
            block_sum = 0;
            previous_value = d_in[tid];
        }

        __syncthreads();

        if (d_in[tid] != previous_value) {
            block_sum++;
            previous_value = d_in[tid];
        }

        d_out[tid] = block_sum;
    }
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

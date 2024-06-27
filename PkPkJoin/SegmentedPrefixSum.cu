#include <stdio.h>

#define BLOCK_SIZE 8

__global__ void segmentedPrefixSum(int *input, int *output, int n) {
    __shared__ int temp[BLOCK_SIZE];
    __shared__ int flags[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (gid < n) {
        temp[tid] = input[gid];
        flags[tid] = (tid == 0) ? 1 : (input[gid] != input[gid - 1]);
    } else {
        temp[tid] = 0;
        flags[tid] = 0;
    }
    __syncthreads();

    // Perform segmented prefix sum in shared memory
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            if (flags[tid] == 0) {
                val = temp[tid - offset];
            }
        }
        __syncthreads();
        temp[tid] += val;
        flags[tid] |= (tid >= offset) ? flags[tid - offset] : 0;
        __syncthreads();
    }

    // Write results to output array
    if (gid < n) {
        output[gid] = temp[tid];
    }
}

int main() {
    const int n = 24;
    int h_input[n] = {0, 0, 0, 1, 1, 1, 2, 2,
                      0, 0, 1, 1, 1, 1, 2, 2,
                      0, 0, 0, 0, 0, 0, 1, 1};
    int h_output[n];

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    segmentedPrefixSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
        if ((i + 1) % BLOCK_SIZE == 0) printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

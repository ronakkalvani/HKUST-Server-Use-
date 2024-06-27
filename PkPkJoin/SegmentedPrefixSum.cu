#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

__global__ void segmentedPrefixSum(int *d_in, int *d_out, int n) {
    __shared__ int temp[BLOCK_SIZE];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        temp[threadIdx.x] = d_in[idx];
    }
    __syncthreads();

    // Step 1: Compute prefix sum within each segment
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int val = 0;
        if (threadIdx.x >= stride) {
            if (temp[threadIdx.x] == temp[threadIdx.x - stride]) {
                val = d_out[idx - stride];
            }
        }
        __syncthreads();

        if (threadIdx.x >= stride) {
            if (temp[threadIdx.x] == temp[threadIdx.x - stride]) {
                d_out[idx] = val + 1;
            }
        }
        __syncthreads();
    }

    // Step 2: Handle the first element of each segment
    if (threadIdx.x == 0) {
        d_out[idx] = 0;
    } else if (temp[threadIdx.x] != temp[threadIdx.x - 1]) {
        d_out[idx] = 0;
    }
}

int main() {
    const int n = 24;
    int h_in[n] = {
        0, 0, 0, 1, 1, 1, 2, 2,
        0, 0, 1, 1, 1, 1, 2, 2,
        0, 0, 0, 0, 0, 0, 1, 1
    };
    int h_out[n];

    int *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));

    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    segmentedPrefixSum<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_out[i]);
        if ((i + 1) % BLOCK_SIZE == 0) {
            printf("\n");
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

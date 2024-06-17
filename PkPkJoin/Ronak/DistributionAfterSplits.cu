#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) <<    \
            " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Kernel to print array
__global__ void printArray(int* arr, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Kernel to determine the correct partition for each element
__global__ void partitionElements(const int* d_subarrays, int* d_partitions, const int* d_pivots, int n, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int partition = 0;
        while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
            partition++;
        }
        atomicAdd(&d_partitions[partition * n + tid], d_subarrays[tid]);
    }
}

// Kernel to merge partitions into the output array
__global__ void mergePartitions(const int* d_partitions, int* d_output, int n, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        for (int i = 0; i < p; i++) {
            if (d_partitions[i * n + tid] != 0) {
                d_output[tid] = d_partitions[i * n + tid];
                break;
            }
        }
    }
}

void merge(int* h_subarrays, int* h_pivots, int n, int p) {
    // Device pointers
    int *d_subarrays, *d_output, *d_pivots, *d_partitions;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_subarrays, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivots, (p - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partitions, n * p * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_subarrays, h_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pivots, h_pivots, (p - 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partitions, 0, n * p * sizeof(int)));

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel to partition elements
    partitionElements<<<numBlocks, blockSize>>>(d_subarrays, d_partitions, d_pivots, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch kernel to merge partitions
    mergePartitions<<<numBlocks, blockSize>>>(d_partitions, d_output, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    int* h_output = new int[n];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    for (int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_subarrays));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_pivots));
    CUDA_CHECK(cudaFree(d_partitions));

    delete[] h_output;
}

int main() {
    // Example data
    int h_subarrays[] = {1, 3, 5, 7, 2, 4, 6, 8};
    int h_pivots[] = {4};

    int n = 8;
    int p = 2;

    merge(h_subarrays, h_pivots, n, p);

    return 0;
}

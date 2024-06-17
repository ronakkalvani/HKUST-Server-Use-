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

__global__ void mergePartitions(
    int* d_subarrays, int* d_partitions, int* d_output, int* d_pivots, 
    int* d_partition_counts, int n, int p) 
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Determine the partition for each element
    if (tid < n) {
        int partition = 0;
        while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
            partition++;
        }

        // Step 2: Count the number of elements in each partition
        atomicAdd(&d_partition_counts[partition], 1);
    }

    // Synchronize threads to ensure all counts are computed
    __syncthreads();

    // Step 3: Compute the starting index for each partition
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < p; ++i) {
            int temp = d_partition_counts[i];
            d_partition_counts[i] = sum;
            sum += temp;
        }
    }

    // Synchronize threads to ensure starting indices are computed
    __syncthreads();

    // Step 4: Distribute elements to the output array
    if (tid < n) {
        int partition = 0;
        while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
            partition++;
        }
        int pos = atomicAdd(&d_partition_counts[partition], 1);
        d_output[pos] = d_subarrays[tid];
    }
}


// __global__ void mergePartitions(int* d_subarrays, int* d_partitions, int* d_output, int* d_pivots, int n, int p) {
//     // Calculate global thread ID
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid < n) {
//         // Determine which partition the element belongs to
//         int partition = 0;
//         while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
//             partition++;
//         }
//         // Compute the global position of the element in the output array
//         atomicAdd(&d_partitions[partition], 1);
//         d_output[atomicAdd(&d_partitions[partition], 1)] = d_subarrays[tid];
//     }
// }

void merge(int* h_subarrays, int* h_pivots, int n, int p) {
    // Device pointers
    int *d_subarrays, *d_output, *d_pivots, *d_partitions;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_subarrays, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivots, (p - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partitions, p * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_subarrays, h_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pivots, h_pivots, (p - 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partitions, 0, p * sizeof(int)));

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel to merge partitions
    mergePartitions<<<numBlocks, blockSize>>>(d_subarrays, d_partitions, d_output, d_pivots, n, p);
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

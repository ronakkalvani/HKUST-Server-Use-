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

__global__ void countElements(
    int* d_subarrays, int* d_pivots, int* d_partition_counts, int n, int p) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int partition = 0;
        while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
            partition++;
        }
        atomicAdd(&d_partition_counts[partition], 1);
    }
}

__global__ void computeStarts(int* d_partition_counts, int* d_partition_starts, int p) {
    int tid = threadIdx.x;

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < p; ++i) {
            d_partition_starts[i] = sum;
            sum += d_partition_counts[i];
        }
    }
}

__global__ void distributeElements(
    int* d_subarrays, int* d_output, int* d_pivots, 
    int* d_partition_starts, int* d_partition_offsets, int n, int p) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int partition = 0;
        while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
            partition++;
        }
        int pos = atomicAdd(&d_partition_offsets[partition], 1);
        d_output[d_partition_starts[partition] + pos] = d_subarrays[tid];
    }
}

int main() {
    const int n = 512;
    int p = 8;
    int h_subarrays[n];
    int h_pivots[p-1];
    for (int i = 0; i < n; i++) {
        h_subarrays[i] = rand() % n;
    }
    for (int i = 0; i < p-1; i++) {
        h_pivots[i] = (i + 1) * (n / p);
    }

    // Device pointers
    int *d_subarrays, *d_output, *d_pivots, *d_partition_counts, *d_partition_starts, *d_partition_offsets;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_subarrays, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivots, (p - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_offsets, p * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_subarrays, h_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pivots, h_pivots, (p - 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_partition_offsets, 0, p * sizeof(int)));

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernels in sequence to ensure synchronization
    countElements<<<numBlocks, blockSize>>>(d_subarrays, d_pivots, d_partition_counts, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    computeStarts<<<1, p>>>(d_partition_counts, d_partition_starts, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    distributeElements<<<numBlocks, blockSize>>>(d_subarrays, d_output, d_pivots, d_partition_starts, d_partition_offsets, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printArray<<<1,1>>>(d_partition_counts, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    printArray<<<1,1>>>(d_partition_starts, p);
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
    CUDA_CHECK(cudaFree(d_partition_counts));
    CUDA_CHECK(cudaFree(d_partition_starts));
    CUDA_CHECK(cudaFree(d_partition_offsets));

    delete[] h_output;

    return 0;
}


// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <cub/cub.cuh>

// // Error checking macro
// #define CUDA_CHECK(call)                                                   \
//     do {                                                                   \
//         cudaError_t error = call;                                          \
//         if (error != cudaSuccess) {                                        \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(error) <<    \
//             " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
//             exit(1);                                                       \
//         }                                                                  \
//     } while (0)

// // Kernel to print array
// __global__ void printArray(int* arr, int size) {
//     for (int i = 0; i < size; ++i) {
//         printf("%d ", arr[i]);
//     }
//     printf("\n");
// }

// __global__ void mergePartitions(
//     int* d_subarrays, int* d_output, int* d_pivots, 
//     int* d_partition_counts, int* d_partition_starts, int* d_partition_offsets, int n, int p) 
// {
//     // Calculate global thread ID
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Step 1: Determine the partition for each element
//     if (tid < n) {
//         int partition = 0;
//         while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
//             partition++;
//         }

//         printf("Element: %d, Partition: %d\n", tid, partition);

//         // Step 2: Count the number of elements in each partition
//         atomicAdd(&d_partition_counts[partition], 1);
//     }

//     // Synchronize threads to ensure all counts are computed
//     __syncthreads();

//     // Step 3: Compute the starting index for each partition
//     if (tid == 0) {
//         int sum = 0;
//         for (int i = 0; i < p; ++i) {
//             d_partition_starts[i] = sum;
//             sum += d_partition_counts[i];
//         }
//     }

//     // Synchronize threads to ensure starting indices are computed

//     if (tid == 0) {
//         for (int i = 0; i < p; ++i) {
//             printf("%d ", d_partition_offsets[i]);
//         }
//         printf("\n");
//     }
//     __syncthreads();

//     // Step 4: Distribute elements to the output array
//     if (tid < n) {
//         int partition = 0;
//         while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
//             partition++;
//         }
//         int pos = atomicAdd(&d_partition_offsets[partition], 1);
//         d_output[d_partition_starts[partition] + pos] = d_subarrays[tid];
//         // d_output[d_partition_starts[partition] + atomicAdd(&d_partition_offsets[partition], 1)] = d_subarrays[tid];
//     }
//     if (tid == 0) {
//         for (int i = 0; i < p; ++i) {
//             printf("%d ", d_partition_offsets[i]);
//         }
//         printf("\n");
//     }
// }

// int main() {
//     // Example data
//     const int n = 512;
//     int p = 8;
//     int h_subarrays[n];
//     int h_pivots[p-1];
//     for (int i = 0; i < n; i++) {
//         h_subarrays[i] = i % n;
//     }
//     for (int i = 0; i < p-1; i++) {
//         h_pivots[i] = (i + 1) * (n / p);
//     }

//     // Device pointers
//     int *d_subarrays, *d_output, *d_pivots, *d_partition_counts, *d_partition_starts, *d_partition_offsets;

//     // Allocate device memory
//     CUDA_CHECK(cudaMalloc(&d_subarrays, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_pivots, (p - 1) * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_offsets, p * sizeof(int)));

//     // Copy data to device
//     CUDA_CHECK(cudaMemcpy(d_subarrays, h_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_pivots, h_pivots, (p - 1) * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_offsets, 0, p * sizeof(int)));

//     // Kernel launch parameters
//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     // Launch kernel to merge partitions
//     mergePartitions<<<numBlocks, blockSize>>>(
//         d_subarrays, d_output, d_pivots, 
//         d_partition_counts, d_partition_starts, d_partition_offsets, n, p
//     );
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     printArray<<<1,1>>>(d_partition_counts, p);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     printArray<<<1,1>>>(d_partition_starts, p);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Copy result back to host
//     int* h_output = new int[n];
//     CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

//     // Print result
//     for (int i = 0; i < n; ++i) {
//         std::cout << h_output[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free device memory
//     CUDA_CHECK(cudaFree(d_subarrays));
//     CUDA_CHECK(cudaFree(d_output));
//     CUDA_CHECK(cudaFree(d_pivots));
//     CUDA_CHECK(cudaFree(d_partition_counts));
//     CUDA_CHECK(cudaFree(d_partition_starts));
//     CUDA_CHECK(cudaFree(d_partition_offsets));

//     delete[] h_output;

//     return 0;
// }



// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <cub/cub.cuh>

// // Error checking macro
// #define CUDA_CHECK(call)                                                   \
//     do {                                                                   \
//         cudaError_t error = call;                                          \
//         if (error != cudaSuccess) {                                        \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(error) <<    \
//             " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
//             exit(1);                                                       \
//         }                                                                  \
//     } while (0)

// // Kernel to print array
// __global__ void printArray(int* arr, int size) {
//     for (int i = 0; i < size; ++i) {
//         printf("%d ", arr[i]);
//     }
//     printf("\n");
// }

// __global__ void mergePartitions(
//     int* d_subarrays, int* d_partitions, int* d_output, int* d_pivots, 
//     int* d_partition_counts, int* d_partition_starts, int n, int p) 
// {
//     // Calculate global thread ID
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Step 1: Determine the partition for each element
//     if (tid < n) {
//         int partition = 0;
//         while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
//             partition++;
//         }

//         printf("Element: %d, Partition: %d\n", tid, partition);

//         // Step 2: Count the number of elements in each partition
//         atomicAdd(&d_partition_counts[partition], 1);
//     }

//     // Synchronize threads to ensure all counts are computed
//     __syncthreads();

//     // Step 3: Compute the starting index for each partition
//     if (tid == 0) {
//         int sum = 0;
//         for (int i = 0; i < p; ++i) {
//             d_partition_starts[i] = sum;
//             sum += d_partition_counts[i];
//         }
//         for (int i = 0; i < p; ++i) {
//             printf("%d ", d_partition_starts[i]);
//         }
//         printf("\n");
//     }

//     // Synchronize threads to ensure starting indices are computed
//     __syncthreads();

//     // Step 4: Distribute elements to the output array
//     if (tid < n) {
//         int partition = 0;
//         while (partition < p - 1 && d_subarrays[tid] > d_pivots[partition]) {
//             partition++;
//         }
//         int pos = atomicAdd(&d_partition_starts[partition], 1);
//         d_output[pos] = d_subarrays[tid];
//     }
// }

// int main() {
//     // Example data
//     const int n = 512;
//     int p = 8;
//     int h_subarrays[n];
//     int h_pivots[p-1];
//     for (int i = 0; i < n; i++) {
//         h_subarrays[i] = i % n;
//     }
//     for (int i = 0; i < p-1; i++) {
//         h_pivots[i] = (i + 1) * (n / p);
//     }

//     // Device pointers
//     int *d_subarrays, *d_output, *d_pivots, *d_partition_counts, *d_partition_starts;

//     // Allocate device memory
//     CUDA_CHECK(cudaMalloc(&d_subarrays, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_pivots, (p - 1) * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));

//     // Copy data to device
//     CUDA_CHECK(cudaMemcpy(d_subarrays, h_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_pivots, h_pivots, (p - 1) * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));

//     // Kernel launch parameters
//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     // Launch kernel to merge partitions
//     mergePartitions<<<numBlocks, blockSize>>>(
//         d_subarrays, d_partition_counts, d_output, d_pivots, 
//         d_partition_counts, d_partition_starts, n, p
//     );
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     printArray<<<1,1>>>(d_partition_counts, p);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     printArray<<<1,1>>>(d_partition_starts, p);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Copy result back to host
//     int* h_output = new int[n];
//     CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

//     // Print result
//     for (int i = 0; i < n; ++i) {
//         std::cout << h_output[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free device memory
//     CUDA_CHECK(cudaFree(d_subarrays));
//     CUDA_CHECK(cudaFree(d_output));
//     CUDA_CHECK(cudaFree(d_pivots));
//     CUDA_CHECK(cudaFree(d_partition_counts));
//     CUDA_CHECK(cudaFree(d_partition_starts));

//     delete[] h_output;

//     return 0;
// }



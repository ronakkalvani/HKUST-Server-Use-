#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <algorithm>

__global__ void findSplitsKernel(const int *data, int *output, const int *splitters, int numData, int numSplitters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numData) {
        int item = data[tid];
        int left = 0;
        int right = numSplitters - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (splitters[mid] <= item) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        output[tid] = left;  // 'left' is the partition index
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "Error: " << message << " (" << cudaGetErrorString(error) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Example setup for large dataset
    const int numData = 1000000; // 1 million data points
    const int numSplitters = 1000; // 1000 splitters

    // Generating example data and splitters
    std::vector<int> h_data(numData);
    std::vector<int> h_splitters(numSplitters);

    // Fill data with sorted values for simplicity
    for (int i = 0; i < numData; ++i) {
        h_data[i] = rand()%numData;
    }

    // Fill splitters with sorted values
    for (int i = 0; i < numSplitters; ++i) {
        h_splitters[i] = (i + 1) * (numData / numSplitters);
    }

    // Device memory pointers
    int *d_data, *d_splitters, *d_output;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_data, numData * sizeof(int)), "Failed to allocate device memory for data");
    checkCudaError(cudaMalloc(&d_splitters, numSplitters * sizeof(int)), "Failed to allocate device memory for splitters");
    checkCudaError(cudaMalloc(&d_output, numData * sizeof(int)), "Failed to allocate device memory for output");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_data, h_data.data(), numData * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy data to device");
    checkCudaError(cudaMemcpy(d_splitters, h_splitters.data(), numSplitters * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy splitters to device");

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numData + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    findSplitsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, d_splitters, numData, numSplitters);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Copy result back to host
    std::vector<int> h_output(numData);
    checkCudaError(cudaMemcpy(h_output.data(), d_output, numData * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy output to host");

    // Optionally print out first few results
    for (int i = 0; i < numData; ++i) {
        std::cout << "Data: " << h_data[i] << " -> Partition: " << h_output[i] << std::endl;
    }

    // Free device memory
    checkCudaError(cudaFree(d_data), "Failed to free device memory for data");
    checkCudaError(cudaFree(d_splitters), "Failed to free device memory for splitters");
    checkCudaError(cudaFree(d_output), "Failed to free device memory for output");

    return 0;
}


// int main() {
//     const int n = 1e6;
//     int p = n/(1024);
//     int h_subarrays[n];
//     int h_pivots[p-1];
//     for (int i = 0; i < n; i++) {
//         h_subarrays[i] = rand() % 12715;
//     }
//     for (int i = 0; i < p-1; i++) {
//         h_pivots[i] = (i + 1) * (12715 / p);
//         std::cout<<h_pivots[i]<<" ";
//     }
//     std::cout<<"\n";

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
//     int blockSize = n/p;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     // Launch kernels in sequence to ensure synchronization
//     countElements<<<numBlocks, blockSize>>>(d_subarrays, d_pivots, d_partition_counts, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     computeStarts<<<1, 1>>>(d_partition_counts, d_partition_starts, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     distributeElements<<<numBlocks, blockSize>>>(d_subarrays, d_output, d_pivots, d_partition_starts, d_partition_offsets, n, p);
//     CUDA_CHECK(cudaGetLastError());
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



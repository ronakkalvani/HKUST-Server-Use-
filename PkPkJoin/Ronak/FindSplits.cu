#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// #define BLOCK_THREADS 32

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

__global__ void Splitterss(int* d_splitters,int* d_samples,int sample_size,int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < p - 1; ++i) {
            d_splitters[i] = d_samples[(i + 1) * sample_size / p];
        }
    }
}

// Kernel to sample elements
__global__ void sampleElements(int* d_sorted_subarrays, int* d_samples, int n, int sample_size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < sample_size) {
        int ind=rand()%n;
        d_samples[tid] = d_sorted_subarrays[ind];
    }
}

void FindSplit(int* d_sorted_data, int* d_samples, int* d_splitters, int n, int p,int sample_size) {
    int blockSize = BLOCK_THREADS;
    int numBlocks = (sample_size + blockSize - 1) / blockSize;
    int stride = n / sample_size;

    sampleElements<<<numBlocks, blockSize>>>(d_sorted_data, d_samples, n, sample_size, stride);
    cudaDeviceSynchronize();
    
    // Sort samples using CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_samples, d_samples, sample_size);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_samples, d_samples, sample_size);
    
    // Free temporary storage
    cudaFree(d_temp_storage);
}

// int main() {
//     // Example data
//     const int n = 256;
//     int p = 8;
//     int sample_size = n/p; // Adjust sample size as needed
//     // int h_sorted_subarrays[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
//     int h_sorted_subarrays[n];
//     for (int i=0;i<n;i++) {
//         h_sorted_subarrays[i] = rand() % 123;
//     }
//     int h_splitters[p - 1];

//     // Device pointers
//     int *d_sorted_subarrays, *d_samples, *d_splitters;
    
//     // Allocate device memory
//     CUDA_CHECK(cudaMalloc(&d_sorted_subarrays, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_samples, sample_size * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_splitters, (p - 1) * sizeof(int)));
    
//     // Copy data to device
//     CUDA_CHECK(cudaMemcpy(d_sorted_subarrays, h_sorted_subarrays, n * sizeof(int), cudaMemcpyHostToDevice));
    
//     // Launch kernel to sample elements
//     int blockSize = n/p;
//     int numBlocks = (sample_size + blockSize - 1) / blockSize;
//     int stride = n / sample_size;
//     sampleElements<<<numBlocks, blockSize>>>(d_sorted_subarrays, d_samples, n, sample_size, stride);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
    
//     // Sort samples using CUB
//     void* d_temp_storage = nullptr;
//     size_t temp_storage_bytes = 0;
    
//     // Determine temporary device storage requirements
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_samples, d_samples, sample_size);
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
//     // Run sorting operation
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_samples, d_samples, sample_size);
    
//     // Free temporary storage
//     CUDA_CHECK(cudaFree(d_temp_storage));
    
//     // Select splitters
//     int* h_samples = new int[sample_size];
//     CUDA_CHECK(cudaMemcpy(h_samples, d_samples, sample_size * sizeof(int), cudaMemcpyDeviceToHost));
    
//     for (int i = 0; i < p - 1; ++i) {
//         h_splitters[i] = h_samples[(i + 1) * sample_size / p];
//     }
    
//     // Free device memory
//     CUDA_CHECK(cudaFree(d_sorted_subarrays));
//     CUDA_CHECK(cudaFree(d_samples));
//     CUDA_CHECK(cudaFree(d_splitters));
    
//     delete[] h_samples;
    
//     // Print splitters
//     for (int i = 0; i < p - 1; ++i) {
//         std::cout << "Splitter " << i << ": " << h_splitters[i] << std::endl;
//     }

//     return 0;
// }

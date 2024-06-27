#include <iostream>
#include <vector>
#include <cub/cub.cuh>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(err); \
        } \
    } while (0)
__global__ void generateData(int* data, int blockSize, int p) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < blockSize * p) {
        data[tid] = tid % 100;  // Random data, replace with actual data generation
    }
}
void sortWithCub(int* d_data, int blockSize, int p) {
    for (int i = 0; i < p; i++) {
        int* d_block = d_data + i * blockSize;
        int* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Determine temporary device storage requirements
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_block, d_block, blockSize));

        // Allocate temporary storage
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Sort data
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_block, d_block, blockSize));

        // Free temporary storage
        CUDA_CHECK(cudaFree(d_temp_storage));
    }
}
__device__ int binarySearch(int* splitters, int numSplitters, int value) {
    int left = 0;
    int right = numSplitters - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (splitters[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}

__global__ void distributeData(int* d_data, int* d_splitters, int* d_out, int* d_prefixSum, int blockSize, int p, int numSplitters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < blockSize * p) {
        int data = d_data[tid];
        int blockIdx = binarySearch(d_splitters, numSplitters, data);
        
        // Find the position using segmented prefix sum
        int index = d_prefixSum[tid];
        d_out[blockIdx * blockSize + index] = data;
    }
}

__global__ void computePrefixSum(int* d_data, int* d_splitters, int* d_prefixSum, int blockSize, int p, int numSplitters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < blockSize * p) {
        int data = d_data[tid];
        int blockIdx = binarySearch(d_splitters, numSplitters, data);
        d_prefixSum[tid] = (blockIdx * blockSize) + tid;  // Initial position
    }
}

int main() {
    const int p = 4;  // Number of blocks
    const int blockSize = 1024;  // Size of each block
    const int dataSize = blockSize * p;
    const int numSplitters = p - 1;
    
    // Allocate memory
    int* h_data = (int*)malloc(dataSize * sizeof(int));
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, dataSize * sizeof(int)));

    // Generate sample data
    generateData<<<(dataSize + 255) / 256, 256>>>(d_data, blockSize, p);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sort data locally in each block using CUB
    sortWithCub(d_data, blockSize, p);

    // Define splitters (example splitters, should be computed based on your logic)
    int h_splitters[numSplitters] = {25, 50, 75};
    int* d_splitters;
    CUDA_CHECK(cudaMalloc(&d_splitters, numSplitters * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_splitters, h_splitters, numSplitters * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for output and prefix sum
    int* d_out;
    int* d_prefixSum;
    CUDA_CHECK(cudaMalloc(&d_out, dataSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefixSum, dataSize * sizeof(int)));

    // Compute prefix sum
    computePrefixSum<<<(dataSize + 255) / 256, 256>>>(d_data, d_splitters, d_prefixSum, blockSize, p, numSplitters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use cub for prefix sum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_prefixSum, d_prefixSum, dataSize));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_prefixSum, d_prefixSum, dataSize));
    CUDA_CHECK(cudaFree(d_temp_storage));

    // Distribute data according to splitters
    distributeData<<<(dataSize + 255) / 256, 256>>>(d_data, d_splitters, d_out, d_prefixSum, blockSize, p, numSplitters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_out, dataSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Print sorted data
    for (int i = 0; i < dataSize; i++) {
        std::cout << h_data[i] << " ";
        if ((i + 1) % blockSize == 0) {
            std::cout << std::endl;
        }
    }

    // Free memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_splitters));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_prefixSum));

    return 0;
}

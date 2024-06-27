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

// int main() {
//     // Example setup for large dataset
//     const int numData = 1000000; // 1 million data points
//     const int numSplitters = 1000; // 1000 splitters

//     // Generating example data and splitters
//     std::vector<int> h_data(numData);
//     std::vector<int> h_splitters(numSplitters);

//     // Fill data with sorted values for simplicity
//     for (int i = 0; i < numData; ++i) {
//         h_data[i] = rand()%numData;
//     }

//     // Fill splitters with sorted values
//     for (int i = 0; i < numSplitters; ++i) {
//         h_splitters[i] = (i + 1) * (numData / numSplitters);
//     }

//     // Device memory pointers
//     int *d_data, *d_splitters, *d_output;

//     // Allocate device memory
//     checkCudaError(cudaMalloc(&d_data, numData * sizeof(int)), "Failed to allocate device memory for data");
//     checkCudaError(cudaMalloc(&d_splitters, numSplitters * sizeof(int)), "Failed to allocate device memory for splitters");
//     checkCudaError(cudaMalloc(&d_output, numData * sizeof(int)), "Failed to allocate device memory for output");

//     // Copy data to device
//     checkCudaError(cudaMemcpy(d_data, h_data.data(), numData * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy data to device");
//     checkCudaError(cudaMemcpy(d_splitters, h_splitters.data(), numSplitters * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy splitters to device");

//     // Kernel launch parameters
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (numData + threadsPerBlock - 1) / threadsPerBlock;

//     // Launch kernel
//     findSplitsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, d_splitters, numData, numSplitters);
//     checkCudaError(cudaGetLastError(), "Kernel launch failed");

//     // Copy result back to host
//     std::vector<int> h_output(numData);
//     checkCudaError(cudaMemcpy(h_output.data(), d_output, numData * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy output to host");

//     // Optionally print out first few results
//     for (int i = 0; i < numData; ++i) {
//         std::cout << "Data: " << h_data[i] << " -> Partition: " << h_output[i] << std::endl;
//     }

//     // Free device memory
//     checkCudaError(cudaFree(d_data), "Failed to free device memory for data");
//     checkCudaError(cudaFree(d_splitters), "Failed to free device memory for splitters");
//     checkCudaError(cudaFree(d_output), "Failed to free device memory for output");

//     return 0;
// }


#include <cuda_runtime.h>
#include <iostream>

const int N = 100; // Total number of elements
const int P = 4;    // Number of sorted blocks

// Function to combine sorted blocks into a globally sorted array
__global__ void merge_sorted_blocks(int* d_data, int* d_output, int* d_splitters, int num_splitters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = N / P;

    // Determine the range for this block
    int start = tid * block_size;
    int end = (tid + 1) * block_size;
    if (tid == P - 1) {
        end = N;
    }

    // Perform merging using binary search on splitters
    int idx = tid * block_size;
    for (int i = start; i < end; ++i) {
        int value = d_data[i];
        // Binary search to find the correct position in d_splitters
        int low = 0, high = num_splitters;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (value > d_splitters[mid]) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        int target_block = low;

        // Calculate the position in the output array
        int output_pos = idx + target_block;
        d_output[output_pos] = value;
        idx++;
    }
}

int main() {
    // Generate example data
    int h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 1000; // Random values (0-999)
    }

    std::cout << "Data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Sort blocks (in this example, assume each block is already sorted)
    // Here you would normally perform sorting on each block using a sorting algorithm like quicksort or mergesort

    // Allocate memory on the device
    int* d_data, *d_output, *d_splitters;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    cudaMalloc((void**)&d_splitters, (P - 1) * sizeof(int));

    // Copy data and splitters to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Assume splitters are known and provided (p-1 values)
    int h_splitters[P - 1] = {250, 500, 750}; // Example splitters
    cudaMemcpy(d_splitters, h_splitters, (P - 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch merge kernel
    int block_size = 256; // Adjust as needed
    merge_sorted_blocks<<<(P + block_size - 1) / block_size, block_size>>>(d_data, d_output, d_splitters, P - 1);
    cudaDeviceSynchronize();

    // Copy sorted data back to host
    int h_output[N];
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted data for verification
    std::cout << "Sorted Data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_splitters);

    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 3

__global__ void mergeSortedBlocks(int *input, int *output, int *split_points, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_blocks) {
        // Calculate the starting indices for the current block and next block
        int start = (idx == 0) ? 0 : split_points[idx - 1];
        int end = split_points[idx];
        int mid = (start + end) / 2;

        // Merge the two sorted halves [start, mid] and [mid+1, end] into output[start, end]
        int i = start;
        int j = mid + 1;
        int k = start;

        while (i <= mid && j <= end) {
            if (input[i] <= input[j]) {
                output[k++] = input[i++];
            } else {
                output[k++] = input[j++];
            }
        }

        // Copy the remaining elements from the first half, if any
        while (i <= mid) {
            output[k++] = input[i++];
        }

        // Copy the remaining elements from the second half, if any
        while (j <= end) {
            output[k++] = input[j++];
        }
    }
}

int main() {
    // Example input data (blockwise sorted)
    int num_blocks = 3;
    int block_size = BLOCK_SIZE;
    int input_size = num_blocks * block_size;
    int input[] = {1, 3, 5, 2, 4, 6, 7, 8, 9};  // Example data (blockwise sorted)
    int output[input_size];
    int split_points[] = {3, 6};  // Split points indicating the end of each sorted block

    int *d_input, *d_output, *d_split_points;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_output, input_size * sizeof(int));
    cudaMalloc((void**)&d_split_points, num_blocks * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_split_points, split_points, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((num_blocks + blockDim.x - 1) / blockDim.x);

    // Call kernel to merge sorted blocks
    mergeSortedBlocks<<<gridDim, blockDim>>>(d_input, d_output, d_split_points, num_blocks);

    // Copy result back to host
    cudaMemcpy(output, d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print merged sorted array
    printf("Merged sorted array:\n");
    for (int i = 0; i < input_size; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Free allocated memory on GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_split_points);

    return 0;
}


// #include <cuda_runtime.h>
// #include <iostream>

// const int N = 12; // Total number of elements
// const int P = 4;    // Number of sorted blocks

// // Function to combine sorted blocks into a globally sorted array
// __global__ void merge_sorted_blocks(int* d_data, int* d_output, int* d_splitters, int num_splitters) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int block_size = N / P;

//     // Determine the range for this block
//     int start = tid * block_size;
//     int end = (tid + 1) * block_size;
//     if (tid == P - 1) {
//         end = N;
//     }

//     // Perform merging using binary search on splitters
//     int idx = tid * block_size;
//     for (int i = start; i < end; ++i) {
//         int value = d_data[i];
//         // Binary search to find the correct position in d_splitters
//         int low = 0, high = num_splitters;
//         while (low < high) {
//             int mid = low + (high - low) / 2;
//             if (value > d_splitters[mid]) {
//                 low = mid + 1;
//             } else {
//                 high = mid;
//             }
//         }
//         int target_block = low;

//         // Calculate the position in the output array
//         int output_pos = idx + target_block;
//         d_output[output_pos] = value;
//         idx++;
//     }
// }

// int main() {
//     // Generate example data
//     int h_data[N];
//     for (int i = 0; i < N; ++i) {
//         h_data[i] = rand() % 1000; // Random values (0-999)
//     }

//     std::cout << "Data:" << std::endl;
//     for (int i = 0; i < N; ++i) {
//         std::cout << h_data[i] << " ";
//     }
//     std::cout << std::endl;

//     // Sort blocks (in this example, assume each block is already sorted)
//     // Here you would normally perform sorting on each block using a sorting algorithm like quicksort or mergesort

//     // Allocate memory on the device
//     int* d_data, *d_output, *d_splitters;
//     cudaMalloc((void**)&d_data, N * sizeof(int));
//     cudaMalloc((void**)&d_output, N * sizeof(int));
//     cudaMalloc((void**)&d_splitters, (P - 1) * sizeof(int));

//     // Copy data and splitters to device
//     cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

//     // Assume splitters are known and provided (p-1 values)
//     int h_splitters[P - 1] = {250, 500, 750}; // Example splitters
//     cudaMemcpy(d_splitters, h_splitters, (P - 1) * sizeof(int), cudaMemcpyHostToDevice);

//     // Launch merge kernel
//     int block_size = 256; // Adjust as needed
//     merge_sorted_blocks<<<(P + block_size - 1) / block_size, block_size>>>(d_data, d_output, d_splitters, P - 1);
//     cudaDeviceSynchronize();

//     // Copy sorted data back to host
//     int h_output[N];
//     cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print sorted data for verification
//     std::cout << "Sorted Data:" << std::endl;
//     for (int i = 0; i < N; ++i) {
//         std::cout << h_output[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free allocated memory
//     cudaFree(d_data);
//     cudaFree(d_output);
//     cudaFree(d_splitters);

//     return 0;
// }

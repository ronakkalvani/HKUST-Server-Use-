#include <cuda_runtime.h>
#include <iostream>

__global__ void countSplits(int* split_indices, int* split_counts, int num_elements, int num_splits) {
    extern __shared__ int shared_counts[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;

    // Initialize shared memory
    for (int i = tid; i < num_splits; i += block_size) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    // Count split indices in shared memory
    for (int i = tid + bid * block_size; i < num_elements; i += block_size * gridDim.x) {
        int split_idx = split_indices[i];
        atomicAdd(&shared_counts[split_idx], 1);
    }
    __syncthreads();

    // Write shared memory counts to global memory
    for (int i = tid; i < num_splits; i += block_size) {
        atomicAdd(&split_counts[i * gridDim.x + bid], shared_counts[i]);
    }
}

int main() {
    const int num_elements = 1e6;
    const int block_size = 256;
    const int num_splits = num_elements/block_size;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    int* h_split_indices = new int[num_elements];
    int* h_split_counts = new int[num_splits * num_blocks]();
    for (int i = 0; i < num_elements; ++i) {
        h_split_indices[i] = i % num_splits;
    }

    int* d_split_indices;
    int* d_split_counts;
    cudaMalloc(&d_split_indices, num_elements * sizeof(int));
    cudaMalloc(&d_split_counts, num_splits * num_blocks * sizeof(int));
    cudaMemcpy(d_split_indices, h_split_indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_split_counts, 0, num_splits * num_blocks * sizeof(int));

    countSplits<<<num_blocks, block_size, num_splits * sizeof(int)>>>(d_split_indices, d_split_counts, num_elements, num_splits);

    cudaMemcpy(h_split_counts, d_split_counts, num_splits * num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int s = 0; s < num_splits; ++s) {
        std::cout << "Split " << s << " counts:\n";
        for (int b = 0; b < num_blocks; ++b) {
            std::cout << "  Block " << b << ": " << h_split_counts[s * num_blocks + b] << "\n";
        }
    }

    delete[] h_split_indices;
    delete[] h_split_counts;
    cudaFree(d_split_indices);
    cudaFree(d_split_counts);

    return 0;
}


// #include <cuda_runtime.h>
// #include <iostream>

// __global__ void countSplits(int* split_indices, int* split_counts, int num_elements, int num_splits) {
//     extern __shared__ int shared_counts[];

//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int block_size = blockDim.x;

//     // Initialize shared memory
//     for (int i = tid; i < num_splits; i += block_size) {
//         shared_counts[i] = 0;
//     }
//     __syncthreads();

//     // Count split indices in shared memory
//     for (int i = tid + bid * block_size; i < num_elements; i += block_size * gridDim.x) {
//         int split_idx = split_indices[i];
//         atomicAdd(&shared_counts[split_idx], 1);
//     }
//     __syncthreads();

//     // Write shared memory counts to global memory
//     for (int i = tid; i < num_splits; i += block_size) {
//         atomicAdd(&split_counts[i + bid * num_splits], shared_counts[i]);
//     }
// }

// int main() {
//     const int num_elements = 1024;
//     const int num_splits = 10;
//     const int block_size = 256;
//     const int num_blocks = (num_elements + block_size - 1) / block_size;

//     int* h_split_indices = new int[num_elements];
//     int* h_split_counts = new int[num_blocks * num_splits]();
//     for (int i = 0; i < num_elements; ++i) {
//         h_split_indices[i] = i % num_splits;
//     }

//     int* d_split_indices;
//     int* d_split_counts;
//     cudaMalloc(&d_split_indices, num_elements * sizeof(int));
//     cudaMalloc(&d_split_counts, num_blocks * num_splits * sizeof(int));
//     cudaMemcpy(d_split_indices, h_split_indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemset(d_split_counts, 0, num_blocks * num_splits * sizeof(int));

//     countSplits<<<num_blocks, block_size, num_splits * sizeof(int)>>>(d_split_indices, d_split_counts, num_elements, num_splits);

//     cudaMemcpy(h_split_counts, d_split_counts, num_blocks * num_splits * sizeof(int), cudaMemcpyDeviceToHost);

//     for (int b = 0; b < num_blocks; ++b) {
//         std::cout << "Block " << b << " counts:\n";
//         for (int s = 0; s < num_splits; ++s) {
//             std::cout << "  Split " << s << ": " << h_split_counts[b * num_splits + s] << "\n";
//         }
//     }

//     delete[] h_split_indices;
//     delete[] h_split_counts;
//     cudaFree(d_split_indices);
//     cudaFree(d_split_counts);

//     return 0;
// }

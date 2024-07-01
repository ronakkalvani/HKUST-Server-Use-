#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to copy data from one array to another
__global__ void Assign(int * NewBlockId,int* segment_sum,int* d_split_counts_prefixsum,int* d_src, int* d_dst, int size,int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        int ind_in_cnt = blockIdx.x  + p * NewBlockId[tid];
        int final_ind = segment_sum[tid]+d_split_counts_prefixsum[ind_in_cnt];
        d_dst[final_ind] = d_src[tid];
    }

}

__global__ void partitions(int* d_split_counts,int* d_partition_counts,int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < p*p) {
        if (tid/p+1<p) atomicAdd(&d_partition_counts[tid/p+1], d_split_counts[tid]);
    }

}

// int main() {
//     int size = 10;
//     int bytes = size * sizeof(int);

//     // Host arrays
//     int h_src[size], h_dst[size];

//     // Initialize host array with some data
//     for (int i = 0; i < size; i++) {
//         h_src[i] = i;
//     }

//     // Device arrays
//     int *d_src, *d_dst;
//     cudaMalloc(&d_src, bytes);
//     cudaMalloc(&d_dst, bytes);

//     // Copy data from host to device (h_src to d_src)
//     cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);

//     // Define block and grid sizes
//     int blockSize = 256;
//     int gridSize = (size + blockSize - 1) / blockSize;

//     // Launch kernel to copy data from d_src to d_dst
//     copyKernel<<<gridSize, blockSize>>>(d_src, d_dst, size);

//     // Copy result from device to host (d_dst to h_dst)
//     cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

//     // Print result
//     std::cout << "Copied data:" << std::endl;
//     for (int i = 0; i < size; i++) {
//         std::cout << h_dst[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free device memory
//     cudaFree(d_src);
//     cudaFree(d_dst);

//     return 0;
// }

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define BLOCK_THREADS 8
#define ITEMS_PER_THREAD 1

// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out, int num_elements)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    // Obtain this block's segment of consecutive keys (blocked across threads)
    int thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int valid_items = num_elements - block_offset > BLOCK_THREADS * ITEMS_PER_THREAD ? BLOCK_THREADS * ITEMS_PER_THREAD : num_elements - block_offset;

    // Initialize thread_keys with a known value for safer debugging
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_keys[i] = (block_offset + threadIdx.x * ITEMS_PER_THREAD + i) < num_elements ? d_in[block_offset + threadIdx.x * ITEMS_PER_THREAD + i] : INT_MAX;
    }

    // Load data
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys, valid_items);

    __syncthreads(); // Barrier for smem reuse

    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

    __syncthreads(); // Barrier for smem reuse

    // Store the sorted segment
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys, valid_items);
}

// int main() {
//     // Initialize host data
//     // std::vector<int> h_data = {34, 78, 12, 56, 89, 21, 90, 34, 23, 45, 67, 11, 23, 56, 78, 99, 123, 45, 67, 89, 23, 45, 67, 34, 78};
//     std::vector<int> h_data(786);
//     for (int i=0;i<h_data.size();i++) {
//         h_data[i]=rand()%37;
//     }
//     int n = h_data.size();

//     // Allocate device memory
//     int* d_data;
//     cudaMalloc(&d_data, n * sizeof(int));
//     int* d_sorted_data;
//     cudaMalloc(&d_sorted_data, n * sizeof(int));

//     // Copy data to device
//     cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

//     int numBlocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

//     // Launch kernel to sort blocks
//     BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_data, d_sorted_data, n);

//     // Copy sorted data back to host
//     cudaMemcpy(h_data.data(), d_sorted_data, n * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print sorted blocks
//     for (int i = 0; i < h_data.size(); i++) {
//         std::cout << h_data[i] << " ";
//     }
//     std::cout << std::endl;

//     // Free device memory
//     cudaFree(d_data);
//     cudaFree(d_sorted_data);

//     return 0;
// }


#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define BLOCK_THREADS 32
#define ITEMS_PER_THREAD 4

// Block-sorting CUDA kernel
__global__ void BlockSortKernel2(int *d_in, int *d_out, int *d_block_starts, int num_blocks, int num_elements)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE> BlockLoadT;
    typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> BlockStoreT;
    typedef cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    // Determine the range of elements this block should sort
    int block_start = d_block_starts[blockIdx.x];
    int block_end = (blockIdx.x == num_blocks - 1) ? num_elements : d_block_starts[blockIdx.x + 1];
    int num_items = block_end - block_start;

    // Load data
    int thread_keys[ITEMS_PER_THREAD];
    BlockLoadT(temp_storage.load).Load(d_in + block_start, thread_keys, num_items);

    __syncthreads(); // Barrier for smem reuse

    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

    __syncthreads(); // Barrier for smem reuse

    // Store the sorted segment
    BlockStoreT(temp_storage.store).Store(d_out + block_start, thread_keys, num_items);
}

// int main() {
//     // Initialize host data
//     std::vector<int> h_data(786);
//     for (int i = 0; i < h_data.size(); i++) {
//         h_data[i] = rand() % 37;
//     }
//     int n = h_data.size();

//     // Define block starting indices
//     std::vector<int> h_block_starts = {0, 128, 256, 384, 512, 640}; // Example block starts
//     int num_blocks = h_block_starts.size();

//     // Allocate device memory
//     int *d_data, *d_sorted_data, *d_block_starts;
//     cudaMalloc(&d_data, n * sizeof(int));
//     cudaMalloc(&d_sorted_data, n * sizeof(int));
//     cudaMalloc(&d_block_starts, num_blocks * sizeof(int));

//     // Copy data to device
//     cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_block_starts, h_block_starts.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);

//     // Launch kernel to sort blocks
//     BlockSortKernel2<<<num_blocks, BLOCK_THREADS>>>(d_data, d_sorted_data, d_block_starts, num_blocks, n);

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
//     cudaFree(d_block_starts);

//     return 0;
// }
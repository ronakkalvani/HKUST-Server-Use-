#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <algorithm>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Kernel to sort each block
__global__ void sortBlocks(int* data, int* block_starts, int num_blocks, int total_elements) {
    // Calculate the current block's size
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int start_index = block_starts[block_id];
    int end_index = (block_id == num_blocks - 1) ? total_elements : block_starts[block_id + 1];
    int block_size = end_index - start_index;

    if (block_size <= 0) return;

    // Allocate shared memory for the block
    extern __shared__ int shared_data[];

    // Copy block data to shared memory
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        shared_data[i] = data[start_index + i];
    }
    __syncthreads();

    // Use CUB to sort the shared memory
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, shared_data, shared_data, block_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, shared_data, shared_data, block_size);
    cudaFree(d_temp_storage);
    __syncthreads();

    // Copy sorted data back to the global memory
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        data[start_index + i] = shared_data[i];
    }
}

int main() {
    // Example data
    std::vector<int> h_data = {3, 2, 1, 7, 6, 5, 9, 8};
    std::vector<int> h_block_starts = {0, 3, 6}; // Each block starts at these indices
    int num_blocks = h_block_starts.size();
    int total_elements = h_data.size();

    // Allocate device memory
    int* d_data;
    int* d_block_starts;
    CUDA_CHECK(cudaMalloc(&d_data, h_data.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_starts, h_block_starts.size() * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_starts, h_block_starts.data(), h_block_starts.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the sorting kernel
    int threads_per_block = 256;
    sortBlocks<<<num_blocks, threads_per_block, threads_per_block * sizeof(int)>>>(d_data, d_block_starts, num_blocks, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy sorted data back to host
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, h_data.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Print sorted data
    std::cout << "Sorted data: ";
    for (int val : h_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_block_starts));

    return 0;
}


// #include <cuda_runtime.h>
// #include <cub/cub.cuh>
// #include <iostream>
// #include <vector>

// #define BLOCK_THREADS 32
// #define ITEMS_PER_THREAD 1

// // Block-sorting CUDA kernel
// __global__ void BlockSortKernel2(int *d_in, int *d_out, int *d_block_starts, int num_blocks, int num_elements)
// {
//     // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
//     typedef cub::BlockLoad<int, 2*BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE> BlockLoadT;
//     typedef cub::BlockStore<int, 2*BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> BlockStoreT;
//     typedef cub::BlockRadixSort<int, 2*BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

//     // Allocate type-safe, repurposable shared memory for collectives
//     __shared__ union {
//         typename BlockLoadT::TempStorage load;
//         typename BlockStoreT::TempStorage store;
//         typename BlockRadixSortT::TempStorage sort;
//     } temp_storage;

//     // Determine the range of elements this block should sort
//     int block_start = d_block_starts[blockIdx.x];
//     int block_end = (blockIdx.x == num_blocks - 1) ? num_elements : d_block_starts[blockIdx.x + 1];
//     int num_items = block_end - block_start;

//     // Ensure we do not access out of bounds memory
//     if (block_start >= num_elements) return;

//     // Load data
//     int thread_keys[ITEMS_PER_THREAD];
//     int valid_items = min(num_items - threadIdx.x * ITEMS_PER_THREAD, ITEMS_PER_THREAD);
//     BlockLoadT(temp_storage.load).Load(d_in + block_start, thread_keys, num_items);

//     __syncthreads(); // Barrier for smem reuse

//     // Collectively sort the keys
//     BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

//     __syncthreads(); // Barrier for smem reuse

//     // Store the sorted segment
//     BlockStoreT(temp_storage.store).Store(d_out + block_start, thread_keys, num_items);
    
// }


// int main() {
//     // Initialize host data
//     std::vector<int> h_data(1024);
//     for (int i = 0; i < h_data.size(); i++) {
//         h_data[i] = rand() % 127;
//         std::cout<<h_data[i]<<" ";
//     }
//     std::cout<<"\n";
//     int n = h_data.size();

//     // Define block starting indices
//     // std::vector<int> h_block_starts = {0, 1000, 2000, 3000, 4000, 4500}; // Example block starts
//     std::vector<int> h_block_starts(n/BLOCK_THREADS);
//     for(int i=0;i<n/BLOCK_THREADS;i++) {
//         if (i%2) h_block_starts[i] = (i)*(BLOCK_THREADS)+7;
//         else h_block_starts[i] = (i)*(BLOCK_THREADS);
//         std::cout<<h_block_starts[i]<<" ";
//     }
//     std::cout<<"\n";
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
//     BlockSortKernel2<<<num_blocks, BLOCK_THREADS*2>>>(d_data, d_sorted_data, d_block_starts, num_blocks, n);

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

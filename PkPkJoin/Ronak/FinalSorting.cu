#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define BLOCK_THREADS 32
#define ITEMS_PER_THREAD 1

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

    // Ensure we do not access out of bounds memory
    if (block_start >= num_elements) return;

    // Load data
    int thread_keys[ITEMS_PER_THREAD];
    int items_to_load = min(num_items - threadIdx.x * ITEMS_PER_THREAD, ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(d_in + block_start + threadIdx.x * ITEMS_PER_THREAD, thread_keys, items_to_load);

    __syncthreads(); // Barrier for smem reuse

    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

    __syncthreads(); // Barrier for smem reuse

    // Store the sorted segment
    BlockStoreT(temp_storage.store).Store(d_out + block_start + threadIdx.x * ITEMS_PER_THREAD, thread_keys, items_to_load);
}

int main() {
    // Initialize host data
    std::vector<int> h_data(1024);
    for (int i = 0; i < h_data.size(); i++) {
        h_data[i] = rand() % 127;
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";
    int n = h_data.size();

    // Define block starting indices
    std::vector<int> h_block_starts(n / BLOCK_THREADS);
    for (int i = 0; i < n / BLOCK_THREADS; i++) {
        if (i % 2) h_block_starts[i] = (i) * (BLOCK_THREADS) + 7;
        else h_block_starts[i] = (i) * (BLOCK_THREADS);
        std::cout << h_block_starts[i] << " ";
    }
    std::cout << "\n";
    int num_blocks = h_block_starts.size();

    // Allocate device memory
    int *d_data, *d_sorted_data, *d_block_starts;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_sorted_data, n * sizeof(int));
    cudaMalloc(&d_block_starts, num_blocks * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_starts, h_block_starts.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to sort blocks
    BlockSortKernel2<<<num_blocks, BLOCK_THREADS>>>(d_data, d_sorted_data, d_block_starts, num_blocks, n);

    // Copy sorted data back to host
    cudaMemcpy(h_data.data(), d_sorted_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted blocks
    for (int i = 0; i < h_data.size(); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_block_starts);

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

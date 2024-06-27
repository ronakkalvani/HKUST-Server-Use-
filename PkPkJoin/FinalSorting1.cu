#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <climits>

#define BLOCK_THREAD 1024
#define ITEMS_PER_THREAD 1

__global__ void printArray0(int* arr, int size) {
    for (int i = 0; i < size; ++i) {
        if (arr[i] == 0)
            printf("%d ", i);
    }
    printf("\n");
}

__global__ void BlockSortKernel2(int *d_in, int *d_out, int *block_indices, int num_blocks, int num_elements) {
    typedef cub::BlockLoad<int, BLOCK_THREAD, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<int, BLOCK_THREAD, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<int, BLOCK_THREAD, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    int thread_keys[ITEMS_PER_THREAD];
    int block_idx = blockIdx.x;
    int block_start = block_indices[block_idx];
    int block_end = (block_idx + 1 < num_blocks) ? block_indices[block_idx + 1] : num_elements;
    int block_size = block_end - block_start;
    int valid_items = min(block_size, BLOCK_THREAD * ITEMS_PER_THREAD);

    // Ensure we do not access out of bounds
    if (threadIdx.x < valid_items) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int idx = block_start + threadIdx.x * ITEMS_PER_THREAD + i;
            thread_keys[i] = (idx < block_end) ? d_in[idx] : INT_MAX;
        }

        // Load data
        BlockLoadT(temp_storage.load).Load(d_in + block_start, thread_keys, valid_items);

        __syncthreads();

        // Collectively sort the keys
        BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

        __syncthreads();

        // Store the sorted segment
        BlockStoreT(temp_storage.store).Store(d_out + block_start, thread_keys, valid_items);
    }
}

void checkCudaError(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line << " in " << func << "\n";
        exit(result);
    }
}

#define CHECK_CUDA(call) checkCudaError((call), #call, __FILE__, __LINE__)

int main() {
    std::vector<int> h_data(1e6);
    for (int i = 0; i < h_data.size(); i++) {
        h_data[i] = rand() % 453483;
    }
    int n = h_data.size();

    int s = BLOCK_THREAD / 2;
    std::vector<int> h_block_indices(n / s);
    for (int i = 0; i < n / s; i++) {
        if (i % 2) h_block_indices[i] = (i) * (s) - 256;
        else h_block_indices[i] = (i) * (s) + 256;
    }
    int num_blocks = h_block_indices.size();

    int* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(int)));
    int* d_sorted_data;
    CHECK_CUDA(cudaMalloc(&d_sorted_data, n * sizeof(int)));
    int* d_block_indices;
    CHECK_CUDA(cudaMalloc(&d_block_indices, h_block_indices.size() * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_block_indices, h_block_indices.data(), h_block_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    BlockSortKernel2<<<num_blocks, BLOCK_THREAD>>>(d_data, d_sorted_data, d_block_indices, num_blocks, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_data.data(), d_sorted_data, n * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_data.size(); i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    printArray0<<<1, 1>>>(d_sorted_data, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_sorted_data));
    CHECK_CUDA(cudaFree(d_block_indices));

    return 0;
}

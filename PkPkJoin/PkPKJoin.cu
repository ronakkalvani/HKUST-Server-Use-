#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/DistributionAfterSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FindSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/JoinAfterSort.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/SortDataBlockWise.cu"

#define BLOCK_THREADS 16
#define ITEMS_PER_THREAD 1

int main() {
    std::vector<int> h_data(100);
    for (int i=0;i<h_data.size();i++) {
        h_data[i]=rand()%7;
    }
    int n = h_data.size();

    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    int* d_sorted_data;
    cudaMalloc(&d_sorted_data, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

    // Launch kernel to sort blocks
    BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_data, d_sorted_data, n);

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

    return 0;
}
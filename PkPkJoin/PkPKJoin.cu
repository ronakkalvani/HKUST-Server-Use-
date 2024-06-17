#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/SortDataBlockWise.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FindSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/DistributionAfterSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/JoinAfterSort.cu"


int main() {
    std::vector<int> h_data(100);
    for (int i=0;i<h_data.size();i++) {
        h_data[i]=rand() % 37;
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

    int p = numBlocks;
    int sample_size = n / p;
    int *d_samples, *d_splitters;
    cudaMalloc(&d_samples, sample_size * sizeof(int));
    cudaMalloc(&d_splitters, (p - 1) * sizeof(int));

    FindSplit(d_sorted_data,d_samples, d_splitters, n, numBlocks, sample_size);

    // Select splitters
    int* h_samples = new int[sample_size];
    CUDA_CHECK(cudaMemcpy(h_samples, d_samples, sample_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    int h_splitters[p - 1];
    for (int i = 0; i < p - 1; ++i) {
        h_splitters[i] = h_samples[(i + 1) * sample_size / p];
    }
    
    delete[] h_samples;
    
    // Print splitters
    for (int i = 0; i < p - 1; ++i) {
        std::cout << "Splitter " << i << ": " << h_splitters[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_samples);
    cudaFree(d_splitters);

    return 0;
}
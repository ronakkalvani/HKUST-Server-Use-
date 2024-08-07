#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cmath>
#include <unordered_set>
#include <random>

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 1
#define BLOCK_THREAD 4 * BLOCK_THREADS

#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/SortDataBlockWise.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/FindSplits.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/AssigningBlocks.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/SegmentedPrefixSum.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/CountSplitValues.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/PrefixSum.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/GlobalArrayAssignment.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/DistributionAfterSplits.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/FinalSorting.cu"
#include "/csproject/yike/intern/ronak/MPC-PKPKJOIN/PkPkJoin/JoinAfterSort.cu"

const int n1 = 1e6;
const int n2 = 1e6;
const int mx = 1e8;
const int n = n1 + n2;
std::vector<int> keys1(n1);
std::vector<int> keys2(n2);
std::vector<int> hmap1(mx, 0);
std::vector<int> hmap2(mx, 0);
int h_results[3 * n];

void PkPkJoin(const std::vector<int>& keys1, const std::vector<int>& keys2, const std::vector<int>& hmap1, const std::vector<int>& hmap2, int* h_results, int n1, int n2, int n) {
    // Allocate host memory
    std::vector<int> h_data(n);
    std::copy(keys1.begin(), keys1.end(), h_data.begin());
    std::copy(keys2.begin(), keys2.end(), h_data.begin() + n1);

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

    int blockSize = BLOCK_THREADS;
    int p = numBlocks;
    int sample_size = p * int(log2(p));
    int* d_samples, *d_splitters;
    curandState* d_state;
    cudaMalloc(&d_samples, sample_size * sizeof(int));
    cudaMalloc(&d_splitters, (p - 1) * sizeof(int));
    cudaMalloc(&d_state, sample_size * sizeof(curandState));

    initCurand<<<numBlocks, blockSize>>>(d_state, time(NULL), sample_size);

    FindSplit(d_sorted_data, d_samples, d_splitters, n, p, sample_size, d_state);

    Splitterss<<<numBlocks, blockSize>>>(d_splitters, d_samples, sample_size, p);

    int* d_Blocks;
    cudaMalloc(&d_Blocks, n * sizeof(int));

    findSplitsKernel<<<numBlocks, blockSize>>>(d_sorted_data, d_Blocks, d_splitters, n, p-1);

    int* d_segment_sum;
    cudaMalloc(&d_segment_sum, n * sizeof(int));

    segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_Blocks, d_segment_sum, n, blockSize);

    int* d_split_counts;
    cudaMalloc(&d_split_counts, p * p * sizeof(int));
    cudaMemset(d_split_counts, 0, p * p * sizeof(int));

    countSplits<<<numBlocks, blockSize, p * sizeof(int)>>>(d_Blocks, d_split_counts, n, p);

    int* d_split_counts_prefixsum;
    cudaMalloc(&d_split_counts_prefixsum, p * p * sizeof(int));

    exclusive_prefix_sum(d_split_counts, d_split_counts_prefixsum, p * p);

    int* d_output;
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemset(d_output, 0, n * sizeof(int));

    Assign<<<numBlocks, blockSize>>>(d_Blocks, d_segment_sum, d_split_counts_prefixsum, d_sorted_data, d_output, n, p);

    int* d_partition_starts;
    cudaMalloc(&d_partition_starts, p * sizeof(int));

    int* d_final_array;
    cudaMalloc(&d_final_array, n * sizeof(int));

    partitions<<<numBlocks, BLOCK_THREADS>>>(d_split_counts_prefixsum, d_partition_starts, p);

    BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts, p, n);

    int* d_results;
    cudaMalloc(&d_results, 3 * n * sizeof(int));
    cudaMemset(d_results, -1, 3 * n * sizeof(int));

    int* d_hmap1;
    cudaMalloc(&d_hmap1, mx * sizeof(int));
    cudaMemcpy(d_hmap1, hmap1.data(), mx * sizeof(int), cudaMemcpyHostToDevice);

    int* d_hmap2;
    cudaMalloc(&d_hmap2, mx * sizeof(int));
    cudaMemcpy(d_hmap2, hmap2.data(), mx * sizeof(int), cudaMemcpyHostToDevice);

    JoinKernel<<<numBlocks, BLOCK_THREADS>>>(d_final_array, d_results, n, d_hmap1, d_hmap2);

    // cudaMemcpy(h_results, d_results, 3 * n * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 3 * n; i += 3) {
    //     if (h_results[i] != -1)
    //         std::cout << "Key: " << h_results[i] << " Values: " << h_results[i + 1] << " " << h_results[i + 2] << std::endl;
    // }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_samples);
    cudaFree(d_splitters);
    cudaFree(d_output);
    cudaFree(d_partition_starts);
    cudaFree(d_final_array);
    cudaFree(d_results);
    cudaFree(d_hmap1);
    cudaFree(d_hmap2);
}

void generateUniqueKeys(std::vector<int>& keys, int mx) {
    std::unordered_set<int> unique_keys;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, mx - 1);

    for (int i = 0; i < keys.size(); i++) {
        int key;
        do {
            key = dist(rng);
        } while (unique_keys.find(key) != unique_keys.end());
        unique_keys.insert(key);
        keys[i] = key;
    }
}

int main() {
    generateUniqueKeys(keys1, mx);
    generateUniqueKeys(keys2, mx);

    for (int i = 0; i < n1; i++) {
        hmap1[keys1[i]] = rand() % 355;
    }
    for (int i = 0; i < n2; i++) {
        hmap2[keys2[i]] = 500 + (rand() % 326);
    }

    // Process data and perform join
    PkPkJoin(keys1, keys2, hmap1, hmap2, h_results, n1, n2, n);

    return 0;
}


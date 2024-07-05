
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cmath>

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 1
#define BLOCK_THREAD 4*BLOCK_THREADS

#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/SortDataBlockWise.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/FindSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/AssigningBlocks.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/SegmentedPrefixSum.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/CountSplitValues.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/PrefixSum.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/GlobalArrayAssignment.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/DistributionAfterSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/FinalSorting.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/JoinAfterSort.cu"

const int n1 = 1e6;
const int n2 = 1e6;
const int mx = 1e7;
std::vector<int> keys1(n1);
std::vector<int> keys2(n2);
std::vector<int> hmap1(mx, 0);
std::vector<int> hmap2(mx, 0);
const int n = n1 + n2;
std::vector<int> h_data(n);

int h_results[3 * n];

int main() {

    for (int i = 0; i < n1; i++) {
        keys1[i] = 2*(n1-i+1);
    }
    for (int i = 0; i < n2; i++) {
        keys2[i] = 3*(n2-i+1);
    }

    for (int i = 0; i < n1; i++) {
        hmap1[keys1[i]] = rand() % 355;
    }
    for (int i = 0; i < n2; i++) {
        hmap2[keys2[i]] = 500 + (rand() % 326);
    }

    for (int i = 0; i < n; i++) {
        if (i < n1) h_data[i] = keys1[i];
        else h_data[i] = keys2[i - n1];
    }

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

    // printArray<<<1, 1>>>(d_sorted_data, n);

    int blockSize = BLOCK_THREADS;
    int p = numBlocks;
    int sample_size = p * int(log2(p));
    int *d_samples, *d_splitters;
    curandState* d_state;
    cudaMalloc(&d_samples, sample_size * sizeof(int));
    cudaMalloc(&d_splitters, (p - 1) * sizeof(int));
    cudaMalloc(&d_state, sample_size * sizeof(curandState));

    initCurand<<<numBlocks, blockSize>>>(d_state, time(NULL), sample_size);

    FindSplit(d_sorted_data, d_samples, d_splitters, n, p, sample_size, d_state);

    Splitterss<<<numBlocks, blockSize>>>(d_splitters, d_samples, sample_size, p);

    // printArray<<<1, 1>>>(d_splitters, p-1);

    int  *d_Blocks;
    cudaMalloc(&d_Blocks, n * sizeof(int));

    findSplitsKernel<<<numBlocks, blockSize>>>(d_sorted_data, d_Blocks, d_splitters, n, p-1);

    // printArray<<<1, 1>>>(d_Blocks, n);

    int  *d_segment_sum;
    cudaMalloc(&d_segment_sum, n * sizeof(int));

    segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_Blocks, d_segment_sum, n, blockSize);
    
    // printArray<<<1, 1>>>(d_segment_sum, 1000);

    int  *d_split_counts;
    cudaMalloc(&d_split_counts, p * p * sizeof(int));
    cudaMemset(d_split_counts, 0, p * p * sizeof(int));

    countSplits<<<numBlocks, blockSize, p * sizeof(int)>>>(d_Blocks, d_split_counts, n, p);

    int  *d_split_counts_prefixsum;
    cudaMalloc(&d_split_counts_prefixsum, p * p * sizeof(int));

    exclusive_prefix_sum(d_split_counts, d_split_counts_prefixsum, p*p);

    // printArray<<<1, 1>>>(d_split_counts_prefixsum, 500);

    int *d_output;
    cudaMalloc(&d_output, n* sizeof(int));
    cudaMemset(d_output, 0, n * sizeof(int));

    Assign<<<numBlocks, blockSize>>>(d_Blocks,d_segment_sum,d_split_counts_prefixsum,d_sorted_data,d_output,n,p);

    int *d_partition_starts;
    cudaMalloc(&d_partition_starts, p * sizeof(int));

    int* d_final_array;
    cudaMalloc(&d_final_array, n * sizeof(int));

    partitions<<<numBlocks, BLOCK_THREADS>>>(d_split_counts_prefixsum,d_partition_starts,p);

    // printArray0<<<1, 1>>>(d_output, n);

    // printArray<<<1, 1>>>(d_partition_starts, p);
    
    BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts, p, n);

    // printArray0<<<1, 1>>>(d_final_array, n);

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

    // printArray<<<1, 1>>>(d_results, 3*n);

    cudaMemcpy(h_results, d_results, 3 * n * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 101; i += 3) {
    for (int i = 0; i < 3 * n; i += 3) {
        if (h_results[i] != -1)
            std::cout << "Key: " << h_results[i] << " Values: " << h_results[i + 1] << " " << h_results[i + 2] << std::endl;
    }

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

    return 0;
}



// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <curand_kernel.h>
// #include <cub/cub.cuh>
// #include <cmath>

// #define BLOCK_THREADS 256
// #define ITEMS_PER_THREAD 1
// #define BLOCK_THREAD 4*BLOCK_THREADS

// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/SortDataBlockWise.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/FindSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/AssigningBlocks.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/SegmentedPrefixSum.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/CountSplitValues.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/PrefixSum.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/GlobalArrayAssignment.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/DistributionAfterSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/FinalSorting.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/JoinAfterSort.cu"

// const int n1 = 1e6;
// const int n2 = 1e6;
// const int mx = 1e7;
// std::vector<int> keys1(n1);
// std::vector<int> keys2(n2);
// std::vector<int> hmap1(mx, 0);
// std::vector<int> hmap2(mx, 0);
// const int n = n1 + n2;
// std::vector<int> h_data(n);

// int h_results[3 * n];

// int main() {

//     for (int i = 0; i < n1; i++) {
//         // keys1[i] = (53*keys1[i-1]-5)%(n1+10);
//         keys1[i] = 2*(n1-i+1);
//     }
//     for (int i = 0; i < n2; i++) {
//         keys2[i] = 3*(n2-i+1);
//         // keys2[i] = (71*keys1[i-1]-3)%(n2+10);
//     }

//     for (int i = 0; i < n1; i++) {
//         hmap1[keys1[i]] = rand() % 355;
//     }
//     for (int i = 0; i < n2; i++) {
//         hmap2[keys2[i]] = 500 + (rand() % 326);
//     }

//     for (int i = 0; i < n; i++) {
//         if (i < n1) h_data[i] = keys1[i];
//         else h_data[i] = keys2[i - n1];
//     }

//     // Allocate device memory
//     int* d_data;
//     CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
//     int* d_sorted_data;
//     CUDA_CHECK(cudaMalloc(&d_sorted_data, n * sizeof(int)));

//     // Copy data to device
//     CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));


//     int numBlocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

//     // Launch kernel to sort blocks
//     BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_data, d_sorted_data, n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1, 1>>>(d_sorted_data, n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int blockSize = BLOCK_THREADS;
//     int p = numBlocks;
//     int sample_size = p * int(log2(p));
//     int *d_samples, *d_splitters;
//     curandState* d_state;
//     CUDA_CHECK(cudaMalloc(&d_samples, sample_size * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_splitters, (p - 1) * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_state, sample_size * sizeof(curandState)));

//     initCurand<<<numBlocks, blockSize>>>(d_state, time(NULL), sample_size);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     FindSplit(d_sorted_data, d_samples, d_splitters, n, p, sample_size, d_state);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     Splitterss<<<numBlocks, blockSize>>>(d_splitters, d_samples, sample_size, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1, 1>>>(d_splitters, p-1);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int  *d_Blocks;
//     checkCudaError(cudaMalloc(&d_Blocks, n * sizeof(int)), "Failed to allocate device memory for output");

//     findSplitsKernel<<<numBlocks, blockSize>>>(d_sorted_data, d_Blocks, d_splitters, n, p-1);

//     // printArray<<<1, 1>>>(d_Blocks, n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int  *d_segment_sum;
//     checkCudaError(cudaMalloc(&d_segment_sum, n * sizeof(int)), "Failed to allocate device memory for output");

//     segmentedPrefixSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_Blocks, d_segment_sum, n, blockSize);
    
//     // printArray<<<1, 1>>>(d_segment_sum, 1000);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int  *d_split_counts;
//     checkCudaError(cudaMalloc(&d_split_counts, p * p * sizeof(int)), "Failed to allocate device memory for output");
//     cudaMemset(d_split_counts, 0, p * p * sizeof(int));

//     countSplits<<<numBlocks, blockSize, p * sizeof(int)>>>(d_Blocks, d_split_counts, n, p);
//     // int numb = (p*p + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);
//     // countSplits<<<numb, blockSize, p * sizeof(int) >>>(d_Blocks, d_split_counts, n, p);
//     // countSplits<<<numBlocks, blockSize, p * sizeof(int)>>>(d_Blocks, d_split_counts, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     int  *d_split_counts_prefixsum;
//     checkCudaError(cudaMalloc(&d_split_counts_prefixsum, p * p * sizeof(int)), "Failed to allocate device memory for output");

//     exclusive_prefix_sum(d_split_counts, d_split_counts_prefixsum, p*p);

//     // printArray<<<1, 1>>>(d_split_counts_prefixsum, 500);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int *d_output;
//     checkCudaError(cudaMalloc(&d_output, n* sizeof(int)), "Failed to allocate device memory for output");
//     CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(int)));

//     Assign<<<numBlocks, blockSize>>>(d_Blocks,d_segment_sum,d_split_counts_prefixsum,d_sorted_data,d_output,n,p);

//     int *d_partition_starts;
//     CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));

//     int* d_final_array;
//     CUDA_CHECK(cudaMalloc(&d_final_array, n * sizeof(int)));

//     partitions<<<numBlocks, BLOCK_THREADS>>>(d_split_counts_prefixsum,d_partition_starts,p);

//     // printArray0<<<1, 1>>>(d_output, n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1, 1>>>(d_partition_starts, p);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());
    
//     BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts, p, n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray0<<<1, 1>>>(d_final_array, n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     int* d_results;
//     CUDA_CHECK(cudaMalloc(&d_results, 3 * n * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_results, -1, 3 * n * sizeof(int)));

//     int* d_hmap1;
//     CUDA_CHECK(cudaMalloc(&d_hmap1, mx * sizeof(int)));
//     CUDA_CHECK(cudaMemcpy(d_hmap1, hmap1.data(), mx * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaDeviceSynchronize());

//     int* d_hmap2;
//     CUDA_CHECK(cudaMalloc(&d_hmap2, mx * sizeof(int)));
//     CUDA_CHECK(cudaMemcpy(d_hmap2, hmap2.data(), mx * sizeof(int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaDeviceSynchronize());

//     JoinKernel<<<numBlocks, BLOCK_THREADS>>>(d_final_array, d_results, n, d_hmap1, d_hmap2);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1, 1>>>(d_results, 3*n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     // CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * n * sizeof(int), cudaMemcpyDeviceToHost));
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     // // for (int i = 0; i < 101; i += 3) {
//     // for (int i = 0; i < 3 * n; i += 3) {
//     //     if (h_results[i] != -1)
//     //         std::cout << "Key: " << h_results[i] << " Values: " << h_results[i + 1] << " " << h_results[i + 2] << std::endl;
//     // }

//     // Free device memory
//     cudaFree(d_data);
//     cudaFree(d_sorted_data);
//     cudaFree(d_samples);
//     cudaFree(d_splitters);
//     cudaFree(d_output);
//     cudaFree(d_partition_starts);
//     cudaFree(d_final_array);
//     cudaFree(d_results);
//     cudaFree(d_hmap1);
//     cudaFree(d_hmap2);

//     return 0;
// }


#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cmath>

#define BLOCK_THREADS 512
#define ITEMS_PER_THREAD 1
#define BLOCK_THREAD 2*BLOCK_THREADS

#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/SortDataBlockWise.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FindSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/DistributionAfterSplits.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FinalSorting.cu"
#include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/JoinAfterSort.cu"

int main() {
    int n1 = 512*100;
    int n2 = 512;

    std::vector<int> keys1(n1);
    std::vector<int> keys2(n2);

    for (int i = 0; i < n1; i++) {
        keys1[i] = 2 * i;
    }
    for (int i = 0; i < n2; i++) {
        keys2[i] = 3 * i;
    }

    int mx = 2*1e6;
    std::vector<int> hmap1(mx, 0);
    std::vector<int> hmap2(mx, 0);

    for (int i = 0; i < n1; i++) {
        hmap1[keys1[i]] = rand() % 355;
    }
    for (int i = 0; i < n2; i++) {
        hmap2[keys2[i]] = 500 + (rand() % 326);
    }

    const int n = n1 + n2;
    std::vector<int> h_data(n);
    for (int i = 0; i < n; i++) {
        if (i < n1) h_data[i] = keys1[i];
        else h_data[i] = keys2[i - n1];
    }

    // Allocate device memory
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    int* d_sorted_data;
    CUDA_CHECK(cudaMalloc(&d_sorted_data, n * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int numBlocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

    // Launch kernel to sort blocks
    BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_data, d_sorted_data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int blockSize = BLOCK_THREADS;
    int p = numBlocks;
    int sample_size = p * int(log2(p));
    int *d_samples, *d_splitters;
    curandState* d_state;
    CUDA_CHECK(cudaMalloc(&d_samples, sample_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_splitters, (p - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_state, sample_size * sizeof(curandState)));

    initCurand<<<numBlocks, blockSize>>>(d_state, time(NULL), sample_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    FindSplit(d_sorted_data, d_samples, d_splitters, n, p, sample_size, d_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Splitterss<<<1, 1>>>(d_splitters, d_samples, sample_size, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printArray<<<1, 1>>>(d_splitters, p - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_output, *d_partition_counts, *d_partition_starts, *d_partition_offsets;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partition_offsets, p * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_partition_offsets, 0, p * sizeof(int)));

    countElements<<<numBlocks, blockSize>>>(d_sorted_data, d_splitters, d_partition_counts, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    computeStarts<<<1, 1>>>(d_partition_counts, d_partition_starts, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    distributeElements<<<numBlocks, blockSize>>>(d_sorted_data, d_output, d_splitters, d_partition_starts, d_partition_offsets, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printArray<<<1, 1>>>(d_partition_counts, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printArray<<<1, 1>>>(d_partition_offsets, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* d_final_array;
    CUDA_CHECK(cudaMalloc(&d_final_array, n * sizeof(int)));

    BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts, p, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // printArray<<<1, 1>>>(d_final_array, n);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // int* d_results;
    // CUDA_CHECK(cudaMalloc(&d_results, 3 * n * sizeof(int)));
    // CUDA_CHECK(cudaMemset(d_results, -1, 3 * n * sizeof(int)));

    // int* d_hmap1;
    // CUDA_CHECK(cudaMalloc(&d_hmap1, mx * sizeof(int)));
    // CUDA_CHECK(cudaMemcpy(d_hmap1, hmap1.data(), mx * sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaDeviceSynchronize());

    // int* d_hmap2;
    // CUDA_CHECK(cudaMalloc(&d_hmap2, mx * sizeof(int)));
    // CUDA_CHECK(cudaMemcpy(d_hmap2, hmap2.data(), mx * sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaDeviceSynchronize());

    // JoinKernel<<<numBlocks, BLOCK_THREADS>>>(d_final_array, d_results, n, d_hmap1, d_hmap2);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // int h_results[3 * n];
    // CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * n * sizeof(int), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaDeviceSynchronize());

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
    cudaFree(d_partition_offsets);
    cudaFree(d_partition_counts);
    cudaFree(d_final_array);
    // cudaFree(d_results);
    // cudaFree(d_hmap1);
    // cudaFree(d_hmap2);

    return 0;
}


// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <cub/cub.cuh>
// #include <cmath>

// #define BLOCK_THREADS 512
// #define ITEMS_PER_THREAD 1
// #define BLOCK_THREAD 2*BLOCK_THREADS

// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/SortDataBlockWise.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FindSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/DistributionAfterSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FinalSorting.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/JoinAfterSort.cu"

// int main() {
//     int n1 = 1e3;
//     int n2 = 2*1e3;

//     std::vector<int> keys1(n1);
//     std::vector<int> keys2(n2);

//     for (int i = 0; i < n1; i++) {
//         keys1[i] = 2 * i;
//     }
//     for (int i = 0; i < n2; i++) {
//         keys2[i] = 3 * i;
//     }

//     int mx = 2 * 1e5;
//     std::vector<int> hmap1(mx, 0);
//     std::vector<int> hmap2(mx, 0);

//     for (int i = 0; i < n1; i++) {
//         hmap1[keys1[i]] = rand() % 355;
//     }
//     for (int i = 0; i < n2; i++) {
//         hmap2[keys2[i]] = 500 + (rand() % 326);
//     }

//     const int n = n1 + n2;
//     std::vector<int> h_data(n);
//     for (int i = 0; i < n; i++) {
//         if (i < n1) h_data[i] = keys1[i];
//         else h_data[i] = keys2[i - n1];
//     }

//     int* d_data;
//     CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
//     int* d_sorted_data;
//     CUDA_CHECK(cudaMalloc(&d_sorted_data, n * sizeof(int)));

//     CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));

//     int numBlocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

//     // Launch kernel to sort blocks
//     BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_data, d_sorted_data, n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

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

//     Splitterss<<<1, 1>>>(d_splitters, d_samples, sample_size, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     printArray<<<1, 1>>>(d_splitters, p - 1);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     int *d_output, *d_partition_counts, *d_partition_starts, *d_partition_offsets;
//     CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_offsets, p * sizeof(int)));

//     CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_offsets, 0, p * sizeof(int)));

//     countElements<<<numBlocks, blockSize>>>(d_sorted_data, d_splitters, d_partition_counts, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     computeStarts<<<1, 1>>>(d_partition_counts, d_partition_starts, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     distributeElements<<<numBlocks, blockSize>>>(d_sorted_data, d_output, d_splitters, d_partition_starts, d_partition_offsets, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     int* d_final_array;
//     CUDA_CHECK(cudaMalloc(&d_final_array, n * sizeof(int)));

//     BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts, p, n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     printArray<<<1, 1>>>(d_final_array, n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Free device memory
//     cudaFree(d_data);
//     cudaFree(d_sorted_data);
//     cudaFree(d_samples);
//     cudaFree(d_splitters);
//     cudaFree(d_output);
//     cudaFree(d_partition_starts);
//     cudaFree(d_partition_offsets);
//     cudaFree(d_partition_counts);
//     cudaFree(d_final_array);

//     return 0;
// }



// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <cub/cub.cuh>
// #include <cmath>

// #define BLOCK_THREADS 512
// #define ITEMS_PER_THREAD 1
// #define BLOCK_THREAD 2*BLOCK_THREADS

// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/SortDataBlockWise.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FindSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/DistributionAfterSplits.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/FinalSorting.cu"
// #include "/csproject/yike/intern/ronak/HKUST-Server-Use-/PkPkJoin/Ronak/JoinAfterSort.cu"

// int main() {
//     // int n1=9;
//     // int n2=5;
//     // int hmap1[15];
//     // int hmap2[10];
//     // int keys1[n1] = {6, 7, 8, 9, 1, 2, 3, 4, 5};
//     // int values1[n1] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
//     // int keys2[n2] = { 3, 6, 9, 1, 2};
//     // int values2[n2] = {101, 102, 103, 108, 110};
//     int n1=1e3;
//     int n2=2*1e3;
//     int keys1[n1];
//     int keys2[n2];
//     for (int i = 0; i < n1; i++) {
//         keys1[i] = 2*i;
//     }
//     for (int i = 0; i < n1; i++) {
//         keys2[i] = 3*i;
//     }
//     int mx=2*1e5;
//     int hmap1[mx];
//     int hmap2[mx];
//     for(int i =0;i<n1;i++)
//     {
//         hmap1[keys1[i]] = rand() % 355;
//     }
//     for(int i =0;i<n2;i++)
//     {
//         hmap2[keys2[i]] = 500+(rand() % 326);
//     }

//     // std::vector<int> h_data(1e5);
//     // for (int i=0;i<h_data.size();i++) {
//     //     h_data[i]=rand()%12574;
//     // }
//     const int n = n1+n2;
//     std::vector<int> h_data(n);
//     for (int i=0;i<n;i++) {
//         if (i<n1) h_data[i] = keys1[i];
//         else h_data[i] = keys2[i-n1];
//     }

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

//     int blockSize = BLOCK_THREADS;
//     int p = numBlocks;
//     int sample_size = p*int(log2(p));
//     int *d_samples, *d_splitters;
//     curandState* d_state;
//     cudaMalloc(&d_samples, sample_size * sizeof(int));
//     cudaMalloc(&d_splitters, (p - 1) * sizeof(int));
//     CUDA_CHECK(cudaMalloc(&d_state, sample_size * sizeof(curandState)));

//     initCurand<<<numBlocks, blockSize>>>(d_state, time(NULL), sample_size);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     FindSplit(d_sorted_data, d_samples, d_splitters, n, p, sample_size, d_state);

//     Splitterss<<<1,1>>> (d_splitters,d_samples,sample_size,p);
//     cudaDeviceSynchronize();
//     printArray<<<1,1>>> (d_splitters,p-1);

//     // Device pointers
//     int  *d_output, *d_partition_counts, *d_partition_starts, *d_partition_offsets;

//     // Allocate device memory
//     CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_counts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_starts, p * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_partition_offsets, p * sizeof(int)));

//     // Copy data to device
//     CUDA_CHECK(cudaMemset(d_partition_counts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_starts, 0, p * sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_partition_offsets, 0, p * sizeof(int)));

//     // Launch kernels in sequence to ensure synchronization
//     countElements<<<numBlocks, blockSize>>>(d_sorted_data, d_splitters, d_partition_counts, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     computeStarts<<<1, 1>>>(d_partition_counts, d_partition_starts, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     distributeElements<<<numBlocks, blockSize>>>(d_sorted_data, d_output, d_splitters, d_partition_starts, d_partition_offsets, n, p);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1,1>>>(d_output,10000);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());

//     // printArray<<<1,1>>>(d_partition_starts,n);
//     // CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());
    

//     int* d_final_array;
//     cudaMalloc(&d_final_array, n * sizeof(int));
//     // BlockSortKernel<<<numBlocks, BLOCK_THREADS>>>(d_output, d_final_array,n);
//     BlockSortKernel2<<<numBlocks, BLOCK_THREAD>>>(d_output, d_final_array, d_partition_starts,p,n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     printArray<<<1,1>>>(d_final_array,n);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     int *d_results;
//     cudaMalloc(&d_results, 3*n * sizeof(int));
//     cudaMemset(d_results, -1, 3*n * sizeof(int));
    
//     int *d_hmap1;
//     cudaMalloc(&d_hmap1, sizeof(hmap1));
//     cudaMemcpy(d_hmap1, hmap1, sizeof(hmap1), cudaMemcpyHostToDevice);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     int *d_hmap2;
//     cudaMalloc(&d_hmap2, sizeof(hmap2));
//     cudaMemcpy(d_hmap2, hmap2, sizeof(hmap2), cudaMemcpyHostToDevice);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     JoinKernel<<<numBlocks, BLOCK_THREADS>>>(d_final_array, d_results, n, d_hmap1, d_hmap2);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
//     // printArray<<<1,1>>>(d_results,3*n);

//     int h_results[3*n];
//     cudaMemcpy(h_results, d_results,3*n * sizeof(int), cudaMemcpyDeviceToHost);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     for (int i=0;i<3*n;i+=3) {
//         if(h_results[i]!=-1)
//             std::cout<<"Key: "<<h_results[i]<<" Values: "<<h_results[i+1]<<" "<<h_results[i+2]<<std::endl;
//     }

//     // Free device memory
//     cudaFree(d_data);
//     cudaFree(d_sorted_data);
//     cudaFree(d_samples);
//     cudaFree(d_splitters);
//     CUDA_CHECK(cudaFree(d_output));
//     CUDA_CHECK(cudaFree(d_partition_starts));
//     CUDA_CHECK(cudaFree(d_partition_offsets));
//     CUDA_CHECK(cudaFree(d_partition_counts));

//     return 0;
// }
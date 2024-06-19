#include <cuda_runtime.h>
#include <iostream>

// Define your data types as needed
// typedef int KeyType;
// typedef int ValueType;

// Kernel to perform hash join-like operation on sorted data
// __global__ void hashJoinKernel(const KeyType* keys, const ValueType* values1, const ValueType* values2, ValueType* results, int numElements)
__global__ void JoinKernel(int * d_final_array, std::vector<vector<int>>& results, int numElements, int* hmap1, int* hmap2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numElements - 1)
    {
        // Check if current element is equal to the next element
        if (keys[tid] == keys[tid + 1])
        {
            int k=keys[tid];
            std::results.push_back({k, hmap1[k], hmap2[k]});
        }

    }
}

// int main()
// {
//     // Example data setup (replace with your actual data)
//     const int numElements = 10; // Example number of elements
//     const int blockSize = 256;
//     const int numBlocks = (numElements + blockSize - 1) / blockSize;

//     // Example sorted data (keys and values)
//     KeyType keys1[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//     ValueType values1[numElements] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
//     KeyType keys1[numElements] = {1, 2, 3, 6, 9};
//     ValueType values2[numElements] = {101, 102, 103, 108, 110};

//     // Allocate device memory
//     KeyType* d_keys;
//     ValueType* d_values1;
//     ValueType* d_values2;
//     ValueType* d_results;

//     cudaMalloc((void**)&d_keys, numElements * sizeof(KeyType));
//     cudaMalloc((void**)&d_values1, numElements * sizeof(ValueType));
//     cudaMalloc((void**)&d_values2, numElements * sizeof(ValueType));
//     cudaMalloc((void**)&d_results, numElements * 3 * sizeof(ValueType)); // Allocate enough space for results

//     // Copy data to device
//     cudaMemcpy(d_keys, keys, numElements * sizeof(KeyType), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values1, values1, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values2, values2, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

//     // Launch kernel
//     hashJoinKernel<<<numBlocks, blockSize>>>(d_keys, d_values1, d_values2, d_results, numElements);

//     // Copy results back to host
//     ValueType results[numElements][3];
//     cudaMemcpy(results, d_results, numElements * 3 * sizeof(ValueType), cudaMemcpyDeviceToHost);

//     // Print results (adjust as needed)
//     std::cout << "Results:" << std::endl;
//     for (int i = 0; i < numElements; ++i)
//     {
//         if (results[i][0] != 0)
//         {
//             std::cout << "Key: " << keys[i] << ", Joined Value: " << results[i][1] << " " << results[i][2] << std::endl;
//         }
//     }

//     // Free device memory
//     cudaFree(d_keys);
//     cudaFree(d_values1);
//     cudaFree(d_values2);
//     cudaFree(d_results);

//     return 0;
// }

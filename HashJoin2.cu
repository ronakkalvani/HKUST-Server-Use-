#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ int hash(int key, int num_buckets) {
    return key % num_buckets;
}

__global__ void buildHashTable(int *keys, int *values, int *hash_table_keys, int *hash_table_values, int num_elements, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int key = keys[idx];
        int value = values[idx];
        int bucket = hash(key, num_buckets);
        // Simple linear probing for collision resolution
        while (atomicCAS(&hash_table_keys[bucket], -1, key) != -1) {
            bucket = (bucket + 1) % num_buckets;
        }
        hash_table_values[bucket] = value;
    }
}

__global__ void probeHashTable(int *keys, int *hash_table_keys, int *hash_table_values, int *results, int num_elements, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int key = keys[idx];
        int bucket = hash(key, num_buckets);
        while (hash_table_keys[bucket] != key && hash_table_keys[bucket] != -1) {
            bucket = (bucket + 1) % num_buckets;
        }
        if (hash_table_keys[bucket] == key) {
            results[idx] = hash_table_values[bucket];
        } else {
            results[idx] = -1; // Key not found
        }
    }
}

void checkCudaError(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
}

int main() {
    const int num_elements = 1024;
    const int num_buckets = 2048;

    int h_keys[num_elements], h_values[num_elements];
    int *d_keys, *d_values, *d_hash_table_keys, *d_hash_table_values, *d_results;

    // Initialize input data
    for (int i = 0; i < num_elements; ++i) {
        h_keys[i] = i;
        h_values[i] = i * 2;
    }

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_keys, num_elements * sizeof(int)), "cudaMalloc d_keys");
    checkCudaError(cudaMalloc(&d_values, num_elements * sizeof(int)), "cudaMalloc d_values");
    checkCudaError(cudaMalloc(&d_hash_table_keys, num_buckets * sizeof(int)), "cudaMalloc d_hash_table_keys");
    checkCudaError(cudaMalloc(&d_hash_table_values, num_buckets * sizeof(int)), "cudaMalloc d_hash_table_values");
    checkCudaError(cudaMalloc(&d_results, num_elements * sizeof(int)), "cudaMalloc d_results");

    // Initialize hash table keys to -1 (indicating empty)
    checkCudaError(cudaMemset(d_hash_table_keys, -1, num_buckets * sizeof(int)), "cudaMemset d_hash_table_keys");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_keys, h_keys, num_elements * sizeof(int)), "cudaMemcpy d_keys");
    checkCudaError(cudaMemcpy(d_values, h_values, num_elements * sizeof(int)), "cudaMemcpy d_values");

    // Kernel launch for building hash table
    buildHashTable<<<(num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_values, d_hash_table_keys, d_hash_table_values, num_elements, num_buckets);
    checkCudaError(cudaGetLastError(), "Kernel launch failed (buildHashTable)");

    // Kernel launch for probing hash table
    probeHashTable<<<(num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_keys, d_hash_table_keys, d_hash_table_values, d_results, num_elements, num_buckets);
    checkCudaError(cudaGetLastError(), "Kernel launch failed (probeHashTable)");

    // Copy results back to host
    int h_results[num_elements];
    checkCudaError(cudaMemcpy(h_results, d_results, num_elements * sizeof(int)), "cudaMemcpy h_results");

    // Verify results
    for (int i = 0; i < num_elements; ++i) {
        if (h_results[i] != -1) {
            std::cout << "Key: " << h_keys[i] << ", Value: " << h_results[i] << std::endl;
        }
    }

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_hash_table_keys);
    cudaFree(d_hash_table_values);
    cudaFree(d_results);

    return 0;
}

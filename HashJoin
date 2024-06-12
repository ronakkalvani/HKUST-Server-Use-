#include <cuda.h>
#include <stdio.h>
#include <cuCollections/cuhash.h>

#define SIZE 1000
#define THREADS_PER_BLOCK 100

__global__ void hash_join(int *arr1, int *arr2, int *result, cuh::unordered_set<int> hash_table) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < SIZE) {
        if (hash_table.contains(arr2[index])) {
            result[index] = arr2[index];
        }
    }
}

int main() {
    int arr1[SIZE], arr2[SIZE], result[SIZE];
    int *d_arr1, *d_arr2, *d_result;

    // Initialize arrays and result
    for (int i = 0; i < SIZE; i++) {
        arr1[i] = i;
        arr2[i] = SIZE - i;
        result[i] = -1;
    }

    cudaMalloc((void **)&d_arr1, SIZE * sizeof(int));
    cudaMalloc((void **)&d_arr2, SIZE * sizeof(int));
    cudaMalloc((void **)&d_result, SIZE * sizeof(int));

    cudaMemcpy(d_arr1, arr1, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cuh::unordered_set<int> hash_table(2 * SIZE);
    for (int i = 0; i < SIZE; i++) {
        hash_table.insert(arr1[i]);
    }

    hash_join<<<SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_arr1, d_arr2, d_result, hash_table);

    cudaMemcpy(result, d_result, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        if (result[i] != -1) {
            printf("%d ", result[i]);
        }
    }

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_result);

    return 0;
}

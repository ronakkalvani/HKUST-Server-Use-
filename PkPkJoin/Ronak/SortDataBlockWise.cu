#include <stdio.h>
#include <cuda.h>

#define N 1024 // Total number of elements in the dataset
#define BLOCK_SIZE 256 // Number of elements in each block

// CUDA kernel to sort a block of data
__global__ void sort_block(int *data, int blockSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread handles sorting within a block
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize - 1; j++) {
            int k = idx * blockSize + j;
            if (data[k] > data[k + 1]) {
                // Swap elements
                int temp = data[k];
                data[k] = data[k + 1];
                data[k + 1] = temp;
            }
        }
    }
}

// Helper function to print the dataset
void print_data(int *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

int main() {
    int *h_data;
    int *d_data;
    size_t size = N * sizeof(int);

    // Allocate memory on the host
    h_data = (int *)malloc(size);

    // Initialize the dataset with random values
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 100;
    }

    printf("Original Data:\n");
    print_data(h_data, N);

    // Allocate memory on the device
    cudaMalloc((void **)&d_data, size);

    // Copy the dataset from the host to the device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch the kernel to sort each block
    sort_block<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, BLOCK_SIZE);

    // Copy the sorted dataset from the device back to the host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("Sorted Data (block-wise):\n");
    print_data(h_data, N);

    // Free the device memory
    cudaFree(d_data);

    // Free the host memory
    free(h_data);

    return 0;
}

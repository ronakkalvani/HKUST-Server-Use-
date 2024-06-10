#include <cub/cub.cuh>
#include <iostream>

__global__ void allocate_data(int* device_data, int num_items)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    device_data[ind]=num_items-ind;
}

// Kernel function to print the sorted data
__global__ void print_sorted_data(int* device_data, int num_items)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    if (ind == 0)
    {
        printf("Sorted data: ");
        for (int i = 0; i < num_items; i++)
            printf("%d ", device_data[i]);
        printf("\n");
    }
}

int main()
{
    // Initialize host data
    int num_items = 1e4;

    // Initialize device data
    int* d_data;
    cudaMalloc(&d_data, num_items);

    // Allocate device memory for sorted data
    int* d_sorted_data;
    cudaMalloc(&d_sorted_data, num_items);

    allocate_data<<<1000,10>>>(d_data, num_items);

    // Allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort data
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_items);

    // Print sorted data
    print_sorted_data<<<1, 1>>>(d_sorted_data, num_items);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_temp_storage);
    
    return 0;
}


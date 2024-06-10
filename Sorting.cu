#include <cub/cub.cuh>
#include <iostream>

// Kernel function to print the sorted data
__global__ void print_sorted_data(int* device_data, int num_items)
{
    if (threadIdx.x == 0)
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
    int h_data[] = {1, 5, 2, 4, 3};
    int num_items = sizeof(h_data) / sizeof(h_data[0]);

    // Initialize device data
    int* d_data;
    cudaMalloc(&d_data, sizeof(h_data));
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    // Allocate device memory for sorted data
    int* d_sorted_data;
    cudaMalloc(&d_sorted_data, sizeof(h_data));

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
    
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    cudaProfilerStop();
    return 0;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_temp_storage);

    return 0;
}


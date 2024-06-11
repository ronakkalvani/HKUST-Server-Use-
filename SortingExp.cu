#include <cub/cub.cuh>
#include <iostream>

__global__ void get_attr(int* device_data, int num_items) {
    int ind=threadIdx.x+blockDim.x*blockIdx.x;
    device_data[ind]=rand()%num_items;
}

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
    const int num_items = 1000;
    int h_data[num_items];

    // Initialize device data
    int* d_data;
    cudaMalloc(&d_data, sizeof(h_data));
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    get_attr<<<1,1000>>>(d_data,num_items);

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
    // print_sorted_data<<<1, 1>>>(d_sorted_data, num_items);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sorted_data);
    cudaFree(d_temp_storage);

    return 0;
}





// #include <cub/cub.cuh>
// #include <iostream>

// __global__ void allocate_data(int* device_data, int num_items)
// {
//     int ind = threadIdx.x + blockIdx.x * blockDim.x;
//     device_data[ind]=num_items-ind;
// }

// // Kernel function to print the sorted data
// __global__ void print_sorted_data(int* device_data, int num_items)
// {
//     int ind = threadIdx.x + blockIdx.x * blockDim.x;
//     if (ind == 0)
//     {
//         printf("Sorted data: ");
//         for (int i = 0; i < num_items; i++)
//             printf("%d ", device_data[i]);
//         printf("\n");
//     }
// }

// int main()
// {
//     // Initialize host data
//     int num_items = 1e6;

//     // Initialize device data
//     int* d_data;
//     cudaMalloc(&d_data, num_items);

//     // Allocate device memory for sorted data
//     int* d_sorted_data;
//     cudaMalloc(&d_sorted_data, num_items);

//     allocate_data<<<1000,1000>>>(d_data, num_items);

//     // Allocate temporary storage
//     void* d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_items);
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);

//     // Sort data
//     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_items);

//     // Print sorted data
//     // print_sorted_data<<<1, 1>>>(d_sorted_data, num_items);
//     cudaDeviceSynchronize();

//     // Cleanup
//     cudaFree(d_data);
//     cudaFree(d_sorted_data);
//     cudaFree(d_temp_storage);
    
//     return 0;
// }


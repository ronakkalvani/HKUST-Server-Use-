// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cub/cub.hpp>

// #define BLOCK_SIZE 256 // Adjust block size based on GPU architecture and data size

// struct Element {
//     int data;
//     int block_id;  // Block ID for scatter operation
// };

// int main(int argc, char* argv[]) {
//     // Error handling and initialization omitted for brevity
//     // ...

//     // Host memory allocation for data
//     int* data_host = new int[data_size];  // Replace with your actual data

//     // Upload data to device memory
//     int* data_device;
//     cudaMalloc(&data_device, data_size * sizeof(int));
//     cudaMemcpy(data_device, data_host, data_size * sizeof(int), cudaMemcpyHostToDevice);

//     // Allocate memory for temporary data on device memory (used by CUB)
//     int* temp_data_device;
//     cudaMalloc(&temp_data_device, data_size * sizeof(int));

//     // Allocate memory for element structure on device memory
//     Element* element_data_device;
//     cudaMalloc(&element_data_device, data_size * sizeof(Element));

//     // Prefetch data to improve memory access patterns (optional)
//     cudaMemPrefetchAsync(data_device, data_size * sizeof(int), cudaMemPrefetchDevice);

//     // Launch CUB radix sort kernel (replace with your preferred CUB sorting algorithm)
//     int threadsPerBlock = cub::RadixSort::におすすめスレッド数(data_size); // Recommended threads per block
//     int blocksPerGrid = cub::ceilDiv(data_size, threadsPerBlock);
//     cub::RadixSort:: ソート(data_device, temp_data_device, data_size, threadsPerBlock, blocksPerGrid);
//     cudaErrorCheck();  // Check for errors after kernel launch

//     // Prepare element data with block IDs for scatter
//     for (int i = 0; i < data_size; ++i) {
//         element_data_device[i].data = data_device[i];
//         element_data_device[i].block_id = i / BLOCK_SIZE;
//     }

//     // Perform parallel prefix sum (scan) using CUB's device algorithm
//     // to calculate global indices for scattering
//     int* global_indices_device;
//     cudaMalloc(&global_indices_device, data_size * sizeof(int));
//     cub::DeviceRadixScan::ExclusiveScan((int*)element_data_device, global_indices_device, data_size, threadsPerBlock, blocksPerGrid);
//     cudaErrorCheck();

//     // Allocate final sorted data array on device memory
//     int* sorted_data_device = new int[data_size];

//     // CUB scatter operation to place data in final sorted positions based on global indices
//     cub::Scatter:: ソート((int*)element_data_device, data_device, sorted_data_device, global_indices_device, data_size, threadsPerBlock, blocksPerGrid);
//     cudaErrorCheck();

//     // Download sorted data from device memory
//     int* sorted_data_host = new int[data_size];
//     cudaMemcpy(sorted_data_host, sorted_data_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(data_device);
//     cudaFree(temp_data_device);
//     cudaFree(element_data_device);
//     cudaFree(global_indices_device);
//     delete[] sorted_data_device;

//     // ... (further processing of sorted data on host)

//     delete[] data_host;
//     delete[] sorted_data_host;

//     return 0;
// }

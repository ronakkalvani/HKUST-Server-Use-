#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

struct DataElement {
    int key;
    int value; // Additional data associated with the key
};

__global__ void createKeyValuePairs(int *keys, int *values, DataElement *data, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        keys[tid] = data[tid].key;
        values[tid] = data[tid].value;
    }
}

__global__ void joinElements(int *keys, int *values, int *joined_keys, int *joined_values1, int *joined_values2, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0 && tid < num_elements) {
        if (keys[tid] == keys[tid - 1]) {
            joined_keys[tid] = keys[tid];
            joined_values1[tid] = values[tid];
            joined_values2[tid] = values[tid - 1];
        }
    }
}

void joinDatasets(DataElement *d_data1, int num_elements1, DataElement *d_data2, int num_elements2) {
    int total_elements = num_elements1 + num_elements2;

    // Allocate device memory for keys and values
    int *d_keys, *d_values;
    cudaMalloc(&d_keys, total_elements * sizeof(int));
    cudaMalloc(&d_values, total_elements * sizeof(int));

    // Copy data to combined arrays
    createKeyValuePairs<<<(num_elements1 + 255) / 256, 256>>>(d_keys, d_values, d_data1, num_elements1);
    createKeyValuePairs<<<(num_elements2 + 255) / 256, 256>>>(d_keys + num_elements1, d_values + num_elements1, d_data2, num_elements2);

    // Sort keys and values together using CUB
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_values, d_values, total_elements);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_values, d_values, total_elements);
    cudaFree(d_temp_storage);

    // Allocate memory for joined elements
    int *d_joined_keys, *d_joined_values1, *d_joined_values2;
    cudaMalloc(&d_joined_keys, total_elements * sizeof(int));
    cudaMalloc(&d_joined_values1, total_elements * sizeof(int));
    cudaMalloc(&d_joined_values2, total_elements * sizeof(int));

    // Join elements
    joinElements<<<(total_elements + 255) / 256, 256>>>(d_keys, d_values, d_joined_keys, d_joined_values1, d_joined_values2, total_elements);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_joined_keys);
    cudaFree(d_joined_values1);
    cudaFree(d_joined_values2);
}

int main() {
    std::vector<DataElement> h_data1 = {{1, 10}, {3, 30}, {5, 50}};
    std::vector<DataElement> h_data2 = {{1, 100}, {2, 200}, {3, 300}};

    int num_elements1 = h_data1.size();
    int num_elements2 = h_data2.size();

    DataElement *d_data1, *d_data2;
    cudaMalloc(&d_data1, num_elements1 * sizeof(DataElement));
    cudaMalloc(&d_data2, num_elements2 * sizeof(DataElement));

    cudaMemcpy(d_data1, h_data1.data(), num_elements1 * sizeof(DataElement), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data2.data(), num_elements2 * sizeof(DataElement), cudaMemcpyHostToDevice);

    joinDatasets(d_data1, num_elements1, d_data2, num_elements2);

    cudaFree(d_data1);
    cudaFree(d_data2);

    return 0;
}

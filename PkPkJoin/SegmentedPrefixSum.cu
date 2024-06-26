#include <iostream>
#include <cub/cub.cuh>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

// Function to print array
template <typename T>
void PrintArray(const T* array, int size, const char* label) {
    std::cout << label << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

__global__ void InitFlags(const int* d_input, int* d_flags, int num_items) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_items) {
        if (tid == 0) {
            d_flags[tid] = 1;
        } else {
            d_flags[tid] = (d_input[tid] != d_input[tid - 1]) ? 1 : 0;
        }
    }
}

int main() {
    const int num_items = 10;
    int h_input[num_items] = {1, 1, 1, 2, 2, 1, 1, 3, 3, 1};
    int h_output[num_items];

    int* d_input = nullptr;
    int* d_output = nullptr;
    int* d_flags = nullptr;
    int* d_segment_offsets = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cudaMalloc((void**)&d_input, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_flags, num_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_segment_offsets, (num_items + 1) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_items * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_items + threads - 1) / threads;

    InitFlags<<<blocks, threads>>>(d_input, d_flags, num_items);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute the segment offsets
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_flags, d_segment_offsets + 1, num_items);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_flags, d_segment_offsets + 1, num_items);

    // Set the first element of segment_offsets to 0
    CUDA_CHECK(cudaMemset(d_segment_offsets, 0, sizeof(int)));

    // Perform the segmented prefix sum
    cub::DeviceSegmentedScan::InclusiveSum(nullptr, temp_storage_bytes, d_input, d_output, num_items, num_items, d_segment_offsets, d_segment_offsets + 1);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceSegmentedScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items, num_items, d_segment_offsets, d_segment_offsets + 1);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_items * sizeof(int), cudaMemcpyDeviceToHost));

    PrintArray(h_input, num_items, "Input");
    PrintArray(h_output, num_items, "Output");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_segment_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));

    return 0;
}

#include <cub/cub.cuh>
#include <iostream>
#include <vector>

// Prefix sum function
void prefix_sum(int* d_input, int* d_output, size_t num_items) {
    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    (cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

    // Allocate temporary storage
    (cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Perform inclusive prefix sum
    (cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, num_items));

    // Clean up temporary storage
    (cudaFree(d_temp_storage));
}

int main() {
    // Initialize host input
    std::vector<int> h_input = {1, 2, 3, 4, 5};

    // Allocate device input and output arrays
    int *d_input = nullptr;
    int *d_output = nullptr;
    (cudaMalloc(&d_input, h_input.size() * sizeof(int)));
    (cudaMalloc(&d_output, h_input.size() * sizeof(int)));

    // Copy input data to device
    (cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Perform prefix sum using the function
    prefix_sum(d_input, d_output, h_input.size());

    // Copy results back to host
    std::vector<int> h_output(h_input.size());
    (cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "Input:  ";
    for (int val : h_input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int val : h_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Clean up
    (cudaFree(d_input));
    (cudaFree(d_output));

    return 0;
}

#include <iostream>
#include <cub/cub.cuh>  // Include CUB header for CUDA utilities

const int blockSize = 8;
const int numBlocks = 3;
const int numElements = blockSize * numBlocks;

__global__ void segmentedPrefixSum(int *data, int *output, int *segmentOffsets, int numElements) {
    typedef cub::BlockScan<int, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input data
    int value = (idx < numElements) ? data[idx] : 0;
    int prefix = 0;

    // Compute inclusive prefix sum within the block
    BlockScan(temp_storage).InclusiveScan(value, value, cub::Sum(), prefix);

    // Store the result in output array
    if (idx < numElements) {
        output[idx] = prefix;

        // Compute segment offsets (start of each new segment)
        if (idx > 0 && data[idx] != data[idx - 1]) {
            segmentOffsets[blockIdx.x] = output[idx - 1] + data[idx - 1];
        }
    }
}

int main() {
    // Example input data
    int input[numElements] = {0, 0, 0, 0, 1, 1, 1, 2,
                               0, 0, 1, 1, 1, 2, 2, 2,
                               0, 1, 1, 1, 2, 2, 2, 2};

    int *d_input, *d_output, *d_segmentOffsets;
    int segmentOffsets[numBlocks];

    // Allocate device memory
    cudaMalloc((void **)&d_input, numElements * sizeof(int));
    cudaMalloc((void **)&d_output, numElements * sizeof(int));
    cudaMalloc((void **)&d_segmentOffsets, numBlocks * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input, numElements * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    segmentedPrefixSum<<<numBlocks, blockSize>>>(d_input, d_output, d_segmentOffsets, numElements);

    // Copy output data and segment offsets back to host
    cudaMemcpy(segmentOffsets, d_segmentOffsets, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output (segmented prefix sums)
    std::cout << "Segmented Prefix Sums:" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        std::cout << output[i] << " ";
        if ((i + 1) % blockSize == 0) {
            std::cout << std::endl;
        }
    }

    // Print segment offsets
    std::cout << "Segment Offsets:" << std::endl;
    for (int i = 0; i < numBlocks; ++i) {
        std::cout << segmentOffsets[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_segmentOffsets);

    return 0;
}

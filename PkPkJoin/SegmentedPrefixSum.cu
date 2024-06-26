#include <iostream>
#include <cub/cub.cuh>  // CUB header file

const int blockSize = 8;
const int numBlocks = 3;
const int numElements = blockSize * numBlocks;

// Kernel to perform segmented prefix sum
__global__ void segmentedPrefixSum(const int *input, int *output, int *segmentOffsets, int numElements)
{
    typedef cub::BlockScan<int, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int value = (idx > 0) ? input[idx] : 0; // get the value, default to 0 for idx 0

    // Compute block-wide prefix sum
    BlockScan(temp_storage).ExclusiveSum(value, value);

    // Write the block-wide prefix sum to output
    if (idx < numElements)
        output[idx] = value;

    // Store segment offset
    if (idx % blockSize == blockSize - 1)
    {
        int segmentIdx = idx / blockSize;
        segmentOffsets[segmentIdx] = output[idx];
    }

    __syncthreads();

    // Adjust output using segment offsets
    if (idx >= blockSize)
    {
        int segmentIdx = idx / blockSize - 1;
        output[idx] += segmentOffsets[segmentIdx];
    }
}

int main()
{
    int input[numElements] = {0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2};
    int output[numElements];
    int segmentOffsets[numBlocks];

    int *d_input, *d_output, *d_segmentOffsets;
    cudaMalloc((void **)&d_input, numElements * sizeof(int));
    cudaMalloc((void **)&d_output, numElements * sizeof(int));
    cudaMalloc((void **)&d_segmentOffsets, numBlocks * sizeof(int));

    cudaMemcpy(d_input, input, numElements * sizeof(int), cudaMemcpyHostToDevice);

    segmentedPrefixSum<<<numBlocks, blockSize>>>(d_input, d_output, d_segmentOffsets, numElements);

    cudaMemcpy(output, d_output, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "Output segmented prefix sums:\n";
    for (int i = 0; i < numElements; ++i)
    {
        std::cout << output[i] << " ";
        if ((i + 1) % blockSize == 0)
            std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_segmentOffsets);

    return 0;
}

__global__ void segmentedPrefixSum(int *input, int *output, int n, int blockSize) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    if (global_tid < n) {
        shared[tid] = input[global_tid];
    }
    __syncthreads();

    // Initialize prefix sum within segments
    if (global_tid < n) {
        if (tid == 0 || shared[tid] != shared[tid - 1]) {
            shared[tid] = 0;
        } else {
            shared[tid] = shared[tid - 1] + 1;
        }
    }
    __syncthreads();

    // Perform prefix sum within segments in shared memory
    for (int stride = 1; stride < blockSize; stride *= 2) {
        int temp = 0;
        if (tid >= stride && shared[tid] == shared[tid - stride]) {
            temp = shared[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && shared[tid] == shared[tid - stride]) {
            shared[tid] += temp;
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (global_tid < n) {
        output[global_tid] = shared[tid];
    }
}

int main() {
    const int n = 10;
    const int blockSize = 4;
    int h_input[n] = {1, 1, 1, 2, 2, 3, 3, 3, 3, 4};
    int h_output[n];

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int sharedMemSize = blockSize * sizeof(int);
    segmentedPrefixSum<<<(n + blockSize - 1) / blockSize, blockSize, sharedMemSize>>>(d_input, d_output, n, blockSize);

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

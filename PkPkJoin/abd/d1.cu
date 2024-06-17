#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;

// Function to initialize data (for testing purposes)
void initializeData(std::vector<int>& data) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(0, 999);

  for (auto& val : data) {
      val = dis(gen);
      cout<<val<<" ";
      }
      cout<<endl;
}

// CUDA error check macro
#define CUDA_CHECK(call) \
  do {\
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
          cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                    << cudaGetErrorString(err) << endl; \
          exit(EXIT_FAILURE); \
      } \
  } while (0)

// Kernel to initialize CURAND states
__global__ void initCurandStates(curandState* states, unsigned int seed, int numStates) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < numStates) {
      curand_init(seed, tid, 0, &states[tid]);
      //printf("%d ", tid);
      //cout<<tid<<endl;
  }
}

// Kernel to generate random sample indices
__global__ void generateSampleIndices(curandState* states, int* sampleIndices, int blockSize, int numSamples) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numSamples) {
        curandState localState = states[tid];
        sampleIndices[tid] = curand(&localState) % blockSize;
        states[tid] = localState;
    }
}

// Kernel to gather samples based on generated indices
__global__ void gatherSamples(int* data, int* sampleIndices, int* samples, int blockSize, int numSamples, int numBlocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numSamples * numBlocks) {
        int blockIdx = tid / numSamples;
        int sampleIdx = tid % numSamples;
        samples[tid] = data[blockIdx * blockSize + sampleIndices[sampleIdx]];
    }
}

// Function to perform local radix sort using CUB
void localRadixSort(int* d_data, int numElements) {
    int* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, numElements);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sort the data
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, numElements);

    // Free temporary storage
    CUDA_CHECK(cudaFree(d_temp_storage));
}

// Kernel to classify data into buckets based on splitters
__global__ void classifyData(int* data, int* splitters, int* blockIndices, int N, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int value = data[tid];
        int bucket = 0;
        while (bucket < p - 1 && value > splitters[bucket]) {
            ++bucket;
        }
        blockIndices[tid] = bucket;
    }
}

// Kernel to count elements per bucket
__global__ void countBuckets(int* blockIndices, int* bucketCounts, int N, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&bucketCounts[blockIndices[tid]], 1);
    }
}

// Kernel to prefix sum for bucket offsets
__global__ void prefixSum(int* bucketCounts, int* bucketOffsets, int p) {
    __shared__ int temp[1024];
    int tid = threadIdx.x;
    if (tid < p) {
        temp[tid] = bucketCounts[tid];
    }
    __syncthreads();

    for (int offset = 1; offset < p; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid < p) {
        bucketOffsets[tid] = temp[tid];
    }
}

// Kernel to scatter data into buckets based on splitters
__global__ void scatterData(int* data, int* blockIndices, int* bucketOffsets, int* sortedData, int* bucketCount, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int bucket = blockIndices[tid];
        int offset = atomicAdd(&bucketCount[bucket], 1);
        sortedData[bucketOffsets[bucket] + offset] = data[tid];
    }
}

int main() {
    const int N = 1024; // Total number of elements
    const int p = 8; // Number of blocks
  const int s = 4; // Number of samples per block
  const int blockSize = N / p;
  
  std::vector<int> h_data(N);
  initializeData(h_data);
  
  int* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  
  // Allocate memory for sample indices and samples
  int* d_sampleIndices;
  int* d_samples;
  CUDA_CHECK(cudaMalloc(&d_sampleIndices, p * s * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_samples, p * s * sizeof(int)));
  
  // Initialize CURAND states
  curandState* d_states;
  CUDA_CHECK(cudaMalloc(&d_states, p * s * sizeof(curandState)));
  initCurandStates<<<(p * s + 255) / 256, 256>>>(d_states, time(0), p * s);
  
  // Perform local radix sort on each block
  for (int i = 0; i < p; ++i) {
      localRadixSort(d_data + i * blockSize, blockSize);
  }
  
  // Generate random sample indices
  generateSampleIndices<<<(p * s + 255) / 256, 256>>>(d_states, d_sampleIndices, blockSize, p * s);
  
  // Gather samples based on generated indices
  gatherSamples<<<(p * s + 255) / 256, 256>>>(d_data, d_sampleIndices, d_samples, blockSize, s, p);
  
  // Copy samples back to host
  std::vector<int> h_samples(p * s);
  CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples, p * s * sizeof(int), cudaMemcpyDeviceToHost));
  
  // Sort the samples and find the (p-1) splitters
  std::sort(h_samples.begin(), h_samples.end());
  std::vector<int> h_splitters(p - 1);
  for (int i = 0; i < p - 1; ++i) {
      h_splitters[i] = h_samples[(i + 1) * s];
      cout<<h_splitters[i]<<endl;
  }
}


#include <cuda.h>
#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from CPU!\n");
    cuda_hello<<<10,1024>>>();
    cudaDeviceSynchronize();
    return 0;
}



#include <stdio.h>
#include <cuda_runtime.h>
#include <volk.h>

__global__ void HelloCuda(int i, int b) {
    int a = i + b;
    printf("Hello, CUDA!\n");
}

int main() {
    HelloCuda<<<1,1>>>(1, 1);

    cudaDeviceSynchronize();
    return 0;
}
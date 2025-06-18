#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("simple kernel running on GPU block %d, thread %d!\n", blockIdx.x, idx);
}

int main () {
    printf("Launching simple_kernel program on CPU...\n");
    // Launch GPU kernel with 3 blocks, 4 threads each
    simpleKernel<<<3, 4>>>();
    // Wait for GPU to finish before proceeding
    cudaDeviceSynchronize();
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Kernel execution completed successfully!\n");

}
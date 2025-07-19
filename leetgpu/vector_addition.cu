#include "solve.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) { 
    // unique thread is which block you're on * threadsPerBlockSize * threadIdx within the current block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // exit early if our thread i exceeds length N since we already have enough threads for this
    if (i >= N) {
        return;
    }
    // else we compute C[i] = A[i] + B[i] for each thread
    C[i] = A[i] + B[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

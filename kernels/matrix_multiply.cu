// Basic matrix multiplication kernel
// C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void initialize_matrix(float* matrix, int rows, int cols) {
    // matrices are stored as flat 1D arrays in CUDA
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0f; // Random values between 0 and 10
    }
}

int main() {
    const int M = 512;
    const int K = 256;
    const int N = 512;
    // calc each matrix size in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    // initialize matrices
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);

    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    // since we're in 2D grid we must defined 2D block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 = 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    // TODO: launch kernel 

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    // TODO: print small examples and verify

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
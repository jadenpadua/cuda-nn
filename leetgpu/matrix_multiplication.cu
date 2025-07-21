#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // ensure row, col are in bounds of matrix
    if (row < M && col < K) {
        float sum = 0.0f;
        // calc dot product of each row A col B and store the sum
        for (int i = 0; i < N; i++) {
            // proper 1D indexing of A,B
            const int a_index = row * N + i; // move aross the same row of A
            const int b_index = i * K + col; // move down rows in the same column of B
            sum += A[a_index] * B[b_index];
        }
        const int c_index = row * K + col;
        C[c_index] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

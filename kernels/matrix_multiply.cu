// Basic matrix multiplication kernel, parallelize such that each thread computes an element c (dot prod of a row A, col B) in result C matrix
// C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// each thread will compute one element of the output matrix C which is dot product of a row A, col B
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int K, int N) {
    // calc so that each thread gets a unique (row, col) coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Check bounds since we may have more threads than elements in the matrix
    if (row < M && col < N) {
        // compute one matrix element
        float sum = 0.0f;
        // dot product of row of A and column of B
        for (int k = 0; k < K; k++) {
            // both A and B are 1D flat arrays so access like this
            sum += A[row * K + k] * B[k * N + col];
        }
        // Store result in 1D flattened array
        C[row * N + col] = sum;
    }
}

void initialize_matrix(float* matrix, int rows, int cols) {
    // matrices are stored as flat 1D arrays in CUDA
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0f; // Random values between 0 and 10
    }
}

void print_matrix(float* matrix, int rows, int cols, const char* name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

float time_kernel(void (*kernel)(float*, float*, float*, int, int, int), 
    float* d_A, float* d_B, float* d_C, int M, int K, int N, 
    dim3 gridDim, dim3 blockDim) {
    // create events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start time
    cudaEventRecord(start);
    // launch the kernel
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    // record stop time
    cudaEventRecord(stop);
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    const int M = 8;
    const int K = 8;
    const int N = 8;
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

    matrix_multiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    // Time the kernel execution after warmup
    float kernel_time = time_kernel(matrix_multiply, d_A, d_B, d_C, M, K, N, gridDim, blockDim);
    printf("Kernel execution time: %.2f ms\n", kernel_time);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    if (M <= 8 && N <= 8) {
        print_matrix(h_A, M, K, "Matrix A");
        print_matrix(h_B, K, N, "Matrix B");
        print_matrix(h_C, M, N, "Matrix C (Result)");
    }
    else {
        printf("Matrix multiplication completed successfully.\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
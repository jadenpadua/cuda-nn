// Basic matrix multiplication kernel, parallelize such that each thread computes an element c (dot prod of a row A, col B) in result C matrix
// C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16 // Width of the tile, can be adjusted based on hardware capabilities

// tiled matrix multiplication kernel using shared memory
__global__ void matrix_multipl_tiled(float* A, float* B, float* C, int M, int K, int N) {
    // shared memory tiles for A and B submatrices
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];
    // calc so that each thread gets a unique (row, col) coordinate
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    // loop over tiles in the K dimension
    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        // load tile of A into shared memory
        int a_row = row;
        int a_col = tile * TILE_WIDTH + threadIdx.x;
        if (a_row < M && a_col < K) {
            tile_A[threadIdx.x][threadIdx.y] = A[a_row * K + a_col];
        } else {
            tile_A[threadIdx.x][threadIdx.y] = 0.0f; // out of bounds, set to 0
        }
        // load tile of B into shared memory
        int b_row = tile * TILE_WIDTH + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f; // out of bounds, set to 0
        }
        // synchronize threads to ensure all data is loaded
        __syncthreads();
        // compute partial dot product using shared memory
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        // synchronize threads before loading next tile
        __syncthreads();
    }







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

    matrix_multipl_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    // Time the kernel execution after warmup
    float kernel_time = time_kernel(matrix_multipl_tiled, d_A, d_B, d_C, M, K, N, gridDim, blockDim);
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
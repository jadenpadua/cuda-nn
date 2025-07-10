#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple reduction kernel - each block reduces its portion
__global__ void vectorSum(float* input, float* output, int n) {
    // Shared memory for this block
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // load data into shared memory
    if (global_id < n) {
        shared_data[tid] = input[global_id];
    } else {
        shared_data[tid] = 0.0f; // handle out of bounds
    }
    // synchronize threads in this block
    __syncthreads();
    // binary tree reduction (threads pair with each other to sum values)
    // only run this on the first half of threads in the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    // Thread 0 writes the result for this block as the final result
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// function to perform complete reduction
float performReduction(float* h_input, int n) {
    float *d_input, *d_temp;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    // allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_temp, grid_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform first reduction on GPU kernel
    vectorSum<<<grid_size, block_size>>>(d_input, d_temp, n);
    // If we have more than one block we need to reduce again
    while (grid_size > 1) {
        int new_grid_size = (grid_size + block_size - 1) / block_size;
        vectorSum<<<new_grid_size, block_size>>>(d_temp, d_temp, grid_size);
        grid_size = new_grid_size;
    }
    // Copy result back
    float result;
    cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    // clean up
    cudaFree(d_input);
    cudaFree(d_temp);

    return result;
}

int main() {
    int n = 1000000; // 1 million floats
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    // init input array
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    // perform gpu reduction on input, performReduction is still on the host
    float gpu_result = performReduction(h_input, n);
    // CPU verification
    float cpu_result = 0.0f;
    for (int i = 0; i < n; i++) {
        cpu_result += h_input[i];
    }

    printf("Array size: %d\n", n);
    printf("GPU Result: %f\n", gpu_result);
    printf("CPU Result: %f\n", cpu_result);
    printf("Results match: %s\n", (fabs(gpu_result - cpu_result) < 1e-5) ? "Yes" : "No");

    free(h_input);
    return 0;
}
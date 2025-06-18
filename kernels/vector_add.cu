#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // check if index is within bounds of number of elemnts
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main () {
    // number of elements in vectors and total size in bytes
    const int n = 1024;
    size_t size = n * sizeof(float);
    // host and device pointers to vectors on heap
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *d_a, *d_b, *d_c;
    // initialize host vectors with some values 
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    // allocate device memory and assign device pointers to GPU addresses 
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // copy input host vectors to device memory (h2d)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // calculate number of threads and blocks needed to cover all elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host (d2h)
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // Verify first 10 results
    for (int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    // Free host and device memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
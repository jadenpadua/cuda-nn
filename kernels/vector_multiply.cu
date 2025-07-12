#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorMultiply(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);
    // Host arrays
    float *h_a, *h_b, *h_c;
    // Device arrays
    float *d_a, *d_b, *d_c;
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.5f;
        h_b[i] = i * 2.0f;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vectorMultiply<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    // wait for kernel to complete
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // Verify results 
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f * %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    // cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
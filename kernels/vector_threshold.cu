// Vector threshold creates a sparse version of your data by zeroing out all values below a cutoff point, keeping only "significant" elements.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_threshold(float *input, float *output, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // set to input value if above threshold, otherwise 0
        output[idx] = (input[idx] > threshold) ? input[idx] : 0.0f;
    }
}

int main() {
    const int n = 1024;
    const float threshold = 0.5f;

    float *h_input, *h_output;
    float *d_input, *d_output;

    h_input = (float*)malloc(n * sizeof(float));
    h_output = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    vector_threshold<<<numBlocks, blockSize>>>(d_input, d_output, n, threshold);
    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    // Print first 10 results for verification
    printf("Threshold: %f\n", threshold);
    printf("Input --> Output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f --> %f\n", h_input[i], h_output[i]);
    }
    // count how many values were aboce threshold
    int count_above = 0;
    for (int i = 0; i < n; i++) {
        if (h_output[i] > 0) {
            count_above++;
        }
    }
    printf("Number of values above threshold: %d\n", count_above);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
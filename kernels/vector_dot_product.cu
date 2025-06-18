#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void vector_dot_product(const float *a, const float *b, float *partial_results, int n) {
    // Shared memory for current block's partial result
    __shared__ float temp[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // each tid will compute one element of the dot product and store in shared temp
    if (tid < n) {
        temp[threadIdx.x] = a[tid] * b[tid];
    } else {
        temp[threadIdx.x] = 0.0f; // Handle out-of-bounds threads safely with 0.0f
    }
    // barrier until all threads in the block have written their results
    __syncthreads();
    // binary tree reduction with final result in temp[0]
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        // only run this on first half of threads
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // let thread idx 0 write the block's partial result to global memory
    if (threadIdx.x == 0) {
        partial_results[blockIdx.x] = temp[0];
    }
}

float cpu_dot_product(const float *a, const float *b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    const int n = 1000000; // 1 million floats
    const size_t size = n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float h_result = 0.0f;
    // partial results array for each block that will be summed up later
    float *d_a, *d_b, *d_partial_results;

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_partial_results, num_blocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    printf("Launching kernel with %d blocks and %d threads per block...\n", num_blocks, THREADS_PER_BLOCK);
    vector_dot_product<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial_results, n);
    CHECK_CUDA(cudaGetLastError());

    float *h_partial_results = (float*)malloc(num_blocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_partial_results, d_partial_results, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    // Sum up the partial results on CPU
    for (int i = 0; i < num_blocks; i++) {
        h_result += h_partial_results[i];
    }
    // Verify result with CPU computation
    float cpu_result = cpu_dot_product(h_a, h_b, n);
    float difference = fabs(h_result - cpu_result);
    
    printf("\n=== RESULTS ===\n");
    printf("GPU result: %.6f\n", h_result);
    printf("CPU result: %.6f\n", cpu_result);
    printf("Difference: %.9f\n", difference);
    printf("Tolerance:  %.9f\n", 1e-5f);
    
    if (difference < 1e-5) {
        printf("\n✅ SUCCESS: Results match within tolerance!\n");
        printf("   GPU and CPU computations are consistent.\n");
    } else {
        printf("\n❌ FAILURE: Results don't match!\n");
        printf("   Difference (%.9f) exceeds tolerance (%.9f)\n", difference, 1e-5f);
        printf("   Check for bugs in kernel implementation.\n");
    }

    free(h_a); free(h_b); free(h_partial_results);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial_results);
    
    return 0;
}
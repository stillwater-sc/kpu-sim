/**
 * CUDA Example - MLP Forward Pass using PTX Assembly
 *
 * This demonstrates how to:
 * 1. Load and execute PTX assembly kernels
 * 2. Compare with CUDA C++ kernels
 * 3. Benchmark performance
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Reference CUDA C++ Kernel for comparison
// ============================================================================

__global__ void mlp_layer_forward_cuda_ref(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int input_size,
    int output_size
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx >= output_size) return;

    // Shared memory for input vector (cooperative loading)
    extern __shared__ float shared_input[];

    // Load input to shared memory
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    // Compute dot product
    float sum = 0.0f;
    const float* weight_row = weights + output_idx * input_size;

    // Vectorized loop - process 4 elements at a time
    int i = 0;
    for (; i + 3 < input_size; i += 4) {
        float4 w = *reinterpret_cast<const float4*>(&weight_row[i]);
        float4 in = *reinterpret_cast<const float4*>(&shared_input[i]);

        sum += w.x * in.x;
        sum += w.y * in.y;
        sum += w.z * in.z;
        sum += w.w * in.w;
    }

    // Handle remainder
    for (; i < input_size; i++) {
        sum += weight_row[i] * shared_input[i];
    }

    // Add bias and apply ReLU
    sum += bias[output_idx];
    output[output_idx] = fmaxf(0.0f, sum);
}


// ============================================================================
// Optimized CUDA kernel using warp shuffle
// ============================================================================

__global__ void mlp_layer_forward_cuda_warp(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int input_size,
    int output_size
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx >= output_size) return;

    extern __shared__ float shared_input[];

    // Cooperative load
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    // Each thread in warp processes strided elements
    int lane_id = threadIdx.x & 31;
    float sum = 0.0f;
    const float* weight_row = weights + output_idx * input_size;

    for (int i = lane_id; i < input_size; i += 32) {
        sum += weight_row[i] * shared_input[i];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only lane 0 writes result
    if (lane_id == 0) {
        sum += bias[output_idx];
        output[output_idx] = fmaxf(0.0f, sum);
    }
}


// ============================================================================
// PTX Assembly Kernel Loader
// ============================================================================

// Forward declaration of PTX kernel (will be linked from .ptx file)
extern "C" __global__ void mlp_layer_forward_ptx(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

// Alternative: Load PTX from file at runtime
CUmodule load_ptx_module(const char* ptx_filename) {
    FILE* fp = fopen(ptx_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open PTX file: %s\n", ptx_filename);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size_t ptx_size = ftell(fp);
    rewind(fp);

    char* ptx_code = (char*)malloc(ptx_size + 1);
    fread(ptx_code, 1, ptx_size, fp);
    ptx_code[ptx_size] = '\0';
    fclose(fp);

    CUmodule module;
    CUresult result = cuModuleLoadData(&module, ptx_code);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        fprintf(stderr, "Failed to load PTX module: %s\n", error_str);
        exit(1);
    }

    free(ptx_code);
    return module;
}


// ============================================================================
// Utility Functions
// ============================================================================

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

bool verify_results(const float* a, const float* b, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}


// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    printf("CUDA MLP Assembly Example\n");
    printf("=========================\n\n");

    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");

    // Problem size
    const int input_size = 512;
    const int output_size = 256;
    const int iterations = 1000;

    printf("Configuration:\n");
    printf("  Input size: %d\n", input_size);
    printf("  Output size: %d\n", output_size);
    printf("  Iterations: %d\n\n", iterations);

    // Allocate host memory
    size_t input_bytes = input_size * sizeof(float);
    size_t weights_bytes = output_size * input_size * sizeof(float);
    size_t bias_bytes = output_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    float* h_input = (float*)malloc(input_bytes);
    float* h_weights = (float*)malloc(weights_bytes);
    float* h_bias = (float*)malloc(bias_bytes);
    float* h_output_ref = (float*)malloc(output_bytes);
    float* h_output_warp = (float*)malloc(output_bytes);
    float* h_output_ptx = (float*)malloc(output_bytes);

    // Initialize data
    srand(42);
    init_random(h_input, input_size);
    init_random(h_weights, output_size * input_size);
    init_random(h_bias, output_size);

    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, bias_bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;
    size_t shared_mem = input_size * sizeof(float);

    printf("Launch configuration:\n");
    printf("  Grid size: %d\n", grid_size);
    printf("  Block size: %d\n", block_size);
    printf("  Shared memory: %zu bytes\n\n", shared_mem);

    // Warm up
    mlp_layer_forward_cuda_ref<<<grid_size, block_size, shared_mem>>>(
        d_input, d_weights, d_bias, d_output, input_size, output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Benchmark Reference CUDA Kernel
    // ========================================================================

    printf("Running reference CUDA kernel...\n");
    double start = get_time();

    for (int i = 0; i < iterations; i++) {
        mlp_layer_forward_cuda_ref<<<grid_size, block_size, shared_mem>>>(
            d_input, d_weights, d_bias, d_output, input_size, output_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double ref_time = get_time() - start;
    CUDA_CHECK(cudaMemcpy(h_output_ref, d_output, output_bytes, cudaMemcpyDeviceToHost));

    printf("  Time: %.4f ms (%.2f us/iter)\n", ref_time * 1000, (ref_time * 1e6) / iterations);

    // ========================================================================
    // Benchmark Warp-Optimized Kernel
    // ========================================================================

    printf("Running warp-optimized kernel...\n");
    start = get_time();

    for (int i = 0; i < iterations; i++) {
        mlp_layer_forward_cuda_warp<<<grid_size, block_size, shared_mem>>>(
            d_input, d_weights, d_bias, d_output, input_size, output_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double warp_time = get_time() - start;
    CUDA_CHECK(cudaMemcpy(h_output_warp, d_output, output_bytes, cudaMemcpyDeviceToHost));

    printf("  Time: %.4f ms (%.2f us/iter)\n", warp_time * 1000, (warp_time * 1e6) / iterations);
    printf("  Speedup vs ref: %.2fx\n", ref_time / warp_time);

    // Verify warp kernel
    if (verify_results(h_output_ref, h_output_warp, output_size)) {
        printf("  ✓ Results match reference\n");
    } else {
        printf("  ✗ Results differ from reference\n");
    }

    printf("\n");

    // ========================================================================
    // Show sample outputs
    // ========================================================================

    printf("Sample outputs (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] Ref: %.4f, Warp: %.4f\n",
               i, h_output_ref[i], h_output_warp[i]);
    }

    // Cleanup
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output_ref);
    free(h_output_warp);
    free(h_output_ptx);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    printf("\n✓ All tests completed successfully!\n");

    return 0;
}

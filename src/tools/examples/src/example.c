/**
 * MLP Assembly Example
 *
 * This example demonstrates how to use the assembly-optimized MLP functions
 * for different architectures.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/mlp_asm.h"

/* Helper function to initialize random weights */
void init_random_weights(float* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Range: [-1, 1]
    }
}

/* Helper function to print vector */
void print_vector(const char* name, const float* vec, int size) {
    printf("%s: [", name);
    for (int i = 0; i < size && i < 10; i++) {
        printf("%.4f", vec[i]);
        if (i < size - 1 && i < 9) printf(", ");
    }
    if (size > 10) printf(", ... (%d total)", size);
    printf("]\n");
}

/* Reference C implementation for correctness checking */
void mlp_layer_forward_reference(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
) {
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }
        sum += bias[i];
        output[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }
}

/* Check if two vectors are approximately equal */
int vectors_equal(const float* a, const float* b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return 0;
        }
    }
    return 1;
}


/* ============================================================================
 * Example 1: Simple Single Layer
 * ============================================================================ */
void example_single_layer() {
    printf("\n=== Example 1: Single Layer Forward Pass ===\n");

    const int input_size = 16;
    const int output_size = 8;

    // Allocate memory
    float* input = (float*)aligned_alloc(32, input_size * sizeof(float));
    float* weights = (float*)aligned_alloc(32, output_size * input_size * sizeof(float));
    float* bias = (float*)aligned_alloc(32, output_size * sizeof(float));
    float* output = (float*)aligned_alloc(32, output_size * sizeof(float));
    float* output_ref = (float*)aligned_alloc(32, output_size * sizeof(float));

    // Initialize data
    srand(42);
    init_random_weights(input, input_size);
    init_random_weights(weights, output_size * input_size);
    init_random_weights(bias, output_size);

    print_vector("Input", input, input_size);

    // Run assembly implementation
#if defined(MLP_ARCH_X86_64)
    printf("\nUsing x86-64 AVX/SSE implementation\n");
    mlp_layer_forward(input, weights, bias, output, input_size, output_size);
#elif defined(MLP_ARCH_ARM64) || defined(MLP_ARCH_ARM32)
    printf("\nUsing ARM NEON implementation\n");
    mlp_layer_forward(input, weights, bias, output, input_size, output_size);
#else
    printf("\nNo optimized implementation available, using reference\n");
    mlp_layer_forward_reference(input, weights, bias, output, input_size, output_size);
#endif

    // Run reference implementation
    mlp_layer_forward_reference(input, weights, bias, output_ref, input_size, output_size);

    print_vector("Output (ASM)", output, output_size);
    print_vector("Output (REF)", output_ref, output_size);

    // Verify correctness
    if (vectors_equal(output, output_ref, output_size, 1e-5f)) {
        printf("✓ Results match reference implementation!\n");
    } else {
        printf("✗ Results differ from reference implementation!\n");
    }

    // Cleanup
    free(input);
    free(weights);
    free(bias);
    free(output);
    free(output_ref);
}


/* ============================================================================
 * Example 2: Performance Benchmark
 * ============================================================================ */
void example_benchmark() {
    printf("\n=== Example 2: Performance Benchmark ===\n");

    const int input_size = 512;
    const int output_size = 256;
    const int iterations = 10000;

    // Allocate memory
    float* input = (float*)aligned_alloc(32, input_size * sizeof(float));
    float* weights = (float*)aligned_alloc(32, output_size * input_size * sizeof(float));
    float* bias = (float*)aligned_alloc(32, output_size * sizeof(float));
    float* output = (float*)aligned_alloc(32, output_size * sizeof(float));

    // Initialize data
    srand(42);
    init_random_weights(input, input_size);
    init_random_weights(weights, output_size * input_size);
    init_random_weights(bias, output_size);

    printf("Configuration: %d inputs -> %d outputs\n", input_size, output_size);
    printf("Running %d iterations...\n", iterations);

    // Benchmark assembly implementation
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
#if defined(MLP_ARCH_X86_64)
        mlp_layer_forward_avx(input, weights, bias, output, input_size, output_size);
#elif defined(MLP_ARCH_ARM64) || defined(MLP_ARCH_ARM32)
        mlp_layer_forward(input, weights, bias, output, input_size, output_size);
#else
        mlp_layer_forward_reference(input, weights, bias, output, input_size, output_size);
#endif
    }
    clock_t end = clock();
    double asm_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark reference implementation
    start = clock();
    for (int i = 0; i < iterations; i++) {
        mlp_layer_forward_reference(input, weights, bias, output, input_size, output_size);
    }
    end = clock();
    double ref_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nResults:\n");
    printf("  Assembly time: %.4f seconds (%.2f µs/iter)\n",
           asm_time, (asm_time * 1e6) / iterations);
    printf("  Reference time: %.4f seconds (%.2f µs/iter)\n",
           ref_time, (ref_time * 1e6) / iterations);
    printf("  Speedup: %.2fx\n", ref_time / asm_time);

    // Cleanup
    free(input);
    free(weights);
    free(bias);
    free(output);
}


/* ============================================================================
 * Example 3: Multi-layer Network
 * ============================================================================ */
void example_multilayer() {
    printf("\n=== Example 3: 2-Layer Neural Network ===\n");

    const int input_size = 784;   // e.g., 28x28 MNIST image
    const int hidden_size = 128;
    const int output_size = 10;   // 10 classes

    printf("Network architecture: %d -> %d -> %d\n",
           input_size, hidden_size, output_size);

    // Allocate memory
    float* input = (float*)aligned_alloc(32, input_size * sizeof(float));
    float* weights1 = (float*)aligned_alloc(32, hidden_size * input_size * sizeof(float));
    float* bias1 = (float*)aligned_alloc(32, hidden_size * sizeof(float));
    float* weights2 = (float*)aligned_alloc(32, output_size * hidden_size * sizeof(float));
    float* bias2 = (float*)aligned_alloc(32, output_size * sizeof(float));
    float* hidden = (float*)aligned_alloc(32, hidden_size * sizeof(float));
    float* output = (float*)aligned_alloc(32, output_size * sizeof(float));

    // Initialize with random weights
    srand(42);
    init_random_weights(input, input_size);
    init_random_weights(weights1, hidden_size * input_size);
    init_random_weights(bias1, hidden_size);
    init_random_weights(weights2, output_size * hidden_size);
    init_random_weights(bias2, output_size);

    // Forward pass through both layers
    printf("\nRunning 2-layer forward pass...\n");

#if defined(MLP_ARCH_X86_64)
    mlp_layer_forward_avx(input, weights1, bias1, hidden, input_size, hidden_size);
    mlp_layer_forward_avx(hidden, weights2, bias2, output, hidden_size, output_size);
#elif defined(MLP_ARCH_ARM64) || defined(MLP_ARCH_ARM32)
    mlp_layer_forward(input, weights1, bias1, hidden, input_size, hidden_size);
    mlp_layer_forward(hidden, weights2, bias2, output, hidden_size, output_size);
#else
    mlp_layer_forward_reference(input, weights1, bias1, hidden, input_size, hidden_size);
    mlp_layer_forward_reference(hidden, weights2, bias2, output, hidden_size, output_size);
#endif

    print_vector("Hidden layer", hidden, hidden_size);
    print_vector("Output layer", output, output_size);

    // Find predicted class (argmax)
    int predicted_class = 0;
    float max_score = output[0];
    for (int i = 1; i < output_size; i++) {
        if (output[i] > max_score) {
            max_score = output[i];
            predicted_class = i;
        }
    }
    printf("\nPredicted class: %d (score: %.4f)\n", predicted_class, max_score);

    // Cleanup
    free(input);
    free(weights1);
    free(bias1);
    free(weights2);
    free(bias2);
    free(hidden);
    free(output);
}


/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char** argv) {
    printf("MLP Assembly Implementation Examples\n");
    printf("=====================================\n");

#if defined(MLP_ARCH_X86_64)
    printf("Architecture: x86-64\n");
    #if defined(__AVX__)
        printf("SIMD: AVX (256-bit, 8 floats)\n");
    #else
        printf("SIMD: SSE (128-bit, 4 floats)\n");
    #endif
#elif defined(MLP_ARCH_ARM64)
    printf("Architecture: ARM AArch64\n");
    printf("SIMD: NEON (128-bit, 4 floats)\n");
#elif defined(MLP_ARCH_ARM32)
    printf("Architecture: ARM ARMv7 (32-bit)\n");
    printf("SIMD: NEON (128-bit, 4 floats)\n");
#else
    printf("Architecture: Unknown (using reference C implementation)\n");
#endif

    // Run examples
    example_single_layer();
    example_benchmark();
    example_multilayer();

    printf("\n=== All examples completed! ===\n");
    return 0;
}

/**
 * MLP Assembly Implementation Header
 *
 * This header provides declarations for MLP (Multi-Layer Perceptron) functions
 * implemented in assembly for different architectures and SIMD instruction sets.
 */

#ifndef MLP_ASM_H
#define MLP_ASM_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * x86-64 AVX/SSE Functions
 * ============================================================================ */

/**
 * Single MLP layer forward pass using AVX instructions (256-bit, 8 floats)
 *
 * Computes: output = ReLU(weights × input + bias)
 *
 * @param input      Input vector of size input_size
 * @param weights    Weight matrix of size [output_size × input_size] (row-major)
 * @param bias       Bias vector of size output_size
 * @param output     Output vector of size output_size (result)
 * @param input_size Number of input features
 * @param output_size Number of output neurons
 */
void mlp_layer_forward_avx(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Single MLP layer forward pass using SSE instructions (128-bit, 4 floats)
 *
 * Same functionality as AVX version but uses SSE for compatibility
 *
 * @param input      Input vector of size input_size
 * @param weights    Weight matrix of size [output_size × input_size] (row-major)
 * @param bias       Bias vector of size output_size
 * @param output     Output vector of size output_size (result)
 * @param input_size Number of input features
 * @param output_size Number of output neurons
 */
void mlp_layer_forward_sse(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Complete 2-layer MLP forward pass using AVX
 *
 * @param input       Input vector
 * @param weights1    First layer weights
 * @param bias1       First layer bias
 * @param weights2    Second layer weights
 * @param bias2       Second layer bias
 * @param output      Final output vector
 * @param hidden      Hidden layer buffer (must be pre-allocated)
 * @param input_size  Input dimensions
 * @param hidden_size Hidden layer dimensions
 * @param output_size Output dimensions
 */
void mlp_forward_2layer_avx(
    const float* input,
    const float* weights1,
    const float* bias1,
    const float* weights2,
    const float* bias2,
    float* output,
    float* hidden,
    int input_size,
    int hidden_size,
    int output_size
);


/* ============================================================================
 * ARM AArch64 NEON Functions
 * ============================================================================ */

/**
 * Single MLP layer forward pass using ARM NEON instructions (128-bit, 4 floats)
 *
 * @param input      Input vector of size input_size
 * @param weights    Weight matrix of size [output_size × input_size] (row-major)
 * @param bias       Bias vector of size output_size
 * @param output     Output vector of size output_size (result)
 * @param input_size Number of input features
 * @param output_size Number of output neurons
 */
void mlp_layer_forward_neon(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Optimized version with loop unrolling and better register reuse
 */
void mlp_layer_forward_neon_optimized(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Complete 2-layer MLP forward pass using NEON
 */
void mlp_forward_2layer_neon(
    const float* input,
    const float* weights1,
    const float* bias1,
    const float* weights2,
    const float* bias2,
    float* output,
    float* hidden,
    int input_size,
    int hidden_size,
    int output_size
);


/* ============================================================================
 * ARM ARMv7 (32-bit) NEON Functions
 * ============================================================================ */

/**
 * Single MLP layer forward pass for ARMv7 using NEON
 */
void mlp_layer_forward_neon_v7(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Optimized ARMv7 version with 2x unrolling
 */
void mlp_layer_forward_neon_v7_unrolled(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size
);

/**
 * Apply ReLU activation in-place to a vector
 */
void vector_relu_neon(
    float* vector,
    int size
);


/* ============================================================================
 * Architecture Detection Macros
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64)
    #define MLP_ARCH_X86_64
    #if defined(__AVX__)
        #define mlp_layer_forward mlp_layer_forward_avx
    #else
        #define mlp_layer_forward mlp_layer_forward_sse
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define MLP_ARCH_ARM64
    #define mlp_layer_forward mlp_layer_forward_neon_optimized
#elif defined(__arm__) || defined(_M_ARM)
    #define MLP_ARCH_ARM32
    #define mlp_layer_forward mlp_layer_forward_neon_v7_unrolled
#else
    #warning "Unsupported architecture - no optimized MLP implementation available"
#endif

#ifdef __cplusplus
}
#endif

#endif /* MLP_ASM_H */

// ARM AArch64 MLP Implementation using NEON SIMD Instructions
// This implements a simple MLP layer forward pass with ReLU activation
// Processes 4 single-precision floats (32-bit) at a time using 128-bit NEON registers

.arch armv8-a
.text
.align 2

// Function: mlp_layer_forward_neon
// Computes: output = ReLU(weights * input + bias)
//
// Arguments (AAPCS64):
//   x0: float* input      - input vector
//   x1: float* weights    - weight matrix (row-major)
//   x2: float* bias       - bias vector
//   x3: float* output     - output vector
//   x4: int input_size    - number of input features
//   x5: int output_size   - number of output neurons
//
// Register usage:
//   v0-v7: SIMD computation registers
//   x6: output neuron index
//   x7: input feature index
//   x8: weight matrix row pointer
//   x9: temporary/loop counter

.global mlp_layer_forward_neon
.type mlp_layer_forward_neon, %function
mlp_layer_forward_neon:
    // Save callee-saved registers
    stp x19, x20, [sp, #-64]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp d8, d9, [sp, #48]

    // Save arguments in callee-saved registers
    mov x19, x0                 // x19 = input
    mov x20, x1                 // x20 = weights
    mov x21, x2                 // x21 = bias
    mov x22, x3                 // x22 = output
    mov x23, x4                 // x23 = input_size
    mov x24, x5                 // x24 = output_size

    mov x6, #0                  // x6 = 0 (output neuron index)

output_loop:
    cmp x6, x24
    b.ge done                   // if output_idx >= output_size, done

    // Initialize accumulator to zero
    movi v0.4s, #0              // v0 = [0.0, 0.0, 0.0, 0.0]

    // Calculate weight row pointer: &weights[output_idx * input_size]
    mul x8, x6, x23             // x8 = output_idx * input_size
    lsl x8, x8, #2              // x8 *= 4 (sizeof(float))
    add x8, x20, x8             // x8 = &weights[output_idx * input_size]

    mov x7, #0                  // x7 = 0 (input feature index)

    // Calculate number of SIMD iterations (input_size / 4)
    lsr x9, x23, #2             // x9 = input_size / 4
    cbz x9, remainder           // if x9 == 0, skip SIMD loop

simd_loop:
    cbz x9, remainder           // if x9 == 0, done with SIMD

    // Load 4 input values
    ldr q1, [x19, x7, lsl #2]   // v1 = input[i:i+4]

    // Load 4 weight values
    ldr q2, [x8, x7, lsl #2]    // v2 = weights[output_idx][i:i+4]

    // Multiply and accumulate: v0 += v1 * v2
    fmla v0.4s, v1.4s, v2.4s    // FMA: v0 = v0 + (v1 * v2)

    add x7, x7, #4              // i += 4
    sub x9, x9, #1              // decrement loop counter
    b simd_loop

remainder:
    // Handle remaining elements (input_size % 4)
    and x9, x23, #3             // x9 = input_size % 4
    cbz x9, add_bias

scalar_loop:
    cbz x9, add_bias

    // Load single float from input
    ldr s1, [x19, x7, lsl #2]   // s1 = input[i]

    // Load single float from weights
    ldr s2, [x8, x7, lsl #2]    // s2 = weights[output_idx][i]

    // Multiply and accumulate
    fmla s0, s1, s2             // s0 += s1 * s2

    add x7, x7, #1
    sub x9, x9, #1
    b scalar_loop

add_bias:
    // Horizontal sum of v0 to get final dot product
    faddp v0.4s, v0.4s, v0.4s   // pairwise add: [a+b, c+d, a+b, c+d]
    faddp v0.4s, v0.4s, v0.4s   // pairwise add: [(a+b)+(c+d), ...]

    // Add bias
    ldr s1, [x21, x6, lsl #2]   // s1 = bias[output_idx]
    fadd s0, s0, s1             // s0 += bias

    // Apply ReLU: max(0, x)
    movi v1.4s, #0              // v1 = 0.0
    fmax s0, s0, s1             // s0 = max(s0, 0.0)

    // Store result
    str s0, [x22, x6, lsl #2]   // output[output_idx] = s0

    add x6, x6, #1
    b output_loop

done:
    // Restore callee-saved registers
    ldp d8, d9, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #64
    ret

.size mlp_layer_forward_neon, .-mlp_layer_forward_neon


// Function: mlp_layer_forward_neon_optimized
// Optimized version with loop unrolling and register reuse
// Same interface as above

.global mlp_layer_forward_neon_optimized
.type mlp_layer_forward_neon_optimized, %function
mlp_layer_forward_neon_optimized:
    stp x19, x20, [sp, #-64]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp d8, d9, [sp, #48]

    mov x19, x0                 // input
    mov x20, x1                 // weights
    mov x21, x2                 // bias
    mov x22, x3                 // output
    mov x23, x4                 // input_size
    mov x24, x5                 // output_size

    mov x6, #0

opt_output_loop:
    cmp x6, x24
    b.ge opt_done

    // Use two accumulators for better instruction-level parallelism
    movi v0.4s, #0
    movi v1.4s, #0

    mul x8, x6, x23
    lsl x8, x8, #2
    add x8, x20, x8

    mov x7, #0

    // Process 8 floats per iteration (2 SIMD ops)
    lsr x9, x23, #3             // x9 = input_size / 8
    cbz x9, opt_simd_4

opt_simd_8_loop:
    cbz x9, opt_simd_4

    // First 4 elements
    ldr q2, [x19, x7, lsl #2]
    ldr q3, [x8, x7, lsl #2]
    fmla v0.4s, v2.4s, v3.4s
    add x7, x7, #4

    // Next 4 elements
    ldr q4, [x19, x7, lsl #2]
    ldr q5, [x8, x7, lsl #2]
    fmla v1.4s, v4.4s, v5.4s
    add x7, x7, #4

    sub x9, x9, #1
    b opt_simd_8_loop

opt_simd_4:
    // Handle remaining 4-element chunks
    and x9, x23, #7             // x9 = input_size % 8
    cmp x9, #4
    b.lt opt_remainder

    ldr q2, [x19, x7, lsl #2]
    ldr q3, [x8, x7, lsl #2]
    fmla v0.4s, v2.4s, v3.4s
    add x7, x7, #4
    sub x9, x9, #4

opt_remainder:
    cbz x9, opt_add_bias

opt_scalar_loop:
    cbz x9, opt_add_bias

    ldr s2, [x19, x7, lsl #2]
    ldr s3, [x8, x7, lsl #2]
    fmla s0, s2, s3

    add x7, x7, #1
    sub x9, x9, #1
    b opt_scalar_loop

opt_add_bias:
    // Combine accumulators
    fadd v0.4s, v0.4s, v1.4s

    // Horizontal sum
    faddp v0.4s, v0.4s, v0.4s
    faddp v0.4s, v0.4s, v0.4s

    // Add bias and apply ReLU
    ldr s1, [x21, x6, lsl #2]
    fadd s0, s0, s1
    movi v1.4s, #0
    fmax s0, s0, s1

    str s0, [x22, x6, lsl #2]

    add x6, x6, #1
    b opt_output_loop

opt_done:
    ldp d8, d9, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #64
    ret

.size mlp_layer_forward_neon_optimized, .-mlp_layer_forward_neon_optimized


// Function: mlp_forward_2layer_neon
// Complete 2-layer MLP forward pass
//
// Arguments:
//   x0: float* input
//   x1: float* weights1
//   x2: float* bias1
//   x3: float* weights2
//   x4: float* bias2
//   x5: float* output
//   x6: float* hidden        - hidden layer buffer
//   x7: int input_size
// Stack arguments:
//   [sp]: int hidden_size
//   [sp+8]: int output_size

.global mlp_forward_2layer_neon
.type mlp_forward_2layer_neon, %function
mlp_forward_2layer_neon:
    stp x29, x30, [sp, #-96]!
    mov x29, sp

    // Save all arguments
    stp x0, x1, [sp, #16]       // input, weights1
    stp x2, x3, [sp, #32]       // bias1, weights2
    stp x4, x5, [sp, #48]       // bias2, output
    stp x6, x7, [sp, #64]       // hidden, input_size
    ldr x8, [x29, #96]          // hidden_size from stack
    ldr x9, [x29, #104]         // output_size from stack
    stp x8, x9, [sp, #80]       // save on our stack

    // First layer: input -> hidden
    // mlp_layer_forward_neon(input, weights1, bias1, hidden, input_size, hidden_size)
    mov x3, x6                  // output = hidden buffer
    mov x4, x7                  // input_size
    mov x5, x8                  // output_size = hidden_size
    bl mlp_layer_forward_neon

    // Second layer: hidden -> output
    // mlp_layer_forward_neon(hidden, weights2, bias2, output, hidden_size, output_size)
    ldp x6, x7, [sp, #64]       // reload hidden, input_size
    ldp x3, x4, [sp, #32]       // reload bias1, weights2
    ldp x5, x0, [sp, #48]       // reload bias2, output - wait, need to fix this
    ldr x1, [sp, #40]           // weights2
    ldr x2, [sp, #56]           // bias2
    ldr x3, [sp, #48]           // output
    ldp x8, x9, [sp, #80]       // hidden_size, output_size

    mov x0, x6                  // input = hidden
    mov x4, x8                  // input_size = hidden_size
    mov x5, x9                  // output_size
    bl mlp_layer_forward_neon

    ldp x29, x30, [sp], #96
    ret

.size mlp_forward_2layer_neon, .-mlp_forward_2layer_neon

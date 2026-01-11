@ ARM ARMv7 (32-bit) MLP Implementation using NEON SIMD Instructions
@ This implements a simple MLP layer forward pass with ReLU activation
@ Processes 4 single-precision floats (32-bit) at a time using 128-bit NEON registers

.syntax unified
.arch armv7-a
.fpu neon
.text
.align 2

@ Function: mlp_layer_forward_neon_v7
@ Computes: output = ReLU(weights * input + bias)
@
@ Arguments (AAPCS):
@   r0: float* input      - input vector
@   r1: float* weights    - weight matrix (row-major)
@   r2: float* bias       - bias vector
@   r3: float* output     - output vector
@ Stack arguments:
@   [sp]: int input_size  - number of input features
@   [sp+4]: int output_size - number of output neurons
@
@ Register usage:
@   q0-q7: SIMD computation registers (d0-d15)
@   r4: output neuron index
@   r5: input feature index
@   r6: weight matrix row pointer
@   r7: temporary/loop counter
@   r8: input_size
@   r9: output_size

.global mlp_layer_forward_neon_v7
.type mlp_layer_forward_neon_v7, %function
mlp_layer_forward_neon_v7:
    push {r4-r11, lr}
    vpush {d8-d15}              @ Save NEON callee-saved registers

    @ Load stack arguments
    ldr r8, [sp, #100]          @ r8 = input_size (36 bytes regs + 64 bytes neon)
    ldr r9, [sp, #104]          @ r9 = output_size

    @ Save input arguments
    mov r4, r0                  @ r4 = input
    mov r5, r1                  @ r5 = weights
    mov r6, r2                  @ r6 = bias
    mov r7, r3                  @ r7 = output

    mov r10, #0                 @ r10 = 0 (output neuron index)

.L_output_loop:
    cmp r10, r9
    bge .L_done                 @ if output_idx >= output_size, done

    @ Initialize accumulator to zero
    vmov.f32 q0, #0.0           @ q0 = [0.0, 0.0, 0.0, 0.0]

    @ Calculate weight row pointer: &weights[output_idx * input_size]
    mul r11, r10, r8            @ r11 = output_idx * input_size
    lsl r11, r11, #2            @ r11 *= 4 (sizeof(float))
    add r11, r5, r11            @ r11 = &weights[output_idx * input_size]

    mov r0, #0                  @ r0 = 0 (input feature index)

    @ Calculate number of SIMD iterations (input_size / 4)
    lsr r1, r8, #2              @ r1 = input_size / 4
    cmp r1, #0
    beq .L_remainder

.L_simd_loop:
    cmp r1, #0
    beq .L_remainder

    @ Load 4 input values
    add r2, r4, r0, lsl #2      @ r2 = &input[i]
    vld1.32 {q1}, [r2]          @ q1 = input[i:i+4]

    @ Load 4 weight values
    add r2, r11, r0, lsl #2     @ r2 = &weights[output_idx][i]
    vld1.32 {q2}, [r2]          @ q2 = weights[output_idx][i:i+4]

    @ Multiply and accumulate: q0 += q1 * q2
    vmla.f32 q0, q1, q2         @ q0 = q0 + (q1 * q2)

    add r0, r0, #4              @ i += 4
    sub r1, r1, #1
    b .L_simd_loop

.L_remainder:
    @ Handle remaining elements (input_size % 4)
    and r1, r8, #3              @ r1 = input_size % 4
    cmp r1, #0
    beq .L_add_bias

.L_scalar_loop:
    cmp r1, #0
    beq .L_add_bias

    @ Load single float from input
    add r2, r4, r0, lsl #2
    vldr s4, [r2]               @ s4 = input[i]

    @ Load single float from weights
    add r2, r11, r0, lsl #2
    vldr s5, [r2]               @ s5 = weights[output_idx][i]

    @ Multiply and accumulate
    vmla.f32 s0, s4, s5         @ s0 += s4 * s5

    add r0, r0, #1
    sub r1, r1, #1
    b .L_scalar_loop

.L_add_bias:
    @ Horizontal sum of q0 to get final dot product
    @ q0 = [a, b, c, d]
    vadd.f32 d0, d0, d1         @ d0 = [a+c, b+d]
    vpadd.f32 d0, d0, d0        @ d0 = [a+b+c+d, a+b+c+d]

    @ Add bias
    add r2, r6, r10, lsl #2
    vldr s1, [r2]               @ s1 = bias[output_idx]
    vadd.f32 s0, s0, s1         @ s0 += bias

    @ Apply ReLU: max(0, x)
    vmov.f32 s1, #0.0           @ s1 = 0.0
    vmax.f32 s0, s0, s1         @ s0 = max(s0, 0.0)

    @ Store result
    add r2, r7, r10, lsl #2
    vstr s0, [r2]               @ output[output_idx] = s0

    add r10, r10, #1
    b .L_output_loop

.L_done:
    @ Restore registers
    vpop {d8-d15}
    pop {r4-r11, pc}

.size mlp_layer_forward_neon_v7, .-mlp_layer_forward_neon_v7


@ Function: mlp_layer_forward_neon_v7_unrolled
@ Optimized version with 2x loop unrolling
@ Same interface as above

.global mlp_layer_forward_neon_v7_unrolled
.type mlp_layer_forward_neon_v7_unrolled, %function
mlp_layer_forward_neon_v7_unrolled:
    push {r4-r11, lr}
    vpush {d8-d15}

    ldr r8, [sp, #100]          @ input_size
    ldr r9, [sp, #104]          @ output_size

    mov r4, r0                  @ input
    mov r5, r1                  @ weights
    mov r6, r2                  @ bias
    mov r7, r3                  @ output

    mov r10, #0                 @ output index

.L_unroll_output_loop:
    cmp r10, r9
    bge .L_unroll_done

    @ Two accumulators for better ILP
    vmov.f32 q0, #0.0
    vmov.f32 q1, #0.0

    mul r11, r10, r8
    lsl r11, r11, #2
    add r11, r5, r11

    mov r0, #0

    @ Process 8 floats per iteration
    lsr r1, r8, #3              @ r1 = input_size / 8
    cmp r1, #0
    beq .L_unroll_simd_4

.L_unroll_simd_8:
    cmp r1, #0
    beq .L_unroll_simd_4

    @ First 4 elements
    add r2, r4, r0, lsl #2
    vld1.32 {q2}, [r2]
    add r2, r11, r0, lsl #2
    vld1.32 {q3}, [r2]
    vmla.f32 q0, q2, q3
    add r0, r0, #4

    @ Next 4 elements
    add r2, r4, r0, lsl #2
    vld1.32 {q4}, [r2]
    add r2, r11, r0, lsl #2
    vld1.32 {q5}, [r2]
    vmla.f32 q1, q4, q5
    add r0, r0, #4

    sub r1, r1, #1
    b .L_unroll_simd_8

.L_unroll_simd_4:
    @ Handle remaining 4-element chunk
    and r1, r8, #7
    cmp r1, #4
    blt .L_unroll_remainder

    add r2, r4, r0, lsl #2
    vld1.32 {q2}, [r2]
    add r2, r11, r0, lsl #2
    vld1.32 {q3}, [r2]
    vmla.f32 q0, q2, q3
    add r0, r0, #4
    sub r1, r1, #4

.L_unroll_remainder:
    cmp r1, #0
    beq .L_unroll_add_bias

.L_unroll_scalar:
    cmp r1, #0
    beq .L_unroll_add_bias

    add r2, r4, r0, lsl #2
    vldr s4, [r2]
    add r2, r11, r0, lsl #2
    vldr s5, [r2]
    vmla.f32 s0, s4, s5

    add r0, r0, #1
    sub r1, r1, #1
    b .L_unroll_scalar

.L_unroll_add_bias:
    @ Combine accumulators
    vadd.f32 q0, q0, q1

    @ Horizontal sum
    vadd.f32 d0, d0, d1
    vpadd.f32 d0, d0, d0

    @ Add bias and ReLU
    add r2, r6, r10, lsl #2
    vldr s1, [r2]
    vadd.f32 s0, s0, s1
    vmov.f32 s1, #0.0
    vmax.f32 s0, s0, s1

    add r2, r7, r10, lsl #2
    vstr s0, [r2]

    add r10, r10, #1
    b .L_unroll_output_loop

.L_unroll_done:
    vpop {d8-d15}
    pop {r4-r11, pc}

.size mlp_layer_forward_neon_v7_unrolled, .-mlp_layer_forward_neon_v7_unrolled


@ Function: vector_relu_neon
@ Applies ReLU activation to a vector in-place
@
@ Arguments:
@   r0: float* vector
@   r1: int size

.global vector_relu_neon
.type vector_relu_neon, %function
vector_relu_neon:
    vmov.f32 q1, #0.0           @ q1 = zero vector for comparison
    mov r2, #0                  @ index

    @ Process 4 elements at a time
    lsr r3, r1, #2
    cmp r3, #0
    beq .L_relu_remainder

.L_relu_simd:
    cmp r3, #0
    beq .L_relu_remainder

    add r12, r0, r2, lsl #2
    vld1.32 {q0}, [r12]         @ load 4 elements
    vmax.f32 q0, q0, q1         @ max(x, 0)
    vst1.32 {q0}, [r12]         @ store back

    add r2, r2, #4
    sub r3, r3, #1
    b .L_relu_simd

.L_relu_remainder:
    and r3, r1, #3
    cmp r3, #0
    beq .L_relu_done

.L_relu_scalar:
    cmp r3, #0
    beq .L_relu_done

    add r12, r0, r2, lsl #2
    vldr s0, [r12]
    vmax.f32 s0, s0, s4         @ s4 is low part of q1 (0.0)
    vstr s0, [r12]

    add r2, r2, #1
    sub r3, r3, #1
    b .L_relu_scalar

.L_relu_done:
    bx lr

.size vector_relu_neon, .-vector_relu_neon

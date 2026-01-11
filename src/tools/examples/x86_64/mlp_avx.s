# x86-64 MLP Implementation using AVX SIMD Instructions
# This implements a simple MLP layer forward pass with ReLU activation
# Processes 8 single-precision floats (32-bit) at a time using 256-bit AVX registers

.intel_syntax noprefix
.text

# Function: mlp_layer_forward_avx
# Computes: output = ReLU(weights * input + bias)
#
# Arguments (System V AMD64 ABI):
#   rdi: float* input      - input vector
#   rsi: float* weights    - weight matrix (row-major)
#   rdx: float* bias       - bias vector
#   rcx: float* output     - output vector
#   r8:  int input_size    - number of input features
#   r9:  int output_size   - number of output neurons
#
# Register usage:
#   ymm0-ymm7: SIMD computation registers
#   rax: output neuron index
#   r10: input feature index
#   r11: weight matrix offset

.globl mlp_layer_forward_avx
.type mlp_layer_forward_avx, @function
mlp_layer_forward_avx:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15

    # Save arguments
    mov r12, rdi           # r12 = input
    mov r13, rsi           # r13 = weights
    mov r14, rdx           # r14 = bias
    mov r15, rcx           # r15 = output

    # r8 already has input_size
    # r9 already has output_size

    xor rax, rax           # rax = 0 (output neuron index)

.output_loop:
    cmp rax, r9
    jge .done              # if output_idx >= output_size, done

    # Initialize accumulator to zero
    vxorps ymm0, ymm0, ymm0    # ymm0 = 0.0 (accumulator for dot product)

    # Calculate weight row offset: weights[output_idx * input_size]
    mov r11, rax
    imul r11, r8           # r11 = output_idx * input_size
    shl r11, 2             # r11 *= 4 (sizeof(float))
    add r11, r13           # r11 = &weights[output_idx * input_size]

    xor r10, r10           # r10 = 0 (input feature index)

    # Main SIMD loop - process 8 floats at a time
.simd_loop:
    mov rcx, r8
    sub rcx, r10           # rcx = remaining elements
    cmp rcx, 8
    jl .remainder          # if less than 8 elements, handle remainder

    # Load 8 input values
    vmovups ymm1, [r12 + r10*4]      # ymm1 = input[i:i+8]

    # Load 8 weight values
    vmovups ymm2, [r11 + r10*4]      # ymm2 = weights[output_idx][i:i+8]

    # Multiply and accumulate: ymm0 += ymm1 * ymm2
    vfmadd231ps ymm0, ymm1, ymm2     # FMA: ymm0 = ymm0 + (ymm1 * ymm2)

    add r10, 8
    jmp .simd_loop

.remainder:
    # Handle remaining elements (less than 8)
    cmp r10, r8
    jge .add_bias

    # Scalar loop for remaining elements
.scalar_loop:
    cmp r10, r8
    jge .add_bias

    vmovss xmm1, [r12 + r10*4]       # xmm1 = input[i]
    vmovss xmm2, [r11 + r10*4]       # xmm2 = weights[output_idx][i]
    vfmadd231ss xmm0, xmm1, xmm2     # xmm0 += xmm1 * xmm2

    inc r10
    jmp .scalar_loop

.add_bias:
    # Horizontal sum of ymm0 to get final dot product
    vextractf128 xmm1, ymm0, 1       # xmm1 = upper 128 bits of ymm0
    vaddps xmm0, xmm0, xmm1          # xmm0 = lower + upper
    vhaddps xmm0, xmm0, xmm0         # horizontal add
    vhaddps xmm0, xmm0, xmm0         # horizontal add again

    # Add bias
    vmovss xmm1, [r14 + rax*4]       # xmm1 = bias[output_idx]
    vaddss xmm0, xmm0, xmm1          # xmm0 += bias

    # Apply ReLU: max(0, x)
    vxorps xmm1, xmm1, xmm1          # xmm1 = 0.0
    vmaxss xmm0, xmm0, xmm1          # xmm0 = max(xmm0, 0.0)

    # Store result
    vmovss [r15 + rax*4], xmm0       # output[output_idx] = xmm0

    inc rax
    jmp .output_loop

.done:
    # Restore callee-saved registers
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

.size mlp_layer_forward_avx, .-mlp_layer_forward_avx


# Function: mlp_layer_forward_sse
# SSE version (128-bit) - processes 4 floats at a time
# Same interface as AVX version

.globl mlp_layer_forward_sse
.type mlp_layer_forward_sse, @function
mlp_layer_forward_sse:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi           # r12 = input
    mov r13, rsi           # r13 = weights
    mov r14, rdx           # r14 = bias
    mov r15, rcx           # r15 = output

    xor rax, rax           # rax = 0 (output neuron index)

.sse_output_loop:
    cmp rax, r9
    jge .sse_done

    # Initialize accumulator
    xorps xmm0, xmm0           # xmm0 = 0.0

    # Calculate weight row offset
    mov r11, rax
    imul r11, r8
    shl r11, 2
    add r11, r13

    xor r10, r10               # r10 = 0

    # Main SSE loop - process 4 floats at a time
.sse_simd_loop:
    mov rcx, r8
    sub rcx, r10
    cmp rcx, 4
    jl .sse_remainder

    # Load 4 input values
    movups xmm1, [r12 + r10*4]

    # Load 4 weight values
    movups xmm2, [r11 + r10*4]

    # Multiply
    mulps xmm1, xmm2

    # Accumulate
    addps xmm0, xmm1

    add r10, 4
    jmp .sse_simd_loop

.sse_remainder:
    cmp r10, r8
    jge .sse_add_bias

.sse_scalar_loop:
    cmp r10, r8
    jge .sse_add_bias

    movss xmm1, [r12 + r10*4]
    movss xmm2, [r11 + r10*4]
    mulss xmm1, xmm2
    addss xmm0, xmm1

    inc r10
    jmp .sse_scalar_loop

.sse_add_bias:
    # Horizontal sum of xmm0
    haddps xmm0, xmm0
    haddps xmm0, xmm0

    # Add bias
    movss xmm1, [r14 + rax*4]
    addss xmm0, xmm1

    # Apply ReLU
    xorps xmm1, xmm1
    maxss xmm0, xmm1

    # Store result
    movss [r15 + rax*4], xmm0

    inc rax
    jmp .sse_output_loop

.sse_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

.size mlp_layer_forward_sse, .-mlp_layer_forward_sse


# Function: mlp_forward_2layer_avx
# Complete 2-layer MLP forward pass
#
# Arguments:
#   rdi: float* input          - input vector
#   rsi: float* weights1       - first layer weights
#   rdx: float* bias1          - first layer bias
#   rcx: float* weights2       - second layer weights
#   r8:  float* bias2          - second layer bias
#   r9:  float* output         - final output vector
# Stack arguments:
#   [rbp+16]: float* hidden    - hidden layer buffer
#   [rbp+24]: int input_size   - input dimensions
#   [rbp+32]: int hidden_size  - hidden layer dimensions
#   [rbp+40]: int output_size  - output dimensions

.globl mlp_forward_2layer_avx
.type mlp_forward_2layer_avx, @function
mlp_forward_2layer_avx:
    push rbp
    mov rbp, rsp
    sub rsp, 64                # Allocate stack space

    # Save all arguments
    mov [rbp-8], rdi           # input
    mov [rbp-16], rsi          # weights1
    mov [rbp-24], rdx          # bias1
    mov [rbp-32], rcx          # weights2
    mov [rbp-40], r8           # bias2
    mov [rbp-48], r9           # output

    # First layer forward pass
    mov rdi, [rbp-8]           # input
    mov rsi, [rbp-16]          # weights1
    mov rdx, [rbp-24]          # bias1
    mov rcx, [rbp+16]          # hidden buffer
    mov r8, [rbp+24]           # input_size
    mov r9, [rbp+32]           # hidden_size
    call mlp_layer_forward_avx

    # Second layer forward pass
    mov rdi, [rbp+16]          # hidden (output from layer 1)
    mov rsi, [rbp-32]          # weights2
    mov rdx, [rbp-40]          # bias2
    mov rcx, [rbp-48]          # output
    mov r8, [rbp+32]           # hidden_size
    mov r9, [rbp+40]           # output_size
    call mlp_layer_forward_avx

    add rsp, 64
    pop rbp
    ret

.size mlp_forward_2layer_avx, .-mlp_forward_2layer_avx

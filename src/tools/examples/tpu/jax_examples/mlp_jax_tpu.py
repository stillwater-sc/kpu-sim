"""
JAX MLP Implementation for Google TPU

JAX compiles Python code to XLA HLO, which then compiles to TPU machine code.
This demonstrates how to write TPU-optimized neural network code.

Requirements:
    pip install jax[tpu] jaxlib
    # Or for CPU/GPU: pip install jax jaxlib

TPU Access:
    - Google Cloud TPU VMs
    - Google Colab (free TPU access)
    - Kaggle notebooks (free TPU)
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
import time
import numpy as np

# Check TPU availability
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())

# ============================================================================
# Basic MLP Layer
# ============================================================================

@jit  # JIT compile to XLA → HLO → TPU code
def mlp_layer_forward(input, weights, bias):
    """
    Single MLP layer forward pass with ReLU activation.

    Computes: output = ReLU(input @ weights.T + bias)

    Args:
        input: Array of shape [batch_size, input_size]
        weights: Array of shape [output_size, input_size]
        bias: Array of shape [output_size]

    Returns:
        output: Array of shape [batch_size, output_size]
    """
    # Matrix multiplication: uses TPU MXU (systolic array)
    # JAX/XLA will automatically tile this for the 128x128 array
    logits = jnp.dot(input, weights.T)  # [batch, output_size]

    # Broadcast and add bias: uses TPU VPU
    logits = logits + bias  # Broadcasting: [batch, out] + [out]

    # ReLU activation: uses TPU VPU
    output = jnp.maximum(0.0, logits)

    return output


# ============================================================================
# Optimized: Using lower precision (BF16)
# ============================================================================

@jit
def mlp_layer_forward_bf16(input, weights, bias):
    """
    BF16 (Brain Float 16) version for 2x throughput on TPU.

    TPU v2/v3/v4 have native BF16 support with same dynamic range as FP32
    but only 16 bits (7-bit exponent, 8-bit mantissa vs FP32's 23-bit mantissa)
    """
    # Convert to BF16
    input_bf16 = input.astype(jnp.bfloat16)
    weights_bf16 = weights.astype(jnp.bfloat16)
    bias_bf16 = bias.astype(jnp.bfloat16)

    # Compute in BF16 (2x faster on TPU)
    logits = jnp.dot(input_bf16, weights_bf16.T)
    logits = logits + bias_bf16
    output = jnp.maximum(0.0, logits)

    # Convert back to FP32 if needed
    return output.astype(jnp.float32)


# ============================================================================
# 2-Layer MLP
# ============================================================================

@jit
def mlp_2layer_forward(input, weights1, bias1, weights2, bias2):
    """Two-layer MLP"""
    hidden = mlp_layer_forward(input, weights1, bias1)
    output = mlp_layer_forward(hidden, weights2, bias2)
    return output


# ============================================================================
# Batched Processing (optimal for TPU)
# ============================================================================

@jit
def mlp_batched(inputs, weights, bias):
    """
    Process multiple inputs in parallel (vectorized).

    TPUs excel at batched operations!
    Uses vmap for automatic vectorization.
    """
    # vmap automatically batches the computation
    return vmap(lambda x: mlp_layer_forward(x[None, :], weights, bias)[0])(inputs)


# ============================================================================
# Fused Operations (XLA optimization)
# ============================================================================

@jit
def mlp_layer_fused(input, weights, bias):
    """
    Fused version where XLA combines operations.

    XLA will automatically fuse:
    - matmul + bias_add + relu into fewer memory operations
    - Reduces HBM traffic
    """
    # This will be fused by XLA into a single kernel
    return jax.nn.relu(jnp.dot(input, weights.T) + bias)


# ============================================================================
# Extract XLA HLO IR
# ============================================================================

def show_hlo_ir():
    """Display the XLA HLO intermediate representation"""

    # Create dummy inputs
    key = random.PRNGKey(0)
    input = random.normal(key, (1, 512))
    weights = random.normal(key, (256, 512)) * 0.1
    bias = jnp.zeros((256,))

    # Get the compiled function
    compiled = jax.jit(mlp_layer_forward).lower(input, weights, bias)

    # Print HLO (similar to what we wrote manually)
    print("=" * 80)
    print("XLA HLO Intermediate Representation:")
    print("=" * 80)
    print(compiled.as_text())
    print("=" * 80)


# ============================================================================
# Benchmark and Comparison
# ============================================================================

def benchmark_mlp():
    """Benchmark different MLP implementations"""

    # Problem size
    batch_size = 128
    input_size = 512
    output_size = 256
    iterations = 1000

    # Initialize random weights
    key = random.PRNGKey(42)
    key, *subkeys = random.split(key, 4)

    inputs = random.normal(subkeys[0], (batch_size, input_size))
    weights = random.normal(subkeys[1], (output_size, input_size)) * 0.1
    bias = random.normal(subkeys[2], (output_size,)) * 0.01

    print(f"\nBenchmark Configuration:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Bias shape: {bias.shape}")
    print(f"  Output shape: ({batch_size}, {output_size})")
    print(f"  Iterations: {iterations}")
    print()

    # ========================================================================
    # FP32 Version
    # ========================================================================

    print("Warming up FP32 version...")
    # Warm up (compile and first execution)
    _ = mlp_layer_forward(inputs, weights, bias)

    # Wait for completion (JAX is asynchronous)
    jax.block_until_ready(_)

    print("Benchmarking FP32...")
    start = time.time()

    for _ in range(iterations):
        result_fp32 = mlp_layer_forward(inputs, weights, bias)

    # Ensure all operations complete
    jax.block_until_ready(result_fp32)

    fp32_time = time.time() - start
    print(f"  FP32 Time: {fp32_time*1000:.2f} ms ({fp32_time*1e6/iterations:.2f} µs/iter)")

    # ========================================================================
    # BF16 Version
    # ========================================================================

    print("\nWarming up BF16 version...")
    _ = mlp_layer_forward_bf16(inputs, weights, bias)
    jax.block_until_ready(_)

    print("Benchmarking BF16...")
    start = time.time()

    for _ in range(iterations):
        result_bf16 = mlp_layer_forward_bf16(inputs, weights, bias)

    jax.block_until_ready(result_bf16)

    bf16_time = time.time() - start
    print(f"  BF16 Time: {bf16_time*1000:.2f} ms ({bf16_time*1e6/iterations:.2f} µs/iter)")
    print(f"  Speedup: {fp32_time/bf16_time:.2f}x")

    # ========================================================================
    # Fused Version
    # ========================================================================

    print("\nWarming up Fused version...")
    _ = mlp_layer_fused(inputs, weights, bias)
    jax.block_until_ready(_)

    print("Benchmarking Fused...")
    start = time.time()

    for _ in range(iterations):
        result_fused = mlp_layer_fused(inputs, weights, bias)

    jax.block_until_ready(result_fused)

    fused_time = time.time() - start
    print(f"  Fused Time: {fused_time*1000:.2f} ms ({fused_time*1e6/iterations:.2f} µs/iter)")
    print(f"  Speedup: {fp32_time/fused_time:.2f}x")

    # ========================================================================
    # Verify Results
    # ========================================================================

    print("\nVerifying results...")

    # Check FP32 vs BF16
    max_diff_bf16 = jnp.max(jnp.abs(result_fp32 - result_bf16))
    print(f"  Max diff (FP32 vs BF16): {max_diff_bf16:.6f}")

    # Check FP32 vs Fused
    max_diff_fused = jnp.max(jnp.abs(result_fp32 - result_fused))
    print(f"  Max diff (FP32 vs Fused): {max_diff_fused:.10f}")

    # Sample outputs
    print("\nSample outputs (first 5):")
    print(f"  FP32:  {result_fp32[0, :5]}")
    print(f"  BF16:  {result_bf16[0, :5]}")
    print(f"  Fused: {result_fused[0, :5]}")

    # Compute FLOPs
    flops_per_iter = 2 * batch_size * input_size * output_size  # matmul FLOPs
    throughput_fp32 = (flops_per_iter * iterations) / fp32_time / 1e12  # TFLOPS
    throughput_bf16 = (flops_per_iter * iterations) / bf16_time / 1e12

    print(f"\nThroughput:")
    print(f"  FP32: {throughput_fp32:.2f} TFLOPS")
    print(f"  BF16: {throughput_bf16:.2f} TFLOPS")


# ============================================================================
# Training Example
# ============================================================================

def train_step_example():
    """Example training step with gradient computation"""

    def loss_fn(weights, bias, inputs, targets):
        """Mean squared error loss"""
        predictions = mlp_layer_forward(inputs, weights, bias)
        return jnp.mean((predictions - targets) ** 2)

    # Compute gradients
    grad_fn = jit(grad(loss_fn, argnums=(0, 1)))

    # Example usage
    key = random.PRNGKey(0)
    weights = random.normal(key, (256, 512)) * 0.1
    bias = jnp.zeros((256,))
    inputs = random.normal(key, (32, 512))
    targets = random.normal(key, (32, 256))

    # Compute gradients (this will also run on TPU)
    weight_grad, bias_grad = grad_fn(weights, bias, inputs, targets)

    print("Gradient shapes:")
    print(f"  Weight gradient: {weight_grad.shape}")
    print(f"  Bias gradient: {bias_grad.shape}")

    # Update parameters
    learning_rate = 0.01
    new_weights = weights - learning_rate * weight_grad
    new_bias = bias - learning_rate * bias_grad

    return new_weights, new_bias


# ============================================================================
# Multi-Core TPU Example
# ============================================================================

def multi_core_tpu():
    """Demonstrate data parallelism across TPU cores"""

    num_devices = len(jax.devices())
    print(f"\nNumber of TPU cores: {num_devices}")

    if num_devices > 1:
        # Shard data across devices
        key = random.PRNGKey(0)

        # Create data sharded across devices
        inputs = random.normal(key, (num_devices, 128, 512))
        weights = random.normal(key, (256, 512)) * 0.1
        bias = jnp.zeros((256,))

        # pmap: parallel map across devices
        @jax.pmap
        def parallel_mlp(inputs_shard):
            return mlp_layer_forward(inputs_shard, weights, bias)

        # Execute on all cores in parallel
        outputs = parallel_mlp(inputs)

        print(f"  Input shape: {inputs.shape}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Each core processed: {inputs.shape[1]} examples")
    else:
        print("  Only 1 device available, skipping multi-core demo")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("JAX MLP on TPU")
    print("=" * 80)

    # Show XLA HLO
    print("\n1. Extracting XLA HLO IR:")
    show_hlo_ir()

    # Benchmark
    print("\n2. Running Benchmarks:")
    benchmark_mlp()

    # Training example
    print("\n3. Training Step Example:")
    train_step_example()

    # Multi-core
    print("\n4. Multi-Core TPU:")
    multi_core_tpu()

    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


# ============================================================================
# TPU-Specific Optimization Tips
# ============================================================================

"""
TIPS FOR TPU OPTIMIZATION:

1. Use Large Batch Sizes:
   - TPUs excel with batch sizes of 128-1024
   - Small batches underutilize the hardware

2. Use BF16 Precision:
   - 2x faster than FP32 on TPU v2/v3/v4
   - Same dynamic range as FP32
   - Minimal accuracy loss for most models

3. Pad to Multiples of 128:
   - MXU works on 128x128 tiles
   - Padding dimensions to 128 improves efficiency
   - e.g., use 256, 384, 512, not 250, 300

4. Fuse Operations:
   - JAX/XLA automatically fuses operations
   - Write natural Python, let compiler optimize

5. Minimize Host-TPU Transfers:
   - Keep data on TPU as long as possible
   - Transfer in large batches, not element-wise

6. Use JIT Compilation:
   - Always use @jit decorator
   - Compiles Python → XLA → HLO → TPU

7. Data Parallelism:
   - Use pmap for multi-core TPUs
   - Each core gets a shard of data

8. Matrix Shapes:
   - Prefer square matrices for MXU efficiency
   - Avoid very skinny/tall matrices

9. Avoid Conditionals:
   - if/else in tight loops can be slow
   - Use jnp.where() instead

10. Profile with TensorBoard:
    - Use tf.profiler to see TPU utilization
    - Identify bottlenecks (MXU vs VPU vs HBM)

TYPICAL TPU v4 PERFORMANCE FOR MLP:
- 512x256 layer, batch=128: ~5-10 µs/iteration (BF16)
- Peak throughput: ~100-200 TFLOPS for this workload
- Memory bound for small networks, compute bound for large
"""

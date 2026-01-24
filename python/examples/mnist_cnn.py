#!/usr/bin/env python3
"""
MNIST CNN Example for KPU Simulator.

This example demonstrates a simple convolutional neural network for MNIST
digit classification using the KPU Python package.

Architecture:
  - Conv1: 1 -> 16 channels, 3x3 kernel, ReLU
  - MaxPool: 2x2
  - Conv2: 16 -> 32 channels, 3x3 kernel, ReLU
  - MaxPool: 2x2
  - Flatten
  - FC1: 32*5*5 -> 128, ReLU
  - FC2: 128 -> 10 (logits)

Usage:
    python examples/mnist_cnn.py                   # BEHAVIORAL mode (default)
    python examples/mnist_cnn.py --transactional   # TRANSACTIONAL mode with XUE analysis
"""

import sys
import os
import argparse

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST CNN on KPU Simulator")
    parser.add_argument("--transactional", action="store_true",
                        help="Use TRANSACTIONAL mode with XUE performance analysis")
    parser.add_argument("--clock", type=float, default=1.0,
                        help="Clock frequency in GHz (default: 1.0)")
    return parser.parse_args()


# Parse command line arguments
args = parse_args()


def create_cnn_weights():
    """Create random weights for the CNN."""
    np.random.seed(42)

    # Xavier initialization scale
    def xavier(shape):
        fan_in = np.prod(shape[1:])
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * scale

    weights = {
        # Conv1: [out_channels=16, in_channels=1, kh=3, kw=3]
        'conv1_w': kpu.Tensor(xavier((16, 1, 3, 3))),
        'conv1_b': kpu.Tensor(np.zeros(16, dtype=np.float32)),

        # Conv2: [out_channels=32, in_channels=16, kh=3, kw=3]
        'conv2_w': kpu.Tensor(xavier((32, 16, 3, 3))),
        'conv2_b': kpu.Tensor(np.zeros(32, dtype=np.float32)),

        # FC1: After two 2x2 pools on 28x28, we get 5x5 spatial
        # So: 32 * 5 * 5 = 800 -> 128
        'fc1_w': kpu.Tensor(xavier((800, 128))),
        'fc1_b': kpu.Tensor(np.zeros(128, dtype=np.float32)),

        # FC2: 128 -> 10
        'fc2_w': kpu.Tensor(xavier((128, 10))),
        'fc2_b': kpu.Tensor(np.zeros(10, dtype=np.float32)),
    }

    return weights


@kpu.compile
def mnist_cnn(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    """
    MNIST CNN forward pass.

    Args:
        x: Input image [batch, 1, 28, 28]
        conv1_w, conv1_b: First conv layer weights
        conv2_w, conv2_b: Second conv layer weights
        fc1_w, fc1_b: First FC layer weights
        fc2_w, fc2_b: Output FC layer weights

    Returns:
        Logits [batch, 10]
    """
    # Conv1: [batch, 1, 28, 28] -> [batch, 16, 26, 26]
    h = kpu.relu(kpu.conv2d(x, conv1_w) + conv1_b.reshape(1, -1, 1, 1))

    # MaxPool1: [batch, 16, 26, 26] -> [batch, 16, 13, 13]
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Conv2: [batch, 16, 13, 13] -> [batch, 32, 11, 11]
    h = kpu.relu(kpu.conv2d(h, conv2_w) + conv2_b.reshape(1, -1, 1, 1))

    # MaxPool2: [batch, 32, 11, 11] -> [batch, 32, 5, 5]
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Flatten: [batch, 32, 5, 5] -> [batch, 800]
    h = h.reshape(h.shape[0], -1)

    # FC1: [batch, 800] -> [batch, 128]
    h = kpu.relu(h @ fc1_w + fc1_b)

    # FC2: [batch, 128] -> [batch, 10]
    logits = h @ fc2_w + fc2_b

    return logits


def numpy_reference(x, weights):
    """NumPy reference implementation for validation."""
    conv1_w = weights['conv1_w'].numpy()
    conv1_b = weights['conv1_b'].numpy()
    conv2_w = weights['conv2_w'].numpy()
    conv2_b = weights['conv2_b'].numpy()
    fc1_w = weights['fc1_w'].numpy()
    fc1_b = weights['fc1_b'].numpy()
    fc2_w = weights['fc2_w'].numpy()
    fc2_b = weights['fc2_b'].numpy()

    def np_conv2d(x, w):
        """Simple NumPy conv2d (no padding)."""
        N, C_in, H, W = x.shape
        C_out, _, K, _ = w.shape
        H_out = H - K + 1
        W_out = W - K + 1
        out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
        for n in range(N):
            for co in range(C_out):
                for h in range(H_out):
                    for wi in range(W_out):
                        val = 0.0
                        for ci in range(C_in):
                            for kh in range(K):
                                for kw in range(K):
                                    val += x[n, ci, h+kh, wi+kw] * w[co, ci, kh, kw]
                        out[n, co, h, wi] = val
        return out

    def np_maxpool2d(x, k=2):
        """Simple NumPy max pool."""
        N, C, H, W = x.shape
        H_out = H // k
        W_out = W // k
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        out[n, c, h, w] = np.max(x[n, c, h*k:(h+1)*k, w*k:(w+1)*k])
        return out

    # Conv1 + ReLU + Pool
    h = np_conv2d(x, conv1_w) + conv1_b.reshape(1, -1, 1, 1)
    h = np.maximum(h, 0)
    h = np_maxpool2d(h)

    # Conv2 + ReLU + Pool
    h = np_conv2d(h, conv2_w) + conv2_b.reshape(1, -1, 1, 1)
    h = np.maximum(h, 0)
    h = np_maxpool2d(h)

    # Flatten + FC1 + ReLU
    h = h.reshape(h.shape[0], -1)
    h = np.maximum(h @ fc1_w + fc1_b, 0)

    # FC2
    logits = h @ fc2_w + fc2_b

    return logits


def main():
    print("=" * 60)
    print("MNIST CNN Example - KPU Python Package")
    print("=" * 60)

    # Set fidelity based on command line
    # NOTE: TRANSACTIONAL mode currently only supports matmul operations.
    # CNN operations (conv2d, pooling) are not yet supported in native simulation.
    if args.transactional:
        print(f"\nNote: TRANSACTIONAL mode for CNN not yet supported (conv2d pending)")
        print("  Using BEHAVIORAL mode instead. See minimal_mlp.py for TRANSACTIONAL example.")
        kpu.set_fidelity(kpu.BEHAVIORAL)
        print(f"\nFidelity: BEHAVIORAL (computes actual values)")
    else:
        kpu.set_fidelity(kpu.BEHAVIORAL)
        print(f"\nFidelity: BEHAVIORAL (computes actual values)")

    # Create random weights
    print("\nCreating CNN weights...")
    weights = create_cnn_weights()

    # Create random "image" batch [batch=4, channels=1, height=28, width=28]
    batch_size = 4
    print(f"Creating random input batch: [{batch_size}, 1, 28, 28]")
    x = kpu.Tensor(np.random.randn(batch_size, 1, 28, 28).astype(np.float32))

    # Run through KPU CNN
    print("\nRunning CNN through KPU...")
    logits = mnist_cnn(
        x,
        weights['conv1_w'], weights['conv1_b'],
        weights['conv2_w'], weights['conv2_b'],
        weights['fc1_w'], weights['fc1_b'],
        weights['fc2_w'], weights['fc2_b']
    )

    print(f"Output shape: {logits.shape}")

    # Validate against NumPy reference
    print("\nValidating against NumPy reference...")
    expected = numpy_reference(x.numpy(), weights)
    max_diff = np.max(np.abs(logits.numpy() - expected))
    print(f"Max difference from NumPy reference: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    # XUE Performance Analysis (TRANSACTIONAL mode only)
    stats = mnist_cnn.get_stats()
    if args.transactional and stats and stats.elapsed_cycles > 0:
        T = stats.elapsed_cycles  # XUE elapsed time in cycles
        print()
        print("=" * 60)
        print("XUE Performance Analysis")
        print("=" * 60)
        print(f"  Clock Frequency:     {stats.clock_frequency_ghz:.1f} GHz")
        print(f"  T (Elapsed Cycles):  {T:,} cycles")
        print(f"  Wall Time:           {T / (stats.clock_frequency_ghz * 1e9) * 1e6:.2f} us")
        print()
        print("Memory Hierarchy (from XUE Summary):")
        # Get per-level stats from xue_summary
        mem_hierarchy = stats.xue_summary.get('memory_hierarchy', {}) if stats.xue_summary else {}
        for level in ['dram', 'l3', 'l2', 'l1']:
            level_stats = mem_hierarchy.get(level, {})
            level_bytes = level_stats.get('bytes', 0)
            level_events = level_stats.get('events', 0)
            service_rate = level_bytes / T if T > 0 else 0.0
            print(f"  {level.upper():4}: {level_bytes:,} bytes | {level_events:,} txns | {service_rate:.2f} B/cycle")
        print()
        print("Compute Performance:")
        print(f"  MatMul FLOPs:  {stats.matmul_flops:,}")
        print(f"  GFLOPS:        {stats.gflops:.1f} @ {stats.clock_frequency_ghz:.1f} GHz")
        print(f"  FLOPs/Cycle:   {stats.matmul_flops / T:.1f}")

    # Print DFX IR statistics
    print("\n" + "-" * 60)
    print("DFX IR Statistics:")
    print("-" * 60)

    dfx = mnist_cnn.get_dfx()
    op_counts = {}
    for op in dfx.ops:
        op_type = op.opcode.value
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    print(f"Total operations: {len(dfx.ops)}")
    for op_type, count in sorted(op_counts.items()):
        print(f"  {op_type}: {count}")

    # Print tensor memory footprint
    total_params = 0
    for name, tensor in weights.items():
        params = tensor.size
        total_params += params
        print(f"  {name}: {tensor.shape} = {params:,} params")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Memory footprint: {total_params * 4 / 1024:.1f} KB (float32)")

    # Sample output (softmax probabilities)
    print("\n" + "-" * 60)
    print("Sample predictions (first batch element):")
    print("-" * 60)
    logits_0 = logits.numpy()[0]
    # Numerically stable softmax
    logits_0 = logits_0 - np.max(logits_0)
    probs = np.exp(logits_0) / np.sum(np.exp(logits_0))
    for i, p in enumerate(probs):
        bar = "#" * int(p * 50)
        print(f"  Digit {i}: {p:.4f} {bar}")

    predicted = np.argmax(probs)
    print(f"\nPredicted digit: {predicted} (random weights, so meaningless)")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

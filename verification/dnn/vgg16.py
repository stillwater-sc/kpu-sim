#!/usr/bin/env python3
"""
VGG-16 on the native @kpu.compile path.

VGG-16 architecture (simplified for KPU testing):
  Block 1: Conv3x3(3->64) x2  + MaxPool
  Block 2: Conv3x3(64->128) x2 + MaxPool
  Block 3: Conv3x3(128->256) x3 + MaxPool
  Block 4: Conv3x3(256->512) x3 + MaxPool
  Block 5: Conv3x3(512->512) x3 + MaxPool
  Classifier: FC(512*S*S -> 4096) -> FC(4096->4096) -> FC(4096->num_classes)

This is the canonical Class I (compute-bound, GEMM-dominant) benchmark model.

Usage:
    python verification/dnn/vgg16.py                    # BEHAVIORAL (64x64 input)
    python verification/dnn/vgg16.py --transactional    # TRANSACTIONAL with timing
    python verification/dnn/vgg16.py --input-size 224   # Full 224x224 input
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu


def parse_args():
    parser = argparse.ArgumentParser(description="VGG-16 native @kpu.compile")
    parser.add_argument("--transactional", action="store_true")
    parser.add_argument("--clock", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=64,
                        help="Input spatial size (default: 64)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _conv_w(c_out, c_in, k=3):
    """Create conv weight with Kaiming init."""
    fan_in = c_in * k * k
    return kpu.Tensor(np.random.randn(c_out, c_in, k, k).astype(np.float32) * np.sqrt(2.0 / fan_in))


def _conv_b(c_out):
    return kpu.Tensor(np.zeros(c_out, dtype=np.float32))


def _fc_w(out_dim, in_dim):
    return kpu.Tensor(np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim))


def _fc_b(out_dim):
    return kpu.Tensor(np.zeros(out_dim, dtype=np.float32))


@kpu.compile(optimize=False)
def vgg16(
    x,
    # Block 1: 2 conv layers (3->64)
    c1_1_w, c1_1_b, c1_2_w, c1_2_b,
    # Block 2: 2 conv layers (64->128)
    c2_1_w, c2_1_b, c2_2_w, c2_2_b,
    # Block 3: 3 conv layers (128->256)
    c3_1_w, c3_1_b, c3_2_w, c3_2_b, c3_3_w, c3_3_b,
    # Block 4: 3 conv layers (256->512)
    c4_1_w, c4_1_b, c4_2_w, c4_2_b, c4_3_w, c4_3_b,
    # Block 5: 3 conv layers (512->512)
    c5_1_w, c5_1_b, c5_2_w, c5_2_b, c5_3_w, c5_3_b,
    # Classifier: 3 FC layers
    fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b,
):
    """VGG-16 forward pass."""
    # Block 1
    x = kpu.relu(kpu.conv2d(x, c1_1_w, c1_1_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c1_2_w, c1_2_b, padding=1))
    x = kpu.max_pool2d(x, kernel_size=2, stride=2)

    # Block 2
    x = kpu.relu(kpu.conv2d(x, c2_1_w, c2_1_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c2_2_w, c2_2_b, padding=1))
    x = kpu.max_pool2d(x, kernel_size=2, stride=2)

    # Block 3
    x = kpu.relu(kpu.conv2d(x, c3_1_w, c3_1_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c3_2_w, c3_2_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c3_3_w, c3_3_b, padding=1))
    x = kpu.max_pool2d(x, kernel_size=2, stride=2)

    # Block 4
    x = kpu.relu(kpu.conv2d(x, c4_1_w, c4_1_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c4_2_w, c4_2_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c4_3_w, c4_3_b, padding=1))
    x = kpu.max_pool2d(x, kernel_size=2, stride=2)

    # Block 5
    x = kpu.relu(kpu.conv2d(x, c5_1_w, c5_1_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c5_2_w, c5_2_b, padding=1))
    x = kpu.relu(kpu.conv2d(x, c5_3_w, c5_3_b, padding=1))
    x = kpu.max_pool2d(x, kernel_size=2, stride=2)

    # Flatten via adaptive avg pool + reshape, or direct flatten
    # After 5 maxpools with stride 2: spatial = input_size / 32
    # For 64x64 input -> 2x2, for 224x224 -> 7x7
    x = kpu.adaptive_avg_pool2d(x, (1, 1))
    x = kpu.reshape(x, (x.shape[0], -1))  # [N, 512]

    # FC layers (using matmul since we flattened)
    x = kpu.relu(kpu.matmul(x, fc1_w) + fc1_b)
    x = kpu.relu(kpu.matmul(x, fc2_w) + fc2_b)
    x = kpu.matmul(x, fc3_w) + fc3_b

    return x


def numpy_vgg16(x, weights):
    """NumPy reference VGG-16."""
    from kpu.ops import _conv2d_im2col

    idx = 0
    def get_wb():
        nonlocal idx
        w, b = weights[idx], weights[idx + 1]
        idx += 2
        return w, b

    def conv_relu(x, w, b):
        N, C, H, W = x.shape
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        out = _conv2d_im2col(x_pad, w, (1, 1), (1, 1), 1,
                             N, w.shape[0], w.shape[1], 3, 3, H, W)
        out = out + b.reshape(1, -1, 1, 1)
        return np.maximum(0, out)

    def maxpool(x, k=2, s=2):
        N, C, H, W = x.shape
        H_out, W_out = H // s, W // s
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = x[:, :, i*s:i*s+k, j*s:j*s+k].max(axis=(2, 3))
        return out

    # Block 1
    w, b = get_wb(); x = conv_relu(x, w, b)
    w, b = get_wb(); x = conv_relu(x, w, b)
    x = maxpool(x)

    # Block 2
    w, b = get_wb(); x = conv_relu(x, w, b)
    w, b = get_wb(); x = conv_relu(x, w, b)
    x = maxpool(x)

    # Block 3
    for _ in range(3):
        w, b = get_wb(); x = conv_relu(x, w, b)
    x = maxpool(x)

    # Block 4
    for _ in range(3):
        w, b = get_wb(); x = conv_relu(x, w, b)
    x = maxpool(x)

    # Block 5
    for _ in range(3):
        w, b = get_wb(); x = conv_relu(x, w, b)
    x = maxpool(x)

    # Adaptive avg pool + flatten
    x = x.mean(axis=(2, 3))  # [N, 512]

    # FC layers
    fc1_w, fc1_b = get_wb()
    fc2_w, fc2_b = get_wb()
    fc3_w, fc3_b = get_wb()

    x = np.maximum(0, x @ fc1_w + fc1_b)
    x = np.maximum(0, x @ fc2_w + fc2_b)
    x = x @ fc3_w + fc3_b

    return x


def main():
    args = parse_args()

    if args.transactional:
        kpu.set_fidelity(kpu.TRANSACTIONAL)
        kpu.set_clock_frequency(args.clock)
    else:
        kpu.set_fidelity(kpu.BEHAVIORAL)

    mode = "TRANSACTIONAL" if args.transactional else "BEHAVIORAL"
    S = args.input_size
    nc = args.num_classes

    print("=" * 60)
    print(f"VGG-16 — native @kpu.compile ({mode})")
    print(f"  Input: [1, 3, {S}, {S}]  Classes: {nc}")
    print("=" * 60)

    np.random.seed(42)

    # Build weight list
    all_weights = []

    # Conv blocks: channels per block
    configs = [
        (3, 64, 2),     # block 1
        (64, 128, 2),   # block 2
        (128, 256, 3),  # block 3
        (256, 512, 3),  # block 4
        (512, 512, 3),  # block 5
    ]

    for c_in, c_out, n_conv in configs:
        for i in range(n_conv):
            ci = c_in if i == 0 else c_out
            all_weights.extend([_conv_w(c_out, ci), _conv_b(c_out)])

    # FC layers: after adaptive avg pool, feature dim = 512
    fc_dim = 512  # using adaptive avg pool to (1,1)
    # Use smaller FC for testing (original VGG uses 4096)
    hidden = 256 if S <= 64 else 4096
    all_weights.extend([_fc_w(hidden, fc_dim), _fc_b(hidden)])
    all_weights.extend([_fc_w(hidden, hidden), _fc_b(hidden)])
    all_weights.extend([_fc_w(nc, hidden), _fc_b(nc)])

    x = kpu.Tensor(np.random.randn(1, 3, S, S).astype(np.float32) * 0.01)

    total_params = sum(np.prod(np.array(t.data).shape) for t in all_weights)
    print(f"  Total weight tensors: {len(all_weights)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  FC hidden dim: {hidden}")

    print("\n  Executing...")
    all_args = [x] + all_weights
    result = vgg16(*all_args)
    print(f"  Output shape: {result.shape}")

    # NumPy reference
    print("  Computing NumPy reference...")
    np_weights = [np.array(t.data) for t in all_weights]
    ref = numpy_vgg16(np.array(x.data), np_weights)

    result_np = np.array(result.data)
    max_diff = float(np.max(np.abs(result_np - ref)))

    print()
    print(f"  Max difference vs NumPy: {max_diff:.6e}")
    if max_diff < 1e-3:
        print("  [PASS] Functional correctness verified")
    else:
        print(f"  [FAIL] Results differ (threshold: 1e-3)")

    if args.transactional:
        stats = vgg16.stats
        if stats and stats.elapsed_cycles > 0:
            print(f"\n  Performance:")
            print(f"    Elapsed cycles: {stats.elapsed_cycles:,}")
            print(f"    Compute cycles: {stats.compute_cycles:,}")
            print(f"    Memory cycles:  {stats.memory_cycles:,}")

    print("=" * 60)
    return 0 if max_diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())

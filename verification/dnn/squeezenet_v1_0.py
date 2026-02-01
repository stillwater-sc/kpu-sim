#!/usr/bin/env python3
"""
SqueezeNet 1.0 on the native @kpu.compile path.

This ports the class-based kpu.models.SqueezeNet to a flat function that can be
traced and compiled to DFX IR, enabling BEHAVIORAL and TRANSACTIONAL execution
through the C++ simulator.

Architecture (SqueezeNet 1.0):
  Conv1(3->96, 7x7, stride=2) -> ReLU -> MaxPool(3x3, stride=2)
  Fire2(96->128)  -> Fire3(128->128) -> Fire4(128->256) -> MaxPool
  Fire5(256->256) -> Fire6(256->384) -> Fire7(384->384) -> Fire8(384->512) -> MaxPool
  Fire9(512->512) -> Classifier Conv(512->num_classes, 1x1) -> AdaptiveAvgPool -> Flatten

Usage:
    python verification/dnn/squeezenet_v1_0.py                    # BEHAVIORAL
    python verification/dnn/squeezenet_v1_0.py --transactional    # TRANSACTIONAL with timing
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import kpu


def parse_args():
    parser = argparse.ArgumentParser(description="SqueezeNet 1.0 native @kpu.compile")
    parser.add_argument("--transactional", action="store_true")
    parser.add_argument("--clock", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of output classes (default: 10 for small test)")
    parser.add_argument("--input-size", type=int, default=64,
                        help="Input spatial size (default: 64 for faster test)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def fire_module(x, sq_w, sq_b, e1_w, e1_b, e3_w, e3_b):
    """Fire module: squeeze 1x1 -> expand 1x1 + expand 3x3 -> concat."""
    s = kpu.relu(kpu.conv2d(x, sq_w, sq_b))
    e1 = kpu.relu(kpu.conv2d(s, e1_w, e1_b))
    e3 = kpu.relu(kpu.conv2d(s, e3_w, e3_b, padding=1))
    return kpu.concat([e1, e3], dim=1)


def _make_fire_weights(c_in, sq, e1, e3):
    """Create random weights for a fire module (6 tensors)."""
    scale = np.sqrt(2.0)
    return [
        kpu.Tensor(np.random.randn(sq, c_in, 1, 1).astype(np.float32) * scale / np.sqrt(c_in)),
        kpu.Tensor(np.zeros(sq, dtype=np.float32)),
        kpu.Tensor(np.random.randn(e1, sq, 1, 1).astype(np.float32) * scale / np.sqrt(sq)),
        kpu.Tensor(np.zeros(e1, dtype=np.float32)),
        kpu.Tensor(np.random.randn(e3, sq, 3, 3).astype(np.float32) * scale / np.sqrt(sq * 9)),
        kpu.Tensor(np.zeros(e3, dtype=np.float32)),
    ]


@kpu.compile(optimize=False)  # disable fusion — conv2d fusion not yet supported
def squeezenet_v1_0(
    x,
    # conv1
    conv1_w, conv1_b,
    # fire2
    f2_sq_w, f2_sq_b, f2_e1_w, f2_e1_b, f2_e3_w, f2_e3_b,
    # fire3
    f3_sq_w, f3_sq_b, f3_e1_w, f3_e1_b, f3_e3_w, f3_e3_b,
    # fire4
    f4_sq_w, f4_sq_b, f4_e1_w, f4_e1_b, f4_e3_w, f4_e3_b,
    # fire5
    f5_sq_w, f5_sq_b, f5_e1_w, f5_e1_b, f5_e3_w, f5_e3_b,
    # fire6
    f6_sq_w, f6_sq_b, f6_e1_w, f6_e1_b, f6_e3_w, f6_e3_b,
    # fire7
    f7_sq_w, f7_sq_b, f7_e1_w, f7_e1_b, f7_e3_w, f7_e3_b,
    # fire8
    f8_sq_w, f8_sq_b, f8_e1_w, f8_e1_b, f8_e3_w, f8_e3_b,
    # fire9
    f9_sq_w, f9_sq_b, f9_e1_w, f9_e1_b, f9_e3_w, f9_e3_b,
    # classifier
    cls_w, cls_b,
):
    """SqueezeNet 1.0 forward pass as a flat compiled function."""
    # Initial conv + pool
    x = kpu.relu(kpu.conv2d(x, conv1_w, conv1_b, stride=2))
    x = kpu.max_pool2d(x, kernel_size=3, stride=2)

    # Fire modules + pools
    x = fire_module(x, f2_sq_w, f2_sq_b, f2_e1_w, f2_e1_b, f2_e3_w, f2_e3_b)
    x = fire_module(x, f3_sq_w, f3_sq_b, f3_e1_w, f3_e1_b, f3_e3_w, f3_e3_b)
    x = fire_module(x, f4_sq_w, f4_sq_b, f4_e1_w, f4_e1_b, f4_e3_w, f4_e3_b)
    x = kpu.max_pool2d(x, kernel_size=3, stride=2)

    x = fire_module(x, f5_sq_w, f5_sq_b, f5_e1_w, f5_e1_b, f5_e3_w, f5_e3_b)
    x = fire_module(x, f6_sq_w, f6_sq_b, f6_e1_w, f6_e1_b, f6_e3_w, f6_e3_b)
    x = fire_module(x, f7_sq_w, f7_sq_b, f7_e1_w, f7_e1_b, f7_e3_w, f7_e3_b)
    x = fire_module(x, f8_sq_w, f8_sq_b, f8_e1_w, f8_e1_b, f8_e3_w, f8_e3_b)
    x = kpu.max_pool2d(x, kernel_size=3, stride=2)

    x = fire_module(x, f9_sq_w, f9_sq_b, f9_e1_w, f9_e1_b, f9_e3_w, f9_e3_b)

    # Classifier
    x = kpu.conv2d(x, cls_w, cls_b)
    x = kpu.relu(x)
    x = kpu.adaptive_avg_pool2d(x, (1, 1))

    # Flatten: reshape from [N, C, 1, 1] to [N, C]
    return kpu.reshape(x, (x.shape[0], -1))


def numpy_fire(x, sq_w, sq_b, e1_w, e1_b, e3_w, e3_b):
    """NumPy reference fire module."""
    from kpu.ops import _conv2d_im2col
    N, C, H, W = x.shape

    # Squeeze 1x1
    s = _conv2d_im2col(x, sq_w, (1, 1), (1, 1), 1,
                       N, sq_w.shape[0], sq_w.shape[1], 1, 1, H, W)
    s = s + sq_b.reshape(1, -1, 1, 1)
    s = np.maximum(0, s)

    # Expand 1x1
    sN, sC, sH, sW = s.shape
    e1 = _conv2d_im2col(s, e1_w, (1, 1), (1, 1), 1,
                        sN, e1_w.shape[0], e1_w.shape[1], 1, 1, sH, sW)
    e1 = e1 + e1_b.reshape(1, -1, 1, 1)
    e1 = np.maximum(0, e1)

    # Expand 3x3 (padding=1)
    s_pad = np.pad(s, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    e3 = _conv2d_im2col(s_pad, e3_w, (1, 1), (1, 1), 1,
                        sN, e3_w.shape[0], e3_w.shape[1], 3, 3, sH, sW)
    e3 = e3 + e3_b.reshape(1, -1, 1, 1)
    e3 = np.maximum(0, e3)

    return np.concatenate([e1, e3], axis=1)


def numpy_maxpool(x, kernel_size=3, stride=2):
    """NumPy reference max pooling."""
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            out[:, :, i, j] = x[:, :, h_start:h_start + kernel_size,
                                  w_start:w_start + kernel_size].max(axis=(2, 3))
    return out


def numpy_adaptive_avgpool(x, output_size=(1, 1)):
    """NumPy reference adaptive average pooling."""
    return x.mean(axis=(2, 3), keepdims=True)


def numpy_squeezenet(x, weights):
    """Full NumPy reference SqueezeNet 1.0."""
    from kpu.ops import _conv2d_im2col

    idx = 0
    def get_wb():
        nonlocal idx
        w, b = weights[idx], weights[idx + 1]
        idx += 2
        return w, b

    def get_fire():
        nonlocal idx
        tensors = weights[idx:idx + 6]
        idx += 6
        return tensors

    # Conv1 (stride=2, no padding in our test — 7x7 kernel)
    conv1_w, conv1_b = get_wb()
    N, C, H, W = x.shape
    H_out = (H - 7) // 2 + 1
    W_out = (W - 7) // 2 + 1
    x = _conv2d_im2col(x, conv1_w, (2, 2), (1, 1), 1,
                       N, conv1_w.shape[0], conv1_w.shape[1], 7, 7, H_out, W_out)
    x = x + conv1_b.reshape(1, -1, 1, 1)
    x = np.maximum(0, x)
    x = numpy_maxpool(x, 3, 2)

    # Fire modules 2-4 + pool
    for _ in range(3):
        fire_w = get_fire()
        x = numpy_fire(x, *fire_w)
    x = numpy_maxpool(x, 3, 2)

    # Fire modules 5-8 + pool
    for _ in range(4):
        fire_w = get_fire()
        x = numpy_fire(x, *fire_w)
    x = numpy_maxpool(x, 3, 2)

    # Fire9
    fire_w = get_fire()
    x = numpy_fire(x, *fire_w)

    # Classifier conv 1x1
    cls_w, cls_b = get_wb()
    N, C, H, W = x.shape
    x = _conv2d_im2col(x, cls_w, (1, 1), (1, 1), 1,
                       N, cls_w.shape[0], cls_w.shape[1], 1, 1, H, W)
    x = x + cls_b.reshape(1, -1, 1, 1)
    x = np.maximum(0, x)

    # Adaptive avg pool + flatten
    x = x.mean(axis=(2, 3))
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
    num_classes = args.num_classes

    print("=" * 60)
    print(f"SqueezeNet 1.0 — native @kpu.compile ({mode})")
    print(f"  Input: [1, 3, {S}, {S}]  Classes: {num_classes}")
    print("=" * 60)

    # Create weights
    np.random.seed(42)

    # SqueezeNet 1.0 architecture:
    # conv1: 3->96, 7x7
    # fire2: 96, sq=16, e1=64, e3=64 -> 128
    # fire3: 128, sq=16, e1=64, e3=64 -> 128
    # fire4: 128, sq=32, e1=128, e3=128 -> 256
    # fire5: 256, sq=32, e1=128, e3=128 -> 256
    # fire6: 256, sq=48, e1=192, e3=192 -> 384
    # fire7: 384, sq=48, e1=192, e3=192 -> 384
    # fire8: 384, sq=64, e1=256, e3=256 -> 512
    # fire9: 512, sq=64, e1=256, e3=256 -> 512
    # classifier: 512->num_classes, 1x1

    scale = np.sqrt(2.0)
    conv1_w = kpu.Tensor(np.random.randn(96, 3, 7, 7).astype(np.float32) * scale / np.sqrt(3 * 49))
    conv1_b = kpu.Tensor(np.zeros(96, dtype=np.float32))

    fire_specs = [
        (96, 16, 64, 64),    # fire2
        (128, 16, 64, 64),   # fire3
        (128, 32, 128, 128), # fire4
        (256, 32, 128, 128), # fire5
        (256, 48, 192, 192), # fire6
        (384, 48, 192, 192), # fire7
        (384, 64, 256, 256), # fire8
        (512, 64, 256, 256), # fire9
    ]

    fire_weights = []
    for c_in, sq, e1, e3 in fire_specs:
        fire_weights.extend(_make_fire_weights(c_in, sq, e1, e3))

    cls_w = kpu.Tensor(np.random.randn(num_classes, 512, 1, 1).astype(np.float32) * scale / np.sqrt(512))
    cls_b = kpu.Tensor(np.zeros(num_classes, dtype=np.float32))

    # Input
    x = kpu.Tensor(np.random.randn(1, 3, S, S).astype(np.float32) * 0.01)

    # All args: x, conv1_w, conv1_b, then 8 fire modules * 6 weights, then cls_w, cls_b
    all_args = [x, conv1_w, conv1_b] + fire_weights + [cls_w, cls_b]

    print(f"  Total weight tensors: {len(all_args) - 1}")
    total_params = sum(np.prod(np.array(t.data).shape) for t in all_args[1:])
    print(f"  Total parameters: {total_params:,}")

    # Execute
    print("\n  Executing...")
    result = squeezenet_v1_0(*all_args)
    print(f"  Output shape: {result.shape}")

    # NumPy reference
    print("  Computing NumPy reference...")
    np_weights = [np.array(t.data) for t in all_args[1:]]
    ref = numpy_squeezenet(np.array(x.data), np_weights)

    result_np = np.array(result.data)
    max_diff = float(np.max(np.abs(result_np - ref)))

    print()
    print(f"  Max difference vs NumPy: {max_diff:.6e}")
    if max_diff < 1e-3:
        print("  [PASS] Functional correctness verified")
    else:
        print(f"  [FAIL] Results differ (threshold: 1e-3)")

    # Transactional stats
    if args.transactional:
        stats = squeezenet_v1_0.stats
        if stats and stats.elapsed_cycles > 0:
            print(f"\n  Performance:")
            print(f"    Elapsed cycles: {stats.elapsed_cycles:,}")
            print(f"    Compute cycles: {stats.compute_cycles:,}")
            print(f"    Memory cycles:  {stats.memory_cycles:,}")
            if hasattr(stats, 'gflops') and stats.gflops > 0:
                print(f"    GFLOPS:         {stats.gflops:.1f}")

    print("=" * 60)
    return 0 if max_diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())

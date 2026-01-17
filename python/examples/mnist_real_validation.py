#!/usr/bin/env python3
"""
MNIST Real Data Validation Example.

This example validates the KPU CNN implementation on real MNIST data.
It downloads the MNIST dataset and runs inference through the CNN.

Usage:
    python examples/mnist_real_validation.py
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import kpu


def create_cnn():
    """Create CNN with random weights."""
    np.random.seed(42)

    def xavier(shape):
        fan_in = np.prod(shape[1:])
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * scale

    return {
        'conv1_w': kpu.Tensor(xavier((16, 1, 3, 3))),
        'conv1_b': kpu.Tensor(np.zeros(16, dtype=np.float32)),
        'conv2_w': kpu.Tensor(xavier((32, 16, 3, 3))),
        'conv2_b': kpu.Tensor(np.zeros(32, dtype=np.float32)),
        'fc1_w': kpu.Tensor(xavier((800, 128))),
        'fc1_b': kpu.Tensor(np.zeros(128, dtype=np.float32)),
        'fc2_w': kpu.Tensor(xavier((128, 10))),
        'fc2_b': kpu.Tensor(np.zeros(10, dtype=np.float32)),
    }


@kpu.compile
def mnist_cnn(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    """MNIST CNN forward pass."""
    # Conv1 + ReLU + Pool
    h = kpu.relu(kpu.conv2d(x, conv1_w) + conv1_b.reshape(1, -1, 1, 1))
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Conv2 + ReLU + Pool
    h = kpu.relu(kpu.conv2d(h, conv2_w) + conv2_b.reshape(1, -1, 1, 1))
    h = kpu.max_pool2d(h, kernel_size=2, stride=2)

    # Flatten + FC1 + ReLU
    h = h.reshape(h.shape[0], -1)
    h = kpu.relu(h @ fc1_w + fc1_b)

    # FC2 (logits)
    return h @ fc2_w + fc2_b


def softmax(x):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    print("=" * 60)
    print("MNIST Real Data Validation")
    print("=" * 60)

    kpu.set_fidelity(kpu.BEHAVIORAL)

    # Try to load MNIST
    print("\nLoading MNIST test dataset...")
    try:
        images, labels = kpu.load_mnist(split='test', normalize=True, limit=1000)
        print(f"  Loaded {len(images)} test images: {images.shape}")
    except Exception as e:
        print(f"  Failed to download MNIST: {e}")
        print("  Using random synthetic data instead...")
        np.random.seed(0)
        images = np.random.randn(1000, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 1000)

    # Create CNN
    print("\nCreating CNN...")
    weights = create_cnn()

    # Run inference
    print("\nRunning inference...")
    batch_size = 32
    n_batches = (len(images) + batch_size - 1) // batch_size

    all_predictions = []
    all_logits = []

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end]

        x = kpu.Tensor(batch)
        logits = mnist_cnn(
            x,
            weights['conv1_w'], weights['conv1_b'],
            weights['conv2_w'], weights['conv2_b'],
            weights['fc1_w'], weights['fc1_b'],
            weights['fc2_w'], weights['fc2_b']
        )

        all_logits.append(logits.numpy())
        predictions = np.argmax(logits.numpy(), axis=-1)
        all_predictions.extend(predictions)

        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  Processed {end}/{len(images)} images")

    all_predictions = np.array(all_predictions)
    all_logits = np.concatenate(all_logits, axis=0)

    # Compute accuracy (with random weights, this will be ~10%)
    accuracy = np.mean(all_predictions == labels)
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy * 100:.2f}% (random weights expected ~10%)")

    # Distribution of predictions
    print("\n  Prediction distribution:")
    unique, counts = np.unique(all_predictions, return_counts=True)
    for digit, count in zip(unique, counts):
        pct = count / len(all_predictions) * 100
        bar = "#" * int(pct / 2)
        print(f"    Digit {digit}: {count:4d} ({pct:5.1f}%) {bar}")

    # Show some sample predictions
    print("\n  Sample predictions (first 10 images):")
    probs = softmax(all_logits[:10])
    for i in range(10):
        pred = all_predictions[i]
        true = labels[i]
        conf = probs[i, pred]
        status = "OK" if pred == true else "X "
        print(f"    Image {i}: predicted={pred}, true={true}, conf={conf:.2f} [{status}]")

    # DFX stats
    print("\n" + "-" * 60)
    print("DFX IR Statistics:")
    dfx = mnist_cnn.get_dfx()
    print(f"  Total operations: {len(dfx.ops)}")

    op_counts = {}
    for op in dfx.ops:
        op_type = op.opcode.value
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    for op_type, count in sorted(op_counts.items()):
        print(f"    {op_type}: {count}")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# python/kpu/models/simple.py
"""
Simple models for testing and MNIST classification.

Provides basic MLP and CNN architectures suitable for:
- Testing the KPU simulator
- MNIST digit classification
- Tutorial examples

Example:
    >>> from kpu.models import mnist_mlp, mnist_cnn
    >>> from kpu.datasets import load_mnist
    >>>
    >>> # Create models
    >>> mlp = mnist_mlp()
    >>> cnn = mnist_cnn()
    >>>
    >>> # Load MNIST
    >>> (x_train, y_train), (x_test, y_test) = load_mnist()
    >>>
    >>> # Run inference
    >>> output = mlp(x_test[:32])
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from ..model import Model, Sequential, Linear, Conv2d, MaxPool2d, Flatten, ReLU, Dropout
from ..tensor import Tensor


class SimpleMLP(Model):
    """Simple Multi-Layer Perceptron.

    Args:
        input_size: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension (number of classes)
        dropout: Dropout probability (default: 0.0)
        name: Optional model name
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "simple_mlp")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            # Linear layer
            self.add_child(f"fc{i}", Linear(sizes[i], sizes[i + 1]))

            # Activation (except for last layer)
            if i < len(sizes) - 2:
                self.add_child(f"relu{i}", ReLU())
                if dropout > 0:
                    self.add_child(f"dropout{i}", Dropout(p=dropout))

        self._num_layers = len(sizes) - 1

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output logits (batch_size, output_size)
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        for i in range(self._num_layers):
            x = self._children[f"fc{i}"](x)

            if i < self._num_layers - 1:
                x = self._children[f"relu{i}"](x)
                if f"dropout{i}" in self._children:
                    x = self._children[f"dropout{i}"](x)

        return x


class SimpleCNN(Model):
    """Simple Convolutional Neural Network.

    Architecture:
        Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> FC

    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        conv_channels: List of conv layer output channels
        fc_size: Size of fully connected hidden layer
        name: Optional model name
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: Optional[List[int]] = None,
        fc_size: int = 128,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "simple_cnn")

        if conv_channels is None:
            conv_channels = [32, 64]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.fc_size = fc_size

        # Convolutional layers
        channels = [in_channels] + conv_channels
        for i in range(len(conv_channels)):
            self.add_child(f"conv{i}", Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            self.add_child(f"relu{i}", ReLU())
            self.add_child(f"pool{i}", MaxPool2d(kernel_size=2, stride=2))

        # Flatten
        self.add_child("flatten", Flatten())

        # FC layers - size depends on input image size
        # For MNIST 28x28: after 2 pools -> 7x7, so fc_input = 64 * 7 * 7 = 3136
        # We'll compute dynamically in forward pass
        self._fc_input_computed = False
        self._fc_input_size = 0

        self.add_child("fc1", Linear(conv_channels[-1] * 7 * 7, fc_size))
        self.add_child("fc1_relu", ReLU())
        self.add_child("fc2", Linear(fc_size, num_classes))

        self._num_conv_layers = len(conv_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Conv layers
        for i in range(self._num_conv_layers):
            x = self._children[f"conv{i}"](x)
            x = self._children[f"relu{i}"](x)
            x = self._children[f"pool{i}"](x)

        # Flatten and FC
        x = self._children["flatten"](x)
        x = self._children["fc1"](x)
        x = self._children["fc1_relu"](x)
        x = self._children["fc2"](x)

        return x


def mnist_mlp(
    hidden_sizes: Optional[List[int]] = None,
    dropout: float = 0.0,
) -> SimpleMLP:
    """Create MLP for MNIST classification.

    Args:
        hidden_sizes: Hidden layer sizes (default: [128, 64])
        dropout: Dropout probability

    Returns:
        SimpleMLP model
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]

    return SimpleMLP(
        input_size=784,  # 28 * 28
        hidden_sizes=hidden_sizes,
        output_size=10,
        dropout=dropout,
        name="mnist_mlp",
    )


def mnist_cnn(
    conv_channels: Optional[List[int]] = None,
    fc_size: int = 128,
) -> SimpleCNN:
    """Create CNN for MNIST classification.

    Args:
        conv_channels: Conv layer channels (default: [32, 64])
        fc_size: FC hidden layer size

    Returns:
        SimpleCNN model
    """
    return SimpleCNN(
        in_channels=1,
        num_classes=10,
        conv_channels=conv_channels,
        fc_size=fc_size,
        name="mnist_cnn",
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SimpleMLP",
    "SimpleCNN",
    "mnist_mlp",
    "mnist_cnn",
]

# python/kpu/models/__init__.py
"""
KPU Pre-built Models - Reference model implementations for KPU simulator.

Provides implementations of common neural network architectures:
- SqueezeNet 1.0/1.1
- MobileNetV2
- Simple MLP and CNN for MNIST

Example:
    >>> from kpu.models import squeezenet1_0, mobilenet_v2
    >>>
    >>> # Create model (random weights)
    >>> model = squeezenet1_0()
    >>>
    >>> # Create model with PyTorch pretrained weights
    >>> model = squeezenet1_0(pretrained=True)  # Requires torch
    >>>
    >>> # Run inference
    >>> output = model(input_tensor)
"""

from .squeezenet import squeezenet1_0, squeezenet1_1, SqueezeNet, Fire
from .mobilenet import mobilenet_v2, MobileNetV2, InvertedResidual
from .simple import mnist_mlp, mnist_cnn, SimpleMLP, SimpleCNN

__all__ = [
    # SqueezeNet
    "squeezenet1_0",
    "squeezenet1_1",
    "SqueezeNet",
    "Fire",
    # MobileNetV2
    "mobilenet_v2",
    "MobileNetV2",
    "InvertedResidual",
    # Simple models
    "mnist_mlp",
    "mnist_cnn",
    "SimpleMLP",
    "SimpleCNN",
]

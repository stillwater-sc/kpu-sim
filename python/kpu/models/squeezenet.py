# python/kpu/models/squeezenet.py
"""
SqueezeNet implementation for KPU simulator.

SqueezeNet: AlexNet-level accuracy with 50x fewer parameters.
Paper: https://arxiv.org/abs/1602.07360

Reference:
- SqueezeNet 1.0: Original architecture
- SqueezeNet 1.1: 2.4x less computation, no accuracy loss

Example:
    >>> from kpu.models import squeezenet1_0
    >>>
    >>> # Create model
    >>> model = squeezenet1_0(num_classes=1000)
    >>>
    >>> # Load PyTorch pretrained weights
    >>> model = squeezenet1_0(pretrained=True)
    >>>
    >>> # Run inference
    >>> output = model(input_tensor)  # (1, 3, 224, 224) -> (1, 1000)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..model import Model, Layer, Sequential, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Dropout, ReLU, Flatten
from ..tensor import Tensor
from .. import ops


class Fire(Layer):
    """Fire module - the building block of SqueezeNet.

    Architecture:
        input -> squeeze (1x1 conv) -> expand1x1 + expand3x3 -> concat

    Args:
        in_channels: Number of input channels
        squeeze_channels: Number of channels in squeeze layer
        expand1x1_channels: Number of channels in 1x1 expand
        expand3x3_channels: Number of channels in 3x3 expand
        name: Optional layer name
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels

        # Squeeze layer
        self.add_child("squeeze", Conv2d(in_channels, squeeze_channels, kernel_size=1))
        self.add_child("squeeze_activation", ReLU())

        # Expand 1x1
        self.add_child("expand1x1", Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1))
        self.add_child("expand1x1_activation", ReLU())

        # Expand 3x3
        self.add_child("expand3x3", Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1))
        self.add_child("expand3x3_activation", ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Output tensor (N, expand1x1 + expand3x3, H, W)
        """
        # Squeeze
        x = self._children["squeeze"](x)
        x = self._children["squeeze_activation"](x)

        # Expand (parallel branches)
        out1x1 = self._children["expand1x1"](x)
        out1x1 = self._children["expand1x1_activation"](out1x1)

        out3x3 = self._children["expand3x3"](x)
        out3x3 = self._children["expand3x3_activation"](out3x3)

        # Concatenate along channel dimension
        return ops.concat([out1x1, out3x3], dim=1)

    @property
    def out_channels(self) -> int:
        """Output channels."""
        return self.expand1x1_channels + self.expand3x3_channels

    def __repr__(self) -> str:
        return (f"Fire(in={self.in_channels}, squeeze={self.squeeze_channels}, "
                f"expand1x1={self.expand1x1_channels}, expand3x3={self.expand3x3_channels})")


class SqueezeNet(Model):
    """SqueezeNet architecture.

    Args:
        version: SqueezeNet version ("1_0" or "1_1")
        num_classes: Number of output classes (default: 1000)
        dropout: Dropout probability (default: 0.5)
        name: Optional model name
    """

    def __init__(
        self,
        version: str = "1_0",
        num_classes: int = 1000,
        dropout: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"squeezenet{version}")

        self.version = version
        self.num_classes = num_classes

        if version == "1_0":
            self._init_v1_0()
        elif version == "1_1":
            self._init_v1_1()
        else:
            raise ValueError(f"Unknown SqueezeNet version: {version}")

        # Classifier
        self.add_child("dropout", Dropout(p=dropout))
        self.add_child("classifier_conv", Conv2d(512, num_classes, kernel_size=1))
        self.add_child("classifier_relu", ReLU())
        self.add_child("avgpool", AdaptiveAvgPool2d((1, 1)))
        self.add_child("flatten", Flatten())

    def _init_v1_0(self):
        """Initialize SqueezeNet 1.0 architecture."""
        # Features
        self.add_child("conv1", Conv2d(3, 96, kernel_size=7, stride=2))
        self.add_child("relu1", ReLU())
        self.add_child("maxpool1", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire2", Fire(96, 16, 64, 64))
        self.add_child("fire3", Fire(128, 16, 64, 64))
        self.add_child("fire4", Fire(128, 32, 128, 128))
        self.add_child("maxpool4", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire5", Fire(256, 32, 128, 128))
        self.add_child("fire6", Fire(256, 48, 192, 192))
        self.add_child("fire7", Fire(384, 48, 192, 192))
        self.add_child("fire8", Fire(384, 64, 256, 256))
        self.add_child("maxpool8", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire9", Fire(512, 64, 256, 256))

    def _init_v1_1(self):
        """Initialize SqueezeNet 1.1 architecture.

        SqueezeNet 1.1 has 2.4x less computation than 1.0
        with no loss in accuracy.
        """
        # Features
        self.add_child("conv1", Conv2d(3, 64, kernel_size=3, stride=2))
        self.add_child("relu1", ReLU())
        self.add_child("maxpool1", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire2", Fire(64, 16, 64, 64))
        self.add_child("fire3", Fire(128, 16, 64, 64))
        self.add_child("maxpool3", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire4", Fire(128, 32, 128, 128))
        self.add_child("fire5", Fire(256, 32, 128, 128))
        self.add_child("maxpool5", MaxPool2d(kernel_size=3, stride=2))

        self.add_child("fire6", Fire(256, 48, 192, 192))
        self.add_child("fire7", Fire(384, 48, 192, 192))
        self.add_child("fire8", Fire(384, 64, 256, 256))
        self.add_child("fire9", Fire(512, 64, 256, 256))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, 224, 224)

        Returns:
            Output logits (N, num_classes)
        """
        # Feature extraction
        if self.version == "1_0":
            x = self._forward_v1_0(x)
        else:
            x = self._forward_v1_1(x)

        # Classifier
        x = self._children["dropout"](x)
        x = self._children["classifier_conv"](x)
        x = self._children["classifier_relu"](x)
        x = self._children["avgpool"](x)
        x = self._children["flatten"](x)

        return x

    def _forward_v1_0(self, x: Tensor) -> Tensor:
        """Forward through v1.0 features."""
        x = self._children["conv1"](x)
        x = self._children["relu1"](x)
        x = self._children["maxpool1"](x)

        x = self._children["fire2"](x)
        x = self._children["fire3"](x)
        x = self._children["fire4"](x)
        x = self._children["maxpool4"](x)

        x = self._children["fire5"](x)
        x = self._children["fire6"](x)
        x = self._children["fire7"](x)
        x = self._children["fire8"](x)
        x = self._children["maxpool8"](x)

        x = self._children["fire9"](x)

        return x

    def _forward_v1_1(self, x: Tensor) -> Tensor:
        """Forward through v1.1 features."""
        x = self._children["conv1"](x)
        x = self._children["relu1"](x)
        x = self._children["maxpool1"](x)

        x = self._children["fire2"](x)
        x = self._children["fire3"](x)
        x = self._children["maxpool3"](x)

        x = self._children["fire4"](x)
        x = self._children["fire5"](x)
        x = self._children["maxpool5"](x)

        x = self._children["fire6"](x)
        x = self._children["fire7"](x)
        x = self._children["fire8"](x)
        x = self._children["fire9"](x)

        return x


def squeezenet1_0(
    pretrained: bool = False,
    num_classes: int = 1000,
    **kwargs,
) -> SqueezeNet:
    """Create SqueezeNet 1.0 model.

    Args:
        pretrained: If True, load PyTorch pretrained weights (requires torch)
        num_classes: Number of output classes
        **kwargs: Additional arguments to SqueezeNet

    Returns:
        SqueezeNet 1.0 model
    """
    model = SqueezeNet(version="1_0", num_classes=num_classes, **kwargs)

    if pretrained:
        _load_pretrained_weights(model, "squeezenet1_0")

    return model


def squeezenet1_1(
    pretrained: bool = False,
    num_classes: int = 1000,
    **kwargs,
) -> SqueezeNet:
    """Create SqueezeNet 1.1 model.

    Args:
        pretrained: If True, load PyTorch pretrained weights (requires torch)
        num_classes: Number of output classes
        **kwargs: Additional arguments to SqueezeNet

    Returns:
        SqueezeNet 1.1 model
    """
    model = SqueezeNet(version="1_1", num_classes=num_classes, **kwargs)

    if pretrained:
        _load_pretrained_weights(model, "squeezenet1_1")

    return model


def _load_pretrained_weights(model: SqueezeNet, model_name: str):
    """Load pretrained weights from PyTorch.

    Args:
        model: SqueezeNet model
        model_name: Model name for torchvision
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        raise ImportError("Loading pretrained weights requires PyTorch. "
                         "Install with: pip install torch torchvision")

    # Get pretrained PyTorch model
    if model_name == "squeezenet1_0":
        torch_model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    else:
        torch_model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

    torch_model.eval()

    # Map PyTorch state dict to KPU model
    torch_state = torch_model.state_dict()
    kpu_state = {}

    # Mapping from PyTorch keys to KPU keys
    # PyTorch: features.0.weight -> KPU: conv1.weight
    key_mapping = _get_squeezenet_key_mapping(model.version)

    for torch_key, tensor in torch_state.items():
        kpu_key = key_mapping.get(torch_key)
        if kpu_key:
            kpu_state[kpu_key] = tensor.numpy()

    model.load_state_dict(kpu_state, strict=False)


def _get_squeezenet_key_mapping(version: str) -> dict:
    """Get key mapping from PyTorch to KPU for SqueezeNet."""
    mapping = {}

    # First conv
    mapping["features.0.weight"] = "conv1.weight"
    mapping["features.0.bias"] = "conv1.bias"

    # Fire modules mapping
    if version == "1_0":
        fire_indices = [(3, 2), (4, 3), (5, 4), (7, 5), (8, 6), (9, 7), (10, 8), (12, 9)]
    else:  # 1_1
        fire_indices = [(3, 2), (4, 3), (6, 4), (7, 5), (9, 6), (10, 7), (11, 8), (12, 9)]

    for torch_idx, kpu_idx in fire_indices:
        fire_name = f"fire{kpu_idx}"

        # Squeeze
        mapping[f"features.{torch_idx}.squeeze.weight"] = f"{fire_name}.squeeze.weight"
        mapping[f"features.{torch_idx}.squeeze.bias"] = f"{fire_name}.squeeze.bias"

        # Expand 1x1
        mapping[f"features.{torch_idx}.expand1x1.weight"] = f"{fire_name}.expand1x1.weight"
        mapping[f"features.{torch_idx}.expand1x1.bias"] = f"{fire_name}.expand1x1.bias"

        # Expand 3x3
        mapping[f"features.{torch_idx}.expand3x3.weight"] = f"{fire_name}.expand3x3.weight"
        mapping[f"features.{torch_idx}.expand3x3.bias"] = f"{fire_name}.expand3x3.bias"

    # Classifier
    mapping["classifier.1.weight"] = "classifier_conv.weight"
    mapping["classifier.1.bias"] = "classifier_conv.bias"

    return mapping


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Fire",
    "SqueezeNet",
    "squeezenet1_0",
    "squeezenet1_1",
]

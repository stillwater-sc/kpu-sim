# python/kpu/model.py
"""
KPU Model Module - nn.Module-like abstractions for KPU-native models.

Provides Layer, Sequential, and Model classes for building neural networks
that can be executed on the KPU simulator.

Example:
    >>> import kpu
    >>> from kpu.model import Sequential, Linear, ReLU
    >>>
    >>> model = Sequential(
    ...     Linear(784, 128),
    ...     ReLU(),
    ...     Linear(128, 10),
    ... )
    >>>
    >>> # Load weights
    >>> model.load_weights({'layer0.weight': w1, 'layer0.bias': b1, ...})
    >>>
    >>> # Execute
    >>> output = model(input_tensor)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .tensor import Tensor
from . import ops


# =============================================================================
# Base Layer Class
# =============================================================================

class Layer(ABC):
    """Base class for all neural network layers.

    Similar to torch.nn.Module but simpler and KPU-focused.

    Attributes:
        name: Optional name for the layer
        training: Whether in training mode (unused in simulation, always False)
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name
        self._training = False
        self._parameters: Dict[str, Tensor] = {}
        self._buffers: Dict[str, Tensor] = {}
        self._children: Dict[str, Layer] = {}

    @property
    def name(self) -> Optional[str]:
        """Layer name."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def training(self) -> bool:
        """Whether in training mode (always False for KPU simulation)."""
        return self._training

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass computation.

        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor
        """
        pass

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Execute forward pass."""
        return self.forward(x, *args, **kwargs)

    def register_parameter(self, name: str, param: Optional[Tensor]):
        """Register a parameter tensor.

        Args:
            name: Parameter name
            param: Parameter tensor (or None)
        """
        if param is not None:
            self._parameters[name] = param
        elif name in self._parameters:
            del self._parameters[name]

    def register_buffer(self, name: str, buffer: Optional[Tensor]):
        """Register a buffer (non-trainable state).

        Args:
            name: Buffer name
            buffer: Buffer tensor (or None)
        """
        if buffer is not None:
            self._buffers[name] = buffer
        elif name in self._buffers:
            del self._buffers[name]

    def add_child(self, name: str, child: Layer):
        """Add a child layer.

        Args:
            name: Child layer name
            child: Child layer instance
        """
        self._children[name] = child
        child.name = name

    def parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over parameters.

        Args:
            recurse: If True, include parameters of child layers

        Yields:
            (name, tensor) tuples
        """
        # Own parameters
        for name, param in self._parameters.items():
            yield name, param

        # Child parameters
        if recurse:
            for child_name, child in self._children.items():
                for param_name, param in child.parameters(recurse=True):
                    yield f"{child_name}.{param_name}", param

    def buffers(self, recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over buffers.

        Args:
            recurse: If True, include buffers of child layers

        Yields:
            (name, tensor) tuples
        """
        for name, buf in self._buffers.items():
            yield name, buf

        if recurse:
            for child_name, child in self._children.items():
                for buf_name, buf in child.buffers(recurse=True):
                    yield f"{child_name}.{buf_name}", buf

    def children(self) -> Iterator[Tuple[str, Layer]]:
        """Iterate over immediate child layers.

        Yields:
            (name, layer) tuples
        """
        for name, child in self._children.items():
            yield name, child

    def modules(self) -> Iterator[Tuple[str, Layer]]:
        """Iterate over all modules (self and descendants).

        Yields:
            (name, layer) tuples
        """
        yield "", self
        for name, child in self._children.items():
            for subname, module in child.modules():
                if subname:
                    yield f"{name}.{subname}", module
                else:
                    yield name, module

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return state dictionary with all parameters and buffers.

        Returns:
            Dictionary mapping names to numpy arrays
        """
        state = {}
        for name, param in self.parameters():
            state[name] = param.numpy()
        for name, buf in self.buffers():
            state[name] = buf.numpy()
        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        """Load state dictionary.

        Args:
            state_dict: Dictionary mapping names to numpy arrays
            strict: If True, raise error on missing/unexpected keys
        """
        own_state = self.state_dict()

        # Check for missing/unexpected keys
        missing_keys = set(own_state.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(own_state.keys())

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")

        # Load parameters
        for name, param in self.parameters():
            if name in state_dict:
                arr = state_dict[name]
                if isinstance(arr, Tensor):
                    arr = arr.numpy()
                param._data = arr.astype(param.dtype)

        # Load buffers
        for name, buf in self.buffers():
            if name in state_dict:
                arr = state_dict[name]
                if isinstance(arr, Tensor):
                    arr = arr.numpy()
                buf._data = arr.astype(buf.dtype)

    def load_weights(self, weights: Dict[str, Union[np.ndarray, Tensor]]):
        """Convenience method to load weights from a dictionary.

        Args:
            weights: Dictionary mapping parameter names to arrays/tensors
        """
        # Convert Tensors to numpy
        state_dict = {}
        for name, w in weights.items():
            if isinstance(w, Tensor):
                state_dict[name] = w.numpy()
            else:
                state_dict[name] = w
        self.load_state_dict(state_dict, strict=False)

    def num_parameters(self) -> int:
        """Count total number of parameters.

        Returns:
            Total parameter count
        """
        total = 0
        for _, param in self.parameters():
            total += param.numel()
        return total

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """Generate model summary.

        Args:
            input_shape: Optional input shape for shape inference

        Returns:
            Summary string
        """
        lines = []
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("-" * 60)
        lines.append(f"{'Layer':<30} {'Output Shape':<15} {'Params':>10}")
        lines.append("=" * 60)

        total_params = 0
        for name, module in self.modules():
            if name == "":
                continue
            params = sum(p.numel() for _, p in module._parameters.items())
            total_params += params
            lines.append(f"{name:<30} {'N/A':<15} {params:>10,}")

        lines.append("=" * 60)
        lines.append(f"Total parameters: {total_params:,}")
        lines.append("-" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Sequential Container
# =============================================================================

class Sequential(Layer):
    """Sequential container for layers.

    Layers are applied in order. Similar to torch.nn.Sequential.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 128),
        ...     ReLU(),
        ...     Linear(128, 10),
        ... )
        >>> output = model(input)
    """

    def __init__(self, *layers: Layer, name: Optional[str] = None):
        """Initialize sequential container.

        Args:
            *layers: Layers to add in order
            name: Optional name for the container
        """
        super().__init__(name=name)
        for i, layer in enumerate(layers):
            self.add_child(f"layer{i}", layer)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Apply layers sequentially.

        Args:
            x: Input tensor

        Returns:
            Output after all layers
        """
        for _, layer in self._children.items():
            x = layer(x)
        return x

    def __getitem__(self, idx: int) -> Layer:
        """Get layer by index."""
        return self._children[f"layer{idx}"]

    def __len__(self) -> int:
        """Number of layers."""
        return len(self._children)

    def __iter__(self) -> Iterator[Layer]:
        """Iterate over layers."""
        for _, layer in self._children.items():
            yield layer

    def append(self, layer: Layer):
        """Append a layer."""
        idx = len(self._children)
        self.add_child(f"layer{idx}", layer)

    def __repr__(self) -> str:
        lines = [f"Sequential("]
        for name, layer in self._children.items():
            lines.append(f"  ({name}): {layer}")
        lines.append(")")
        return "\n".join(lines)


# =============================================================================
# Common Layers
# =============================================================================

class Linear(Layer):
    """Fully connected layer.

    Applies y = x @ W.T + b

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias (default: True)
        name: Optional layer name
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Initialize parameters (Xavier initialization)
        std = np.sqrt(2.0 / (in_features + out_features))
        weight = np.random.randn(out_features, in_features).astype(np.float32) * std
        self.register_parameter("weight", Tensor(weight))

        if bias:
            self.register_parameter("bias", Tensor(np.zeros(out_features, dtype=np.float32)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transformation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        weight = self._parameters["weight"]
        output = ops.linear(x, weight)

        if self.has_bias and "bias" in self._parameters:
            bias = self._parameters["bias"]
            output = output + bias

        return output

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias})"


class Conv2d(Layer):
    """2D Convolution layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        bias: Whether to include bias (default: True)
        name: Optional layer name
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.has_bias = bias

        # Initialize parameters (Kaiming/He initialization)
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        std = np.sqrt(2.0 / fan_in)
        weight = np.random.randn(
            out_channels, in_channels, kernel_size[0], kernel_size[1]
        ).astype(np.float32) * std
        self.register_parameter("weight", Tensor(weight))

        if bias:
            self.register_parameter("bias", Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D convolution.

        Args:
            x: Input tensor of shape (N, C_in, H, W)

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        weight = self._parameters["weight"]
        bias = self._parameters.get("bias", None)

        output = ops.conv2d(
            x, weight,
            stride=self.stride,
            padding=self.padding,
        )

        if self.has_bias and bias is not None:
            # Add bias: (N, C, H, W) + (C,) -> broadcast
            output = output + bias.reshape(1, -1, 1, 1)

        return output

    def __repr__(self) -> str:
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={self.has_bias})")


class BatchNorm2d(Layer):
    """2D Batch Normalization.

    Args:
        num_features: Number of features/channels
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running stats (unused in inference)
        name: Optional layer name
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.register_parameter("weight", Tensor(np.ones(num_features, dtype=np.float32)))
        self.register_parameter("bias", Tensor(np.zeros(num_features, dtype=np.float32)))

        # Running statistics (buffers)
        self.register_buffer("running_mean", Tensor(np.zeros(num_features, dtype=np.float32)))
        self.register_buffer("running_var", Tensor(np.ones(num_features, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        """Apply batch normalization.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized output tensor
        """
        weight = self._parameters["weight"]
        bias = self._parameters["bias"]
        running_mean = self._buffers["running_mean"]
        running_var = self._buffers["running_var"]

        return ops.batch_norm2d(
            x,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
            eps=self.eps
        )

    def __repr__(self) -> str:
        return f"BatchNorm2d({self.num_features}, eps={self.eps})"


class LayerNorm(Layer):
    """Layer Normalization.

    Args:
        normalized_shape: Shape of normalized dimensions
        eps: Small constant for numerical stability (default: 1e-5)
        name: Optional layer name
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        shape = normalized_shape if len(normalized_shape) > 1 else normalized_shape[0]
        self.register_parameter("weight", Tensor(np.ones(shape, dtype=np.float32)))
        self.register_parameter("bias", Tensor(np.zeros(shape, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        weight = self._parameters["weight"]
        bias = self._parameters["bias"]

        return ops.layer_norm(x, self.normalized_shape, weight=weight, bias=bias, eps=self.eps)

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


# =============================================================================
# Activation Layers
# =============================================================================

class ReLU(Layer):
    """ReLU activation."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class GELU(Layer):
    """GELU activation."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return ops.gelu(x)

    def __repr__(self) -> str:
        return "GELU()"


class SiLU(Layer):
    """SiLU (Swish) activation."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return ops.silu(x)

    def __repr__(self) -> str:
        return "SiLU()"


class Sigmoid(Layer):
    """Sigmoid activation."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Layer):
    """Tanh activation."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Layer):
    """Softmax activation.

    Args:
        dim: Dimension to apply softmax over (default: -1)
    """

    def __init__(self, dim: int = -1, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return ops.softmax(x, axis=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


# =============================================================================
# Pooling Layers
# =============================================================================

class MaxPool2d(Layer):
    """2D Max Pooling.

    Args:
        kernel_size: Pooling window size
        stride: Stride (default: same as kernel_size)
        padding: Padding (default: 0)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return ops.max_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Layer):
    """2D Average Pooling.

    Args:
        kernel_size: Pooling window size
        stride: Stride (default: same as kernel_size)
        padding: Padding (default: 0)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return ops.avg_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)

    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AdaptiveAvgPool2d(Layer):
    """2D Adaptive Average Pooling.

    Args:
        output_size: Target output size (H, W) or single int
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return ops.adaptive_avg_pool2d(x, self.output_size)

    def __repr__(self) -> str:
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"


# =============================================================================
# Utility Layers
# =============================================================================

class Flatten(Layer):
    """Flatten layer.

    Args:
        start_dim: First dimension to flatten (default: 1)
        end_dim: Last dimension to flatten (default: -1)
    """

    def __init__(
        self,
        start_dim: int = 1,
        end_dim: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return ops.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


class Dropout(Layer):
    """Dropout layer.

    Note: In KPU simulation (inference mode), dropout is always disabled.

    Args:
        p: Dropout probability (unused in inference)
    """

    def __init__(self, p: float = 0.5, name: Optional[str] = None):
        super().__init__(name=name)
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # In inference mode, dropout is identity
        return x

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Identity(Layer):
    """Identity layer (no-op)."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return x

    def __repr__(self) -> str:
        return "Identity()"


# =============================================================================
# Model Base Class
# =============================================================================

class Model(Layer):
    """Base class for models with additional functionality.

    Provides:
    - Compilation via @kpu.compile
    - State dict save/load
    - Model summary
    - Input/output shape tracking
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self._compiled_fn = None
        self._input_shape = None
        self._output_shape = None

    def compile(self, sample_input: Optional[Tensor] = None):
        """Compile the model for execution.

        Args:
            sample_input: Optional sample input for shape inference
        """
        from .compiler import compile as kpu_compile

        # Create a wrapper function that captures self
        def model_forward(x):
            return self.forward(x)

        self._compiled_fn = kpu_compile(model_forward)

        if sample_input is not None:
            # Run once to trace
            _ = self._compiled_fn(sample_input)
            self._input_shape = sample_input.shape

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Execute the model."""
        if self._compiled_fn is not None:
            return self._compiled_fn(x, *args, **kwargs)
        return self.forward(x, *args, **kwargs)

    @property
    def graph(self):
        """Get the operation graph (if compiled)."""
        if self._compiled_fn is not None:
            return self._compiled_fn.graph
        return None

    @property
    def dfx(self):
        """Get the DFX IR (if compiled)."""
        if self._compiled_fn is not None:
            return self._compiled_fn.dfx
        return None

    @property
    def stats(self):
        """Get execution statistics (if available)."""
        if self._compiled_fn is not None:
            return self._compiled_fn.stats
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model_from_state_dict(
    model_class: type,
    state_dict: Dict[str, np.ndarray],
    **kwargs
) -> Model:
    """Create a model and load state dict.

    Args:
        model_class: Model class to instantiate
        state_dict: State dictionary with weights
        **kwargs: Arguments to pass to model constructor

    Returns:
        Model with loaded weights
    """
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    return model


def save_state_dict(model: Layer, path: str):
    """Save model state dict to numpy file.

    Args:
        model: Model to save
        path: Path to save to (.npz format)
    """
    state_dict = model.state_dict()
    np.savez(path, **state_dict)


def load_state_dict_from_file(path: str) -> Dict[str, np.ndarray]:
    """Load state dict from numpy file.

    Args:
        path: Path to .npz file

    Returns:
        State dictionary
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "Layer",
    "Sequential",
    "Model",
    # Linear layers
    "Linear",
    # Convolution layers
    "Conv2d",
    # Normalization layers
    "BatchNorm2d",
    "LayerNorm",
    # Activation layers
    "ReLU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    # Utility layers
    "Flatten",
    "Dropout",
    "Identity",
    # Functions
    "load_model_from_state_dict",
    "save_state_dict",
    "load_state_dict_from_file",
]

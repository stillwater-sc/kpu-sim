# python/kpu/model_loader.py
"""
KPU Model Loader - Load models from JSON and ONNX formats.

Supports:
- JSON model definition format (custom KPU format)
- ONNX model loading (requires onnx and onnxruntime)
- PyTorch state dict loading

Example:
    >>> from kpu.model_loader import ModelLoader
    >>>
    >>> # Load from JSON
    >>> model = ModelLoader.from_json("model.json")
    >>>
    >>> # Load from ONNX
    >>> model = ModelLoader.from_onnx("model.onnx")
    >>>
    >>> # Load PyTorch weights into KPU model
    >>> model = create_squeezenet()
    >>> ModelLoader.load_pytorch_weights(model, "squeezenet.pth")
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .model import (
    Layer, Sequential, Model, Linear, Conv2d, BatchNorm2d, LayerNorm,
    ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax,
    MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
    Flatten, Dropout, Identity,
)
from .tensor import Tensor


# =============================================================================
# JSON Model Format
# =============================================================================

@dataclass
class LayerSpec:
    """Specification for a layer in JSON format."""
    type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    """Specification for a complete model in JSON format."""
    name: str
    version: str
    layers: List[LayerSpec]
    inputs: List[str]
    outputs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Layer type registry
LAYER_REGISTRY: Dict[str, Type[Layer]] = {
    "linear": Linear,
    "fc": Linear,  # alias
    "conv2d": Conv2d,
    "batchnorm2d": BatchNorm2d,
    "bn2d": BatchNorm2d,  # alias
    "layernorm": LayerNorm,
    "ln": LayerNorm,  # alias
    "relu": ReLU,
    "gelu": GELU,
    "silu": SiLU,
    "swish": SiLU,  # alias
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "maxpool2d": MaxPool2d,
    "avgpool2d": AvgPool2d,
    "adaptiveavgpool2d": AdaptiveAvgPool2d,
    "globalavgpool2d": AdaptiveAvgPool2d,  # alias for (1,1)
    "flatten": Flatten,
    "dropout": Dropout,
    "identity": Identity,
}


def register_layer(name: str, layer_class: Type[Layer]):
    """Register a custom layer type.

    Args:
        name: Layer type name (used in JSON)
        layer_class: Layer class to instantiate
    """
    LAYER_REGISTRY[name.lower()] = layer_class


# =============================================================================
# Model Loader Class
# =============================================================================

class ModelLoader:
    """Load models from various formats."""

    @staticmethod
    def from_json(path: Union[str, Path]) -> Model:
        """Load model from JSON file.

        JSON format:
        {
            "name": "model_name",
            "version": "1.0",
            "inputs": ["input"],
            "outputs": ["output"],
            "layers": [
                {"type": "conv2d", "name": "conv1", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3}},
                {"type": "relu", "name": "relu1"},
                ...
            ],
            "weights": "weights.npz"  // optional external weights file
        }

        Args:
            path: Path to JSON file

        Returns:
            Loaded model
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        # Parse specification
        spec = ModelSpec(
            name=data.get("name", "model"),
            version=data.get("version", "1.0"),
            layers=[
                LayerSpec(
                    type=layer["type"],
                    name=layer.get("name", f"layer{i}"),
                    params=layer.get("params", {}),
                    inputs=layer.get("inputs", []),
                )
                for i, layer in enumerate(data["layers"])
            ],
            inputs=data.get("inputs", ["input"]),
            outputs=data.get("outputs", ["output"]),
            metadata=data.get("metadata", {}),
        )

        # Build model
        model = ModelLoader._build_model_from_spec(spec)

        # Load weights if specified
        weights_path = data.get("weights")
        if weights_path:
            weights_file = path.parent / weights_path
            if weights_file.exists():
                state_dict = np.load(weights_file)
                model.load_state_dict({k: state_dict[k] for k in state_dict.files})

        return model

    @staticmethod
    def _build_model_from_spec(spec: ModelSpec) -> Model:
        """Build model from specification.

        Args:
            spec: Model specification

        Returns:
            Built model
        """
        # For simple sequential models
        layers = []
        for layer_spec in spec.layers:
            layer = ModelLoader._create_layer(layer_spec)
            layers.append(layer)

        # Create sequential model
        model = _SequentialModel(*layers, name=spec.name)
        return model

    @staticmethod
    def _create_layer(spec: LayerSpec) -> Layer:
        """Create a layer from specification.

        Args:
            spec: Layer specification

        Returns:
            Layer instance
        """
        layer_type = spec.type.lower()

        if layer_type not in LAYER_REGISTRY:
            raise ValueError(f"Unknown layer type: {spec.type}. "
                           f"Available types: {list(LAYER_REGISTRY.keys())}")

        layer_class = LAYER_REGISTRY[layer_type]

        # Handle special cases
        params = spec.params.copy()

        # Convert tuple parameters from lists
        for key in ["kernel_size", "stride", "padding", "output_size", "normalized_shape"]:
            if key in params and isinstance(params[key], list):
                params[key] = tuple(params[key])

        # GlobalAvgPool2d is AdaptiveAvgPool2d with output_size=(1,1)
        if layer_type == "globalavgpool2d":
            params["output_size"] = (1, 1)

        # Create layer
        try:
            layer = layer_class(**params, name=spec.name)
        except TypeError as e:
            raise ValueError(f"Failed to create {spec.type} layer '{spec.name}': {e}")

        return layer

    @staticmethod
    def from_onnx(path: Union[str, Path]) -> Model:
        """Load model from ONNX file.

        Requires: pip install onnx

        Args:
            path: Path to ONNX file

        Returns:
            Loaded model
        """
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            raise ImportError("ONNX loading requires 'onnx' package. "
                            "Install with: pip install onnx")

        path = Path(path)
        onnx_model = onnx.load(str(path))

        # Extract graph info
        graph = onnx_model.graph

        # Extract weights
        weights = {}
        for initializer in graph.initializer:
            weights[initializer.name] = numpy_helper.to_array(initializer)

        # Build model from ONNX graph
        model = ModelLoader._build_model_from_onnx_graph(graph, weights)

        return model

    @staticmethod
    def _build_model_from_onnx_graph(graph, weights: Dict[str, np.ndarray]) -> Model:
        """Build model from ONNX graph.

        Args:
            graph: ONNX graph
            weights: Dictionary of weights

        Returns:
            Built model
        """
        # Map ONNX ops to KPU layers
        onnx_to_kpu = {
            "Conv": "conv2d",
            "Gemm": "linear",
            "MatMul": "linear",
            "Relu": "relu",
            "Sigmoid": "sigmoid",
            "Tanh": "tanh",
            "Softmax": "softmax",
            "MaxPool": "maxpool2d",
            "AveragePool": "avgpool2d",
            "GlobalAveragePool": "adaptiveavgpool2d",
            "BatchNormalization": "batchnorm2d",
            "Flatten": "flatten",
            "Dropout": "dropout",
        }

        layers = []

        for node in graph.node:
            op_type = node.op_type

            if op_type not in onnx_to_kpu:
                # Skip unsupported ops or use identity
                continue

            kpu_type = onnx_to_kpu[op_type]

            # Extract attributes
            attrs = {attr.name: ModelLoader._onnx_attr_to_python(attr)
                    for attr in node.attribute}

            # Build layer params based on op type
            params = ModelLoader._onnx_attrs_to_kpu_params(op_type, attrs, node, weights)

            spec = LayerSpec(
                type=kpu_type,
                name=node.name or f"{op_type}_{len(layers)}",
                params=params,
            )

            layer = ModelLoader._create_layer(spec)

            # Load weights for this layer
            ModelLoader._load_onnx_weights(layer, node, weights)

            layers.append(layer)

        model = _SequentialModel(*layers)
        return model

    @staticmethod
    def _onnx_attr_to_python(attr) -> Any:
        """Convert ONNX attribute to Python value."""
        import onnx

        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        else:
            return None

    @staticmethod
    def _onnx_attrs_to_kpu_params(
        op_type: str,
        attrs: Dict[str, Any],
        node,
        weights: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Convert ONNX attributes to KPU layer parameters."""
        params = {}

        if op_type == "Conv":
            # Infer channels from weights
            weight_name = node.input[1]
            if weight_name in weights:
                w = weights[weight_name]
                params["out_channels"] = w.shape[0]
                params["in_channels"] = w.shape[1]
                params["kernel_size"] = (w.shape[2], w.shape[3])

            # Get attributes
            if "strides" in attrs:
                params["stride"] = tuple(attrs["strides"])
            if "pads" in attrs:
                # ONNX uses [top, left, bottom, right], we need (h, w)
                pads = attrs["pads"]
                params["padding"] = (pads[0], pads[1])

            params["bias"] = len(node.input) > 2

        elif op_type in ("Gemm", "MatMul"):
            # Infer features from weights
            for inp in node.input:
                if inp in weights:
                    w = weights[inp]
                    if len(w.shape) == 2:
                        params["in_features"] = w.shape[1]
                        params["out_features"] = w.shape[0]
                        break

        elif op_type == "MaxPool":
            if "kernel_shape" in attrs:
                params["kernel_size"] = tuple(attrs["kernel_shape"])
            if "strides" in attrs:
                params["stride"] = tuple(attrs["strides"])

        elif op_type == "AveragePool":
            if "kernel_shape" in attrs:
                params["kernel_size"] = tuple(attrs["kernel_shape"])
            if "strides" in attrs:
                params["stride"] = tuple(attrs["strides"])

        elif op_type == "GlobalAveragePool":
            params["output_size"] = (1, 1)

        elif op_type == "BatchNormalization":
            # Infer num_features from weights
            for inp in node.input[1:]:  # Skip input, get gamma/beta/mean/var
                if inp in weights:
                    params["num_features"] = weights[inp].shape[0]
                    break
            if "epsilon" in attrs:
                params["eps"] = attrs["epsilon"]

        elif op_type == "Softmax":
            if "axis" in attrs:
                params["dim"] = attrs["axis"]

        elif op_type == "Dropout":
            if "ratio" in attrs:
                params["p"] = attrs["ratio"]

        return params

    @staticmethod
    def _load_onnx_weights(layer: Layer, node, weights: Dict[str, np.ndarray]):
        """Load ONNX weights into a layer."""
        op_type = node.op_type

        if op_type == "Conv":
            # Weight is second input
            if len(node.input) > 1 and node.input[1] in weights:
                layer._parameters["weight"]._data = weights[node.input[1]].astype(np.float32)
            # Bias is third input
            if len(node.input) > 2 and node.input[2] in weights:
                if "bias" in layer._parameters:
                    layer._parameters["bias"]._data = weights[node.input[2]].astype(np.float32)

        elif op_type in ("Gemm", "MatMul"):
            # Weight
            for i, inp in enumerate(node.input):
                if inp in weights:
                    w = weights[inp]
                    if len(w.shape) == 2:
                        layer._parameters["weight"]._data = w.astype(np.float32)
                    elif len(w.shape) == 1 and "bias" in layer._parameters:
                        layer._parameters["bias"]._data = w.astype(np.float32)

        elif op_type == "BatchNormalization":
            # Order: scale (gamma), bias (beta), mean, var
            names = ["weight", "bias", "running_mean", "running_var"]
            for i, inp in enumerate(node.input[1:5]):  # Skip input tensor
                if inp in weights and i < len(names):
                    name = names[i]
                    if name in layer._parameters:
                        layer._parameters[name]._data = weights[inp].astype(np.float32)
                    elif name in layer._buffers:
                        layer._buffers[name]._data = weights[inp].astype(np.float32)

    @staticmethod
    def load_pytorch_weights(
        model: Model,
        path: Union[str, Path],
        strict: bool = False,
        key_map: Optional[Dict[str, str]] = None,
    ):
        """Load PyTorch weights into a KPU model.

        Requires: pip install torch

        Args:
            model: KPU model to load weights into
            path: Path to PyTorch .pth file
            strict: If True, require all keys to match
            key_map: Optional mapping from PyTorch keys to KPU keys
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch weight loading requires 'torch' package. "
                            "Install with: pip install torch")

        path = Path(path)

        # Load state dict
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Handle nested state dicts (e.g., {'model': {...}, 'optimizer': {...}})
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Convert to numpy
        numpy_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                numpy_state_dict[key] = value.numpy()
            else:
                numpy_state_dict[key] = value

        # Apply key mapping if provided
        if key_map:
            mapped_dict = {}
            for old_key, new_key in key_map.items():
                if old_key in numpy_state_dict:
                    mapped_dict[new_key] = numpy_state_dict[old_key]
            numpy_state_dict = mapped_dict

        # Load into model
        model.load_state_dict(numpy_state_dict, strict=strict)

    @staticmethod
    def save_to_json(model: Model, path: Union[str, Path], save_weights: bool = True):
        """Save model to JSON format.

        Args:
            model: Model to save
            path: Path to save JSON file
            save_weights: If True, save weights to separate .npz file
        """
        path = Path(path)

        # Build layer specs
        layers = []
        for name, layer in model.modules():
            if name == "":  # Skip root
                continue

            layer_spec = {
                "type": layer.__class__.__name__.lower(),
                "name": name,
                "params": ModelLoader._get_layer_params(layer),
            }
            layers.append(layer_spec)

        # Build model spec
        model_dict = {
            "name": model.name or "model",
            "version": "1.0",
            "inputs": ["input"],
            "outputs": ["output"],
            "layers": layers,
        }

        # Save weights
        if save_weights:
            weights_path = path.with_suffix(".npz")
            state_dict = model.state_dict()
            np.savez(weights_path, **state_dict)
            model_dict["weights"] = weights_path.name

        # Save JSON
        with open(path, "w") as f:
            json.dump(model_dict, f, indent=2)

    @staticmethod
    def _get_layer_params(layer: Layer) -> Dict[str, Any]:
        """Extract constructor parameters from a layer."""
        params = {}

        if isinstance(layer, Linear):
            params["in_features"] = layer.in_features
            params["out_features"] = layer.out_features
            params["bias"] = layer.has_bias

        elif isinstance(layer, Conv2d):
            params["in_channels"] = layer.in_channels
            params["out_channels"] = layer.out_channels
            params["kernel_size"] = list(layer.kernel_size)
            params["stride"] = list(layer.stride)
            params["padding"] = list(layer.padding)
            params["bias"] = layer.has_bias

        elif isinstance(layer, BatchNorm2d):
            params["num_features"] = layer.num_features
            params["eps"] = layer.eps

        elif isinstance(layer, LayerNorm):
            params["normalized_shape"] = list(layer.normalized_shape)
            params["eps"] = layer.eps

        elif isinstance(layer, (MaxPool2d, AvgPool2d)):
            params["kernel_size"] = list(layer.kernel_size)
            params["stride"] = list(layer.stride)
            params["padding"] = list(layer.padding)

        elif isinstance(layer, AdaptiveAvgPool2d):
            params["output_size"] = list(layer.output_size)

        elif isinstance(layer, Softmax):
            params["dim"] = layer.dim

        elif isinstance(layer, Flatten):
            params["start_dim"] = layer.start_dim
            params["end_dim"] = layer.end_dim

        elif isinstance(layer, Dropout):
            params["p"] = layer.p

        return params


# =============================================================================
# Internal Sequential Model Class
# =============================================================================

class _SequentialModel(Model):
    """Internal sequential model class for loaded models."""

    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        for i, layer in enumerate(layers):
            # Use layer's own name if available, otherwise use layer{i}
            layer_name = layer.name if layer.name else f"layer{i}"
            self.add_child(layer_name, layer)

    def forward(self, x: Tensor) -> Tensor:
        for _, layer in self._children.items():
            x = layer(x)
        return x


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(path: Union[str, Path]) -> Model:
    """Load model from file (auto-detect format).

    Args:
        path: Path to model file (.json or .onnx)

    Returns:
        Loaded model
    """
    path = Path(path)

    if path.suffix == ".json":
        return ModelLoader.from_json(path)
    elif path.suffix == ".onnx":
        return ModelLoader.from_onnx(path)
    else:
        raise ValueError(f"Unknown model format: {path.suffix}. "
                        "Supported formats: .json, .onnx")


def save_model(model: Model, path: Union[str, Path], save_weights: bool = True):
    """Save model to file.

    Args:
        model: Model to save
        path: Path to save to (.json format)
        save_weights: If True, save weights to separate .npz file
    """
    ModelLoader.save_to_json(model, path, save_weights=save_weights)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ModelLoader",
    "LayerSpec",
    "ModelSpec",
    "register_layer",
    "load_model",
    "save_model",
    "LAYER_REGISTRY",
]

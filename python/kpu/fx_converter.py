# python/kpu/fx_converter.py
"""
FX Graph to KPU converter.

Converts PyTorch FX GraphModule to executable KPU operations.
This is the core of the torch.compile backend.

For TRANSACTIONAL and CYCLE_ACCURATE fidelity levels, operations are
routed through the KPU runtime with C++ native bindings to collect
timing statistics.
"""

from __future__ import annotations
import operator
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    import torch.fx as fx
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Module-level storage for last execution stats (for torch.compile paths)
_last_torch_compile_stats = None


@dataclass
class KPUValue:
    """
    Represents a value in the KPU execution graph.

    Can be a tensor (numpy array) or a constant.
    """
    name: str
    data: Optional[np.ndarray] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[np.dtype] = None
    is_param: bool = False  # True for weights/biases

    @classmethod
    def from_torch(cls, name: str, tensor: 'torch.Tensor', is_param: bool = False) -> 'KPUValue':
        """Create KPUValue from PyTorch tensor."""
        return cls(
            name=name,
            data=tensor.detach().cpu().numpy(),
            shape=tuple(tensor.shape),
            dtype=np.float32,  # Convert to float32 for now
            is_param=is_param
        )


class FXToKPUConverter:
    """
    Converts FX GraphModule to KPU executable.

    Walks the FX graph, maps operations to kpu equivalents,
    and produces a callable that executes on the KPU simulator.
    """

    def __init__(self,
                 gm: 'fx.GraphModule',
                 example_inputs: List['torch.Tensor'],
                 fidelity: Optional[int] = None):
        """
        Initialize converter.

        Args:
            gm: FX GraphModule to convert
            example_inputs: Example inputs for shape inference
            fidelity: Simulation fidelity level
        """
        self.gm = gm
        self.example_inputs = example_inputs
        self.fidelity = fidelity

        # Maps FX node names to KPUValue
        self.env: Dict[str, KPUValue] = {}

        # Collected parameters (weights, biases)
        self.params: Dict[str, np.ndarray] = {}

        # Operation sequence for execution
        self.ops: List[Tuple[str, Callable, List[str], str]] = []

        # Track all placeholder names in order
        # Dynamo passes all placeholders (both params and inputs) as args at runtime
        self.placeholder_names: List[str] = []

    def convert(self) -> Callable:
        """
        Convert FX graph to KPU executable.

        Returns:
            Callable that takes PyTorch tensors and returns PyTorch tensors
        """
        import torch

        # Extract parameters from the module
        self._extract_params()

        # Process each node in the graph
        for node in self.gm.graph.nodes:
            self._process_node(node)

        # Build the executable
        return self._build_executable()

    def _extract_params(self):
        """Extract parameters (weights, biases) from the GraphModule."""
        for name, param in self.gm.named_parameters():
            self.params[name] = param.detach().cpu().numpy().astype(np.float32)

        for name, buffer in self.gm.named_buffers():
            self.params[name] = buffer.detach().cpu().numpy().astype(np.float32)

    def _process_node(self, node: 'fx.Node'):
        """Process a single FX node."""
        if node.op == 'placeholder':
            self._handle_placeholder(node)
        elif node.op == 'get_attr':
            self._handle_get_attr(node)
        elif node.op == 'call_function':
            self._handle_call_function(node)
        elif node.op == 'call_method':
            self._handle_call_method(node)
        elif node.op == 'call_module':
            self._handle_call_module(node)
        elif node.op == 'output':
            self._handle_output(node)

    def _handle_placeholder(self, node: 'fx.Node'):
        """
        Handle placeholder node.

        In Dynamo-captured graphs, ALL placeholders (both parameters and inputs)
        are passed as arguments at runtime. We simply record them in order.
        """
        # Record placeholder name in order
        self.placeholder_names.append(node.name)

        # Get example shape if available
        idx = len(self.placeholder_names) - 1
        if idx < len(self.example_inputs):
            example = self.example_inputs[idx]
            self.env[node.name] = KPUValue(
                name=node.name,
                shape=tuple(example.shape),
                dtype=np.float32
            )
        else:
            self.env[node.name] = KPUValue(name=node.name)

    def _handle_get_attr(self, node: 'fx.Node'):
        """Handle attribute access (weights, biases)."""
        attr_name = node.target
        # Navigate the attribute path
        attr = self.gm
        for part in attr_name.split('.'):
            attr = getattr(attr, part)

        if isinstance(attr, torch.Tensor):
            self.env[node.name] = KPUValue.from_torch(node.name, attr, is_param=True)
            self.params[node.name] = attr.detach().cpu().numpy().astype(np.float32)
        elif isinstance(attr, (int, float)):
            self.env[node.name] = KPUValue(name=node.name, data=np.array(attr))

    def _handle_call_function(self, node: 'fx.Node'):
        """Handle function calls (torch.relu, torch.matmul, etc.)."""
        target = node.target
        args = node.args
        kwargs = node.kwargs

        # Get target name for comparison (handles torch._C._nn.linear etc)
        target_name = getattr(target, '__name__', str(target))

        # Map torch functions to kpu ops
        if target in (torch.relu, F.relu):
            self._emit_unary_op(node, 'relu')

        elif target in (torch.sigmoid, F.sigmoid, torch.sigmoid_):
            self._emit_unary_op(node, 'sigmoid')

        elif target in (torch.tanh, F.tanh):
            self._emit_unary_op(node, 'tanh')

        elif target == F.gelu:
            self._emit_unary_op(node, 'gelu')

        elif target == F.silu:
            self._emit_unary_op(node, 'silu')

        elif target == F.softmax:
            self._emit_softmax(node)

        elif target in (torch.matmul, torch.mm, torch.bmm):
            self._emit_matmul(node)

        elif target == F.linear or target_name == 'linear':
            # Handle both F.linear and torch._C._nn.linear
            self._emit_linear(node)

        elif target in (F.conv2d, torch.conv2d):
            self._emit_conv2d(node)

        elif target == F.conv3d:
            self._emit_conv3d(node)

        elif target == F.max_pool2d:
            self._emit_max_pool2d(node)

        elif target == F.max_pool3d:
            self._emit_max_pool3d(node)

        elif target == F.avg_pool2d:
            self._emit_avg_pool2d(node)

        elif target == F.avg_pool3d:
            self._emit_avg_pool3d(node)

        elif target == F.adaptive_avg_pool2d:
            self._emit_adaptive_avg_pool2d(node)

        elif target == F.adaptive_avg_pool3d:
            self._emit_adaptive_avg_pool3d(node)

        elif target == F.batch_norm:
            self._emit_batch_norm(node)

        elif target == F.layer_norm:
            self._emit_layer_norm(node)

        elif target == F.scaled_dot_product_attention:
            self._emit_scaled_dot_product_attention(node)

        elif target in (operator.add, torch.add):
            self._emit_binary_op(node, 'add')

        elif target in (operator.sub, torch.sub):
            self._emit_binary_op(node, 'sub')

        elif target in (operator.mul, torch.mul):
            self._emit_binary_op(node, 'mul')

        elif target in (operator.truediv, torch.div):
            self._emit_binary_op(node, 'div')

        elif target == operator.getitem:
            self._emit_getitem(node)

        elif target == torch.cat:
            self._emit_concat(node)

        elif target in (torch.flatten, torch.flatten):
            self._emit_flatten(node)

        elif target == torch.transpose:
            self._emit_transpose(node)

        elif target == torch.reshape:
            self._emit_reshape(node)

        else:
            # Fallback: try to execute in numpy
            self._emit_fallback(node)

    def _handle_call_method(self, node: 'fx.Node'):
        """Handle method calls (tensor.view, tensor.reshape, etc.)."""
        method = node.target

        if method == 'view':
            self._emit_reshape(node)
        elif method == 'reshape':
            self._emit_reshape(node)
        elif method == 'flatten':
            self._emit_flatten(node)
        elif method == 'transpose':
            self._emit_transpose(node)
        elif method == 'permute':
            self._emit_permute(node)
        elif method == 'contiguous':
            # contiguous is a no-op for us
            self._emit_identity(node)
        elif method in ('size', 'shape'):
            self._emit_shape(node)
        elif method == 'mean':
            self._emit_mean(node)
        elif method == 'sum':
            self._emit_sum(node)
        else:
            self._emit_fallback(node)

    def _handle_call_module(self, node: 'fx.Node'):
        """Handle module calls (nn.Conv2d, nn.Linear, etc.)."""
        module = self.gm.get_submodule(node.target)

        if isinstance(module, nn.Conv2d):
            self._emit_conv2d_module(node, module)
        elif isinstance(module, nn.Conv3d):
            self._emit_conv3d_module(node, module)
        elif isinstance(module, nn.Linear):
            self._emit_linear_module(node, module)
        elif isinstance(module, nn.BatchNorm2d):
            self._emit_batch_norm_module(node, module)
        elif isinstance(module, nn.BatchNorm3d):
            self._emit_batch_norm3d_module(node, module)
        elif isinstance(module, nn.LayerNorm):
            self._emit_layer_norm_module(node, module)
        elif isinstance(module, nn.ReLU):
            self._emit_unary_op(node, 'relu')
        elif isinstance(module, nn.GELU):
            self._emit_unary_op(node, 'gelu')
        elif isinstance(module, nn.SiLU):
            self._emit_unary_op(node, 'silu')
        elif isinstance(module, nn.Sigmoid):
            self._emit_unary_op(node, 'sigmoid')
        elif isinstance(module, nn.Tanh):
            self._emit_unary_op(node, 'tanh')
        elif isinstance(module, nn.Softmax):
            self._emit_softmax(node)
        elif isinstance(module, nn.MaxPool2d):
            self._emit_max_pool2d_module(node, module)
        elif isinstance(module, nn.MaxPool3d):
            self._emit_max_pool3d_module(node, module)
        elif isinstance(module, nn.AvgPool2d):
            self._emit_avg_pool2d_module(node, module)
        elif isinstance(module, nn.AvgPool3d):
            self._emit_avg_pool3d_module(node, module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            self._emit_adaptive_avg_pool2d_module(node, module)
        elif isinstance(module, nn.AdaptiveAvgPool3d):
            self._emit_adaptive_avg_pool3d_module(node, module)
        elif isinstance(module, nn.Flatten):
            self._emit_flatten(node)
        elif isinstance(module, nn.Dropout):
            # Dropout is identity in eval mode
            self._emit_identity(node)
        else:
            self._emit_fallback(node)

    def _handle_output(self, node: 'fx.Node'):
        """Handle output node."""
        self.output_names = []
        args = node.args[0]
        if isinstance(args, (list, tuple)):
            for arg in args:
                if hasattr(arg, 'name'):
                    self.output_names.append(arg.name)
        elif hasattr(args, 'name'):
            self.output_names.append(args.name)

    # --- Op emission helpers ---

    def _emit_unary_op(self, node: 'fx.Node', op_name: str):
        """Emit a unary operation (relu, sigmoid, etc.)."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            if op_name == 'relu':
                return np.maximum(x, 0)
            elif op_name == 'sigmoid':
                return 1.0 / (1.0 + np.exp(-x))
            elif op_name == 'tanh':
                return np.tanh(x)
            elif op_name == 'gelu':
                return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            elif op_name == 'silu':
                return x * (1.0 / (1.0 + np.exp(-x)))
            else:
                raise ValueError(f"Unknown unary op: {op_name}")

        self.ops.append((op_name, op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_binary_op(self, node: 'fx.Node', op_name: str):
        """Emit a binary operation (add, mul, etc.)."""
        lhs_name = self._get_arg_name(node.args[0])
        rhs_name = self._get_arg_name(node.args[1])

        def op_fn(tensors, params):
            lhs = self._resolve_value(node.args[0], tensors, params)
            rhs = self._resolve_value(node.args[1], tensors, params)
            if op_name == 'add':
                return lhs + rhs
            elif op_name == 'sub':
                return lhs - rhs
            elif op_name == 'mul':
                return lhs * rhs
            elif op_name == 'div':
                return lhs / rhs
            else:
                raise ValueError(f"Unknown binary op: {op_name}")

        self.ops.append((op_name, op_fn, [lhs_name, rhs_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_matmul(self, node: 'fx.Node'):
        """Emit matrix multiplication."""
        a_name = self._get_arg_name(node.args[0])
        b_name = self._get_arg_name(node.args[1])

        def op_fn(tensors, params):
            a = self._resolve_value(node.args[0], tensors, params)
            b = self._resolve_value(node.args[1], tensors, params)
            return np.matmul(a, b)

        self.ops.append(('matmul', op_fn, [a_name, b_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_linear(self, node: 'fx.Node'):
        """Emit linear layer (y = x @ W.T + b)."""
        # Get input names for DFX program
        input_names = []
        for arg in node.args:
            arg_name = self._get_arg_name(arg)
            if arg_name is not None:
                input_names.append(arg_name)

        def op_fn(tensors, params):
            x = self._resolve_value(node.args[0], tensors, params)
            weight = self._resolve_value(node.args[1], tensors, params)
            bias = self._resolve_value(node.args[2], tensors, params) if len(node.args) > 2 else None

            result = np.matmul(x, weight.T)
            if bias is not None:
                result = result + bias
            return result

        self.ops.append(('linear', op_fn, input_names, node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_linear_module(self, node: 'fx.Node', module: 'nn.Linear'):
        """Emit linear layer from nn.Linear module."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32)
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None

        weight_name = f"{node.target}.weight"
        bias_name = f"{node.target}.bias" if bias is not None else None
        self.params[weight_name] = weight
        if bias is not None:
            self.params[bias_name] = bias

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            w = params[weight_name]
            result = np.matmul(x, w.T)
            if bias_name:
                result = result + params[bias_name]
            return result

        self.ops.append(('linear', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_conv2d(self, node: 'fx.Node'):
        """Emit 2D convolution (supports grouped and depthwise convolutions)."""
        # torch.conv2d signature: input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
        # Dynamo may pass these as positional args or kwargs
        kwargs = dict(node.kwargs)

        def op_fn(tensors, params):
            x = self._resolve_value(node.args[0], tensors, params)
            weight = self._resolve_value(node.args[1], tensors, params)
            bias = self._resolve_value(node.args[2], tensors, params) if len(node.args) > 2 and node.args[2] is not None else None

            # Get stride, padding, dilation, groups from args or kwargs
            # args: input, weight, bias, stride, padding, dilation, groups
            if len(node.args) > 3:
                stride = node.args[3]
            else:
                stride = kwargs.get('stride', (1, 1))

            if len(node.args) > 4:
                padding = node.args[4]
            else:
                padding = kwargs.get('padding', (0, 0))

            # dilation at index 5 (not yet supported, but extract it)
            if len(node.args) > 5:
                dilation = node.args[5]
            else:
                dilation = kwargs.get('dilation', (1, 1))

            # groups at index 6
            if len(node.args) > 6:
                groups = node.args[6]
            else:
                groups = kwargs.get('groups', 1)

            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            return self._numpy_conv2d(x, weight, bias, stride, padding, dilation, groups)

        self.ops.append(('conv2d', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_conv2d_module(self, node: 'fx.Node', module: 'nn.Conv2d'):
        """Emit conv2d from nn.Conv2d module (supports grouped and depthwise)."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32)
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None

        weight_name = f"{node.target}.weight"
        bias_name = f"{node.target}.bias" if bias is not None else None
        self.params[weight_name] = weight
        if bias is not None:
            self.params[bias_name] = bias

        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            w = params[weight_name]
            b = params.get(bias_name) if bias_name else None
            return self._numpy_conv2d(x, w, b, stride, padding, dilation, groups)

        self.ops.append(('conv2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_conv3d(self, node: 'fx.Node'):
        """Emit 3D convolution (for video models)."""
        kwargs = dict(node.kwargs)

        def op_fn(tensors, params):
            x = self._resolve_value(node.args[0], tensors, params)
            weight = self._resolve_value(node.args[1], tensors, params)
            bias = self._resolve_value(node.args[2], tensors, params) if len(node.args) > 2 and node.args[2] is not None else None

            # Get stride, padding, dilation, groups from args or kwargs
            if len(node.args) > 3:
                stride = node.args[3]
            else:
                stride = kwargs.get('stride', (1, 1, 1))

            if len(node.args) > 4:
                padding = node.args[4]
            else:
                padding = kwargs.get('padding', (0, 0, 0))

            if len(node.args) > 5:
                dilation = node.args[5]
            else:
                dilation = kwargs.get('dilation', (1, 1, 1))

            if len(node.args) > 6:
                groups = node.args[6]
            else:
                groups = kwargs.get('groups', 1)

            if isinstance(stride, int):
                stride = (stride, stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation, dilation)

            return self._numpy_conv3d(x, weight, bias, stride, padding, dilation, groups)

        self.ops.append(('conv3d', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_conv3d_module(self, node: 'fx.Node', module: 'nn.Conv3d'):
        """Emit conv3d from nn.Conv3d module."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32)
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None

        weight_name = f"{node.target}.weight"
        bias_name = f"{node.target}.bias" if bias is not None else None
        self.params[weight_name] = weight
        if bias is not None:
            self.params[bias_name] = bias

        stride = module.stride
        padding = module.padding
        groups = module.groups
        dilation = module.dilation

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            w = params[weight_name]
            b = params.get(bias_name) if bias_name else None
            return self._numpy_conv3d(x, w, b, stride, padding, dilation, groups)

        self.ops.append(('conv3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_max_pool2d(self, node: 'fx.Node'):
        """Emit max pooling."""
        kwargs = dict(node.kwargs)
        kernel_size = node.args[1] if len(node.args) > 1 else kwargs.get('kernel_size', 2)
        stride = node.args[2] if len(node.args) > 2 else kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else kwargs.get('padding', 0)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_max_pool2d(x, kernel_size, stride, padding)

        self.ops.append(('max_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_max_pool2d_module(self, node: 'fx.Node', module: 'nn.MaxPool2d'):
        """Emit max pooling from module."""
        kernel_size = module.kernel_size
        stride = module.stride if module.stride else kernel_size
        padding = module.padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_max_pool2d(x, kernel_size, stride, padding)

        self.ops.append(('max_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_avg_pool2d(self, node: 'fx.Node'):
        """Emit average pooling."""
        kwargs = dict(node.kwargs)
        kernel_size = node.args[1] if len(node.args) > 1 else kwargs.get('kernel_size', 2)
        stride = node.args[2] if len(node.args) > 2 else kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else kwargs.get('padding', 0)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_avg_pool2d(x, kernel_size, stride, padding)

        self.ops.append(('avg_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_avg_pool2d_module(self, node: 'fx.Node', module: 'nn.AvgPool2d'):
        """Emit average pooling from module."""
        kernel_size = module.kernel_size
        stride = module.stride if module.stride else kernel_size
        padding = module.padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_avg_pool2d(x, kernel_size, stride, padding)

        self.ops.append(('avg_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_adaptive_avg_pool2d(self, node: 'fx.Node'):
        """Emit adaptive average pooling."""
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get('output_size', (1, 1))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_adaptive_avg_pool2d(x, output_size)

        self.ops.append(('adaptive_avg_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_adaptive_avg_pool2d_module(self, node: 'fx.Node', module: 'nn.AdaptiveAvgPool2d'):
        """Emit adaptive average pooling from module."""
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_adaptive_avg_pool2d(x, output_size)

        self.ops.append(('adaptive_avg_pool2d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    # --- 3D Pooling (Video Models) ---

    def _emit_max_pool3d(self, node: 'fx.Node'):
        """Emit 3D max pooling."""
        kwargs = dict(node.kwargs)
        kernel_size = node.args[1] if len(node.args) > 1 else kwargs.get('kernel_size', 2)
        stride = node.args[2] if len(node.args) > 2 else kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else kwargs.get('padding', 0)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_max_pool3d(x, kernel_size, stride, padding)

        self.ops.append(('max_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_max_pool3d_module(self, node: 'fx.Node', module: 'nn.MaxPool3d'):
        """Emit 3D max pooling from module."""
        kernel_size = module.kernel_size
        stride = module.stride if module.stride else kernel_size
        padding = module.padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_max_pool3d(x, kernel_size, stride, padding)

        self.ops.append(('max_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_avg_pool3d(self, node: 'fx.Node'):
        """Emit 3D average pooling."""
        kwargs = dict(node.kwargs)
        kernel_size = node.args[1] if len(node.args) > 1 else kwargs.get('kernel_size', 2)
        stride = node.args[2] if len(node.args) > 2 else kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else kwargs.get('padding', 0)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_avg_pool3d(x, kernel_size, stride, padding)

        self.ops.append(('avg_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_avg_pool3d_module(self, node: 'fx.Node', module: 'nn.AvgPool3d'):
        """Emit 3D average pooling from module."""
        kernel_size = module.kernel_size
        stride = module.stride if module.stride else kernel_size
        padding = module.padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_avg_pool3d(x, kernel_size, stride, padding)

        self.ops.append(('avg_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_adaptive_avg_pool3d(self, node: 'fx.Node'):
        """Emit 3D adaptive average pooling."""
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get('output_size', (1, 1, 1))
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_adaptive_avg_pool3d(x, output_size)

        self.ops.append(('adaptive_avg_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_adaptive_avg_pool3d_module(self, node: 'fx.Node', module: 'nn.AdaptiveAvgPool3d'):
        """Emit 3D adaptive average pooling from module."""
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return self._numpy_adaptive_avg_pool3d(x, output_size)

        self.ops.append(('adaptive_avg_pool3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_batch_norm(self, node: 'fx.Node'):
        """Emit batch normalization (inference mode).

        F.batch_norm signature:
            batch_norm(input, running_mean, running_var, weight=None, bias=None,
                       training=False, momentum=0.1, eps=1e-05)

        Supports both 4D (N,C,H,W) and 5D (N,C,D,H,W) tensors.
        """
        def op_fn(tensors, params):
            x = self._resolve_value(node.args[0], tensors, params)
            # F.batch_norm: args[1]=running_mean, args[2]=running_var, args[3]=weight, args[4]=bias
            running_mean = self._resolve_value(node.args[1], tensors, params) if len(node.args) > 1 else None
            running_var = self._resolve_value(node.args[2], tensors, params) if len(node.args) > 2 else None
            weight = self._resolve_value(node.args[3], tensors, params) if len(node.args) > 3 else None
            bias = self._resolve_value(node.args[4], tensors, params) if len(node.args) > 4 else None
            eps = node.kwargs.get('eps', 1e-5)

            # Determine reshape based on input dimensions (4D vs 5D)
            ndim = len(x.shape)
            if ndim == 4:
                reshape_dims = (1, -1, 1, 1)
                spatial_axes = (0, 2, 3)
            elif ndim == 5:
                reshape_dims = (1, -1, 1, 1, 1)
                spatial_axes = (0, 2, 3, 4)
            else:
                # Fallback for other dimensions
                reshape_dims = (1, -1) + (1,) * (ndim - 2)
                spatial_axes = tuple([0] + list(range(2, ndim)))

            if running_mean is not None and running_var is not None:
                # Use running stats (inference mode)
                mean = running_mean.reshape(reshape_dims)
                var = running_var.reshape(reshape_dims)
            else:
                # Compute batch stats (training mode fallback)
                mean = np.mean(x, axis=spatial_axes, keepdims=True)
                var = np.var(x, axis=spatial_axes, keepdims=True)

            y = (x - mean) / np.sqrt(var + eps)
            if weight is not None:
                y = y * weight.reshape(reshape_dims)
            if bias is not None:
                y = y + bias.reshape(reshape_dims)
            return y

        self.ops.append(('batch_norm', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_batch_norm_module(self, node: 'fx.Node', module: 'nn.BatchNorm2d'):
        """Emit batch norm from module."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32) if module.weight is not None else None
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None
        running_mean = module.running_mean.detach().cpu().numpy().astype(np.float32)
        running_var = module.running_var.detach().cpu().numpy().astype(np.float32)
        eps = module.eps

        prefix = node.target
        if weight is not None:
            self.params[f"{prefix}.weight"] = weight
        if bias is not None:
            self.params[f"{prefix}.bias"] = bias
        self.params[f"{prefix}.running_mean"] = running_mean
        self.params[f"{prefix}.running_var"] = running_var

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            mean = params[f"{prefix}.running_mean"].reshape(1, -1, 1, 1)
            var = params[f"{prefix}.running_var"].reshape(1, -1, 1, 1)
            y = (x - mean) / np.sqrt(var + eps)
            if f"{prefix}.weight" in params:
                y = y * params[f"{prefix}.weight"].reshape(1, -1, 1, 1)
            if f"{prefix}.bias" in params:
                y = y + params[f"{prefix}.bias"].reshape(1, -1, 1, 1)
            return y

        self.ops.append(('batch_norm', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_batch_norm3d_module(self, node: 'fx.Node', module: 'nn.BatchNorm3d'):
        """Emit 3D batch norm from module (video models)."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32) if module.weight is not None else None
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None
        running_mean = module.running_mean.detach().cpu().numpy().astype(np.float32)
        running_var = module.running_var.detach().cpu().numpy().astype(np.float32)
        eps = module.eps

        prefix = node.target
        if weight is not None:
            self.params[f"{prefix}.weight"] = weight
        if bias is not None:
            self.params[f"{prefix}.bias"] = bias
        self.params[f"{prefix}.running_mean"] = running_mean
        self.params[f"{prefix}.running_var"] = running_var

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            # 5D tensor: (N, C, D, H, W)
            mean = params[f"{prefix}.running_mean"].reshape(1, -1, 1, 1, 1)
            var = params[f"{prefix}.running_var"].reshape(1, -1, 1, 1, 1)
            y = (x - mean) / np.sqrt(var + eps)
            if f"{prefix}.weight" in params:
                y = y * params[f"{prefix}.weight"].reshape(1, -1, 1, 1, 1)
            if f"{prefix}.bias" in params:
                y = y + params[f"{prefix}.bias"].reshape(1, -1, 1, 1, 1)
            return y

        self.ops.append(('batch_norm3d', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_layer_norm(self, node: 'fx.Node'):
        """Emit layer normalization."""
        def op_fn(tensors, params):
            x = self._resolve_value(node.args[0], tensors, params)
            normalized_shape = node.args[1] if len(node.args) > 1 else node.kwargs.get('normalized_shape')
            weight = self._resolve_value(node.args[2], tensors, params) if len(node.args) > 2 else None
            bias = self._resolve_value(node.args[3], tensors, params) if len(node.args) > 3 else None
            eps = node.kwargs.get('eps', 1e-5)

            ndim = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
            axes = tuple(range(-ndim, 0))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            y = (x - mean) / np.sqrt(var + eps)
            if weight is not None:
                y = y * weight
            if bias is not None:
                y = y + bias
            return y

        self.ops.append(('layer_norm', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_layer_norm_module(self, node: 'fx.Node', module: 'nn.LayerNorm'):
        """Emit layer norm from module."""
        weight = module.weight.detach().cpu().numpy().astype(np.float32) if module.weight is not None else None
        bias = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else None
        normalized_shape = module.normalized_shape
        eps = module.eps

        prefix = node.target
        if weight is not None:
            self.params[f"{prefix}.weight"] = weight
        if bias is not None:
            self.params[f"{prefix}.bias"] = bias

        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            ndim = len(normalized_shape)
            axes = tuple(range(-ndim, 0))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            y = (x - mean) / np.sqrt(var + eps)
            if f"{prefix}.weight" in params:
                y = y * params[f"{prefix}.weight"]
            if f"{prefix}.bias" in params:
                y = y + params[f"{prefix}.bias"]
            return y

        self.ops.append(('layer_norm', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_softmax(self, node: 'fx.Node'):
        """Emit softmax."""
        dim = node.kwargs.get('dim', -1)
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
            return exp_x / np.sum(exp_x, axis=dim, keepdims=True)

        self.ops.append(('softmax', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_scaled_dot_product_attention(self, node: 'fx.Node'):
        """Emit scaled dot-product attention."""
        # F.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False, scale=None)
        query_name = self._get_arg_name(node.args[0])
        key_name = self._get_arg_name(node.args[1])
        value_name = self._get_arg_name(node.args[2])

        kwargs = dict(node.kwargs)
        is_causal = kwargs.get('is_causal', False)
        scale = kwargs.get('scale', None)

        def op_fn(tensors, params):
            q = self._resolve_value(node.args[0], tensors, params)
            k = self._resolve_value(node.args[1], tensors, params)
            v = self._resolve_value(node.args[2], tensors, params)

            # Get head dimension
            d_k = q.shape[-1]
            if scale is None:
                s = 1.0 / np.sqrt(d_k)
            else:
                s = scale

            # Q @ K^T
            k_t = np.swapaxes(k, -2, -1)
            scores = np.matmul(q, k_t) * s

            # Handle attention mask if provided
            if len(node.args) > 3 and node.args[3] is not None:
                attn_mask = self._resolve_value(node.args[3], tensors, params)
                scores = scores + attn_mask

            # Apply causal mask if requested
            if is_causal:
                seq_len = scores.shape[-1]
                causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                scores = scores + causal_mask

            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Attention @ V
            return np.matmul(attn_weights, v)

        self.ops.append(('attention', op_fn, [query_name, key_name, value_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_concat(self, node: 'fx.Node'):
        """Emit concatenation."""
        tensors_arg = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)

        def op_fn(tensors, params):
            arrays = [self._resolve_value(t, tensors, params) for t in tensors_arg]
            return np.concatenate(arrays, axis=dim)

        self.ops.append(('concat', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_reshape(self, node: 'fx.Node'):
        """Emit reshape/view."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            if node.op == 'call_method':
                # tensor.view(*shape) or tensor.reshape(*shape)
                shape = tuple(a if isinstance(a, int) else tensors.get(self._get_arg_name(a), a)
                              for a in node.args[1:])
            else:
                # torch.reshape(tensor, shape)
                shape = node.args[1]
            return x.reshape(shape)

        self.ops.append(('reshape', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_flatten(self, node: 'fx.Node'):
        """Emit flatten."""
        input_name = self._get_arg_name(node.args[0])
        start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('start_dim', 0)
        end_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get('end_dim', -1)

        def op_fn(tensors, params):
            x = tensors[input_name]
            shape = x.shape
            ndim = len(shape)
            if start_dim < 0:
                start_dim_resolved = ndim + start_dim
            else:
                start_dim_resolved = start_dim
            if end_dim < 0:
                end_dim_resolved = ndim + end_dim
            else:
                end_dim_resolved = end_dim

            new_shape = list(shape[:start_dim_resolved])
            flat_size = 1
            for i in range(start_dim_resolved, end_dim_resolved + 1):
                flat_size *= shape[i]
            new_shape.append(flat_size)
            new_shape.extend(shape[end_dim_resolved + 1:])
            return x.reshape(new_shape)

        self.ops.append(('flatten', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_transpose(self, node: 'fx.Node'):
        """Emit transpose."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            dim0 = node.args[1] if len(node.args) > 1 else 0
            dim1 = node.args[2] if len(node.args) > 2 else 1
            axes = list(range(x.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            return np.transpose(x, axes)

        self.ops.append(('transpose', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_permute(self, node: 'fx.Node'):
        """Emit permute."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            dims = node.args[1:] if len(node.args) > 1 else node.kwargs.get('dims', ())
            return np.transpose(x, dims)

        self.ops.append(('permute', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_mean(self, node: 'fx.Node'):
        """Emit mean reduction."""
        input_name = self._get_arg_name(node.args[0])

        # Extract dim and keepdim at conversion time to avoid FX immutable types
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
        keepdim = node.kwargs.get('keepdim', False)

        # Convert immutable_list to tuple for numpy compatibility
        if dim is not None and hasattr(dim, '__iter__') and not isinstance(dim, (int, type(None))):
            dim = tuple(dim)

        def op_fn(tensors, params):
            x = tensors[input_name]
            return np.mean(x, axis=dim, keepdims=keepdim)

        self.ops.append(('mean', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_sum(self, node: 'fx.Node'):
        """Emit sum reduction."""
        input_name = self._get_arg_name(node.args[0])

        # Extract dim and keepdim at conversion time to avoid FX immutable types
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', None)
        keepdim = node.kwargs.get('keepdim', False)

        # Convert immutable_list to tuple for numpy compatibility
        if dim is not None and hasattr(dim, '__iter__') and not isinstance(dim, (int, type(None))):
            dim = tuple(dim)

        def op_fn(tensors, params):
            x = tensors[input_name]
            return np.sum(x, axis=dim, keepdims=keepdim)

        self.ops.append(('sum', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_getitem(self, node: 'fx.Node'):
        """Emit getitem (indexing).

        Handles both static indices (int, slice, tuple) and dynamic indices
        (tensors computed at runtime, like relative_position_index in Swin).
        """
        input_name = self._get_arg_name(node.args[0])
        index_arg = node.args[1]

        # Check if index is a dynamic tensor (FX node) that needs runtime resolution
        if hasattr(index_arg, 'name'):
            # Dynamic index - resolve at runtime
            index_name = index_arg.name

            def op_fn(tensors, params):
                x = tensors[input_name]
                # Resolve index from tensors or params (can't use 'or' with arrays)
                if index_name in tensors:
                    idx = tensors[index_name]
                elif index_name in params:
                    idx = params[index_name]
                elif index_name in self.params:
                    idx = self.params[index_name]
                else:
                    raise ValueError(f"Could not resolve index '{index_name}'")
                # Convert to appropriate type for numpy indexing
                if hasattr(idx, 'astype'):
                    idx = idx.astype(np.int64)
                return x[idx]

            self.ops.append(('getitem_dynamic', op_fn, [input_name], node.name))
        else:
            # Static index - use directly
            index = index_arg

            def op_fn(tensors, params):
                x = tensors[input_name]
                return x[index]

            self.ops.append(('getitem', op_fn, [input_name], node.name))

        self.env[node.name] = KPUValue(name=node.name)

    def _emit_identity(self, node: 'fx.Node'):
        """Emit identity (pass-through)."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            return tensors[input_name]

        self.ops.append(('identity', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_shape(self, node: 'fx.Node'):
        """Emit shape query."""
        input_name = self._get_arg_name(node.args[0])

        def op_fn(tensors, params):
            x = tensors[input_name]
            return x.shape

        self.ops.append(('shape', op_fn, [input_name], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    def _emit_fallback(self, node: 'fx.Node'):
        """Fallback: execute using torch and convert."""
        import torch

        def op_fn(tensors, params):
            # Convert numpy to torch, execute, convert back
            def to_torch(x):
                if isinstance(x, np.ndarray):
                    t = torch.from_numpy(x)
                    # Ensure float64 arrays are cast to float32 for consistency
                    # with model weights (which are typically float32)
                    if t.dtype == torch.float64:
                        t = t.float()
                    return t
                return x

            def from_torch(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif isinstance(x, (list, tuple)):
                    # Handle operations that return multiple tensors (like chunk, split)
                    return type(x)(from_torch(item) for item in x)
                return x

            torch_args = [to_torch(self._resolve_value(a, tensors, params))
                          for a in node.args]
            torch_kwargs = {k: to_torch(self._resolve_value(v, tensors, params))
                            for k, v in node.kwargs.items()}

            if node.op == 'call_function':
                result = node.target(*torch_args, **torch_kwargs)
            elif node.op == 'call_method':
                result = getattr(torch_args[0], node.target)(*torch_args[1:], **torch_kwargs)
            else:
                raise ValueError(f"Cannot fallback for op type: {node.op}")

            return from_torch(result)

        self.ops.append(('fallback', op_fn, [], node.name))
        self.env[node.name] = KPUValue(name=node.name)

    # --- NumPy implementations of core ops ---

    def _numpy_conv2d(self, x: np.ndarray, weight: np.ndarray,
                       bias: Optional[np.ndarray],
                       stride: Tuple[int, int],
                       padding: Tuple[int, int],
                       dilation: Tuple[int, int] = (1, 1),
                       groups: int = 1) -> np.ndarray:
        """NumPy implementation of 2D convolution using im2col for performance.

        Supports grouped, depthwise, and dilated convolutions:
        - groups=1: Standard convolution
        - groups=C_in: Depthwise convolution (each channel convolved separately)
        - groups>1: Grouped convolution (channels split into groups)
        - dilation>1: Dilated/atrous convolution (spacing between kernel elements)

        im2col unfolds input patches into columns, enabling convolution via
        a single matrix multiplication. This is much faster than nested loops.
        """
        N, C_in, H_in, W_in = x.shape
        C_out, C_in_per_group, K_h, K_w = weight.shape

        # Validate groups parameter
        assert C_in % groups == 0, f"C_in ({C_in}) must be divisible by groups ({groups})"
        assert C_out % groups == 0, f"C_out ({C_out}) must be divisible by groups ({groups})"
        C_in_per_group_expected = C_in // groups
        C_out_per_group = C_out // groups

        # Compute effective kernel size with dilation
        # dilation spreads out kernel elements: effective_K = dilation * (K - 1) + 1
        K_h_eff = dilation[0] * (K_h - 1) + 1
        K_w_eff = dilation[1] * (K_w - 1) + 1

        # Output dimensions account for effective kernel size
        H_out = (H_in + 2 * padding[0] - K_h_eff) // stride[0] + 1
        W_out = (W_in + 2 * padding[1] - K_w_eff) // stride[1] + 1

        # Pad input if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0),
                                  (padding[0], padding[0]),
                                  (padding[1], padding[1])), mode='constant')
        else:
            x_padded = x

        if groups == 1:
            # Standard convolution (original fast path)
            col = self._im2col(x_padded, K_h, K_w, stride, dilation, H_out, W_out)
            weight_col = weight.reshape(C_out, -1)
            result = np.zeros((N, C_out, H_out * W_out), dtype=x.dtype)
            for n in range(N):
                result[n] = weight_col @ col[n]
            result = result.reshape(N, C_out, H_out, W_out)
        else:
            # Grouped convolution (includes depthwise when groups == C_in)
            result = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

            for g in range(groups):
                # Slice input channels for this group
                c_in_start = g * C_in_per_group_expected
                c_in_end = c_in_start + C_in_per_group_expected
                x_group = x_padded[:, c_in_start:c_in_end, :, :]

                # Slice output channels (weights) for this group
                c_out_start = g * C_out_per_group
                c_out_end = c_out_start + C_out_per_group
                weight_group = weight[c_out_start:c_out_end]

                # im2col for this group
                col_group = self._im2col(x_group, K_h, K_w, stride, dilation, H_out, W_out)
                weight_col = weight_group.reshape(C_out_per_group, -1)

                # Convolve this group
                for n in range(N):
                    result[n, c_out_start:c_out_end] = (weight_col @ col_group[n]).reshape(C_out_per_group, H_out, W_out)

        if bias is not None:
            result = result + bias.reshape(1, -1, 1, 1)

        return result

    def _im2col(self, x: np.ndarray, K_h: int, K_w: int,
                stride: Tuple[int, int], dilation: Tuple[int, int],
                H_out: int, W_out: int) -> np.ndarray:
        """Extract image patches into columns for efficient convolution.

        Supports dilated convolution by sampling at spaced-out positions.

        Args:
            x: Input tensor (N, C, H, W) - already padded
            K_h, K_w: Kernel dimensions
            stride: Stride tuple
            dilation: Dilation tuple (spacing between kernel elements)
            H_out, W_out: Output spatial dimensions

        Returns:
            col: (N, C * K_h * K_w, H_out * W_out)
        """
        N, C, H, W = x.shape

        if dilation == (1, 1):
            # Fast path: use stride tricks for efficient patch extraction
            # Shape of output indices
            shape = (N, C, K_h, K_w, H_out, W_out)

            # Compute strides for the view
            s = x.strides
            strides = (s[0], s[1], s[2], s[3], s[2] * stride[0], s[3] * stride[1])

            # Create view of patches
            patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

            # Reshape to (N, C * K_h * K_w, H_out * W_out)
            col = patches.reshape(N, C * K_h * K_w, H_out * W_out)

            # Need to make contiguous copy since as_strided returns a view
            return np.ascontiguousarray(col)
        else:
            # Dilated convolution: sample at spaced positions
            # This is slower but correct for any dilation
            col = np.zeros((N, C * K_h * K_w, H_out * W_out), dtype=x.dtype)

            for oh in range(H_out):
                for ow in range(W_out):
                    # Starting position in input
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]

                    # Output column index
                    out_idx = oh * W_out + ow

                    # Extract patch with dilation
                    col_idx = 0
                    for c in range(C):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                # Dilated position
                                h_pos = h_start + kh * dilation[0]
                                w_pos = w_start + kw * dilation[1]
                                col[:, col_idx, out_idx] = x[:, c, h_pos, w_pos]
                                col_idx += 1

            return col

    def _numpy_max_pool2d(self, x: np.ndarray,
                          kernel_size: Tuple[int, int],
                          stride: Tuple[int, int],
                          padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """NumPy implementation of max pooling using stride tricks."""
        N, C, H_in, W_in = x.shape
        K_h, K_w = kernel_size

        # Apply padding if needed
        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1])),
                       mode='constant', constant_values=-np.inf)
            H_in = x.shape[2]
            W_in = x.shape[3]

        H_out = (H_in - K_h) // stride[0] + 1
        W_out = (W_in - K_w) // stride[1] + 1

        # Use stride tricks to create a view of all windows
        # Shape: (N, C, H_out, W_out, K_h, K_w)
        shape = (N, C, H_out, W_out, K_h, K_w)
        s = x.strides
        strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[2], s[3])

        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # Max over the last two dimensions (kernel dims)
        result = np.max(windows, axis=(4, 5))
        return result

    def _numpy_avg_pool2d(self, x: np.ndarray,
                          kernel_size: Tuple[int, int],
                          stride: Tuple[int, int],
                          padding: Tuple[int, int] = (0, 0),
                          count_include_pad: bool = True) -> np.ndarray:
        """NumPy implementation of average pooling using stride tricks."""
        N, C, H_in, W_in = x.shape
        K_h, K_w = kernel_size

        # Apply padding if needed
        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1])),
                       mode='constant', constant_values=0)
            H_in = x.shape[2]
            W_in = x.shape[3]

        H_out = (H_in - K_h) // stride[0] + 1
        W_out = (W_in - K_w) // stride[1] + 1

        # Use stride tricks to create a view of all windows
        # Shape: (N, C, H_out, W_out, K_h, K_w)
        shape = (N, C, H_out, W_out, K_h, K_w)
        s = x.strides
        strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[2], s[3])

        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # Mean over the last two dimensions (kernel dims)
        result = np.mean(windows, axis=(4, 5))
        return result

    def _numpy_adaptive_avg_pool2d(self, x: np.ndarray,
                                    output_size: Tuple[int, int]) -> np.ndarray:
        """NumPy implementation of adaptive average pooling.

        Optimized for common case of global average pooling (output_size=1,1).
        """
        N, C, H_in, W_in = x.shape
        H_out, W_out = output_size

        # Special case: global average pooling (very common in CNNs)
        if H_out == 1 and W_out == 1:
            return np.mean(x, axis=(2, 3), keepdims=True)

        # General case: compute adaptive kernel sizes
        # When output divides input evenly, use stride tricks
        if H_in % H_out == 0 and W_in % W_out == 0:
            K_h = H_in // H_out
            K_w = W_in // W_out
            # Reshape and mean
            x_reshaped = x.reshape(N, C, H_out, K_h, W_out, K_w)
            return np.mean(x_reshaped, axis=(3, 5))

        # Non-uniform case: vectorize over batch and channel
        result = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = (h_out * H_in) // H_out
                h_end = ((h_out + 1) * H_in) // H_out
                w_start = (w_out * W_in) // W_out
                w_end = ((w_out + 1) * W_in) // W_out
                # Vectorize over N and C
                window = x[:, :, h_start:h_end, w_start:w_end]
                result[:, :, h_out, w_out] = np.mean(window, axis=(2, 3))
        return result

    # --- 3D Operations (Video Models) ---

    def _numpy_conv3d(self, x: np.ndarray, weight: np.ndarray,
                      bias: Optional[np.ndarray],
                      stride: Tuple[int, int, int],
                      padding: Tuple[int, int, int],
                      dilation: Tuple[int, int, int] = (1, 1, 1),
                      groups: int = 1) -> np.ndarray:
        """NumPy implementation of 3D convolution using im2col.

        Args:
            x: Input tensor (N, C_in, D, H, W)
            weight: Filter weights (C_out, C_in/groups, K_d, K_h, K_w)
            bias: Optional bias (C_out,)
            stride: Stride tuple (stride_d, stride_h, stride_w)
            padding: Padding tuple (pad_d, pad_h, pad_w)
            dilation: Dilation tuple (dil_d, dil_h, dil_w)
            groups: Number of groups for grouped convolution
        """
        N, C_in, D_in, H_in, W_in = x.shape
        C_out, C_in_per_group, K_d, K_h, K_w = weight.shape

        # Validate groups parameter
        assert C_in % groups == 0, f"C_in ({C_in}) must be divisible by groups ({groups})"
        assert C_out % groups == 0, f"C_out ({C_out}) must be divisible by groups ({groups})"
        C_in_per_group_expected = C_in // groups
        C_out_per_group = C_out // groups

        # Compute effective kernel size with dilation
        K_d_eff = dilation[0] * (K_d - 1) + 1
        K_h_eff = dilation[1] * (K_h - 1) + 1
        K_w_eff = dilation[2] * (K_w - 1) + 1

        # Output dimensions
        D_out = (D_in + 2 * padding[0] - K_d_eff) // stride[0] + 1
        H_out = (H_in + 2 * padding[1] - K_h_eff) // stride[1] + 1
        W_out = (W_in + 2 * padding[2] - K_w_eff) // stride[2] + 1

        # Pad input if needed
        if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0),
                                  (padding[0], padding[0]),
                                  (padding[1], padding[1]),
                                  (padding[2], padding[2])), mode='constant')
        else:
            x_padded = x

        if groups == 1:
            # Standard convolution
            col = self._im2col_3d(x_padded, K_d, K_h, K_w, stride, dilation, D_out, H_out, W_out)
            weight_col = weight.reshape(C_out, -1)
            result = np.zeros((N, C_out, D_out * H_out * W_out), dtype=x.dtype)
            for n in range(N):
                result[n] = weight_col @ col[n]
            result = result.reshape(N, C_out, D_out, H_out, W_out)
        else:
            # Grouped convolution
            result = np.zeros((N, C_out, D_out, H_out, W_out), dtype=x.dtype)

            for g in range(groups):
                c_in_start = g * C_in_per_group_expected
                c_in_end = c_in_start + C_in_per_group_expected
                x_group = x_padded[:, c_in_start:c_in_end, :, :, :]

                c_out_start = g * C_out_per_group
                c_out_end = c_out_start + C_out_per_group
                weight_group = weight[c_out_start:c_out_end]

                col_group = self._im2col_3d(x_group, K_d, K_h, K_w, stride, dilation, D_out, H_out, W_out)
                weight_col = weight_group.reshape(C_out_per_group, -1)

                for n in range(N):
                    result[n, c_out_start:c_out_end] = (weight_col @ col_group[n]).reshape(C_out_per_group, D_out, H_out, W_out)

        if bias is not None:
            result = result + bias.reshape(1, -1, 1, 1, 1)

        return result

    def _im2col_3d(self, x: np.ndarray, K_d: int, K_h: int, K_w: int,
                   stride: Tuple[int, int, int], dilation: Tuple[int, int, int],
                   D_out: int, H_out: int, W_out: int) -> np.ndarray:
        """Extract 3D patches into columns for efficient 3D convolution.

        Args:
            x: Input tensor (N, C, D, H, W) - already padded
            K_d, K_h, K_w: Kernel dimensions
            stride: Stride tuple (stride_d, stride_h, stride_w)
            dilation: Dilation tuple
            D_out, H_out, W_out: Output spatial dimensions

        Returns:
            col: (N, C * K_d * K_h * K_w, D_out * H_out * W_out)
        """
        N, C, D, H, W = x.shape

        if dilation == (1, 1, 1):
            # Fast path: use stride tricks
            shape = (N, C, K_d, K_h, K_w, D_out, H_out, W_out)
            s = x.strides
            strides = (s[0], s[1], s[2], s[3], s[4],
                       s[2] * stride[0], s[3] * stride[1], s[4] * stride[2])
            patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
            col = patches.reshape(N, C * K_d * K_h * K_w, D_out * H_out * W_out)
            return np.ascontiguousarray(col)
        else:
            # Dilated: slower but correct
            col = np.zeros((N, C * K_d * K_h * K_w, D_out * H_out * W_out), dtype=x.dtype)

            for od in range(D_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        d_start = od * stride[0]
                        h_start = oh * stride[1]
                        w_start = ow * stride[2]
                        out_idx = od * H_out * W_out + oh * W_out + ow

                        col_idx = 0
                        for c in range(C):
                            for kd in range(K_d):
                                for kh in range(K_h):
                                    for kw in range(K_w):
                                        d_pos = d_start + kd * dilation[0]
                                        h_pos = h_start + kh * dilation[1]
                                        w_pos = w_start + kw * dilation[2]
                                        col[:, col_idx, out_idx] = x[:, c, d_pos, h_pos, w_pos]
                                        col_idx += 1
            return col

    def _numpy_max_pool3d(self, x: np.ndarray,
                          kernel_size: Tuple[int, int, int],
                          stride: Tuple[int, int, int],
                          padding: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """NumPy implementation of 3D max pooling."""
        N, C, D_in, H_in, W_in = x.shape
        K_d, K_h, K_w = kernel_size

        # Apply padding
        if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1]),
                          (padding[2], padding[2])),
                       mode='constant', constant_values=-np.inf)
            D_in = x.shape[2]
            H_in = x.shape[3]
            W_in = x.shape[4]

        D_out = (D_in - K_d) // stride[0] + 1
        H_out = (H_in - K_h) // stride[1] + 1
        W_out = (W_in - K_w) // stride[2] + 1

        # Use stride tricks
        shape = (N, C, D_out, H_out, W_out, K_d, K_h, K_w)
        s = x.strides
        strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[4] * stride[2], s[2], s[3], s[4])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return np.max(windows, axis=(5, 6, 7))

    def _numpy_avg_pool3d(self, x: np.ndarray,
                          kernel_size: Tuple[int, int, int],
                          stride: Tuple[int, int, int],
                          padding: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """NumPy implementation of 3D average pooling."""
        N, C, D_in, H_in, W_in = x.shape
        K_d, K_h, K_w = kernel_size

        # Apply padding
        if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1]),
                          (padding[2], padding[2])),
                       mode='constant', constant_values=0)
            D_in = x.shape[2]
            H_in = x.shape[3]
            W_in = x.shape[4]

        D_out = (D_in - K_d) // stride[0] + 1
        H_out = (H_in - K_h) // stride[1] + 1
        W_out = (W_in - K_w) // stride[2] + 1

        shape = (N, C, D_out, H_out, W_out, K_d, K_h, K_w)
        s = x.strides
        strides = (s[0], s[1], s[2] * stride[0], s[3] * stride[1], s[4] * stride[2], s[2], s[3], s[4])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return np.mean(windows, axis=(5, 6, 7))

    def _numpy_adaptive_avg_pool3d(self, x: np.ndarray,
                                    output_size: Tuple[int, int, int]) -> np.ndarray:
        """NumPy implementation of 3D adaptive average pooling."""
        N, C, D_in, H_in, W_in = x.shape
        D_out, H_out, W_out = output_size

        # Global average pooling (common case)
        if D_out == 1 and H_out == 1 and W_out == 1:
            return np.mean(x, axis=(2, 3, 4), keepdims=True)

        # Uniform case
        if D_in % D_out == 0 and H_in % H_out == 0 and W_in % W_out == 0:
            K_d = D_in // D_out
            K_h = H_in // H_out
            K_w = W_in // W_out
            x_reshaped = x.reshape(N, C, D_out, K_d, H_out, K_h, W_out, K_w)
            return np.mean(x_reshaped, axis=(3, 5, 7))

        # Non-uniform case
        result = np.zeros((N, C, D_out, H_out, W_out), dtype=x.dtype)
        for d_out in range(D_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    d_start = (d_out * D_in) // D_out
                    d_end = ((d_out + 1) * D_in) // D_out
                    h_start = (h_out * H_in) // H_out
                    h_end = ((h_out + 1) * H_in) // H_out
                    w_start = (w_out * W_in) // W_out
                    w_end = ((w_out + 1) * W_in) // W_out
                    window = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    result[:, :, d_out, h_out, w_out] = np.mean(window, axis=(2, 3, 4))
        return result

    # --- Helpers ---

    def _get_arg_name(self, arg) -> str:
        """Get the name of an argument (node or constant)."""
        if hasattr(arg, 'name'):
            return arg.name
        else:
            return str(id(arg))

    def _resolve_value(self, arg, tensors: Dict[str, np.ndarray],
                        params: Dict[str, np.ndarray]) -> Any:
        """Resolve an argument to its value."""
        if hasattr(arg, 'name'):
            name = arg.name
            if name in tensors:
                return tensors[name]
            elif name in params:
                return params[name]
            elif name in self.params:
                return self.params[name]
        if isinstance(arg, (int, float, bool, type(None))):
            return arg
        if isinstance(arg, (list, tuple)):
            return type(arg)(self._resolve_value(a, tensors, params) for a in arg)
        return arg

    def _build_executable(self) -> Callable:
        """Build the final executable function."""
        import torch
        from .runtime import BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE

        ops = self.ops
        params = self.params
        output_names = self.output_names
        fidelity = self.fidelity

        # All placeholders in order - Dynamo passes all args (params + inputs) at runtime
        placeholder_names = self.placeholder_names

        # Check if we should use timed execution
        use_timed_execution = fidelity in (TRANSACTIONAL, CYCLE_ACCURATE)

        def executable(*args):
            global _last_torch_compile_stats

            # Initialize tensors dict
            tensors = {}

            # Map all runtime args to all placeholders in order
            for name, arg in zip(placeholder_names, args):
                if isinstance(arg, torch.Tensor):
                    tensors[name] = arg.detach().cpu().numpy().astype(np.float32)
                elif isinstance(arg, np.ndarray):
                    tensors[name] = arg.astype(np.float32)
                else:
                    tensors[name] = arg

            if use_timed_execution:
                # Route through KPU runtime for timing stats
                result_tensors, stats = _execute_with_timing(
                    ops, tensors, params, output_names, fidelity
                )
                _last_torch_compile_stats = stats

                # Convert outputs to PyTorch
                outputs = []
                for name in output_names:
                    out = result_tensors[name]
                    if isinstance(out, np.ndarray):
                        outputs.append(torch.from_numpy(out))
                    else:
                        outputs.append(out)
            else:
                # Execute operations directly (BEHAVIORAL mode)
                for op_name, op_fn, inputs, output in ops:
                    result = op_fn(tensors, params)
                    tensors[output] = result

                # Collect outputs and convert back to PyTorch
                outputs = []
                for name in output_names:
                    out = tensors[name]
                    if isinstance(out, np.ndarray):
                        outputs.append(torch.from_numpy(out))
                    else:
                        outputs.append(out)

                _last_torch_compile_stats = None

            # Dynamo expects outputs as a tuple matching the FX graph's output format
            return tuple(outputs)

        return executable


def _execute_with_timing(ops, tensors, params, output_names, fidelity):
    """Execute operations through KPU runtime to collect timing stats.

    For TRANSACTIONAL and CYCLE_ACCURATE modes, this builds a DFX program
    from the operations and executes it through the native C++ simulator.
    """
    from .runtime import get_runtime, ExecutionStats
    from .dfx_emitter import DFXProgram, DFXOp, DFXOpCode, DFXTensor, DFXDataType, DFXMemLevel
    from .tensor import Tensor

    runtime = get_runtime()
    original_fidelity = runtime.fidelity
    runtime.set_fidelity(fidelity)

    # Set default clock frequency if not set (required for TRANSACTIONAL/CYCLE_ACCURATE)
    from .runtime import is_clock_frequency_set, set_clock_frequency
    if not is_clock_frequency_set():
        set_clock_frequency(1.0)  # Default 1 GHz

    try:
        # Build DFX program from ops
        dfx_ops = []
        tensor_shapes = {}

        # Track input tensor names and their data
        input_names = []
        input_data = []

        # First pass: identify unique inputs
        seen_inputs = set()
        for op_name, op_fn, op_inputs, output in ops:
            for inp in op_inputs:
                if inp in tensors and inp not in seen_inputs:
                    seen_inputs.add(inp)
                    input_names.append(inp)
                    data = tensors[inp]
                    if isinstance(data, np.ndarray):
                        input_data.append(Tensor(data))
                        tensor_shapes[inp] = data.shape
                    else:
                        input_data.append(Tensor(np.array([data])))
                        tensor_shapes[inp] = ()

        # Also include params as inputs
        for pname, pdata in params.items():
            if pname not in seen_inputs:
                seen_inputs.add(pname)
                input_names.append(pname)
                input_data.append(Tensor(pdata))
                tensor_shapes[pname] = pdata.shape

        # Convert ops to DFX format
        for op_name, op_fn, op_inputs, output in ops:
            dfx_opcode = _map_op_to_dfx(op_name)
            if dfx_opcode is not None:
                # Get all input names for this op
                all_inputs = list(op_inputs) if op_inputs else []

                dfx_op = DFXOp(
                    opcode=dfx_opcode,
                    inputs=all_inputs,
                    outputs=[output],
                    attrs={}
                )
                dfx_ops.append(dfx_op)

        # Create DFX tensors from tracked shapes
        dfx_tensors = {}
        for tensor_name, shape in tensor_shapes.items():
            dfx_tensors[tensor_name] = DFXTensor(
                name=tensor_name,
                shape=tuple(shape) if hasattr(shape, '__iter__') else (),
                dtype=DFXDataType.FLOAT32,
                memory_level=DFXMemLevel.EXTERNAL,
                is_const=tensor_name in params
            )

        # Also add output tensors (we don't know shapes yet, but need entries)
        for out_name in output_names:
            if out_name not in dfx_tensors:
                dfx_tensors[out_name] = DFXTensor(
                    name=out_name,
                    shape=(),  # Unknown shape
                    dtype=DFXDataType.FLOAT32,
                    memory_level=DFXMemLevel.EXTERNAL,
                    is_const=False
                )

        # Create DFX program
        program = DFXProgram(
            name="torch_compile_graph",
            tensors=dfx_tensors,
            ops=dfx_ops,
            inputs=input_names,
            outputs=output_names
        )

        # Execute through runtime
        result, stats = runtime.execute(program, input_data)

        # Reconstruct result tensors dict
        # We need to also execute behaviorally to get intermediate values
        # since the C++ runtime only returns final output
        result_tensors = dict(tensors)
        for pname, pdata in params.items():
            result_tensors[pname] = pdata

        # Execute ops to populate intermediate tensors
        for op_name, op_fn, op_inputs, output in ops:
            res = op_fn(result_tensors, params)
            result_tensors[output] = res

        return result_tensors, stats

    finally:
        runtime.set_fidelity(original_fidelity)


def _map_op_to_dfx(op_name: str):
    """Map FX converter op name to DFXOpCode."""
    from .dfx_emitter import DFXOpCode

    op_map = {
        'matmul': DFXOpCode.MATMUL,
        'linear': DFXOpCode.LINEAR,  # Linear: y = x @ W.T + b
        'attention': DFXOpCode.ATTENTION,  # Scaled dot-product attention
        'relu': DFXOpCode.RELU,
        'sigmoid': DFXOpCode.SIGMOID,
        'tanh': DFXOpCode.TANH,
        'gelu': DFXOpCode.GELU,
        'silu': DFXOpCode.SILU,
        'softmax': DFXOpCode.SOFTMAX,
        'add': DFXOpCode.ADD,
        'sub': DFXOpCode.SUB,
        'mul': DFXOpCode.MUL,
        'div': DFXOpCode.DIV,
        'conv2d': DFXOpCode.CONV2D,
        'max_pool2d': DFXOpCode.MAXPOOL2D,
        'avg_pool2d': DFXOpCode.AVGPOOL2D,
        'adaptive_avg_pool2d': DFXOpCode.ADAPTIVE_AVGPOOL2D,
        'batch_norm': DFXOpCode.BATCH_NORM,
        'layer_norm': DFXOpCode.LAYER_NORM,
        'concat': DFXOpCode.CONCAT,
        'reshape': DFXOpCode.RESHAPE,
        'flatten': DFXOpCode.FLATTEN,
        'transpose': DFXOpCode.TRANSPOSE,
        'mean': DFXOpCode.MEAN,
        'sum': DFXOpCode.SUM,
    }
    return op_map.get(op_name)


def get_last_stats():
    """Get execution stats from the last torch.compile execution.

    Returns:
        ExecutionStats from the last TRANSACTIONAL/CYCLE_ACCURATE execution,
        or None if the last execution was BEHAVIORAL or no execution has occurred.
    """
    return _last_torch_compile_stats

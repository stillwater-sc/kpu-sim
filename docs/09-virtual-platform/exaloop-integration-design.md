# Exaloop/Codon Integration Design for KPU Simulator

## Executive Summary

This document designs the integration between [Exaloop's Codon compiler](https://github.com/exaloop/codon) and the KPU functional/transactional simulators. The goal is to enable the compiler team to write neural networks in Python (NumPy/PyTorch style) and execute them on the KPU simulator with a minimal decorator-based interface.

**Target: MNIST MLP as first proof-of-concept**

```python
import kpu

@kpu.compile
def mnist_mlp(x: kpu.Tensor, w1: kpu.Tensor, w2: kpu.Tensor, w3: kpu.Tensor) -> kpu.Tensor:
    h1 = kpu.relu(x @ w1)        # 784 → 128
    h2 = kpu.relu(h1 @ w2)       # 128 → 64
    out = h2 @ w3                # 64 → 10
    return out

# Execute on functional simulator
result = mnist_mlp(input_data, weights1, weights2, weights3)
```

---

## Exaloop/Codon Technology Overview

### What Codon Provides

[Codon](https://docs.exaloop.io/) is a high-performance Python compiler that:

1. **Decorator-based compilation**: `@codon.jit` marks functions for native compilation
2. **Native NumPy**: [Reimplemented in Codon](https://docs.exaloop.io/libraries/numpy/) with operator fusion, 2.4x average speedup
3. **Plugin system**: [DSL extensions](https://docs.exaloop.io/developers/extend/) via `codon::DSL` class
4. **IR passes**: Custom [CIR transformations](https://docs.exaloop.io/developers/ir/) can intercept operations

### Codon Compilation Pipeline

```
Python Source
    ↓ (parsing)
AST
    ↓ (type checking)
CIR (Codon IR)           ← Plugin passes inject here
    ↓ (optimization)
LLVM IR
    ↓ (codegen)
Native Code (x86/ARM)    ← We redirect to KPU DFX
```

### Key Codon Mechanisms

| Mechanism | How It Works | Our Use |
|-----------|--------------|---------|
| `@codon.jit` | Marks function for compilation | Model for `@kpu.compile` |
| NumPy ops | Expressed as `__matmul__`, `__add__` calls | Intercept for KPU kernels |
| Type inference | Static typing from annotations | Shape/dtype propagation |
| Operator fusion | Combines ops in single pass | Already done, we tile further |
| Plugin DSL | `codon::DSL` class extensions | KPU backend plugin |

---

## Integration Architecture

### Phased Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Quick Start (2-3 weeks)                                            │
│ • Pure Python decorator with AST introspection                              │
│ • Generate DFX IR directly from Python                                      │
│ • No Codon dependency, works immediately                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Codon Integration (4-6 weeks)                                      │
│ • Create Codon DSL plugin for KPU                                           │
│ • CIR pass intercepts NumPy ops                                             │
│ • Full Codon optimizations (fusion, etc.)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Production (8-12 weeks)                                            │
│ • Ahead-of-time compilation to loadable KPU objects                         │
│ • Graph optimization, memory planning                                       │
│ • Integration with PyTorch/ONNX model loading                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1 Architecture (Quick Start)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Python User Code                                  │
│                                                                             │
│  @kpu.compile                                                               │
│  def mnist_mlp(x, w1, w2, w3):                                              │
│      h1 = kpu.relu(x @ w1)                                                  │
│      h2 = kpu.relu(h1 @ w2)                                                 │
│      return h2 @ w3                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         kpu.compile Decorator                               │
│                                                                             │
│  1. AST introspection (identify ops: @, relu, etc.)                         │
│  2. Shape inference (from type hints or runtime)                            │
│  3. Build operation graph                                                   │
│  4. Generate DFX IR                                                         │
│  5. Compile to KPU kernels                                                  │
│  6. Return callable that executes on simulator                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KPU Runtime                                    │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  DFX IR      │───>│ KPU Kernels  │───>│  Simulator   │                   │
│  │  (per layer) │    │ (compiled)   │    │ (fsim/tsim)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 Implementation Design

### Python Package Structure

```
python/
├── kpu/
│   ├── __init__.py           # Public API exports
│   ├── tensor.py             # KPU Tensor class (wraps numpy)
│   ├── ops.py                # Operator definitions (relu, softmax, etc.)
│   ├── compiler.py           # @kpu.compile decorator
│   ├── graph.py              # Operation graph representation
│   ├── dfx_emitter.py        # Emit DFX IR from graph
│   ├── runtime.py            # Execute kernels on simulator
│   └── _native/
│       └── kpu_sim.so        # Python bindings to kpu-sim (pybind11)
├── tests/
│   └── test_mnist_mlp.py     # MNIST test case
└── examples/
    └── mnist_mlp.py          # Complete MNIST example
```

### Core API Design

```python
# kpu/__init__.py
from .tensor import Tensor
from .compiler import compile, jit
from .ops import relu, gelu, softmax, sigmoid, tanh
from .runtime import Device, get_device, set_fidelity

# Fidelity levels
BEHAVIORAL = 0      # Functional correctness, computes actual values
TRANSACTIONAL = 1   # Performance estimation, statistical timing
CYCLE_ACCURATE = 2  # Full timing simulation

__all__ = [
    'Tensor', 'compile', 'jit',
    'relu', 'gelu', 'softmax', 'sigmoid', 'tanh',
    'Device', 'get_device', 'set_fidelity',
    'BEHAVIORAL', 'TRANSACTIONAL', 'CYCLE_ACCURATE'
]
```

### Tensor Class

```python
# kpu/tensor.py
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class TensorMeta:
    """Metadata for shape/type tracking during compilation."""
    shape: Tuple[int, ...]
    dtype: np.dtype
    name: Optional[str] = None
    is_weight: bool = False  # Weights can be pre-loaded to L3

class Tensor:
    """
    KPU Tensor - wraps NumPy array with metadata for compilation.

    During tracing (@kpu.compile), operations on Tensors are recorded
    rather than executed. During execution, operations run on simulator.
    """

    _tracing: bool = False  # Class-level tracing flag
    _trace_graph: 'OpGraph' = None

    def __init__(self, data: Union[np.ndarray, 'TensorMeta'], name: str = None):
        if isinstance(data, np.ndarray):
            self._data = data
            self._meta = TensorMeta(
                shape=data.shape,
                dtype=data.dtype,
                name=name
            )
        else:
            # Symbolic tensor for tracing
            self._data = None
            self._meta = data
            self._meta.name = name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._meta.shape

    @property
    def dtype(self) -> np.dtype:
        return self._meta.dtype

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: C = A @ B"""
        if Tensor._tracing:
            return self._trace_matmul(other)
        else:
            return self._execute_matmul(other)

    def _trace_matmul(self, other: 'Tensor') -> 'Tensor':
        """Record matmul operation during tracing."""
        # Infer output shape
        M, K1 = self.shape[-2], self.shape[-1]
        K2, N = other.shape[-2], other.shape[-1]
        assert K1 == K2, f"Shape mismatch: {self.shape} @ {other.shape}"

        out_shape = (*self.shape[:-2], M, N)
        out_meta = TensorMeta(shape=out_shape, dtype=self.dtype)
        out = Tensor(out_meta)

        # Record operation
        Tensor._trace_graph.add_op('matmul', [self, other], [out])
        return out

    def _execute_matmul(self, other: 'Tensor') -> 'Tensor':
        """Execute matmul on KPU simulator."""
        result = np.matmul(self._data, other._data)  # For BEHAVIORAL
        return Tensor(result)
```

### Compiler Decorator

```python
# kpu/compiler.py
import ast
import inspect
import functools
from typing import Callable, List, Dict, Any
from .graph import OpGraph, OpNode
from .dfx_emitter import DFXEmitter
from .runtime import CompiledKernel, get_runtime

def compile(func: Callable = None, *,
            fidelity: int = None,
            optimize: bool = True) -> Callable:
    """
    Decorator to compile a Python function to KPU kernels.

    Usage:
        @kpu.compile
        def mlp(x, w1, w2):
            h = kpu.relu(x @ w1)
            return h @ w2

        # With options:
        @kpu.compile(fidelity=kpu.BEHAVIORAL)
        def mlp(x, w1, w2):
            ...

    The decorated function:
    1. First call: traces the function to build operation graph
    2. Compiles graph to DFX IR
    3. Compiles DFX to KPU kernels
    4. Subsequent calls: executes compiled kernels
    """
    def decorator(fn: Callable) -> Callable:
        compiled_kernel: CompiledKernel = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled_kernel

            if compiled_kernel is None:
                # First call: trace and compile
                compiled_kernel = _trace_and_compile(fn, args, kwargs,
                                                     fidelity, optimize)

            # Execute on simulator
            return compiled_kernel.execute(*args, **kwargs)

        # Allow explicit compilation without execution
        wrapper.compile = lambda *args: _trace_and_compile(fn, args, {},
                                                            fidelity, optimize)
        wrapper.get_dfx = lambda: compiled_kernel.dfx_program if compiled_kernel else None
        wrapper.get_graph = lambda: compiled_kernel.op_graph if compiled_kernel else None

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _trace_and_compile(fn: Callable,
                       args: tuple,
                       kwargs: dict,
                       fidelity: int,
                       optimize: bool) -> 'CompiledKernel':
    """Trace function execution and compile to KPU kernels."""
    from .tensor import Tensor

    # Create symbolic tensors for tracing
    symbolic_args = []
    for i, arg in enumerate(args):
        if isinstance(arg, Tensor):
            # Create symbolic version
            sym = Tensor(arg._meta, name=f"arg{i}")
            symbolic_args.append(sym)
        elif isinstance(arg, np.ndarray):
            meta = TensorMeta(shape=arg.shape, dtype=arg.dtype, name=f"arg{i}")
            symbolic_args.append(Tensor(meta))
        else:
            symbolic_args.append(arg)

    # Enable tracing
    graph = OpGraph()
    Tensor._tracing = True
    Tensor._trace_graph = graph

    try:
        # Execute function symbolically
        result = fn(*symbolic_args, **kwargs)

        # Mark outputs
        if isinstance(result, Tensor):
            graph.mark_output(result)
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, Tensor):
                    graph.mark_output(r)
    finally:
        Tensor._tracing = False
        Tensor._trace_graph = None

    # Compile graph to DFX
    emitter = DFXEmitter()
    dfx_program = emitter.emit(graph)

    # Compile DFX to KPU kernels
    runtime = get_runtime()
    kernels = runtime.compile_dfx(dfx_program, optimize=optimize)

    return CompiledKernel(
        op_graph=graph,
        dfx_program=dfx_program,
        kernels=kernels,
        fidelity=fidelity
    )
```

### Operation Graph

```python
# kpu/graph.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

class OpType(Enum):
    # Data movement
    INPUT = "input"
    OUTPUT = "output"

    # Compute
    MATMUL = "matmul"
    RELU = "relu"
    GELU = "gelu"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    # Elementwise
    ADD = "add"
    MUL = "mul"
    SUB = "sub"
    DIV = "div"

@dataclass
class OpNode:
    """A single operation in the graph."""
    op_type: OpType
    inputs: List['Tensor']
    outputs: List['Tensor']
    attrs: Dict[str, any] = field(default_factory=dict)
    name: str = ""

    # For scheduling
    _id: int = -1
    _deps: Set[int] = field(default_factory=set)

class OpGraph:
    """
    Computational graph built during tracing.

    Represents the DAG of operations to be compiled to DFX IR.
    """

    def __init__(self):
        self.nodes: List[OpNode] = []
        self.inputs: List['Tensor'] = []
        self.outputs: List['Tensor'] = []
        self._tensor_to_producer: Dict[int, int] = {}  # tensor_id -> node_id
        self._next_id = 0

    def add_op(self, op_type: str, inputs: List['Tensor'],
               outputs: List['Tensor'], **attrs) -> OpNode:
        """Add an operation to the graph."""
        node = OpNode(
            op_type=OpType(op_type),
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            _id=self._next_id
        )
        self._next_id += 1

        # Track dependencies
        for inp in inputs:
            producer_id = self._tensor_to_producer.get(id(inp))
            if producer_id is not None:
                node._deps.add(producer_id)

        # Track producers
        for out in outputs:
            self._tensor_to_producer[id(out)] = node._id

        self.nodes.append(node)
        return node

    def mark_input(self, tensor: 'Tensor'):
        """Mark a tensor as a graph input."""
        self.inputs.append(tensor)

    def mark_output(self, tensor: 'Tensor'):
        """Mark a tensor as a graph output."""
        self.outputs.append(tensor)

    def topological_order(self) -> List[OpNode]:
        """Return nodes in topological order."""
        visited = set()
        order = []

        def visit(node_id: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for dep_id in node._deps:
                visit(dep_id)
            order.append(node)

        for node in self.nodes:
            visit(node._id)

        return order

    def to_dfx_ops(self) -> List['DFXOp']:
        """Convert graph to DFX operations."""
        from .dfx_emitter import DFXEmitter
        emitter = DFXEmitter()
        return emitter.emit_ops(self)
```

### DFX IR Emitter

```python
# kpu/dfx_emitter.py
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from .graph import OpGraph, OpNode, OpType

class DFXDataType(Enum):
    FLOAT32 = "f32"
    FLOAT16 = "f16"
    BFLOAT16 = "bf16"
    INT8 = "i8"

class DFXMemLevel(Enum):
    EXTERNAL = 0  # DRAM
    L3 = 1
    L2 = 2
    L1 = 3
    REGISTER = 4

@dataclass
class DFXTensor:
    """Tensor descriptor in DFX IR."""
    name: str
    shape: Tuple[int, ...]
    dtype: DFXDataType
    memory_level: DFXMemLevel = DFXMemLevel.EXTERNAL

@dataclass
class DFXOp:
    """Base class for DFX operations."""
    pass

@dataclass
class DFXMatmul(DFXOp):
    """Matrix multiplication: C[M,N] = A[M,K] @ B[K,N]"""
    A: DFXTensor
    B: DFXTensor
    C: DFXTensor
    M: int
    N: int
    K: int
    tile_M: int = 0  # 0 = auto
    tile_N: int = 0
    tile_K: int = 0

@dataclass
class DFXActivation(DFXOp):
    """Activation function: Y = act(X)"""
    input: DFXTensor
    output: DFXTensor
    activation: str  # "relu", "gelu", "sigmoid", etc.

@dataclass
class DFXDataMove(DFXOp):
    """Data movement between memory levels."""
    src: DFXTensor
    dst: DFXTensor
    src_level: DFXMemLevel
    dst_level: DFXMemLevel

@dataclass
class DFXProgram:
    """Complete DFX program."""
    name: str
    tensors: List[DFXTensor]
    ops: List[DFXOp]
    inputs: List[str]   # Tensor names
    outputs: List[str]  # Tensor names

    def to_json(self) -> dict:
        """Serialize to JSON for C++ consumption."""
        return {
            "name": self.name,
            "tensors": [self._tensor_to_json(t) for t in self.tensors],
            "ops": [self._op_to_json(op) for op in self.ops],
            "inputs": self.inputs,
            "outputs": self.outputs
        }

    def _tensor_to_json(self, t: DFXTensor) -> dict:
        return {
            "name": t.name,
            "shape": list(t.shape),
            "dtype": t.dtype.value,
            "memory_level": t.memory_level.value
        }

    def _op_to_json(self, op: DFXOp) -> dict:
        if isinstance(op, DFXMatmul):
            return {
                "type": "matmul",
                "A": op.A.name, "B": op.B.name, "C": op.C.name,
                "M": op.M, "N": op.N, "K": op.K,
                "tile_M": op.tile_M, "tile_N": op.tile_N, "tile_K": op.tile_K
            }
        elif isinstance(op, DFXActivation):
            return {
                "type": "activation",
                "input": op.input.name,
                "output": op.output.name,
                "activation": op.activation
            }
        elif isinstance(op, DFXDataMove):
            return {
                "type": "data_move",
                "src": op.src.name, "dst": op.dst.name,
                "src_level": op.src_level.value,
                "dst_level": op.dst_level.value
            }
        else:
            raise ValueError(f"Unknown op type: {type(op)}")


class DFXEmitter:
    """Emits DFX IR from an OpGraph."""

    def __init__(self):
        self._tensor_counter = 0
        self._tensors: Dict[int, DFXTensor] = {}  # Python id -> DFXTensor

    def emit(self, graph: OpGraph) -> DFXProgram:
        """Convert OpGraph to DFX program."""
        dfx_tensors = []
        dfx_ops = []

        # Process nodes in topological order
        for node in graph.topological_order():
            ops = self._emit_node(node)
            dfx_ops.extend(ops)

        # Collect all tensors
        dfx_tensors = list(self._tensors.values())

        # Get input/output names
        input_names = [self._get_dfx_tensor(t).name for t in graph.inputs]
        output_names = [self._get_dfx_tensor(t).name for t in graph.outputs]

        return DFXProgram(
            name="compiled_graph",
            tensors=dfx_tensors,
            ops=dfx_ops,
            inputs=input_names,
            outputs=output_names
        )

    def _emit_node(self, node: OpNode) -> List[DFXOp]:
        """Emit DFX ops for a single graph node."""
        if node.op_type == OpType.MATMUL:
            return self._emit_matmul(node)
        elif node.op_type == OpType.RELU:
            return self._emit_activation(node, "relu")
        elif node.op_type == OpType.GELU:
            return self._emit_activation(node, "gelu")
        elif node.op_type == OpType.SOFTMAX:
            return self._emit_activation(node, "softmax")
        else:
            raise NotImplementedError(f"Op type {node.op_type} not implemented")

    def _emit_matmul(self, node: OpNode) -> List[DFXOp]:
        """Emit DFX matmul operation."""
        A = self._get_dfx_tensor(node.inputs[0])
        B = self._get_dfx_tensor(node.inputs[1])
        C = self._get_dfx_tensor(node.outputs[0])

        M, K = A.shape[-2], A.shape[-1]
        K2, N = B.shape[-2], B.shape[-1]

        return [DFXMatmul(
            A=A, B=B, C=C,
            M=M, N=N, K=K
        )]

    def _emit_activation(self, node: OpNode, act_type: str) -> List[DFXOp]:
        """Emit DFX activation operation."""
        inp = self._get_dfx_tensor(node.inputs[0])
        out = self._get_dfx_tensor(node.outputs[0])

        return [DFXActivation(
            input=inp,
            output=out,
            activation=act_type
        )]

    def _get_dfx_tensor(self, tensor: 'Tensor') -> DFXTensor:
        """Get or create DFX tensor for a Python Tensor."""
        tid = id(tensor)
        if tid not in self._tensors:
            name = tensor._meta.name or f"t{self._tensor_counter}"
            self._tensor_counter += 1

            dtype = self._numpy_to_dfx_dtype(tensor.dtype)

            self._tensors[tid] = DFXTensor(
                name=name,
                shape=tensor.shape,
                dtype=dtype
            )
        return self._tensors[tid]

    def _numpy_to_dfx_dtype(self, dtype) -> DFXDataType:
        """Convert numpy dtype to DFX dtype."""
        import numpy as np
        if dtype == np.float32:
            return DFXDataType.FLOAT32
        elif dtype == np.float16:
            return DFXDataType.FLOAT16
        elif dtype == np.int8:
            return DFXDataType.INT8
        else:
            return DFXDataType.FLOAT32  # Default
```

### Runtime Integration

```python
# kpu/runtime.py
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .dfx_emitter import DFXProgram

# Fidelity levels
BEHAVIORAL = 0
TRANSACTIONAL = 1
CYCLE_ACCURATE = 2

@dataclass
class ExecutionStats:
    """Statistics from kernel execution."""
    cycles: int
    compute_cycles: int
    memory_cycles: int
    l3_accesses: int
    l2_accesses: int
    dram_bytes: int

@dataclass
class CompiledKernel:
    """A compiled KPU kernel ready for execution."""
    op_graph: 'OpGraph'
    dfx_program: DFXProgram
    kernels: List[Any]  # Native kernel handles
    fidelity: int

    def execute(self, *args, **kwargs) -> 'Tensor':
        """Execute the kernel on the simulator."""
        runtime = get_runtime()
        return runtime.execute(self, *args, **kwargs)

class KPURuntime:
    """
    Runtime for executing KPU kernels on the simulator.

    Wraps the C++ kpu-sim library via pybind11 bindings.
    """

    _instance: 'KPURuntime' = None

    def __init__(self, fidelity: int = BEHAVIORAL):
        self.fidelity = fidelity
        self._sim = None  # Will be initialized on first use
        self._stats: Optional[ExecutionStats] = None

    @classmethod
    def get(cls) -> 'KPURuntime':
        """Get singleton runtime instance."""
        if cls._instance is None:
            cls._instance = KPURuntime()
        return cls._instance

    def _init_simulator(self):
        """Initialize the C++ simulator."""
        if self._sim is not None:
            return

        try:
            # Try to load native bindings
            from . import _native
            self._sim = _native.create_simulator(self.fidelity)
        except ImportError:
            # Fall back to pure Python behavioral simulation
            self._sim = PythonBehavioralSim()

    def compile_dfx(self, program: DFXProgram, optimize: bool = True) -> List[Any]:
        """Compile DFX program to native kernels."""
        self._init_simulator()

        if hasattr(self._sim, 'compile_dfx'):
            return self._sim.compile_dfx(program.to_json(), optimize)
        else:
            # Pure Python path - return the program for interpretation
            return [program]

    def execute(self, kernel: CompiledKernel, *args, **kwargs) -> 'Tensor':
        """Execute a compiled kernel."""
        self._init_simulator()

        # Convert inputs to numpy arrays
        input_arrays = []
        for arg in args:
            if hasattr(arg, '_data'):
                input_arrays.append(arg._data)
            elif isinstance(arg, np.ndarray):
                input_arrays.append(arg)
            else:
                raise TypeError(f"Expected Tensor or ndarray, got {type(arg)}")

        # Execute on simulator
        if hasattr(self._sim, 'execute'):
            result, stats = self._sim.execute(kernel.kernels, input_arrays)
            self._stats = stats
        else:
            # Pure Python behavioral execution
            result = self._execute_behavioral(kernel.dfx_program, input_arrays)

        from .tensor import Tensor
        return Tensor(result)

    def _execute_behavioral(self, program: DFXProgram, inputs: List[np.ndarray]) -> np.ndarray:
        """Pure Python behavioral execution (computes actual values)."""
        # Map input names to arrays
        tensors: Dict[str, np.ndarray] = {}
        for name, arr in zip(program.inputs, inputs):
            tensors[name] = arr

        # Execute operations in order
        for op_json in [program._op_to_json(op) for op in program.ops]:
            if op_json['type'] == 'matmul':
                A = tensors[op_json['A']]
                B = tensors[op_json['B']]
                C = np.matmul(A, B)
                tensors[op_json['C']] = C

            elif op_json['type'] == 'activation':
                inp = tensors[op_json['input']]
                act = op_json['activation']

                if act == 'relu':
                    out = np.maximum(inp, 0)
                elif act == 'gelu':
                    out = inp * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (inp + 0.044715 * inp**3)))
                elif act == 'sigmoid':
                    out = 1 / (1 + np.exp(-inp))
                elif act == 'softmax':
                    exp_x = np.exp(inp - np.max(inp, axis=-1, keepdims=True))
                    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                else:
                    out = inp  # Identity

                tensors[op_json['output']] = out

        # Return output tensor
        return tensors[program.outputs[0]]

    def get_stats(self) -> Optional[ExecutionStats]:
        """Get statistics from last execution."""
        return self._stats


class PythonBehavioralSim:
    """Pure Python behavioral simulator fallback."""
    pass


def get_runtime() -> KPURuntime:
    """Get the global runtime instance."""
    return KPURuntime.get()

def set_fidelity(fidelity: int):
    """Set the simulation fidelity level."""
    runtime = get_runtime()
    runtime.fidelity = fidelity
    runtime._sim = None  # Force re-initialization
```

---

## MNIST MLP Test Case

### Complete Example

```python
# examples/mnist_mlp.py
"""
MNIST MLP Example - First KPU functional test.

Network: 784 → 128 → 64 → 10
"""

import numpy as np
import kpu

# Set fidelity (BEHAVIORAL computes actual values)
kpu.set_fidelity(kpu.BEHAVIORAL)

@kpu.compile
def mnist_mlp(x: kpu.Tensor,
              w1: kpu.Tensor, b1: kpu.Tensor,
              w2: kpu.Tensor, b2: kpu.Tensor,
              w3: kpu.Tensor, b3: kpu.Tensor) -> kpu.Tensor:
    """
    3-layer MLP for MNIST classification.

    Args:
        x: Input images [batch, 784]
        w1, b1: Layer 1 weights [784, 128] and bias [128]
        w2, b2: Layer 2 weights [128, 64] and bias [64]
        w3, b3: Layer 3 weights [64, 10] and bias [10]

    Returns:
        Logits [batch, 10]
    """
    # Layer 1: 784 → 128
    h1 = kpu.relu(x @ w1 + b1)

    # Layer 2: 128 → 64
    h2 = kpu.relu(h1 @ w2 + b2)

    # Layer 3: 64 → 10 (no activation - raw logits)
    logits = h2 @ w3 + b3

    return logits


def create_test_data(batch_size: int = 32):
    """Create synthetic test data."""
    np.random.seed(42)

    # Xavier initialization for weights
    def xavier(shape):
        fan_in, fan_out = shape[0], shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape).astype(np.float32) * std

    # Weights and biases
    w1 = kpu.Tensor(xavier((784, 128)), name="w1")
    b1 = kpu.Tensor(np.zeros(128, dtype=np.float32), name="b1")

    w2 = kpu.Tensor(xavier((128, 64)), name="w2")
    b2 = kpu.Tensor(np.zeros(64, dtype=np.float32), name="b2")

    w3 = kpu.Tensor(xavier((64, 10)), name="w3")
    b3 = kpu.Tensor(np.zeros(10, dtype=np.float32), name="b3")

    # Random input (simulating flattened MNIST images)
    x = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32), name="input")

    return x, (w1, b1, w2, b2, w3, b3)


def reference_mlp(x, w1, b1, w2, b2, w3, b3):
    """NumPy reference implementation for verification."""
    h1 = np.maximum(x @ w1 + b1, 0)  # ReLU
    h2 = np.maximum(h1 @ w2 + b2, 0)  # ReLU
    logits = h2 @ w3 + b3
    return logits


def main():
    print("=== MNIST MLP on KPU Simulator ===\n")

    # Create test data
    x, weights = create_test_data(batch_size=32)
    w1, b1, w2, b2, w3, b3 = weights

    print(f"Input shape: {x.shape}")
    print(f"Layer 1: {w1.shape[0]} → {w1.shape[1]}")
    print(f"Layer 2: {w2.shape[0]} → {w2.shape[1]}")
    print(f"Layer 3: {w3.shape[0]} → {w3.shape[1]}")
    print()

    # Execute on KPU simulator
    print("Compiling and executing on KPU (BEHAVIORAL)...")
    logits = mnist_mlp(x, w1, b1, w2, b2, w3, b3)

    print(f"Output shape: {logits.shape}")
    print()

    # Verify against NumPy reference
    print("Verifying against NumPy reference...")
    ref_logits = reference_mlp(
        x._data, w1._data, b1._data,
        w2._data, b2._data, w3._data, b3._data
    )

    max_diff = np.max(np.abs(logits._data - ref_logits))
    print(f"Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ Results match reference implementation!")
    else:
        print("✗ Results do not match - check implementation")

    # Print execution statistics
    stats = kpu.get_runtime().get_stats()
    if stats:
        print(f"\nExecution Statistics:")
        print(f"  Total cycles: {stats.cycles}")
        print(f"  Compute cycles: {stats.compute_cycles}")
        print(f"  Memory cycles: {stats.memory_cycles}")

    # Show generated DFX IR
    print("\n=== Generated DFX IR ===")
    dfx = mnist_mlp.get_dfx()
    if dfx:
        import json
        print(json.dumps(dfx.to_json(), indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
```

### Test Suite

```python
# tests/test_mnist_mlp.py
import pytest
import numpy as np
import kpu

class TestMNISTMLP:
    """Test suite for MNIST MLP on KPU simulator."""

    def test_matmul_basic(self):
        """Test basic matrix multiplication."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

        @kpu.compile
        def matmul(a, b):
            return a @ b

        A = kpu.Tensor(np.random.randn(32, 64).astype(np.float32))
        B = kpu.Tensor(np.random.randn(64, 128).astype(np.float32))

        result = matmul(A, B)
        expected = A._data @ B._data

        np.testing.assert_allclose(result._data, expected, rtol=1e-5)

    def test_relu_activation(self):
        """Test ReLU activation function."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

        @kpu.compile
        def relu_test(x):
            return kpu.relu(x)

        X = kpu.Tensor(np.array([[-1, 2], [3, -4]], dtype=np.float32))
        result = relu_test(X)
        expected = np.array([[0, 2], [3, 0]], dtype=np.float32)

        np.testing.assert_array_equal(result._data, expected)

    def test_single_layer(self):
        """Test single layer: y = relu(x @ w + b)"""
        kpu.set_fidelity(kpu.BEHAVIORAL)

        @kpu.compile
        def single_layer(x, w, b):
            return kpu.relu(x @ w + b)

        X = kpu.Tensor(np.random.randn(16, 784).astype(np.float32))
        W = kpu.Tensor(np.random.randn(784, 128).astype(np.float32))
        B = kpu.Tensor(np.zeros(128, dtype=np.float32))

        result = single_layer(X, W, B)
        expected = np.maximum(X._data @ W._data + B._data, 0)

        np.testing.assert_allclose(result._data, expected, rtol=1e-5)

    def test_full_mnist_mlp(self):
        """Test complete MNIST MLP network."""
        kpu.set_fidelity(kpu.BEHAVIORAL)

        @kpu.compile
        def mnist_mlp(x, w1, b1, w2, b2, w3, b3):
            h1 = kpu.relu(x @ w1 + b1)
            h2 = kpu.relu(h1 @ w2 + b2)
            return h2 @ w3 + b3

        np.random.seed(42)
        batch_size = 8

        X = kpu.Tensor(np.random.randn(batch_size, 784).astype(np.float32))
        W1 = kpu.Tensor(np.random.randn(784, 128).astype(np.float32) * 0.01)
        B1 = kpu.Tensor(np.zeros(128, dtype=np.float32))
        W2 = kpu.Tensor(np.random.randn(128, 64).astype(np.float32) * 0.01)
        B2 = kpu.Tensor(np.zeros(64, dtype=np.float32))
        W3 = kpu.Tensor(np.random.randn(64, 10).astype(np.float32) * 0.01)
        B3 = kpu.Tensor(np.zeros(10, dtype=np.float32))

        result = mnist_mlp(X, W1, B1, W2, B2, W3, B3)

        # NumPy reference
        h1 = np.maximum(X._data @ W1._data + B1._data, 0)
        h2 = np.maximum(h1 @ W2._data + B2._data, 0)
        expected = h2 @ W3._data + B3._data

        np.testing.assert_allclose(result._data, expected, rtol=1e-5)
        assert result.shape == (batch_size, 10)

    def test_dfx_generation(self):
        """Test that DFX IR is correctly generated."""
        @kpu.compile
        def simple_net(x, w):
            return kpu.relu(x @ w)

        X = kpu.Tensor(np.zeros((4, 8), dtype=np.float32))
        W = kpu.Tensor(np.zeros((8, 16), dtype=np.float32))

        _ = simple_net(X, W)

        dfx = simple_net.get_dfx()
        assert dfx is not None

        dfx_json = dfx.to_json()
        assert 'ops' in dfx_json

        # Should have matmul and relu ops
        op_types = [op['type'] for op in dfx_json['ops']]
        assert 'matmul' in op_types
        assert 'activation' in op_types
```

---

## C++ Bindings (pybind11)

```cpp
// python/kpu/_native/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/compiler/kernel_compiler.hpp>
#include <sw/runtime/runtime.hpp>
#include <nlohmann/json.hpp>

namespace py = pybind11;
using json = nlohmann::json;

namespace {

using namespace sw::kpu;

class PyKPUSimulator {
public:
    PyKPUSimulator(int fidelity) {
        SimulatorConfig config;
        config.default_fidelity = static_cast<SimulationFidelity>(fidelity);
        sim_ = std::make_unique<KPUSimulator>(config);
        runtime_ = std::make_unique<runtime::KPURuntime>(sim_.get());
    }

    py::list compile_dfx(const py::dict& dfx_json, bool optimize) {
        // Parse DFX JSON and compile to kernels
        json j = json::parse(py::str(py::module::import("json").attr("dumps")(dfx_json)));

        py::list kernels;
        for (const auto& op : j["ops"]) {
            if (op["type"] == "matmul") {
                int M = op["M"], N = op["N"], K = op["K"];

                compiler::CompileOptions opts;
                opts.optimize_tiles = optimize;

                auto kernel = compiler_.compile_matmul(M, N, K, opts);
                kernels.append(py::cast(kernel));
            }
            // Add other op types as needed
        }
        return kernels;
    }

    std::pair<py::array_t<float>, py::dict> execute(
        const py::list& kernels,
        const std::vector<py::array_t<float>>& inputs) {

        // Allocate device memory and copy inputs
        std::vector<runtime::Address> addrs;
        for (const auto& arr : inputs) {
            size_t size = arr.size() * sizeof(float);
            auto addr = runtime_->malloc(size);
            runtime_->memcpy_h2d(addr, arr.data(), size);
            addrs.push_back(addr);
        }

        // Execute kernels
        for (py::handle kernel_obj : kernels) {
            auto kernel = kernel_obj.cast<Kernel>();
            runtime_->launch(kernel, addrs);
        }

        runtime_->synchronize();

        // Copy result back
        // (Simplified - actual implementation needs output tensor tracking)
        auto& last_output = addrs.back();
        // ...

        // Get stats
        py::dict stats;
        stats["cycles"] = sim_->get_stats().total_cycles;
        stats["compute_cycles"] = sim_->get_stats().compute_cycles;

        return {py::array_t<float>(), stats};  // Placeholder
    }

private:
    std::unique_ptr<KPUSimulator> sim_;
    std::unique_ptr<runtime::KPURuntime> runtime_;
    compiler::KernelCompiler compiler_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "KPU Simulator Python Bindings";

    py::class_<PyKPUSimulator>(m, "Simulator")
        .def(py::init<int>(), py::arg("fidelity") = 0)
        .def("compile_dfx", &PyKPUSimulator::compile_dfx)
        .def("execute", &PyKPUSimulator::execute);

    m.def("create_simulator", [](int fidelity) {
        return std::make_unique<PyKPUSimulator>(fidelity);
    });

    // Fidelity constants
    m.attr("BEHAVIORAL") = 0;
    m.attr("TRANSACTIONAL") = 1;
    m.attr("CYCLE_ACCURATE") = 2;
}
```

---

## Roadmap

### Phase 1: Minimal Viable Product (2-3 weeks)

| Task | Description | Owner |
|------|-------------|-------|
| Python package skeleton | `kpu/` package with Tensor, ops | Week 1 |
| Pure Python behavioral | Execute without C++ bindings | Week 1 |
| AST-based tracing | Build OpGraph from decorated functions | Week 1-2 |
| DFX emission | Convert OpGraph to DFX IR | Week 2 |
| MNIST MLP test | End-to-end test case | Week 2-3 |
| pybind11 bindings | Connect to existing kpu-sim | Week 2-3 |

### Phase 2: Codon Integration (4-6 weeks)

| Task | Description |
|------|-------------|
| Codon DSL plugin | Create `codon-kpu` plugin package |
| CIR pass | Intercept NumPy ops, emit KPU IR |
| Type propagation | Shape/dtype inference in Codon |
| Code generation | DFX emission from Codon |

### Phase 3: Production Features (8-12 weeks)

| Task | Description |
|------|-------------|
| Ahead-of-time compilation | Compile to loadable `.kpu` objects |
| Graph optimization | Operator fusion, memory planning |
| ONNX/PyTorch loading | Load pretrained models |
| Transactional simulation | Performance estimation |

---

## References

- [Exaloop Codon GitHub](https://github.com/exaloop/codon)
- [Codon Documentation](https://docs.exaloop.io/)
- [Codon IR Developer Guide](https://docs.exaloop.io/developers/ir/)
- [Extend Codon (Plugins)](https://docs.exaloop.io/developers/extend/)
- [Codon NumPy Implementation](https://docs.exaloop.io/libraries/numpy/)
- [Codon JIT Decorator](https://docs.exaloop.io/integrations/python/codon-from-python/)
- KPU DFX IR: `include/sw/compiler/dfx/dfx.hpp`
- KPU Runtime API: `include/sw/runtime/runtime.hpp`
- KPU Kernel Compiler: `include/sw/compiler/kernel_compiler.hpp`

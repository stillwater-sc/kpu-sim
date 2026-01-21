# python/kpu/compiler.py
"""
KPU compiler with @kpu.compile decorator.

The decorator traces Python functions to build operation graphs,
then compiles them to DFX IR for execution on the KPU simulator.
"""

from __future__ import annotations
import functools
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .tensor import Tensor
    from .graph import OpGraph
    from .dfx_emitter import DFXProgram


# Fidelity levels
BEHAVIORAL = 0      # Functional correctness, computes actual values
TRANSACTIONAL = 1   # Performance estimation, statistical timing
CYCLE_ACCURATE = 2  # Full timing simulation


@dataclass
class CompilationResult:
    """Result of compiling a function."""
    op_graph: 'OpGraph'
    dfx_program: 'DFXProgram'
    input_names: List[str]
    output_names: List[str]


@dataclass
class ExecutionStats:
    """Statistics from kernel execution."""
    cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    matmul_flops: int = 0
    memory_bytes: int = 0


class CompiledFunction:
    """
    A compiled KPU function ready for execution.

    Wraps the original Python function with tracing and execution on the simulator.
    """

    def __init__(self,
                 func: Callable,
                 fidelity: int = BEHAVIORAL,
                 optimize: bool = True):
        self._func = func
        self._fidelity = fidelity
        self._optimize = optimize
        self._compiled: Optional[CompilationResult] = None
        self._stats: Optional[ExecutionStats] = None

        # Copy function metadata
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> 'Tensor':
        """Execute the compiled function."""
        from .tensor import Tensor

        # Convert inputs to Tensors if needed
        tensor_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_args.append(arg)
            elif isinstance(arg, np.ndarray):
                tensor_args.append(Tensor(arg))
            else:
                tensor_args.append(arg)

        # Compile on first call (lazy compilation)
        if self._compiled is None:
            self._compiled = self._trace_and_compile(tensor_args, kwargs)

        # Execute
        return self._execute(tensor_args, kwargs)

    def _trace_and_compile(self,
                           args: List['Tensor'],
                           kwargs: Dict[str, Any]) -> CompilationResult:
        """Trace function execution and compile to DFX."""
        from .tensor import Tensor, TensorMeta
        from .graph import OpGraph
        from .dfx_emitter import DFXEmitter

        # Create symbolic tensors for tracing
        symbolic_args = []
        input_names = []

        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                # Create symbolic version with same shape/dtype
                name = arg.name or f"arg{i}"
                sym_meta = TensorMeta(
                    shape=arg.shape,
                    dtype=arg.dtype,
                    name=name,
                    is_weight=arg._meta.is_weight if hasattr(arg._meta, 'is_weight') else False,
                )
                sym = Tensor(sym_meta)
                symbolic_args.append(sym)
                input_names.append(name)
            else:
                symbolic_args.append(arg)

        # Enable tracing
        graph = OpGraph(name=self._func.__name__)
        Tensor._tracing = True
        Tensor._trace_graph = graph

        try:
            # Mark inputs
            for sym in symbolic_args:
                if isinstance(sym, Tensor):
                    graph.mark_input(sym)

            # Execute function symbolically
            result = self._func(*symbolic_args, **kwargs)

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

        # Validate graph
        errors = graph.validate()
        if errors:
            raise ValueError(f"Graph validation failed: {errors}")

        # Apply fusion optimization if enabled
        if self._optimize:
            from .fusion import FusionCompiler
            fusion_compiler = FusionCompiler()
            graph = fusion_compiler.optimize(graph)

        # Emit DFX IR
        emitter = DFXEmitter()
        dfx_program = emitter.emit(graph)

        return CompilationResult(
            op_graph=graph,
            dfx_program=dfx_program,
            input_names=input_names,
            output_names=dfx_program.outputs,
        )

    def _execute(self,
                 args: List['Tensor'],
                 kwargs: Dict[str, Any]) -> 'Tensor':
        """Execute the compiled function on inputs."""
        from .runtime import get_runtime

        runtime = get_runtime()
        result, stats = runtime.execute(self._compiled.dfx_program, args)
        self._stats = stats
        return result

    @property
    def graph(self) -> Optional['OpGraph']:
        """Get the operation graph (after compilation)."""
        return self._compiled.op_graph if self._compiled else None

    @property
    def dfx(self) -> Optional['DFXProgram']:
        """Get the DFX program (after compilation)."""
        return self._compiled.dfx_program if self._compiled else None

    @property
    def stats(self) -> Optional[ExecutionStats]:
        """Get execution statistics from last run."""
        return self._stats

    def get_graph(self) -> Optional['OpGraph']:
        """Get the operation graph."""
        return self.graph

    def get_dfx(self) -> Optional['DFXProgram']:
        """Get the DFX program."""
        return self.dfx

    def get_stats(self) -> Optional[ExecutionStats]:
        """Get execution statistics."""
        return self._stats

    def execute_with_stats(self, *args, **kwargs) -> Tuple['Tensor', Optional[ExecutionStats]]:
        """
        Execute the compiled function and return both result and stats.

        Convenience method that returns a tuple of (result, stats) for
        TRANSACTIONAL and CYCLE_ACCURATE modes.

        Returns:
            Tuple of (result Tensor, ExecutionStats or None)

        Example:
            result, stats = compiled_fn.execute_with_stats(x, w1, w2)
            if stats:
                print(f"Cycles: {stats.cycles}")
        """
        result = self.__call__(*args, **kwargs)
        return result, self._stats

    def summary(self) -> str:
        """Get a summary of the compiled function."""
        if self._compiled is None:
            return f"CompiledFunction '{self._func.__name__}' (not yet compiled)"

        return self._compiled.op_graph.summary()


def compile(func: Callable = None, *,
            fidelity: int = BEHAVIORAL,
            optimize: bool = True) -> Union[CompiledFunction, Callable]:
    """
    Decorator to compile a Python function to KPU kernels.

    The decorated function is traced on first call to build an operation graph,
    which is then compiled to DFX IR. Subsequent calls execute the compiled
    version on the KPU simulator.

    Args:
        func: The function to compile (when used without parentheses)
        fidelity: Simulation fidelity level (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE)
        optimize: Whether to apply optimizations

    Returns:
        CompiledFunction wrapper

    Example:
        @kpu.compile
        def mlp(x, w1, w2):
            h = kpu.relu(x @ w1)
            return h @ w2

        # With options:
        @kpu.compile(fidelity=kpu.BEHAVIORAL)
        def mlp(x, w1, w2):
            ...

    The decorated function computes actual values when fidelity=BEHAVIORAL,
    enabling functional verification of neural networks.
    """
    def decorator(fn: Callable) -> CompiledFunction:
        return CompiledFunction(fn, fidelity=fidelity, optimize=optimize)

    if func is not None:
        # Called without parentheses: @kpu.compile
        return decorator(func)
    else:
        # Called with parentheses: @kpu.compile(...)
        return decorator


# Alias for consistency with other frameworks
jit = compile

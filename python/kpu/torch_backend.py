# python/kpu/torch_backend.py
"""
torch.compile backend for KPU simulator.

This module provides a custom backend for torch.compile that executes
PyTorch models on the KPU functional/transactional simulator.

Usage:
    import torch
    import kpu

    model = torchvision.models.squeezenet1_0(pretrained=True)
    compiled = torch.compile(model, backend="kpu")
    output = compiled(input_tensor)

The backend:
1. Receives FX GraphModule from Dynamo
2. Walks FX nodes and maps to kpu operations
3. Executes on kpu simulator (BEHAVIORAL/TRANSACTIONAL/CYCLE_ACCURATE)
4. Returns results compatible with PyTorch

Note: PyTorch C++ warnings (like NNPACK) are filtered in kpu/__init__.py
"""

from __future__ import annotations
import functools
from typing import List, Callable, Any, Dict, Optional, Tuple
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.fx as fx
    # PyTorch 2.9+ uses torch._dynamo, older versions use torch.dynamo
    try:
        from torch._dynamo import register_backend
    except ImportError:
        from torch.dynamo import register_backend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    """Raise error if PyTorch not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for torch.compile backend. "
            "Install with: pip install torch"
        )


class KPUBackend:
    """
    KPU backend for torch.compile.

    Converts FX GraphModule to kpu operations and executes on simulator.
    """

    def __init__(self, fidelity: Optional[int] = None):
        """
        Initialize KPU backend.

        Args:
            fidelity: Simulation fidelity (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE)
                     If None, uses current global fidelity.
        """
        from .runtime import get_runtime, BEHAVIORAL
        self.fidelity = fidelity
        self.runtime = get_runtime()
        self._compiled_cache: Dict[int, Callable] = {}

    def __call__(self, gm: 'fx.GraphModule', example_inputs: List[torch.Tensor]) -> Callable:
        """
        Compile FX GraphModule for KPU execution.

        This is called by torch.compile when the backend is invoked.

        Args:
            gm: FX GraphModule captured by Dynamo
            example_inputs: Example input tensors for shape inference

        Returns:
            Callable that executes the graph on KPU
        """
        from .fx_converter import FXToKPUConverter

        # Convert FX graph to KPU executable
        converter = FXToKPUConverter(gm, example_inputs, fidelity=self.fidelity)
        kpu_executable = converter.convert()

        # Return wrapper that handles torch<->numpy conversion
        @functools.wraps(gm.forward)
        def kpu_forward(*args):
            return kpu_executable(*args)

        return kpu_forward


def kpu_backend(gm: 'fx.GraphModule', example_inputs: List['torch.Tensor']) -> Callable:
    """
    Default KPU backend function for torch.compile.

    Args:
        gm: FX GraphModule from Dynamo
        example_inputs: Example inputs for tracing

    Returns:
        Compiled function that runs on KPU
    """
    _check_torch()
    backend = KPUBackend()
    return backend(gm, example_inputs)


def register_kpu_backend():
    """
    Register 'kpu' as a torch.compile backend.

    After calling this, you can use:
        torch.compile(model, backend="kpu")
    """
    _check_torch()

    # Register with torch._dynamo
    register_backend(name="kpu", compiler_fn=kpu_backend)

    # Also register variants for different fidelities
    def make_fidelity_backend(fidelity: int) -> Callable:
        def backend_fn(gm, example_inputs):
            backend = KPUBackend(fidelity=fidelity)
            return backend(gm, example_inputs)
        return backend_fn

    from .runtime import BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE

    register_backend(name="kpu_behavioral", compiler_fn=make_fidelity_backend(BEHAVIORAL))
    register_backend(name="kpu_transactional", compiler_fn=make_fidelity_backend(TRANSACTIONAL))
    register_backend(name="kpu_cycle_accurate", compiler_fn=make_fidelity_backend(CYCLE_ACCURATE))


# Auto-register on import if torch is available
if TORCH_AVAILABLE:
    try:
        register_kpu_backend()
    except Exception:
        # Registration may fail in some environments, that's OK
        pass


def compile(model: Any, *,
            fidelity: Optional[int] = None,
            fullgraph: bool = False,
            **kwargs) -> Any:
    """
    Convenience function to compile a PyTorch model for KPU.

    This is equivalent to torch.compile(model, backend="kpu") but with
    additional KPU-specific options.

    Args:
        model: PyTorch model or function to compile
        fidelity: Simulation fidelity level
        fullgraph: If True, require entire graph to be capturable
        **kwargs: Additional arguments passed to torch.compile

    Returns:
        Compiled model that executes on KPU simulator

    Example:
        >>> model = nn.Linear(784, 10)
        >>> compiled = kpu.torch_backend.compile(model)
        >>> output = compiled(input_tensor)
    """
    _check_torch()

    if fidelity is not None:
        # Use fidelity-specific backend
        from .runtime import BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE
        fidelity_names = {
            BEHAVIORAL: "kpu_behavioral",
            TRANSACTIONAL: "kpu_transactional",
            CYCLE_ACCURATE: "kpu_cycle_accurate",
        }
        backend = fidelity_names.get(fidelity, "kpu")
    else:
        backend = "kpu"

    return torch.compile(model, backend=backend, fullgraph=fullgraph, **kwargs)


def get_last_stats():
    """Get execution stats from the last torch.compile execution.

    After executing a model compiled with backend="kpu_transactional" or
    backend="kpu_cycle_accurate", this function returns the timing statistics
    collected during execution.

    Returns:
        ExecutionStats from the last TRANSACTIONAL/CYCLE_ACCURATE execution,
        or None if the last execution was BEHAVIORAL or no execution occurred.

    Example:
        >>> compiled = torch.compile(model, backend="kpu_transactional")
        >>> output = compiled(x)
        >>> stats = kpu.torch_backend.get_last_stats()
        >>> print(f"Cycles: {stats.cycles}")
        >>> print(f"GFLOPS: {stats.gflops:.1f}")
    """
    from .fx_converter import get_last_stats as _get_stats
    return _get_stats()

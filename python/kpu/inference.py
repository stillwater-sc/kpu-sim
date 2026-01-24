# python/kpu/inference.py
"""
KPU Inference Pipeline - End-to-end model execution with profiling.

Provides:
- InferencePipeline: High-level inference API with batching
- Layer-by-layer profiling and statistics
- Memory traffic analysis
- Batch processing with automatic chunking

Example:
    >>> from kpu.inference import InferencePipeline
    >>> from kpu.models import squeezenet1_0
    >>>
    >>> # Create pipeline
    >>> model = squeezenet1_0(pretrained=True)
    >>> pipeline = InferencePipeline(model)
    >>>
    >>> # Run inference
    >>> output = pipeline(input_tensor)
    >>>
    >>> # Get layer-by-layer stats
    >>> stats = pipeline.get_layer_stats()
    >>> for layer_name, layer_stats in stats.items():
    ...     print(f"{layer_name}: {layer_stats.cycles} cycles")
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .tensor import Tensor
from .model import Layer, Model
from .runtime import KPURuntime, ExecutionStats, get_runtime, set_fidelity, get_fidelity
from .compiler import compile as kpu_compile, BEHAVIORAL, TRANSACTIONAL


# =============================================================================
# Statistics Data Classes
# =============================================================================

@dataclass
class LayerStats:
    """Statistics for a single layer execution."""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    flops: int = 0
    memory_bytes: int = 0
    execution_time_ms: float = 0.0

    @property
    def compute_intensity(self) -> float:
        """Compute intensity (FLOPS / byte)."""
        if self.memory_bytes == 0:
            return float('inf')
        return self.flops / self.memory_bytes

    @property
    def utilization(self) -> float:
        """Compute utilization (0-1)."""
        if self.cycles == 0:
            return 0.0
        return self.compute_cycles / self.cycles

    def __repr__(self) -> str:
        return (f"LayerStats({self.name}, cycles={self.cycles}, "
                f"flops={self.flops:,}, memory={self.memory_bytes:,}B)")


@dataclass
class InferenceStats:
    """Statistics for a complete inference run."""
    total_cycles: int = 0
    total_compute_cycles: int = 0
    total_memory_cycles: int = 0
    total_flops: int = 0
    total_memory_bytes: int = 0
    total_execution_time_ms: float = 0.0
    batch_size: int = 1
    layer_stats: Dict[str, LayerStats] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Throughput in samples/second."""
        if self.total_execution_time_ms == 0:
            return float('inf')
        return self.batch_size / (self.total_execution_time_ms / 1000)

    @property
    def latency_ms(self) -> float:
        """Per-sample latency in milliseconds."""
        return self.total_execution_time_ms / self.batch_size

    @property
    def gflops(self) -> float:
        """GFLOPS achieved."""
        if self.total_execution_time_ms == 0:
            return float('inf')
        return self.total_flops / (self.total_execution_time_ms / 1000) / 1e9

    @property
    def memory_bandwidth_gbps(self) -> float:
        """Memory bandwidth in GB/s."""
        if self.total_execution_time_ms == 0:
            return float('inf')
        return self.total_memory_bytes / (self.total_execution_time_ms / 1000) / 1e9

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Inference Statistics",
            "=" * 50,
            f"Batch size: {self.batch_size}",
            f"Total cycles: {self.total_cycles:,}",
            f"  Compute: {self.total_compute_cycles:,} ({100*self.total_compute_cycles/max(self.total_cycles,1):.1f}%)",
            f"  Memory:  {self.total_memory_cycles:,} ({100*self.total_memory_cycles/max(self.total_cycles,1):.1f}%)",
            f"Total FLOPS: {self.total_flops:,}",
            f"Total memory: {self.total_memory_bytes:,} bytes",
            f"Execution time: {self.total_execution_time_ms:.2f} ms",
            f"Throughput: {self.throughput:.1f} samples/sec",
            f"Latency: {self.latency_ms:.2f} ms/sample",
            f"GFLOPS: {self.gflops:.1f}",
            f"Memory BW: {self.memory_bandwidth_gbps:.1f} GB/s",
            "=" * 50,
        ]

        if self.layer_stats:
            lines.append("\nPer-Layer Breakdown:")
            lines.append("-" * 80)
            lines.append(f"{'Layer':<30} {'Type':<15} {'Cycles':>10} {'FLOPS':>12} {'Memory':>12}")
            lines.append("-" * 80)

            for name, stats in self.layer_stats.items():
                lines.append(f"{name:<30} {stats.layer_type:<15} {stats.cycles:>10,} "
                           f"{stats.flops:>12,} {stats.memory_bytes:>12,}")

            lines.append("-" * 80)

        return "\n".join(lines)


# =============================================================================
# Inference Pipeline
# =============================================================================

class InferencePipeline:
    """High-level inference pipeline for KPU models.

    Provides:
    - Automatic compilation
    - Batch processing
    - Layer-by-layer profiling
    - Memory traffic analysis

    Args:
        model: KPU model to run
        fidelity: Simulation fidelity (BEHAVIORAL or TRANSACTIONAL)
        profile_layers: Whether to collect per-layer statistics
        max_batch_size: Maximum batch size for chunking (None = no limit)

    Example:
        >>> model = create_model()
        >>> pipeline = InferencePipeline(model)
        >>> output = pipeline(input_tensor)
        >>> print(pipeline.stats.summary())
    """

    def __init__(
        self,
        model: Model,
        fidelity: int = BEHAVIORAL,
        profile_layers: bool = True,
        max_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.fidelity = fidelity
        self.profile_layers = profile_layers
        self.max_batch_size = max_batch_size

        self._compiled = False
        self._compiled_fn = None
        self._layer_hooks: Dict[str, Callable] = {}
        self._last_stats: Optional[InferenceStats] = None

    def compile(self, sample_input: Optional[Tensor] = None):
        """Compile the model for execution.

        Args:
            sample_input: Optional sample input for tracing

        Note:
            For models using custom Layer classes, compilation may not be
            possible and direct execution will be used instead.
        """
        if self._compiled:
            return

        # Set fidelity
        set_fidelity(self.fidelity)

        # Check if model already has a compiled function
        if hasattr(self.model, '_compiled_fn') and self.model._compiled_fn is not None:
            self._compiled_fn = self.model._compiled_fn
            self._compiled = True
            return

        # For models using custom Layer classes, skip compilation
        # and use direct execution (more compatible)
        # Compilation with @kpu.compile works best for simple function-based models
        self._compiled_fn = None
        self._compiled = True

    def __call__(
        self,
        x: Union[Tensor, np.ndarray],
        return_stats: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, InferenceStats]]:
        """Run inference.

        Args:
            x: Input tensor or numpy array
            return_stats: If True, return (output, stats) tuple

        Returns:
            Output tensor, or (output, stats) if return_stats=True
        """
        # Convert numpy to Tensor
        if isinstance(x, np.ndarray):
            x = Tensor(x)

        # Compile if needed
        if not self._compiled:
            self.compile(x)

        # Get batch size
        batch_size = x.shape[0] if len(x.shape) > 1 else 1

        # Run inference (with chunking if needed)
        start_time = time.perf_counter()

        if self.max_batch_size and batch_size > self.max_batch_size:
            output = self._run_chunked(x)
        else:
            output = self._run_single(x)

        end_time = time.perf_counter()

        # Collect statistics
        self._last_stats = self._collect_stats(
            batch_size=batch_size,
            execution_time_ms=(end_time - start_time) * 1000,
        )

        if return_stats:
            return output, self._last_stats
        return output

    def _run_single(self, x: Tensor) -> Tensor:
        """Run inference on single batch."""
        if self._compiled_fn is not None:
            return self._compiled_fn(x)
        return self.model(x)

    def _run_chunked(self, x: Tensor) -> Tensor:
        """Run inference with batch chunking."""
        batch_size = x.shape[0]
        outputs = []

        for i in range(0, batch_size, self.max_batch_size):
            chunk = Tensor(x.numpy()[i:i + self.max_batch_size])
            output = self._run_single(chunk)
            outputs.append(output.numpy())

        return Tensor(np.concatenate(outputs, axis=0))

    def _collect_stats(
        self,
        batch_size: int,
        execution_time_ms: float,
    ) -> InferenceStats:
        """Collect inference statistics."""
        stats = InferenceStats(
            batch_size=batch_size,
            total_execution_time_ms=execution_time_ms,
        )

        # Get stats from compiled function if available
        if self._compiled_fn is not None and hasattr(self._compiled_fn, 'stats'):
            fn_stats = self._compiled_fn.stats
            if fn_stats is not None:
                stats.total_cycles = fn_stats.cycles
                stats.total_compute_cycles = fn_stats.compute_cycles
                stats.total_memory_cycles = fn_stats.memory_cycles

                # Get DRAM bytes from XUE summary
                if fn_stats.xue_summary is not None:
                    mem_hierarchy = fn_stats.xue_summary.get('memory_hierarchy', {})
                    dram_info = mem_hierarchy.get('dram', {})
                    stats.total_memory_bytes = dram_info.get('bytes', 0)

        # Collect layer stats if profiling enabled
        if self.profile_layers:
            stats.layer_stats = self._collect_layer_stats()

            # Sum up FLOPS and memory from layers
            for layer_stats in stats.layer_stats.values():
                stats.total_flops += layer_stats.flops
                if stats.total_memory_bytes == 0:
                    stats.total_memory_bytes += layer_stats.memory_bytes

        return stats

    def _collect_layer_stats(self) -> Dict[str, LayerStats]:
        """Collect per-layer statistics."""
        layer_stats = {}

        # Estimate stats for each layer
        for name, layer in self.model.modules():
            if name == "":
                continue

            # Get layer type
            layer_type = layer.__class__.__name__

            # Estimate FLOPS and memory for this layer
            flops, memory = self._estimate_layer_ops(layer)

            layer_stats[name] = LayerStats(
                name=name,
                layer_type=layer_type,
                input_shape=(),  # Would need tracing to get actual shapes
                output_shape=(),
                flops=flops,
                memory_bytes=memory,
            )

        return layer_stats

    def _estimate_layer_ops(self, layer: Layer) -> Tuple[int, int]:
        """Estimate FLOPS and memory for a layer.

        Returns:
            (flops, memory_bytes)
        """
        from .model import Linear, Conv2d, BatchNorm2d, LayerNorm

        flops = 0
        memory_bytes = 0

        if isinstance(layer, Linear):
            # FLOPS: 2 * in_features * out_features (multiply + add)
            flops = 2 * layer.in_features * layer.out_features
            # Memory: weights + bias + input + output
            memory_bytes = (
                layer.in_features * layer.out_features * 4 +  # weights
                layer.out_features * 4 +  # bias
                layer.in_features * 4 +  # input
                layer.out_features * 4   # output
            )

        elif isinstance(layer, Conv2d):
            # Estimate based on typical feature maps
            # FLOPS: 2 * Cout * Cin * Kh * Kw * Hout * Wout
            kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
            flops = 2 * layer.out_channels * kernel_ops * 100 * 100  # Assume 100x100 output
            memory_bytes = (
                layer.out_channels * layer.in_channels *
                layer.kernel_size[0] * layer.kernel_size[1] * 4
            )

        elif isinstance(layer, (BatchNorm2d, LayerNorm)):
            # Minimal compute
            flops = layer.num_features if hasattr(layer, 'num_features') else 0
            memory_bytes = flops * 4 * 4  # gamma, beta, mean, var

        return flops, memory_bytes

    @property
    def stats(self) -> Optional[InferenceStats]:
        """Get last inference statistics."""
        return self._last_stats

    def get_layer_stats(self) -> Dict[str, LayerStats]:
        """Get per-layer statistics from last inference."""
        if self._last_stats is None:
            return {}
        return self._last_stats.layer_stats

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> InferenceStats:
        """Benchmark inference performance.

        Args:
            input_shape: Input tensor shape (including batch)
            num_iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations

        Returns:
            Aggregated statistics
        """
        # Create random input
        x = Tensor(np.random.randn(*input_shape).astype(np.float32))

        # Warmup
        for _ in range(warmup_iterations):
            _ = self(x)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        # Compute statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Return stats with average time
        stats = self._collect_stats(
            batch_size=input_shape[0],
            execution_time_ms=avg_time,
        )

        return stats

    def profile(
        self,
        x: Union[Tensor, np.ndarray],
    ) -> InferenceStats:
        """Profile inference with detailed layer-by-layer timing.

        Args:
            x: Input tensor

        Returns:
            Detailed inference statistics
        """
        # Enable profiling
        old_profile = self.profile_layers
        self.profile_layers = True

        # Run inference
        _, stats = self(x, return_stats=True)

        # Restore setting
        self.profile_layers = old_profile

        return stats


# =============================================================================
# Batch Inference Helper
# =============================================================================

class BatchInference:
    """Batch inference with automatic batching and progress tracking.

    Args:
        pipeline: Inference pipeline
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Example:
        >>> batch_inference = BatchInference(pipeline, batch_size=32)
        >>> results = batch_inference.run(dataset)
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.show_progress = show_progress

    def run(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Run batch inference on dataset.

        Args:
            data: Input data (N, ...) or list of arrays

        Returns:
            Output predictions
        """
        if isinstance(data, list):
            data = np.stack(data)

        num_samples = len(data)
        outputs = []

        for i in range(0, num_samples, self.batch_size):
            batch = data[i:i + self.batch_size]
            x = Tensor(batch.astype(np.float32))

            output = self.pipeline(x)
            outputs.append(output.numpy())

            if self.show_progress:
                progress = min(i + self.batch_size, num_samples)
                print(f"\rProcessing: {progress}/{num_samples}", end="", flush=True)

        if self.show_progress:
            print()  # Newline after progress

        return np.concatenate(outputs, axis=0)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_inference(
    model: Model,
    x: Union[Tensor, np.ndarray],
    fidelity: int = BEHAVIORAL,
) -> Tensor:
    """Run inference on a model.

    Args:
        model: KPU model
        x: Input tensor
        fidelity: Simulation fidelity

    Returns:
        Output tensor
    """
    pipeline = InferencePipeline(model, fidelity=fidelity)
    return pipeline(x)


def profile_model(
    model: Model,
    input_shape: Tuple[int, ...],
    fidelity: int = BEHAVIORAL,
) -> InferenceStats:
    """Profile a model's inference performance.

    Args:
        model: KPU model
        input_shape: Input tensor shape
        fidelity: Simulation fidelity

    Returns:
        Inference statistics
    """
    pipeline = InferencePipeline(model, fidelity=fidelity, profile_layers=True)
    x = Tensor(np.random.randn(*input_shape).astype(np.float32))
    _, stats = pipeline(x, return_stats=True)
    return stats


def benchmark_model(
    model: Model,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    fidelity: int = BEHAVIORAL,
) -> InferenceStats:
    """Benchmark a model's inference throughput.

    Args:
        model: KPU model
        input_shape: Input tensor shape
        num_iterations: Number of iterations
        fidelity: Simulation fidelity

    Returns:
        Benchmark statistics
    """
    pipeline = InferencePipeline(model, fidelity=fidelity)
    return pipeline.benchmark(input_shape, num_iterations)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LayerStats",
    "InferenceStats",
    "InferencePipeline",
    "BatchInference",
    "run_inference",
    "profile_model",
    "benchmark_model",
]

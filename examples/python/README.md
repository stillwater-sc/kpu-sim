# Native `stillwater_kpu` Examples

These scripts use the native `stillwater_kpu` module (`src/tools/bindings/python`), which exposes the simulator's
components directly. For compiling Python functions and PyTorch models, use the `kpu` package in
[`python/`](../../python/README.md) instead; its examples are in [`fusion/`](../fusion/), [`torch/`](../torch/) and
[`applications/`](../applications/).

## Setup

```bash
# From the repository root, with `build` configured as Release (`cmake --preset release`)
cmake --build build --target stillwater_kpu
export PYTHONPATH=build/src/tools/bindings/python/Release
```

The module goes to the directory of the build's configuration: for a Debug build, use
`build/src/tools/bindings/python/Debug`. Each script falls back to the Release directory only when `PYTHONPATH`
doesn't provide the module, so set `PYTHONPATH` for any other configuration.

## Scripts

| Script | What it does |
|--------|-------------|
| `address_based_dma_example.py` | DMA transfers through the unified address space: Host → External → L3, per level and routed by address. Checks every destination. |
| `kernel_graph_example.py` | Creates matmul and MLP kernels, compiles them, builds a two-layer kernel graph, and validates it. |
| `test_kernel_validation.py` | Kernel creation and metadata checks. |
| `test_graph_validation.py` | Kernel-graph construction and analysis checks. |
| `test_compiler_regression.py` | Kernel compiler regression checks. |

Each exits non-zero when a check fails. CTest runs all five as `python_example_*`
(`ctest --test-dir build -L python`), so an API change that breaks one fails the build's tests. They are registered only when tests and the
Python bindings are enabled (`KPU_BUILD_TESTS`, `KPU_BUILD_PYTHON_BINDINGS`, pybind11 available) and CMake knows the
interpreter the module was built for. Otherwise `ctest -L python` runs none of them, and CMake warns when the
binding is built but no interpreter is found.

The kernel scripts use the kernel path behind `isa::ConcurrentExecutor`, which ADR 0001 D6 freezes (#266). They
report structure and cycles, not values, and they are deleted or ported with that engine.

## Quick reference

```python
import stillwater_kpu as kpu

# Kernels
matmul = kpu.Kernel.create_matmul(M=1024, N=1024, K=1024)
mlp = kpu.Kernel.create_mlp(M=64, N=512, K=256, activation=kpu.ActivationType.GELU, has_bias=True)

# Compile, with tiling statistics
compiler = kpu.KernelCompiler()
kernel = compiler.compile_matmul(1024, 1024, 1024)
stats = compiler.last_stats()
print(f"Tiles: {stats.total_tiles}, arithmetic intensity: {stats.estimated_arithmetic_intensity:.2f}")

# Kernel graphs
graph = kpu.KernelGraph("my_network")
fc1 = graph.add_kernel(kpu.Kernel.create_mlp(64, 256, 128, kpu.ActivationType.RELU, True), "fc1")
fc2 = graph.add_kernel(kpu.Kernel.create_mlp(64, 64, 256, kpu.ActivationType.NONE, True), "fc2")
graph.add_edge(fc1, fc2, "C", "A")
result = graph.compile()
```

## Documentation

- [`docs/07-runtime/python-bindings.md`](../../docs/07-runtime/python-bindings.md): the native bindings
- [`docs/07-runtime/python-integration-guide.md`](../../docs/07-runtime/python-integration-guide.md): the Python API guide
- [`docs/design/multi-kernel-fusion-support.md`](../../docs/design/multi-kernel-fusion-support.md): kernel graphs

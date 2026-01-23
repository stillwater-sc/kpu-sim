# KPU-Simulator Test Guide

This document describes all C++ and Python tests and how to run them.

## Quick Start

```bash
# Run all C++ tests (excluding external domain_flow tests)
ctest --preset release -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)"

# Run all Python tests
cd python && ~/.local/bin/pytest tests/ -v

# Or with venv
source .venv/bin/activate && cd python && python -m pytest tests/ -v
```

## C++ Tests (CTest)

### Running C++ Tests

```bash
# Build first
cmake --preset release && cmake --build --preset release

# Run all KPU tests (recommended - excludes external tests)
ctest --preset release -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure

# Run specific test by name
ctest --preset release -R "graph_loader" -V

# Run tests matching pattern
ctest --preset release -R "memory" -V
ctest --preset release -R "dma" -V
ctest --preset release -R "xue" -V
```

### C++ Test Categories

| Directory | Tests | Description |
|-----------|-------|-------------|
| `tests/system/` | Configuration, formatting | System-level tests |
| `tests/memory/` | Allocation, sparse, map | Memory subsystem |
| `tests/dma/` | Basic, performance, tensor | DMA engine |
| `tests/block_mover/` | Basic ops, tracing | Block mover component |
| `tests/streamer/` | Basic ops, tracing | Streamer component |
| `tests/compute/` | Fabric, systolic array | Compute fabric |
| `tests/storage/` | IDDO, EDDO workflows | Storage scheduler |
| `tests/xue/` | Event recording, analysis | XUE Observation Architecture |
| `tests/integration/` | End-to-end, multi-component | Integration tests |

### XUE Tests (79 assertions)

```bash
ctest --preset release -R xue -V
```

Tests the C++ XUE Observation Architecture:
- Event type hierarchy (45+ event types)
- Event collector singleton
- Atomic thread-safe counters
- Operational analysis (roofline model)

## Python Tests (pytest)

### Running Python Tests

```bash
cd python

# Run all tests
~/.local/bin/pytest tests/ -v

# Run specific test file
~/.local/bin/pytest tests/test_transformer_ops.py -v

# Run specific test class
~/.local/bin/pytest tests/test_fusion.py::TestConv2DFusionPatterns -v

# Run specific test
~/.local/bin/pytest tests/test_native_execution.py::TestCppExecutionProof::test_behavioral_records_xue_events -v

# With coverage
~/.local/bin/pytest tests/ --cov=kpu --cov-report=term-missing
```

### Python Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_native_execution.py` | 16 | Verifies C++ backend execution |
| `test_transformer_ops.py` | 10 | Softmax, layer_norm, attention |
| `test_xue_integration.py` | 18 | XUE API and event recording |
| `test_fusion.py` | 32 | Operation fusion patterns |
| `test_kpu.py` | ~20 | Core KPU operations |
| `test_model.py` | ~10 | Model execution |
| `test_cnn_validation.py` | ~15 | CNN operations |
| `test_v05_kernel_validation.py` | ~10 | v0.5 kernel validation |

### Test Details

#### test_native_execution.py (16 tests)
Verifies that BEHAVIORAL and TRANSACTIONAL modes use C++ BehavioralComputeFabric:
- `TestNativeAvailability`: Check native module is built
- `TestStrictNativeMode`: strict_native mode functionality
- `TestExecutionBackendTracking`: execution_backend field in stats
- `TestVerifyNativeExecution`: verify_native_execution() function
- `TestCppExecutionProof`: XUE events prove C++ execution
- `TestNoPythonFallback`: No fallback when native available
- `TestInfoFunction`: kpu.info() includes native status

#### test_transformer_ops.py (10 tests)
Tests transformer operations via C++ backend:
- `TestSoftmax`: Basic softmax, numerical stability
- `TestLayerNorm`: Basic layer_norm, affine transform
- `TestAttention`: Basic attention, batched, uniform weights
- `TestFusedMatMulOps`: Fused matmul+bias+relu, matmul+gelu
- `TestTransformerMLP`: Complete MLP block

#### test_xue_integration.py (18 tests)
Tests XUE Observation Architecture Python API:
- `TestXUEAPI`: get_xue_summary(), get_operational_analysis(), validate_operational_analysis()
- `TestXUEEventRecording`: Matmul, ReLU, multiple operations
- `TestXUEMemoryHierarchy`: Memory level tracking, DRAM traffic
- `TestXUETransactionalMode`: XUE in transactional mode
- `TestXUERooflineModel`: Ridge point, memory/compute bound prediction
- `TestXUECorrectness`: FLOP counts, event consistency

#### test_fusion.py (32 tests)
Tests operation fusion for performance optimization:
- `TestFusionPatternDetection`: Matmul+bias+relu, matmul+bias+gelu, etc.
- `TestFusedOperationCorrectness`: Fused ops produce correct output
- `TestGraphRewriting`: Operation count reduction, topological order
- `TestConv2DFusionPatterns`: Conv2d+relu, conv2d+bn+relu
- `TestOpTypePredicates`: is_fused, is_fused_conv, is_fused_matmul
- `TestDFXEmission`: Fused ops in DFX format
- `TestMemorySavingsEstimation`: Memory savings from fusion
- `TestFusionAnalyzer`: Find fusion opportunities
- `TestRooflineAnalysis`: Roofline metrics, efficiency
- `TestFusionReport`: Report generation

## Test Summary

| Suite | Tests | Command |
|-------|-------|---------|
| C++ (CTest) | 30 | `ctest --preset release -E "^(dsp_\|nla_\|dfa_\|dnn_\|ctl_\|cnn_)"` |
| Python | 76+ | `cd python && pytest tests/ -v` |
| **Total** | **106+** | |

## Continuous Integration

For CI pipelines:

```bash
# C++ tests
cmake --preset release
cmake --build --preset release
ctest --preset release -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure

# Python tests (requires native module built)
cd python
python -m pytest tests/ -v --tb=short
```

## Troubleshooting

### Python tests skip with "Native module not available"
The native module wasn't built or Python version mismatch:
```bash
# Check Python version matches build
python --version  # Should match what CMake used

# Rebuild native module
cmake --build --preset release --target _native
```

### XUE tests show 0 events
XUE events are recorded in C++ compute fabrics. Ensure:
1. Native module is available: `kpu.is_native_available() == True`
2. Using BEHAVIORAL or TRANSACTIONAL mode
3. Counters reset before test: `kpu.reset_xue_counters()`

### Segfaults in tests
Check the session logs in `docs/sessions/` for known issues and fixes.
Common causes:
- Nullptr dereference in C++ (fixed by null checks)
- numpy interop issues (fixed by pure C++ loops)

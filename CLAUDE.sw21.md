# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Stillwater Knowledge Processing Unit (KPU) Simulator is a C++20 functional simulator for a specialized hardware accelerator targeting knowledge processing and AI workloads. Key characteristics:

- **Data-driven execution model** (not traditional stored-program architecture)
- **Software-managed memory hierarchy** with DMA engines, BlockMovers, and Streamers
- **Systolic array** for matrix operations (tau111_s001 configuration: 8×8 PEs)
- **Integration with domain_flow** computational graphs

## Build Commands

```bash
# Build (release)
cmake --preset=release
cmake --build --preset=release

# Build (debug with sanitizers)
cmake --preset=debug
cmake --build --preset=debug

# Run all tests (excluding external domain_flow tests)
ctest --test-dir build -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure

# Run single test
ctest --test-dir build -R "test_name" -V

# Run tests by category label
ctest --test-dir build -L compiler -V    # compiler tests
ctest --test-dir build -L memory -V      # memory tests
ctest --test-dir build -L integration -V # integration tests

# Build with local domain_flow
cmake --preset=release -DKPU_DOMAIN_FLOW_LOCAL_PATH=~/dev/domain_flow
```

## Architecture

### Memory Hierarchy (Nested)
```
Host DDR → (PCIe) → External Memory (GDDR6/HBM)
  → DMA Engines (50GB/s) → L3 Tiles (128KB × 4)
  → BlockMovers (100GB/s) → L2 Banks (64KB × 8)
  → Streamers (200GB/s) → L1 Buffers (32KB × 4)
  → PE Registers → Systolic Array
```

### Key Subsystems

| Directory | Purpose |
|-----------|---------|
| `src/system/` | Top-level orchestration, configuration loading |
| `src/components/datamovement/` | DMA, BlockMover, Streamer engines |
| `src/components/compute/` | SystolicArray, ComputeFabric |
| `src/components/memory/` | L1/L2/L3 caches, address decoder |
| `src/compiler/` | Graph loading, tile optimization, schedule generation |
| `src/isa/` | Tile layouts, memory interleaving policies |

### Core Abstractions

- **GraphLoader** (`include/sw/compiler/graph_loader.hpp`): Loads domain_flow `.dfg` or JSON graphs
- **TileOptimizer** (`include/sw/compiler/tile_optimizer.hpp`): Determines optimal tile sizes for memory hierarchy
- **ScheduleGenerator** (`include/sw/compiler/schedule_generator.hpp`): Converts graphs to execution schedules
- **TileLayout** (`include/sw/isa/tile_layout.hpp`): Memory channel interleaving policies (MATRIX_PARTITIONED, ROUND_ROBIN, ITERATION_AWARE, HARDWARE_INTERLEAVED)

### Namespaces

- `sw::kpu::*` - KPU simulator components
- `sw::sim::*` - System-level simulation
- `sw::kpu::compiler::*` - Compiler infrastructure

## Testing

Test helper macro in `tests/CMakeLists.txt`: `kpu_add_component_test(NAME category/test_name SOURCES file.cpp LABELS label1 label2)`

Test categories: system, driver, memory, trace, dma, block_mover, streamer, compute, storage, compiler, isa, integration

## Configuration

JSON-based configuration in `configs/`. Key CMake options:
- `KPU_BUILD_TESTS`, `KPU_BUILD_EXAMPLES`, `KPU_BUILD_PYTHON_BINDINGS` (all ON by default)
- `KPU_DOMAIN_FLOW_LOCAL_PATH` - Path to local domain_flow for development

## Python Bindings

```bash
pip install -e .
export PYTHONPATH=$PWD/build:$PYTHONPATH
python tests/test_python.py
```

Module name: `kpu_bindings` (import as `import kpu_bindings`)

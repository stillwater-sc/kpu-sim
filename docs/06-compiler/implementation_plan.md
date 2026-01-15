# KPU Compiler Implementation Plan

## Executive Summary

This document outlines the implementation plan for the KPU kernel compiler workflow, consisting of:
1. **kpu_kernel_compiler** - Compiles DFG files into KPU object files
2. **kpu_loader/driver** - Loads object files to program the KPU simulator

The design follows the NVIDIA PTX model where high-level abstractions remain hardware-agnostic, and the driver/orchestration layer handles micro-architecture-specific bufferization.

---

## 1. Architecture Overview

### 1.1 Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KPU Compilation Pipeline                               │
└─────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌───────────┐
   │  .dfg   │────▶│ kpu_kernel_      │────▶│  .kpu Object    │────▶│  KPU      │
   │  File   │     │ compiler         │     │  File           │     │  Loader   │
   └─────────┘     └──────────────────┘     └─────────────────┘     └───────────┘
                          │                                               │
                          │                                               │
                          ▼                                               ▼
              ┌───────────────────────┐                    ┌─────────────────────────┐
              │  Hardware-Agnostic    │                    │  Micro-Architecture     │
              │  Representation       │                    │  Specific Scheduling    │
              │  (PTX-equivalent)     │                    │  & Bufferization        │
              └───────────────────────┘                    └─────────────────────────┘
```

### 1.2 Abstraction Layers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Domain Flow Graph (DFG)                                                │
│ - High-level operator graph (MATMUL, CONV2D, SOFTMAX, etc.)                    │
│ - Tensor shapes and data types                                                  │
│ - Data dependencies between operators                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Domain Flow Execution (DFX) - "PTX-equivalent"                         │
│ - Tiled computation specification (Ti, Tj, Tk parameters)                       │
│ - Data movement commands (abstract DMA, BlockMover, Streamer ops)              │
│ - Synchronization barriers                                                      │
│ - Memory allocation hints (not concrete addresses)                              │
│ - Dataflow strategy (output/weight/input stationary)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Scheduled Object File (.kpu)                                           │
│ - Hardware-agnostic operation sequence                                          │
│ - Tile iteration order and dependencies                                         │
│ - Abstract data movement operations                                             │
│ - Performance hints and constraints                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Driver/Loader (Runtime)                                                │
│ - Concrete memory address assignment                                            │
│ - Specific DMA engine/BlockMover/Streamer binding                              │
│ - L1/L2/L3 tile allocation based on actual hardware                            │
│ - Cycle-accurate scheduling                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. PTX-Equivalent Intermediate Representation (DFX)

DFX = Domain Flow Execution - the hardware-agnostic intermediate representation for KPU programs,
analogous to NVIDIA's PTX (Parallel Thread Execution).

### 2.1 Design Principles

The DFX must capture computation and data movement at a level that:
1. **Hardware-agnostic**: Same DFX works on different KPU configurations
2. **Expressive**: Captures all necessary scheduling decisions
3. **Optimizable**: Allows driver-level optimization
4. **Serializable**: Can be saved to disk and loaded later

### 2.2 DFX Operations

```cpp
namespace dfx {

// ============================================================================
// Memory Region Specification (Abstract - no concrete addresses)
// ============================================================================
enum class MemoryLevel {
    EXTERNAL,       // External DRAM
    L3,             // L3 cache tiles
    L2,             // L2 cache banks
    L1,             // L1 streaming buffers
    REGISTER        // Systolic array registers
};

struct TensorRegion {
    std::string tensor_name;    // e.g., "A", "B", "C"
    MemoryLevel level;          // Where this region resides
    std::vector<size_t> tile_indices;  // [row_tile, col_tile] for 2D
    std::vector<size_t> tile_shape;    // Actual tile dimensions
    size_t element_size;        // Bytes per element
};

// ============================================================================
// Data Movement Operations (Abstract)
// ============================================================================
enum class DataMoveType {
    LOAD,           // Lower level → Higher level (e.g., DRAM → L3)
    STORE,          // Higher level → Lower level
    PREFETCH,       // Speculative load
    FLUSH           // Force writeback
};

struct DataMoveOp {
    uint64_t op_id;             // Unique operation ID
    DataMoveType type;
    TensorRegion source;
    TensorRegion destination;

    // Transform hints
    bool transpose;             // Apply transpose during move
    bool broadcast;             // Broadcast to multiple destinations

    // Synchronization
    std::vector<uint64_t> depends_on;  // Operations this depends on
    uint64_t produces;          // Synchronization point this produces
};

// ============================================================================
// Compute Operations
// ============================================================================
enum class ComputeType {
    MATMUL_TILE,    // Tile matrix multiplication
    CONV2D_TILE,    // Tile convolution
    ELEMENTWISE,    // Element-wise operation
    REDUCTION,      // Reduction operation
    SOFTMAX_TILE,   // Tile softmax
    LAYERNORM_TILE  // Tile layer normalization
};

struct ComputeOp {
    uint64_t op_id;
    ComputeType type;

    // Input tiles (in L1/registers)
    std::vector<TensorRegion> inputs;
    // Output tile (in L1/registers)
    TensorRegion output;

    // For MATMUL: accumulate flag
    bool accumulate;            // Add to existing output vs overwrite

    // Synchronization
    std::vector<uint64_t> depends_on;
    uint64_t produces;
};

// ============================================================================
// Synchronization Barriers
// ============================================================================
struct BarrierOp {
    uint64_t op_id;
    std::vector<uint64_t> wait_for;  // Wait for these ops to complete
    std::string label;               // For debugging/tracing
};

// ============================================================================
// Loop Constructs (for expressing tiled iteration)
// ============================================================================
struct TileLoop {
    std::string induction_var;  // e.g., "ti", "tj", "tk"
    size_t start;
    size_t end;
    size_t step;                // Usually 1 (iterating over tiles)
    std::vector<size_t> iteration_order;  // Pre-computed iteration sequence
};

} // namespace dfx
```

### 2.3 Hardware-Agnostic Properties

| Property | Specified in DFX | Determined by Driver |
|----------|------------------|----------------------|
| Tile dimensions (Ti, Tj, Tk) | ✓ | |
| Tile iteration order | ✓ | |
| Data dependencies | ✓ | |
| Memory level hints | ✓ | |
| Dataflow strategy | ✓ | |
| Concrete addresses | | ✓ |
| DMA engine assignment | | ✓ |
| L2 bank allocation | | ✓ |
| Cycle timing | | ✓ |
| Prefetch distance | | ✓ |

---

## 3. Object File Format (.kpu)

### 3.1 File Structure

```
┌───────────────────────────────────────────┐
│ KPU Object File (.kpu)                    │
├───────────────────────────────────────────┤
│ Header                                    │
│   - Magic number: "KPU\0"                 │
│   - Version: uint32                       │
│   - Flags: uint32                         │
│   - Section count: uint32                 │
├───────────────────────────────────────────┤
│ Metadata Section                          │
│   - Graph name                            │
│   - Target dataflow (OS/WS/IS)           │
│   - Tile configuration (Ti, Tj, Tk)       │
│   - Matrix dimensions (M, N, K)           │
│   - Element type and size                 │
├───────────────────────────────────────────┤
│ Tensor Descriptors Section                │
│   - List of all tensors                   │
│   - Shapes, dtypes, memory hints          │
├───────────────────────────────────────────┤
│ Operation Sequence Section                │
│   - Serialized DFX operations             │
│   - DataMoveOps, ComputeOps, BarrierOps   │
├───────────────────────────────────────────┤
│ Dependency Graph Section                  │
│   - DAG of operation dependencies         │
│   - Enables parallel execution analysis   │
├───────────────────────────────────────────┤
│ Hints Section (Optional)                  │
│   - Performance hints                     │
│   - Estimated memory traffic              │
│   - Suggested parallelism                 │
└───────────────────────────────────────────┘
```

### 3.2 Serialization Format Options

1. **Binary format** - Compact, fast to load
2. **JSON format** - Human-readable, debuggable
3. **FlatBuffers/Cap'n Proto** - Best of both worlds

Initial implementation will use JSON for debuggability, with binary option later.

---

## 4. Tools Directory Reorganization

### 4.1 Current Structure (to be deprecated)

```
tools/
├── cpp/                    # Awkward location
│   └── CMakeLists.txt
├── compiler/               # Python tools mixed in
├── python/
└── dse/
```

### 4.2 Proposed Structure

```
tools/
├── CMakeLists.txt          # Main tools CMake
│
├── compiler/               # Compilation toolchain
│   ├── CMakeLists.txt
│   ├── kpu-kernel-compiler/
│   │   ├── main.cpp
│   │   ├── dfg_parser.hpp
│   │   ├── dfg_parser.cpp
│   │   ├── dfx_generator.hpp
│   │   ├── dfx_generator.cpp
│   │   ├── object_writer.hpp
│   │   └── object_writer.cpp
│   └── python/             # Python compiler tools
│       ├── visualize_pareto_frontier.py
│       └── ...
│
├── runtime/                # Runtime/loader tools
│   ├── CMakeLists.txt
│   ├── kpu-loader/
│   │   ├── main.cpp
│   │   ├── object_reader.hpp
│   │   ├── object_reader.cpp
│   │   ├── schedule_binder.hpp
│   │   └── schedule_binder.cpp
│   └── kpu-driver/         # High-level driver API
│       └── ...
│
├── analysis/               # Analysis and profiling
│   ├── CMakeLists.txt
│   ├── kpu-profiler/
│   └── kpu-trace-analyzer/
│
├── development/            # Development tools
│   ├── CMakeLists.txt
│   ├── kpu-assembler/
│   ├── kpu-debugger/
│   └── kpu-disassembler/
│
├── configuration/          # Configuration tools
│   ├── CMakeLists.txt
│   └── kpu-config/
│
├── benchmark/              # Benchmarking tools
│   ├── CMakeLists.txt
│   └── kpu-benchmark/
│
└── dse/                    # Design Space Exploration
    └── python scripts
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**5.1.1 Create directory structure**
- Set up new tools directory layout
- Create CMakeLists.txt for each category
- Move existing tools to appropriate locations

**5.1.2 Define DFX data structures**
- Header files for DFX types
- Serialization/deserialization
- Unit tests

**5.1.3 Object file format**
- JSON-based initial format
- Reader/writer classes
- Validation utilities

### Phase 2: Compiler Implementation (Week 3-4)

**5.2.1 DFG Parser enhancements**
- Extend existing GraphLoader
- Support for MATMUL operator extraction
- Tensor shape and dependency analysis

**5.2.2 DFX Generator**
- Transform DFG operators to DFX operations
- Integrate TileOptimizer for tile size selection
- Generate data movement operations
- Create dependency graph

**5.2.3 Object File Writer**
- Serialize DFX to .kpu format
- Include metadata and hints
- Validate output

### Phase 3: Loader/Driver Implementation (Week 5-6)

**5.3.1 Object File Reader**
- Load .kpu files
- Validate format and version
- Extract DFX operations

**5.3.2 Schedule Binder**
- Bind abstract ops to concrete resources
- Allocate L2/L3 tiles
- Assign DMA engines, BlockMovers, Streamers
- Generate cycle-accurate schedule

**5.3.3 Simulator Integration**
- Execute scheduled operations
- Drive KPUSimulator
- Collect traces and statistics

### Phase 4: Advanced Operators (Week 7-8)

**5.4.1 CONV2D Support**
- Im2col transformation in KIR
- Strided data movement
- Multiple tile layouts

**5.4.2 MLP with Softmax**
- Multi-operator fusion
- Intermediate tensor handling
- Softmax tiling strategy

---

## 6. Key Design Decisions

### 6.1 Tiling Strategy Selection

The compiler must select tiling parameters that work across hardware configurations:

```cpp
struct TilingConstraints {
    // Minimum requirements (from DFX)
    size_t min_tile_elements;   // Must have enough work per tile
    size_t max_tiles_per_dim;   // Limit iteration overhead

    // Hints (from compiler analysis)
    double target_reuse_ratio;  // Desired data reuse
    bool prefer_square_tiles;   // For balanced data movement
};
```

### 6.2 Dataflow Selection

The compiler analyzes the DFG to select optimal dataflow:

| Workload Pattern | Recommended Dataflow |
|-----------------|---------------------|
| Large M (batch) | Weight Stationary (B resident) |
| Large N (many outputs) | Input Stationary (A resident) |
| Balanced M=N | Output Stationary |
| Small K (thin matrices) | Output Stationary with full K |

### 6.3 Memory Hierarchy Abstraction

```cpp
// Compiler generates abstract allocations
struct AbstractAllocation {
    std::string tensor_name;
    MemoryLevel level;
    size_t size_bytes;
    AllocationHint hint;  // TEMPORARY, PERSISTENT, SHARED
};

// Driver resolves to concrete addresses
struct ConcreteAllocation {
    AbstractAllocation abstract;
    uint64_t base_address;
    size_t bank_id;       // For L2
    size_t tile_id;       // For L3
    size_t buffer_id;     // For L1
};
```

---

## 7. CLI Interface Design

### 7.1 kpu-kernel-compiler

```bash
# Basic compilation
kpu-kernel-compiler matmul.dfg -o matmul.kpu

# With options
kpu-kernel-compiler matmul.dfg \
    --output matmul.kpu \
    --dataflow output-stationary \
    --tile-strategy analytical \
    --format json \
    --verbose

# Generate human-readable output
kpu-kernel-compiler matmul.dfg --emit-dfx --dump
```

### 7.2 kpu-loader

```bash
# Load and execute
kpu-loader matmul.kpu \
    --config kpu_config.json \
    --input-a A.bin \
    --input-b B.bin \
    --output-c C.bin

# Profile execution
kpu-loader matmul.kpu --profile --trace-output trace.json

# Dry-run (show schedule without execution)
kpu-loader matmul.kpu --dry-run --verbose
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

- DFX serialization/deserialization
- Object file read/write
- Tile optimizer integration
- Schedule binder correctness

### 8.2 Integration Tests

- End-to-end: DFG → .kpu → execution
- Multiple hardware configurations
- Various matrix sizes

### 8.3 Validation Tests

- Compare results with reference implementation
- Numerical accuracy verification
- Performance regression tests

---

## 9. File Locations Summary

| Component | Location |
|-----------|----------|
| DFX definitions | `include/sw/compiler/dfx/` |
| Object file format | `include/sw/compiler/dfx/dfx_object_file.hpp` |
| Kernel compiler | `tools/compiler/kpu-kernel-compiler/` |
| Loader/driver | `tools/runtime/kpu-loader/` |
| Tests | `tests/compiler/` |
| Examples | `examples/compiler/` |

---

## 10. Dependencies

### 10.1 Required

- domain_flow IR library (for .dfg parsing)
- nlohmann_json (for serialization)
- Existing compiler infrastructure (TileOptimizer, L2TileScheduler)

### 10.2 Optional

- FlatBuffers (for binary format)
- CLI11 (for command-line parsing)
- fmt (for formatting)

---

## 11. Next Steps

1. **Approve this plan** - Review and iterate on design decisions
2. **Create directory structure** - Set up new tools layout
3. **Implement DFX headers** - Define core data structures
4. **Implement object file format** - JSON-based serialization
5. **Build kernel compiler skeleton** - Main CLI with parsing
6. **Integrate existing schedulers** - Connect TileOptimizer and L2TileScheduler
7. **Build loader skeleton** - Object file reading and binding
8. **End-to-end testing** - Verify complete pipeline

---

## Appendix A: Example DFX for MATMUL

```json
{
  "dfx_version": "1.0",
  "graph_name": "non-batched-2-input-matmul",
  "metadata": {
    "M": 4, "N": 4, "K": 16,
    "Ti": 4, "Tj": 4, "Tk": 16,
    "dataflow": "output_stationary",
    "element_type": "float32"
  },
  "tensors": [
    {"name": "A", "shape": [4, 16], "dtype": "f32"},
    {"name": "B", "shape": [16, 4], "dtype": "f32"},
    {"name": "C", "shape": [4, 4], "dtype": "f32"}
  ],
  "operations": [
    {
      "op_id": 1,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "A", "level": "EXTERNAL", "tile": [0, 0]},
      "dest": {"tensor": "A", "level": "L3", "tile": [0, 0]},
      "depends_on": [],
      "produces": 1
    },
    {
      "op_id": 2,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "B", "level": "EXTERNAL", "tile": [0, 0]},
      "dest": {"tensor": "B", "level": "L3", "tile": [0, 0]},
      "depends_on": [],
      "produces": 2
    },
    {
      "op_id": 3,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "A", "level": "L3", "tile": [0, 0]},
      "dest": {"tensor": "A", "level": "L2", "tile": [0, 0]},
      "depends_on": [1],
      "produces": 3
    },
    {
      "op_id": 4,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "B", "level": "L3", "tile": [0, 0]},
      "dest": {"tensor": "B", "level": "L2", "tile": [0, 0]},
      "depends_on": [2],
      "produces": 4
    },
    {
      "op_id": 5,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "A", "level": "L2", "tile": [0, 0]},
      "dest": {"tensor": "A", "level": "L1", "tile": [0, 0]},
      "depends_on": [3],
      "produces": 5
    },
    {
      "op_id": 6,
      "type": "DATA_MOVE",
      "subtype": "LOAD",
      "source": {"tensor": "B", "level": "L2", "tile": [0, 0]},
      "dest": {"tensor": "B", "level": "L1", "tile": [0, 0]},
      "depends_on": [4],
      "produces": 6
    },
    {
      "op_id": 7,
      "type": "COMPUTE",
      "subtype": "MATMUL_TILE",
      "inputs": [
        {"tensor": "A", "level": "L1", "tile": [0, 0]},
        {"tensor": "B", "level": "L1", "tile": [0, 0]}
      ],
      "output": {"tensor": "C", "level": "REGISTER", "tile": [0, 0]},
      "accumulate": false,
      "depends_on": [5, 6],
      "produces": 7
    },
    {
      "op_id": 8,
      "type": "DATA_MOVE",
      "subtype": "STORE",
      "source": {"tensor": "C", "level": "REGISTER", "tile": [0, 0]},
      "dest": {"tensor": "C", "level": "EXTERNAL", "tile": [0, 0]},
      "depends_on": [7],
      "produces": 8
    }
  ],
  "hints": {
    "estimated_dram_bytes": 416,
    "estimated_compute_cycles": 1024,
    "parallelism_degree": 1
  }
}
```

---

## Appendix B: Relationship to Existing Code

| Existing Component | Role in New System |
|-------------------|-------------------|
| `GraphLoader` | Used by kpu-kernel-compiler to parse .dfg files |
| `TileOptimizer` | Generates Ti, Tj, Tk for DFX |
| `L2TileScheduler` | Informs tile iteration order and reuse patterns |
| `L3Scheduler` | Provides DRAM traffic estimates for hints |
| `DMAEngine` | Target for driver binding of DATA_MOVE ops |
| `BlockMover` | Target for L3→L2 moves |
| `Streamer` | Target for L2→L1 moves |
| `KPUSimulator` | Execution target for loaded schedules |

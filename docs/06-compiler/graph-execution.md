# Functional Graph Execution


> We need to build a functional simulator for the KPU. Can you analyze this repo and tell me which components are already here and which components we
still need to design and implement?

● Based on my analysis of the repository, here's a comprehensive breakdown:

  What the KPU Is

  The KPU (Knowledge Processing Unit) is a Domain Flow Architecture (DFA) accelerator - a push-based dataflow pipeline machine with:
  - Tagged token matching for position-independent computation
  - Distributed Content Addressable Memory (CAM) across processing elements
  - Programmable systolic arrays that execute Systems of Uniform Recurrence Equations (SUREs)
  - Self-coordinating execution through token flow (no centralized synchronization)

  ✅ Implemented Components

  Memory Hierarchy (Complete)

  - src/components/memory/external_memory.cpp - External DRAM/GDDR6
  - src/components/memory/l3_tile.cpp - L3 scratchpad tiles (1-8MB)
  - src/components/memory/l2_bank.cpp - L2 scratchpad banks (16-64KB)
  - src/components/memory/l1_buffer.cpp - L1 stream buffers
  - src/components/memory/memory_map.cpp - Unified address space
  - src/components/memory/address_decoder.cpp - Address decoding

  Data Movement (Complete)

  - src/components/datamovement/dma_engine.cpp - DMA engines (DRAM ↔ L3)
  - src/components/datamovement/block_mover.cpp - BlockMovers (L3 ↔ L2)
  - src/components/datamovement/streamer.cpp - Streamers (L2 → L1 → Compute)

  Basic Compute

  - src/components/compute/systolic_array.cpp - Systolic array
  - src/components/compute/compute_fabric.cpp - Fabric orchestration
  - src/components/compute/matmul_tau111_s001.cpp - Matrix multiplication kernel

  System Infrastructure

  - Configuration system (JSON-based)
  - Top-level simulator (src/system/toplevel.cpp)
  - Python/C bindings
  - Comprehensive test suite (40+ tests)

  ❌ Missing/Incomplete Components

  Critical Missing (Core DFA Features)

  1. Processing Element (PE) Implementation - Headers exist but no .cpp files
  2. Token Routing & CAM Matching - The fundamental dataflow mechanism
  3. Domain Flow Program Loader - Loading SURE programs into PEs
  4. Special Function Units (SFUs) - Activation functions, fusion operations
  5. Dynamic Sparsity Gating - Zero-detection and dynamic gating
  6. Quantization Support - Multi-stage quantization in data movement

  Infrastructure Gaps

  7. Memory Controller/Page Cache - cache.cpp is nearly empty (82 bytes)
  8. Network-on-Chip (NoC) - Directory exists, only README present
  9. PCIe Controller - Only arbiter exists, full controller missing
  10. Energy Modeling - Extensively documented but not implemented

  Tooling Gaps

  11. KPU Assembly Compiler (KAC) - Not implemented
  12. KPU Configuration Generator (KCG) - Not found
  13. Benchmark Suite - Directory empty

  Current State

  You have a memory-centric architectural simulator that can model:
  - ✅ Data movement patterns and bandwidth
  - ✅ Memory hierarchy behavior
  - ✅ Basic systolic compute throughput
  - ✅ System configuration and integration

  You do NOT yet have a full Domain Flow Architecture simulator with:
  - ❌ Tagged token dataflow execution
  - ❌ CAM-based operand matching
  - ❌ Self-synchronizing computation
  - ❌ Hardware-driven operator fusion

  Recommendation

  To build a functional KPU simulator, you need to prioritize:

  1. Processing Element implementation (the compute core)
  2. Token routing and CAM matching (the dataflow engine)
  3. Domain flow program representation and loading
  4. Memory controller/page cache completion
  5. Network-on-Chip for inter-tile communication


## System level programming

We have done cycle-accurate simulation of the PE and CAM and token routing to drive the micro-architeture of the KPU, but
that level of modeling is very slow and does not reveal the micro-architecture design required for the compiler. The KPU compute fabric has been modeled
already, but what we are missing is how a compiler would need to program the system level schedules. Remember, the compute fabric is push-based and
adaptive, we do not need to model the low-level dynamics of the compute fabric as that is 'automatic' and governed by an architectural design methodology
 that takes the SURE expressions and transforms them via the scheduling vector and projection into an executable for the compute fabric. However, the
compiler that takes a computational graph and needs to program the system level data movement schedule needs more work, and that is the priority we want
to focus on today.

● we need a system-level compiler that maps computational graphs to data movement schedules, while the
  compute fabric handles its own dataflow execution. Let me analyze what compiler infrastructure exists and what's missing.

  Current State: What Exists for System-Level Compilation

  ✅ Infrastructure (Well-Developed)

  Memory Hierarchy Components:
  - DMAEngine - DRAM ↔ L3 transfers (IDDO-style)
  - BlockMover - L3 ↔ L2 transfers with transforms (IDDO-style)
  - Streamer - L2 → L1 → Fabric streaming (IDDO-style)
  - StorageScheduler - Autonomous multi-bank storage with command sequences
  - MemoryOrchestrator - EDDO-style orchestration layer

  Tiling Infrastructure:
  - tools/compiler/kpu_tiling/matmul_simulator.py - Bandwidth tracking for tiled matmul
  - tools/compiler/kpu_tiling/loop_config.py - Loop nest configuration (result/input/weight-stationary strategies)
  - tools/python/distributed_sequencer.py - Cycle-accurate sequencer simulation for DMA/BlockMover/Streamer coordination

  System Configuration:
  - JSON-based configuration system
  - System topology definitions (T64, T256, T768)
  - Memory hierarchy capacity/bandwidth parameters

  ❌ Missing: Compiler Components

  1. Computational Graph Representation
  Status: DOES NOT EXIST
  Need:
  - IR for representing operator graphs (from PyTorch/TensorFlow/JAX/ONNX)
  - Nodes: Operators (GEMM, Conv, Attention, etc.)
  - Edges: Data dependencies (tensor flow)
  - Metadata: Tensor shapes, dtypes, layout

  2. Graph Ingestion & Lowering
  Status: DOES NOT EXIST
  Need:
  - ONNX → KPU IR converter
  - PyTorch → KPU IR converter
  - Framework-agnostic operator representation

  3. Bufferization & Memory Planning
  Status: PARTIAL (tiling tools exist, but no full bufferization)
  Need:
  - Tensor lifetime analysis
  - L3/L2 memory allocation strategy
  - Double/triple buffering decisions
  - Spilling/eviction policies

  4. Schedule Generation (THE CRITICAL MISSING PIECE)
  Status: DOES NOT EXIST
  Need:
  - Map operators → tiled execution schedules
  - Generate DMA/BlockMover/Streamer command sequences
  - Determine tile sizes (TM, TN, TK) based on:
    * Memory hierarchy capacity
    * Compute fabric dimensions
    * Bandwidth constraints
  - Coordinate system-level timing:
    * When to prefetch B tiles from DRAM
    * When to load A tiles to L2
    * When to stream into L1/fabric

  5. Code Generation
  Status: DOES NOT EXIST
  Need:
  - Emit executable schedules
  - Generate configuration for:
    * DMA descriptors
    * BlockMover transforms
    * Streamer injection patterns
  - Runtime dispatch code

  What We Have: IDDO Building Blocks

  Your existing DMA/BlockMover/Streamer are execution engines - they're the IDDO infrastructure that executes schedules once generated. What's missing is
   the compiler that generates those schedules.

  What We Need: A System-Level Scheduler/Compiler

  Based on docs/kpu_architecture.md:950-1026, the compiler should focus on:

  Compilation Stages (from architecture doc)

  1. Graph Ingestion ✅ (documented) ❌ (not implemented)
    - Parse operator graph from framework
    - Preserve operator semantics
  2. Bufferization ⚠️ (partially - tiling tools exist)
    - Analyze data dependencies
    - Allocate scratchpad memory (L3, L2)
    - Determine buffer lifetimes
    - Minimize on-chip memory
  3. Domain Flow Program Generation ✅ (handled by compute fabric - automatic)
    - Transform operators → SURE representation
    - This is "automatic" per your clarification
  4. Schedule Generation ❌ (THE MISSING PIECE)
    - System-level domain schedule for DMA
    - Tile sizes and blocking strategies
    - BlockMover and Streamer configurations
    - Overlap computation and data movement
  5. Code Generation ❌ (not implemented)
    - Runtime operator dispatch
    - DMA/BlockMover/Streamer configs
    - Domain Flow Programs packaging

  Concrete Example: What a Compiler Needs to Generate

  For a simple C = A @ B matmul:

```cpp
  Input (Computational Graph):

  {
    "op": "matmul",
    "A": {"shape": [1024, 512], "dtype": "float32"},
    "B": {"shape": [512, 1024], "dtype": "float32"},
    "C": {"shape": [1024, 1024], "dtype": "float32"}
  }
```

  Output (System-Level Schedule):

```cpp
  # What the compiler must generate
  schedule = [
    # DMA commands (DRAM → L3)
    {"cycle": 0, "component": "DMA", "op": "prefetch_B", "tile_id": "B[0,0]", "src": 0x1000, "dst": "L3_tile_0"},
    {"cycle": 100, "component": "DMA", "op": "prefetch_B", "tile_id": "B[0,1]", "src": 0x2000, "dst": "L3_tile_1"},

    # BlockMover commands (L3 → L2)
    {"cycle": 110, "component": "BlockMover", "op": "move", "tile_id": "A[0,0]", "src": "L3_tile_2", "dst": "L2_bank_0"},
    {"cycle": 120, "component": "BlockMover", "op": "move", "tile_id": "B[0,0]", "src": "L3_tile_0", "dst": "L2_bank_1"},

    # Streamer commands (L2 → L1 → Fabric)
    {"cycle": 130, "component": "Streamer", "op": "stream_row", "tile_id": "A[0,0]", "src": "L2_bank_0", "dst": "L1_0"},
    {"cycle": 130, "component": "Streamer", "op": "stream_col", "tile_id": "B[0,0]", "src": "L2_bank_1", "dst": "L1_1"},

    # Compute (fabric handles this)
    {"cycle": 132, "component": "Fabric", "op": "accumulate", "result": "C[0,0]"}
  ]
```

  Priority Recommendations

  To build a functional system-level compiler, implement in this order:

  1. Computational Graph IR (highest priority)
    - Simple operator graph representation
    - Start with just GEMM, Conv2D, Elementwise ops
    - Data dependency edges
  2. Simple Bufferization
    - Naive tensor allocation (all tensors resident)
    - Later: lifetime analysis and reuse
  3. Schedule Generator (the core compiler)
    - Take operator graph + KPU config
    - Generate tiling decisions
    - Emit DMA/BlockMover/Streamer command sequences
    - Use your existing distributed_sequencer.py logic as a starting point
  4. Code Generator
    - Emit executable schedules for the simulator
    - Later: emit actual runtime code
  5. Framework Integration
    - ONNX importer
    - PyTorch/JAX integration


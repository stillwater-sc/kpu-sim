# Summary

The KPU compiler workflow infrastructure roadmap:

  1. Implementation Plan Document

  - Created /docs/compiler/KPU_COMPILER_IMPLEMENTATION_PLAN.md
    - Architecture overview with compilation pipeline diagrams
    - PTX-equivalent intermediate representation (DFX) design
    - Object file format specification
    - Tools directory reorganization
    - Implementation phases with concrete steps
    - Example DFX JSON output

  2. Tools Directory Reorganization

  New structure under tools/:
  - compiler/ - Compilation toolchain (kpu-kernel-compiler)
  - runtime/ - Loader/driver tools (kpu-loader)
  - analysis/ - Profiling tools (kpu-profiler, kpu-trace-analyzer)
  - development/ - Dev tools (kpu-assembler, kpu-debugger, kpu-disassembler)
  - configuration/ - Config tools (kpu-config)
  - benchmark/ - Benchmark suite

  3. DFX (Domain Flow Execution) - The PTX-Equivalent Layer

  Created two header files in include/sw/compiler/dfx/:

  dfx.hpp - Core DFX definitions:
  - DataType, MemoryLevel, DataflowStrategy enums
  - TensorDescriptor, TileSpec for data description
  - DataMoveOp, ComputeOp, BarrierOp operation types
  - TilingConfig, PerformanceHints, Program structures
  - Hardware-agnostic abstractions that work across KPU configurations

  object_file.hpp - Serialization format:
  - JSON-based serialization using nlohmann_json
  - Reader/writer functions for .kpu files
  - Version management and validation

  4. KPU Kernel Compiler (kpu-kernel-compiler)

  A fully functional compiler that:
  - Parses DFG files (.dfg via domain_flow, .json for testing)
  - Extracts MATMUL operations with tensor shapes
  - Uses existing TileOptimizer to compute optimal tile sizes
  - Generates complete DFX programs with:
    - Data movement operations (EXTERNAL→L3→L2→L1 chain)
    - Compute operations (MATMUL_TILE with accumulation)
    - Proper dependency tracking between operations
    - Performance hints (DRAM traffic, compute cycles, arithmetic intensity)

  Components:
  - dfg_parser.cpp/hpp - DFG file parsing and operator extraction
  - dfx_generator.cpp/hpp - DFX program generation with scheduling
  - object_writer.cpp/hpp - Object file output
  - main.cpp - CLI with options for dataflow strategy, tile strategy, verbosity

  5. KPU Loader (Skeleton)

  Created tools/runtime/kpu-loader/ with:
  - object_reader.cpp/hpp - Reads .kpu files back
  - schedule_binder.cpp/hpp - Binds abstract DFX ops to concrete resources
  - main.cpp - CLI skeleton

  Key Design Decisions

  Hardware-Agnostic Properties in DFX:
  | Specified in DFX             | Determined by Driver  |
  |------------------------------|-----------------------|
  | Tile dimensions (Ti, Tj, Tk) | Concrete addresses    |
  | Tile iteration order         | DMA engine assignment |
  | Data dependencies            | L2 bank allocation    |
  | Memory level hints           | Cycle timing          |
  | Dataflow strategy            | Prefetch distance     |

  Tested Output:
  $ ./kpu-kernel-compiler matmul.json -v
  DFX Generation Summary:
    Matrix: 1024x1024x512
    Tiles: 80x80x48
    Tile grid: 13x13x11
    Operations: 13689
    Data moves: 11830
    Computes: 1859
  Compiled matmul.json -> matmul.kpu

  The generated .kpu file contains properly formatted JSON with all operations, dependencies, tensors, and performance hints - ready for the loader to
  bind to specific micro-architecture resources.

DFX (Domain Flow Execution) serves the same architectural role as NVIDIA's PTX (Parallel Thread Execution). 

  PTX vs DFX Comparison

  | Aspect               | NVIDIA PTX                                                              | STILLWATER DFX
                                |
  |----------------------|-------------------------------------------------------------------------|-----------------------------------------------------
  ------------------------------|
  | Purpose              | Hardware-agnostic virtual ISA                                           | Hardware-agnostic operation schedule
                                |
  | Abstracts away       | Specific GPU architecture (SM count, register file size, shared memory) | Specific KPU configuration (L1/L2/L3 sizes, DMA
  count, systolic array dimensions) |
  | Compiled from        | CUDA C/C++                                                              | Domain Flow Graphs (.dfg)
                                |
  | Compiled to          | SASS (native GPU assembly) by driver                                    | Bound schedule with concrete addresses by
  loader/driver                           |
  | When binding happens | At runtime (JIT) or install time                                        | At load time
                                |
  | Portability          | Same PTX runs on different GPU generations                              | Same DFX runs on different KPU configurations
                                |

The Key Insight

Just as PTX lets NVIDIA:
  - Ship one binary that works across GPU generations
  - Let the driver optimize for the specific hardware at runtime
  - Hide micro-architectural details from the compiler

DFX lets the KPU:
  - Compile a DFG once, run on different KPU configurations
  - Let the loader/driver decide concrete resource allocation
  - Keep tiling decisions but defer address binding

What DFX Captures (Hardware-Agnostic)

  - Tile dimensions: Ti=80, Tj=80, Tk=48
  - Iteration order: for ti: for tj: for tk
  - Data movement: A[ti,tk] needs to go EXTERNAL→L3→L2→L1
  - Dependencies: Compute[7] depends on Load[3] and Load[6]
  - Dataflow strategy: output-stationary

What the Driver Decides (Hardware-Specific)

  - "L3" → L3 tile #2 at address 0x8000_0000
  - "L2" → L2 bank #5 at offset 0x4000
  - "Load A L3→L2" → BlockMover #1
  - "Load A L2→L1" → Streamer #0
  - Start cycle: 1024, End cycle: 1156

This separation is critical for supporting different KPU SKUs (e.g., a smaller edge device with 2 L3 tiles vs. a datacenter chip with 8 L3 tiles) with
the same compiled kernel.



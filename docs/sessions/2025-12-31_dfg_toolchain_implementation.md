# Session Log: DFG Toolchain Implementation

**Date:** 2025-12-31
**Duration:** ~2 hours
**Focus:** Standalone DFG toolchain for debugging and modular compilation

## Summary

Implemented a complete standalone CLI toolchain for Data Flow Graph (DFG) generation, scheduling, compilation, visualization, and analysis. This work was motivated by the need to isolate the DFG compiler stages for easier debugging after discovering a command priority ordering bug in the BlockMover compiler.

## Context

In a previous session, root cause analysis of an A[0,3] tile delivery order issue revealed that the `assemble_programs()` function in `block_mover_compiler.cpp` was incorrectly ordering commands - putting `PUSH_TO_L2` before `SEND` operations. This violated systolic ordering requirements where tiles must be injected into the NoC before local L2 operations.

The fix was straightforward (changing SEND priority from 3 to 1), but the debugging process highlighted the need for better observability into the compilation pipeline. The user requested: "Given the fact that the DFG compiler was part of the problem, can we isolate the DFG generation, compilation, and scheduling functionality into a standalone tool set?"

## Implementation

### Architecture

Created 5 standalone CLI tools that operate on JSON files:

```
kpu-dfg-gen → dfg.json → kpu-dfg-sched → scheduled.json → kpu-dfg-compile → programs.json
                ↓                ↓
           kpu-dfg-viz     kpu-dfg-analyze
           (DOT, Chrome Trace, Mermaid)
```

### Files Created

**Common Library** (`tools/dfg/common/`):
- `dfg_json.hpp/cpp` - TileDataFlowGraph JSON serialization
- `schedule_json.hpp/cpp` - DFGSchedule JSON serialization
- `compiled_json.hpp/cpp` - CompiledSchedule/BlockMoverProgram serialization

**Tools** (`tools/dfg/kpu-dfg-*/main.cpp`):
- `kpu-dfg-gen` - Generate DFG from templates (matmul implemented)
- `kpu-dfg-sched` - Schedule DFG using ASAP/ALAP/LIST algorithms
- `kpu-dfg-compile` - Compile to BlockMover programs
- `kpu-dfg-viz` - Export to DOT, Chrome Trace, Mermaid formats
- `kpu-dfg-analyze` - Statistics, critical path, validation

**Build System**:
- `tools/dfg/CMakeLists.txt` - Build configuration for all tools
- Updated `tools/CMakeLists.txt` to include dfg subdirectory

**Documentation**:
- `docs/dfg-toolchain.md` - Comprehensive usage documentation (431 lines)

### Technical Details

1. **JSON Format Design**
   - Each stage has a defined JSON schema
   - Scheduled JSON embeds the full DFG for self-containment
   - Programs JSON includes per-L3 BlockMover command sequences

2. **File Type Detection**
   - Tools auto-detect input file type (DFG vs Schedule vs Compiled)
   - Added validation to throw exceptions for mismatched file types
   - Prevents silent failures from type confusion

3. **Edge Reconstruction**
   - Fixed bug where edges weren't being reconstructed from JSON
   - Now properly calls `add_edge()` for each edge in the JSON

4. **Chrome Trace Export**
   - Outputs timeline-compatible JSON for Perfetto visualization
   - Each L3 tile appears as a separate thread (tid)
   - Events categorized by type (DMA, L3_Transfer, Compute)

## Bugs Fixed During Implementation

1. **Namespace Issues** - Added proper `using` declarations for types from `sw::kpu::dataflow`

2. **API Mismatches**:
   - `timing()` → `timing_model()`
   - `set_timing()` → `set_timing_model()`
   - `max_dma_concurrency` → `max_concurrent_dma`

3. **TensorId Enum** - Updated to use correct values (A, B, C, BIAS, WORKSPACE, CUSTOM)

4. **Edge Deserialization** - Added code to reconstruct edges from JSON edges array

5. **File Type Validation** - Added checks to throw exceptions when file doesn't match expected format

## Example Output

```bash
# Full pipeline
$ kpu-dfg-gen --template matmul -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json -v
Generated DFG:
  Nodes: 208
  Edges: 312
  Critical path: 253506 cycles

$ kpu-dfg-sched -i dfg.json -o scheduled.json --algorithm ASAP -v
Schedule generated:
  Makespan: 253506 cycles
  Scheduled nodes: 208

$ kpu-dfg-compile -i scheduled.json -o programs.json -v
Compilation complete:
  Total commands: 744
  L3 transfers: 96
  Compute ops: 64

$ kpu-dfg-analyze -i dfg.json --stats --critical-path
=== DFG Statistics ===
Node Types:
  DMA_LOAD       : 32
  DMA_STORE      : 16
  L3_TRANSFER    : 96
  MATMUL         : 64

Critical path: 253506 cycles
Avg parallelism: 17.0x
```

## Benefits

1. **Debugging**: Can inspect intermediate JSON at each stage
2. **Modularity**: Each tool can be developed/tested independently
3. **Visualization**: Chrome Trace enables Perfetto timeline analysis
4. **Reproducibility**: Save and replay exact configurations
5. **Validation**: Built-in validation at each stage

## Future Enhancements

- YAML spec parser for generic DFG definition
- Additional scheduling algorithms (Genetic, ILP-based)
- Binary .kpubin output format
- Schedule diff tool for comparing algorithms
- Integration with simulation runner

## Files Modified

| File | Change |
|------|--------|
| `tools/CMakeLists.txt` | Added `add_subdirectory(dfg)` |
| `src/dataflow/block_mover_compiler.cpp` | Fixed command priority (earlier session) |

## Test Results

All tools build and execute successfully:
- DFG generation: 208 nodes, 312 edges for 4x4x4 tiled matmul
- Scheduling: 253506 cycle makespan
- Compilation: 744 total commands across 16 L3 programs
- Visualization: DOT and Chrome Trace exports verified
- Analysis: Statistics and critical path analysis working

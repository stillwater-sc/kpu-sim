# DFG Toolchain

A standalone command-line toolchain for Data Flow Graph generation, scheduling, compilation, visualization, and analysis.

## Why

The DFG toolchain was created to address several needs:

1. **Debugging**: When issues arise in systolic dataflow execution (such as incorrect tile ordering), having isolated stages makes it easier to identify where problems originate - in the DFG structure, the schedule, or the compiled programs.

2. **Modularity**: Separating concerns allows each stage to be developed, tested, and optimized independently. The scheduler can be improved without touching the compiler, and vice versa.

3. **Inspection**: JSON interchange format between stages enables human-readable inspection of intermediate results. You can examine exactly what the scheduler produced before compilation.

4. **Visualization**: Chrome Trace and GraphViz exports enable visual debugging of schedules and data dependencies using familiar tools like Perfetto.

5. **Reproducibility**: File-based pipeline means you can save, share, and replay exact configurations for regression testing and issue reproduction.

## What

The toolchain consists of five CLI tools that operate on JSON files:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  kpu-dfg-gen    │───▶│  kpu-dfg-sched   │───▶│  kpu-dfg-compile │
│                 │    │                  │    │                  │
│  Generate DFG   │    │  Schedule nodes  │    │  Emit BlockMover │
│  from template  │    │  to cycles       │    │  programs        │
└─────────────────┘    └──────────────────┘    └──────────────────┘
        │                      │                       │
        ▼                      ▼                       ▼
    dfg.json            scheduled.json           programs.json
        │                      │
        └──────────┬───────────┘
                   ▼
         ┌─────────────────┐     ┌─────────────────┐
         │  kpu-dfg-viz    │     │ kpu-dfg-analyze │
         │                 │     │                 │
         │  Export to DOT, │     │  Stats, critical│
         │  Chrome Trace   │     │  path, validate │
         └─────────────────┘     └─────────────────┘
```

### Tool Summary

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `kpu-dfg-gen` | Template params | `dfg.json` | Generate DFG from templates |
| `kpu-dfg-sched` | `dfg.json` | `scheduled.json` | Schedule nodes to cycles |
| `kpu-dfg-compile` | `scheduled.json` | `programs.json` | Compile to BlockMover ISA |
| `kpu-dfg-viz` | Any JSON | `.dot`, `.json` | Visualization export |
| `kpu-dfg-analyze` | Any JSON | stdout | Statistics and validation |

## How

### kpu-dfg-gen - DFG Generator

Generates a TileDataFlowGraph from a template specification.

```
Usage: kpu-dfg-gen [options] -o output.json

Options:
  -o, --output FILE      Output JSON file (required)
  --template NAME        Template: matmul (default)
  --dataflow PATTERN     Dataflow: output-stationary (default),
                         weight-stationary, input-stationary
  -M SIZE                Matrix M dimension (default: 1024)
  -N SIZE                Matrix N dimension (default: 1024)
  -K SIZE                Matrix K dimension (default: 1024)
  --tiles MxNxK          Tiling factor (default: 4x4x4)
  --mesh ROWSxCOLS       L3 mesh dimensions (default: 4x4)
  -v, --verbose          Verbose output
  -h, --help             Show help
```

**Output format** (`dfg.json`):
```json
{
  "version": "1.0",
  "timing": {
    "mesh_rows": 4,
    "mesh_cols": 4,
    "dma_bandwidth_bytes_per_cycle": 64,
    "l3_bandwidth_bytes_per_cycle": 64
  },
  "nodes": [
    {
      "id": 0,
      "type": "DMA_LOAD",
      "name": "DMA_LOAD_A[0,0,0]",
      "l3_id": 0,
      "duration": 4096,
      "tile": {"tensor": "A", "m_tile": 0, "k_tile": 0, "size": 262144},
      "predecessors": [],
      "successors": [1, 32]
    }
  ],
  "edges": [
    {"from": 0, "to": 1, "type": "CONTROL"}
  ],
  "stats": {
    "num_nodes": 208,
    "num_edges": 312,
    "critical_path_length": 253506
  }
}
```

### kpu-dfg-sched - DFG Scheduler

Schedules DFG nodes to specific cycles using various algorithms.

```
Usage: kpu-dfg-sched -i input.json -o output.json [options]

Options:
  -i, --input FILE       Input DFG JSON file (required)
  -o, --output FILE      Output scheduled JSON file (required)
  --algorithm ALG        ASAP (default), ALAP, LIST, CRITICAL_PATH
  --l3-concurrency N     Max concurrent ops per L3 (default: 2)
  --dma-channels N       Number of DMA channels (default: 8)
  --validate             Validate schedule after generation
  -v, --verbose          Verbose output
  -h, --help             Show help
```

**Algorithms**:
- `ASAP` - As Soon As Possible: Schedule each node at earliest valid cycle
- `ALAP` - As Late As Possible: Schedule nodes as late as possible
- `LIST` - List scheduling with priority queue
- `CRITICAL_PATH` - Prioritize critical path nodes

**Output format** (`scheduled.json`):
```json
{
  "version": "1.0",
  "makespan": 253506,
  "algorithm": "ASAP",
  "schedule": [
    {"node_id": 0, "start_cycle": 0, "end_cycle": 4096, "resource_id": 0}
  ],
  "per_l3_schedule": {
    "0": [0, 1, 2, 16, 32],
    "1": [17, 33, 57]
  },
  "dfg": { /* embedded DFG */ }
}
```

### kpu-dfg-compile - BlockMover Compiler

Compiles a scheduled DFG to BlockMover programs for each L3 tile.

```
Usage: kpu-dfg-compile -i scheduled.json -o programs.json [options]

Options:
  -i, --input FILE       Input scheduled JSON file (required)
  -o, --output FILE      Output programs JSON file (required)
  --no-waits             Don't emit WAIT_UNTIL_CYCLE commands
  --sync-sends           Add WAIT_DELIVERY after sends
  --no-barriers          Don't emit BARRIER commands
  --mesh ROWSxCOLS       Override mesh dimensions
  -v, --verbose          Verbose output
  -h, --help             Show help
```

**Output format** (`programs.json`):
```json
{
  "version": "1.0",
  "estimated_cycles": 253506,
  "compute_cycles": 3728320,
  "data_movement_cycles": 590016,
  "stats": {
    "total_commands": 744,
    "total_dma_ops": 48,
    "total_l3_transfers": 96,
    "total_l2_transfers": 128,
    "total_compute_ops": 64,
    "total_barriers": 16
  },
  "programs": [
    {
      "l3_id": 0,
      "num_commands": 49,
      "commands": [
        {"op": "TRACE_MARKER", "trace_id": 0, "tile": {...}},
        {"op": "SEND_EAST", "tile": {...}},
        {"op": "PUSH_TO_L2", "tile": {...}, "l2_bank_id": 0}
      ]
    }
  ]
}
```

### kpu-dfg-viz - Visualization Export

Exports DFG or schedule to visualization formats.

```
Usage: kpu-dfg-viz -i input.json -o output [options]

Options:
  -i, --input FILE       Input JSON file (DFG or schedule)
  -o, --output FILE      Output file
  --format FORMAT        dot (default), chrome-trace, mermaid
  -v, --verbose          Verbose output
  -h, --help             Show help
```

**Formats**:

1. **DOT** (GraphViz) - Graph structure visualization
   ```bash
   kpu-dfg-viz -i dfg.json -o graph.dot --format dot
   dot -Tpng graph.dot -o graph.png  # Render with GraphViz
   ```

2. **Chrome Trace** - Timeline visualization for Perfetto
   ```bash
   kpu-dfg-viz -i scheduled.json -o timeline.json --format chrome-trace
   # Open in chrome://tracing or https://ui.perfetto.dev
   ```

3. **Mermaid** - Markdown-compatible diagrams
   ```bash
   kpu-dfg-viz -i dfg.json -o diagram.md --format mermaid
   ```

### kpu-dfg-analyze - Analysis Tool

Analyzes and validates DFG, schedule, or compiled programs.

```
Usage: kpu-dfg-analyze -i input.json [options]

Options:
  -i, --input FILE       Input JSON file (required)
  --stats                Show statistics (default if no option specified)
  --critical-path        Show critical path analysis
  --utilization          Show per-L3 utilization
  --validate             Validate schedule/programs
  --check-order          Check systolic ordering
  -v, --verbose          Verbose output
  -h, --help             Show help
```

**Output** (for DFG):
```
=== DFG Statistics ===

Graph Structure:
  Total nodes: 208
  Total edges: 312
  Is acyclic: yes

Node Types:
  DMA_LOAD       : 32
  DMA_STORE      : 16
  L3_TRANSFER    : 96
  MATMUL         : 64

Timing:
  Critical path: 253506 cycles
  Total work: 4318336 cycles
  Avg parallelism: 17.0x

=== Critical Path Analysis ===

Critical path length: 253506 cycles
Critical path nodes: 9

Path:
  [4096] DMA_LOAD_A[0,0,0] (4096 cycles)
  [8194] L3_XFER_A[0,0,0] (4098 cycles)
  ...
  [253506] DMA_STORE_C[0,3,0] (4096 cycles)
```

## Examples

### Basic Matmul Pipeline

```bash
# 1. Generate a 1024x1024 matmul DFG with 4x4x4 tiling on a 4x4 mesh
kpu-dfg-gen --template matmul \
    -M 1024 -N 1024 -K 1024 \
    --tiles 4x4x4 \
    --mesh 4x4 \
    -o matmul.dfg.json -v

# 2. Schedule using ASAP algorithm
kpu-dfg-sched -i matmul.dfg.json \
    -o matmul.sched.json \
    --algorithm ASAP \
    --validate -v

# 3. Compile to BlockMover programs
kpu-dfg-compile -i matmul.sched.json \
    -o matmul.prog.json -v

# 4. Analyze the result
kpu-dfg-analyze -i matmul.prog.json --stats
```

### Debugging a Schedule

```bash
# Generate and schedule
kpu-dfg-gen -M 512 -N 512 -K 512 --tiles 2x2x2 -o small.dfg.json
kpu-dfg-sched -i small.dfg.json -o small.sched.json --algorithm LIST

# Visualize the schedule timeline
kpu-dfg-viz -i small.sched.json -o timeline.json --format chrome-trace

# Open in Perfetto for visual debugging
# Navigate to https://ui.perfetto.dev and load timeline.json
```

### Comparing Scheduling Algorithms

```bash
# Generate DFG once
kpu-dfg-gen -M 2048 -N 2048 -K 2048 --tiles 8x8x8 -o large.dfg.json

# Try different schedulers
for algo in ASAP ALAP LIST; do
    kpu-dfg-sched -i large.dfg.json -o sched_${algo}.json --algorithm $algo
    echo "=== $algo ==="
    kpu-dfg-analyze -i sched_${algo}.json --stats | grep Makespan
done
```

### Generating Graph Visualization

```bash
# Create DOT graph
kpu-dfg-viz -i matmul.dfg.json -o matmul.dot --format dot

# Render with GraphViz (if installed)
dot -Tsvg matmul.dot -o matmul.svg
dot -Tpng matmul.dot -o matmul.png

# Or use online GraphViz viewer
# Copy matmul.dot content to https://dreampuf.github.io/GraphvizOnline/
```

### Validating Critical Path

```bash
# Generate and analyze critical path
kpu-dfg-gen -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json
kpu-dfg-analyze -i dfg.json --critical-path

# The critical path shows the bottleneck:
# - If dominated by L3_XFER: network-bound
# - If dominated by MATMUL: compute-bound
# - If dominated by DMA: memory-bound
```

### Pipeline with Validation

```bash
#!/bin/bash
set -e  # Exit on error

# Full pipeline with validation at each stage
kpu-dfg-gen -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json -v

# Validate DFG is acyclic
kpu-dfg-analyze -i dfg.json --validate

# Schedule with validation
kpu-dfg-sched -i dfg.json -o sched.json --validate -v

# Compile
kpu-dfg-compile -i sched.json -o prog.json -v

# Final analysis
kpu-dfg-analyze -i prog.json --stats

echo "Pipeline completed successfully"
```

## File Locations

After building, the tools are located at:
```
build/tools/dfg/kpu-dfg-gen
build/tools/dfg/kpu-dfg-sched
build/tools/dfg/kpu-dfg-compile
build/tools/dfg/kpu-dfg-viz
build/tools/dfg/kpu-dfg-analyze
```

Source code:
```
tools/dfg/
├── CMakeLists.txt
├── common/
│   ├── dfg_json.hpp/cpp        # DFG serialization
│   ├── schedule_json.hpp/cpp   # Schedule serialization
│   └── compiled_json.hpp/cpp   # Compiled programs serialization
├── kpu-dfg-gen/main.cpp
├── kpu-dfg-sched/main.cpp
├── kpu-dfg-compile/main.cpp
├── kpu-dfg-viz/main.cpp
└── kpu-dfg-analyze/main.cpp
```

## Troubleshooting

### "Not a valid schedule JSON" error
The input file type is auto-detected. If you get this error, ensure you're using the correct file type for the tool. For example, `kpu-dfg-compile` requires a scheduled JSON (output from `kpu-dfg-sched`), not a raw DFG.

### Empty visualization output
Ensure the DFG file was correctly generated with nodes and edges. Check with:
```bash
kpu-dfg-analyze -i dfg.json --stats
```

### Schedule validation fails
This usually indicates a dependency cycle or impossible schedule. Check:
```bash
kpu-dfg-analyze -i dfg.json --validate
```

### Chrome Trace not loading
Ensure the JSON is valid and the file extension is `.json`. Chrome Trace requires the schedule (not plain DFG) for timeline visualization.

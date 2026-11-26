# Trace Output Directory

Generated trace files from memory controller patterns and visualization tools.

## Directory Structure

```
traces/
├── README.md                       # This file
└── memory/
    ├── lpddr5/                     # LPDDR5 memory controller records (.mflow)
    │   ├── tools/                  # Visualization tools
    │   │   └── blockdiagram.html   # Animated block diagram
    │   ├── inventory.md            # The pattern records, by level
    │   ├── single-bank/            # Level 1: page_hits, page_conflicts, mixed_rw
    │   ├── two-bank/               # Level 2: same_group, diff_groups
    │   ├── three-bank/             # Level 3: same_group, mixed_groups
    │   ├── four-bank/              # Level 4: full_group, across_groups, page_hit_burst
    │   ├── dual-channel/           # Level 5: independent, interleaved
    │   ├── complex/                # Level 6: stream, tile_load, multi_dma, random, strided
    │   └── bandwidth/              # Level 7: page_burst, max_bandwidth
    │                               #   each a <name>.mflow record, generated (not committed)
    │
    └── gddr6/                      # GDDR6 memory controller traces
        ├── tools/                  # Visualization tools (shared with LPDDR5)
        ├── single-bank/            # Level 1: Single bank patterns
        │   ├── page_hits_trace.json
        │   ├── page_conflicts_trace.json
        │   └── mixed_rw_trace.json
        ├── two-bank/               # Level 2: Two bank patterns
        │   ├── same_group_trace.json
        │   └── diff_groups_trace.json
        ├── three-bank/             # Level 3: Three bank patterns
        │   ├── same_group_trace.json
        │   └── mixed_groups_trace.json
        ├── four-bank/              # Level 4: Four bank patterns
        │   ├── full_group_trace.json
        │   ├── across_groups_trace.json
        │   └── page_hit_burst_trace.json
        ├── dual-channel/           # Level 5: Dual channel patterns
        │   ├── independent_trace.json
        │   └── interleaved_trace.json
        ├── complex/                # Level 6: Complex patterns
        │   ├── stream_trace.json       # STREAM benchmark
        │   ├── tile_load_trace.json    # ML tile loading
        │   ├── multi_dma_trace.json    # Multi-DMA (16 banks)
        │   ├── random_trace.json       # Random access
        │   └── strided_trace.json      # Strided access
        └── bandwidth/              # Level 7: Bandwidth patterns
            ├── page_burst_trace.json        # Max page hits
            ├── max_bandwidth_trace.json     # Peak bandwidth (16 banks)
            └── eight_bank_bandwidth_trace.json  # 8-bank subset
```

## Memory Technologies

### LPDDR5-6400
- **Clock**: 3.2 GHz
- **Banks**: 8 per channel (2 bank groups × 4 banks)
- **Channels**: 1-2
- **Peak Bandwidth**: 25.6 GB/s (dual channel)

### GDDR6-16000
- **Clock**: 2.0 GHz
- **Banks**: 16 per channel (4 bank groups × 4 banks)
- **Channels**: 2
- **Peak Bandwidth**: 64 GB/s

## Quick Start

```bash
# Generate LPDDR5 records (.mflow bundles; under ctest they go to build/patterns/memory/lpddr5/)
./build/patterns/memory/lpddr5/lpddr5_page_hits
./build/patterns/memory/lpddr5/lpddr5_stream
./build/patterns/memory/lpddr5/lpddr5_multi_dma

# Generate GDDR6 traces
./build/patterns/memory/gddr6/gddr6_page_hits
./build/patterns/memory/gddr6/gddr6_stream
./build/patterns/memory/gddr6/gddr6_multi_dma

# Check a record, and see what the run did
python3 tools/trace/mflow_check.py traces/memory/lpddr5/complex/stream.mflow --summary

# Serve the traces to a browser (from a laptop: ssh -L 8080:localhost:8080 <server>)
python3 tools/debug-server/kpu_debug_server.py --root traces/memory
```

## Pattern Categories

### Level 1: Single Bank
Basic single-bank access patterns for latency characterization.
- `page_hits` - Sequential same-row access (best case)
- `page_conflicts` - Alternating different rows (worst case)
- `mixed_rw` - Interleaved read/write operations

### Level 2: Two Banks
Bank-level parallelism with two concurrent banks.
- `same_group` - Both banks in same bank group (tRRD_L, tCCD_L apply)
- `diff_groups` - Banks in different groups (tRRD_S, tCCD_S apply)

### Level 3: Three Banks
Increased parallelism with three active banks.
- `same_group` - All three in one bank group
- `mixed_groups` - Banks distributed across groups

### Level 4: Four Banks
Full bank group utilization.
- `full_group` - Complete bank group (4 banks)
- `across_groups` - One bank per group (LPDDR5: 2 groups, GDDR6: 4 groups)
- `page_hit_burst` - Sustained page hits across 4 banks

### Level 5: Dual Channel
Multi-channel configurations.
- `independent` - Separate access per channel
- `interleaved` - Alternating channel access

### Level 6: Complex Patterns
Real-world workload simulations.
- `stream` - STREAM benchmark (Copy, Scale, Add, Triad)
- `tile_load` - ML accelerator tile loading patterns
- `multi_dma` - Concurrent DMA engine simulation
- `random` - Random access stress test
- `strided` - Regular strided access patterns

### Level 7: Bandwidth Patterns
Maximum throughput characterization.
- `page_burst` - Full page utilization (128 cache lines/page)
- `max_bandwidth` - Peak achievable bandwidth
- `eight_bank_bandwidth` - GDDR6 8-bank subset comparison

## Visualization Tools

### Web-Based Viewers

The DRAM views are moving to `tools/visualization/dram/`, rebuilt on `.mflow` records
(`docs/plans/cycle-mflow-and-debug-server.md` step 9). The timing diagram has moved:
`tools/visualization/dram/timing.html`, the bank analyzer `tools/visualization/dram/bank_analyzer.html`, and the request swimlanes
`tools/visualization/dram/swimlane.html`.

Located in `traces/memory/{lpddr5,gddr6}/tools/`:

| Tool | Description | Best For |
|------|-------------|----------|
| `blockdiagram.html` | Animated architecture | Data flow visualization |

### CLI Tools

```bash
# Check an LPDDR5 record against the invariants (exit 0 pass, 1 violations, 2 unreadable)
python3 tools/trace/mflow_check.py record.mflow

# ... and summarize the run: throughput, page outcomes, each bank's commands, burst latency
python3 tools/trace/mflow_check.py record.mflow --summary

# JSON output for scripting
python3 tools/trace/mflow_check.py record.mflow --summary --json

# Serve the traces to a browser (tools/debug-server/README.md)
python3 tools/debug-server/kpu_debug_server.py --root traces
```

### Perfetto (Advanced)

For detailed timeline analysis of a Chrome trace (the GDDR6 and other devices' traces, the
cycle core's component trace; the debug server has an Open in Perfetto button for each):
1. Go to https://ui.perfetto.dev
2. Drag and drop the trace JSON file
3. Explore with full zoom/pan/search capabilities

## Keyboard Controls (Web Viewers)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Right Arrow | Step forward |
| Left Arrow | Step backward |
| Home | Go to start |
| End | Go to end |

## Trace File Format

The LPDDR5 patterns write `.mflow` records (`tools/visualization/common/mflow.js` reads them;
`docs/plans/cycle-mflow-and-debug-server.md` describes them). The other devices' patterns write
Chrome Trace Event Format (JSON):

```json
[
  {"name": "process_name", "ph": "M", "pid": 22, "args": {"name": "GDDR6_BANK"}},
  {"name": "thread_name", "ph": "M", "pid": 22, "tid": 0, "args": {"name": "GDDR6_BANK #0"}},
  {"name": "ACTIVATE", "cat": "GDDR6_BANK", "ph": "X",
   "ts": 23000.0, "dur": 9000.0, "pid": 22, "tid": 0,
   "args": {"txn_id": 1, "cycle_issue": 46, "cycle_complete": 64}}
]
```

### Event Types

| Type | Description |
|------|-------------|
| `ACTIVATE` | Row activation (ACT command) |
| `BURST_READ` | Read data burst |
| `BURST_WRITE` | Write data burst |
| `PRECHARGE` | Row precharge (close page) |
| `REFRESH` | Bank refresh operation |

## See Also

- [Memory Characterization](../docs/memory-characterization.md) - Latency and bandwidth analysis
- [Pattern Source](../patterns/memory/) - Pattern implementation code
- [LPDDR5 Invariants](../patterns/memory/lpddr5/INVARIANTS.md) - each invariant and the `mflow_check.py` check that enforces it

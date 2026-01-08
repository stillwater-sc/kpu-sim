# Trace Output Directory

Generated trace files from memory controller patterns and visualization tools.

## Directory Structure

```
traces/
├── README.md                       # This file
└── memory/
    ├── lpddr5/                     # LPDDR5 memory controller traces
    │   ├── tools/                  # Visualization tools
    │   │   ├── swimlane.html       # Swimlane pipeline view
    │   │   ├── timing.html         # DRAM timing diagram view
    │   │   ├── blockdiagram.html   # Animated block diagram
    │   │   └── bank_analyzer.html  # Bank state analyzer
    │   ├── single-bank/            # Level 1: Single bank patterns
    │   │   ├── page_hits_trace.json
    │   │   ├── page_conflicts_trace.json
    │   │   └── mixed_rw_trace.json
    │   ├── two-bank/               # Level 2: Two bank patterns
    │   │   ├── same_group_trace.json
    │   │   └── diff_groups_trace.json
    │   ├── three-bank/             # Level 3: Three bank patterns
    │   │   ├── same_group_trace.json
    │   │   └── mixed_groups_trace.json
    │   ├── four-bank/              # Level 4: Four bank patterns
    │   │   ├── full_group_trace.json
    │   │   ├── across_groups_trace.json
    │   │   └── page_hit_burst_trace.json
    │   ├── dual-channel/           # Level 5: Dual channel patterns
    │   │   ├── independent_trace.json
    │   │   └── interleaved_trace.json
    │   ├── complex/                # Level 6: Complex patterns
    │   │   ├── stream_trace.json       # STREAM benchmark
    │   │   ├── tile_load_trace.json    # ML tile loading
    │   │   ├── multi_dma_trace.json    # Multi-DMA engines
    │   │   ├── random_trace.json       # Random access
    │   │   └── strided_trace.json      # Strided access
    │   └── bandwidth/              # Level 7: Bandwidth patterns
    │       ├── page_burst_trace.json   # Max page hits (128/page)
    │       └── max_bandwidth_trace.json # Peak bandwidth
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
# Generate LPDDR5 traces
./build/patterns/memory/lpddr5/lpddr5_page_hits
./build/patterns/memory/lpddr5/lpddr5_stream
./build/patterns/memory/lpddr5/lpddr5_multi_dma

# Generate GDDR6 traces
./build/patterns/memory/gddr6/gddr6_page_hits
./build/patterns/memory/gddr6/gddr6_stream
./build/patterns/memory/gddr6/gddr6_multi_dma

# View summary (CLI)
./build/tools/trace/kpu-trace-summary traces/memory/lpddr5/complex/stream_trace.json

# Launch web viewer
./tools/trace/serve-trace.py traces/memory/gddr6/complex/multi_dma_trace.json
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

Located in `traces/memory/{lpddr5,gddr6}/tools/`:

| Tool | Description | Best For |
|------|-------------|----------|
| `swimlane.html` | Horizontal pipeline view | Request parallelism |
| `timing.html` | DRAM waveform diagram | Timing validation |
| `blockdiagram.html` | Animated architecture | Data flow visualization |
| `bank_analyzer.html` | Per-bank lanes | Resource utilization |

### CLI Tools

```bash
# Basic summary
./build/tools/trace/kpu-trace-summary trace.json

# Verbose with per-transaction details
./build/tools/trace/kpu-trace-summary trace.json --verbose

# JSON output for scripting
./build/tools/trace/kpu-trace-summary trace.json --json

# Validate timing constraints
./build/tools/trace/kpu-trace-summary trace.json --validate

# Launch local web server
./tools/trace/serve-trace.py trace.json
```

### Perfetto (Advanced)

For detailed timeline analysis:
1. Go to https://ui.perfetto.dev
2. Drag and drop any trace JSON file
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
- [LPDDR5 Invariants](../patterns/memory/lpddr5/INVARIANTS.md) - Timing constraint validation

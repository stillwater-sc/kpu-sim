# Trace Output Directory

Generated trace files from memory controller patterns and visualization tools.

## Directory Structure

```
traces/
├── README.md                    # This file
├── memory/
│   └── lpddr5/
│       ├── tools/               # LPDDR5 visualization tools
│       │   ├── swimlane.html    # Swimlane pipeline view
│       │   ├── timing.html      # DRAM timing diagram view
│       │   ├── blockdiagram.html # Animated block diagram
│       │   └── bank_analyzer.html # Bank state analyzer
│       ├── single-bank/         # Single bank pattern traces
│       ├── two-bank/            # Two bank pattern traces
│       ├── three-bank/          # Three bank pattern traces
│       ├── four-bank/           # Four bank pattern traces
│       ├── dual-channel/        # Dual channel pattern traces
│       └── complex/             # Complex pattern traces
```

## Quick Start

```bash
# Generate a trace
./build/patterns/memory/lpddr5/lpddr5_page_hits

# View summary (CLI - works everywhere)
./build/tools/trace/kpu-trace-summary traces/memory/lpddr5/single-bank/page_hits_trace.json

# Launch web viewer (when GUI available)
./tools/trace/serve-trace.py traces/memory/lpddr5/single-bank/page_hits_trace.json
```

## Visualization Tools

Located in `traces/memory/lpddr5/tools/`:

### 1. Swimlane View (`swimlane.html`)

Horizontal pipeline visualization showing requests flowing through stages:

```
Request Queue → MC Scheduler → Cmd Bus → Bank (ACT) → Bank (CAS) → Data Bus → Return
```

**Best for:**
- Understanding request parallelism
- Seeing resource contention
- Tracking multiple concurrent requests
- Timeline-based analysis

### 2. Timing Diagram (`timing.html`)

Classic waveform-style diagram like DRAM datasheets:

```
CLK:      _|‾|_|‾|_|‾|_|‾|_
CMD:      ────┤ACT├───┤RD├──
Bank 0:   IDLE│ACTIVATING│ACTIVE│READING│
DQ:       ────────────────────┤BURST├──
                  ├──tRCD──┤├─tCL─┤
```

**Best for:**
- Validating timing constraints (tRCD, tCL, tRP, etc.)
- Understanding command sequences
- Comparing page hit vs page miss latencies

### 3. Block Diagram Animation (`blockdiagram.html`)

Animated architectural view with request tokens flowing through components.

**Best for:**
- Understanding physical architecture
- Visualizing data flow paths
- Seeing bank state changes

### 4. Bank Analyzer (`bank_analyzer.html`)

Per-bank horizontal lanes with command/data bus visualization.

**Best for:**
- Detailed bank-level analysis
- Resource utilization over time
- Command scheduling analysis

## Keyboard Controls

All viewers support:
- **Space**: Play/Pause
- **Arrow Right**: Step forward
- **Arrow Left**: Step backward
- **Home**: Go to start
- **End**: Go to end

## CLI Tools

### kpu-trace-summary

Command-line trace analysis (works on headless servers):

```bash
# Basic summary
./build/tools/trace/kpu-trace-summary trace.json

# Verbose output with per-transaction details
./build/tools/trace/kpu-trace-summary trace.json --verbose

# JSON output for scripting
./build/tools/trace/kpu-trace-summary trace.json --json

# Validate timing constraints
./build/tools/trace/kpu-trace-summary trace.json --validate
```

### serve-trace.py

Local HTTP server for HTML visualization:

```bash
# Serve and open browser
./tools/trace/serve-trace.py trace.json

# Specify port
./tools/trace/serve-trace.py trace.json --port 8888

# Server only (no browser)
./tools/trace/serve-trace.py trace.json --no-browser
```

## Generating Traces

```bash
# Build and run a pattern (generates trace automatically)
./build/patterns/memory/lpddr5/lpddr5_page_hits

# Trace written to: traces/memory/lpddr5/single-bank/page_hits_trace.json

# Custom trace path
./build/patterns/memory/lpddr5/lpddr5_page_hits --trace /tmp/custom.json

# Skip trace generation
./build/patterns/memory/lpddr5/lpddr5_page_hits --no-trace
```

## Perfetto (Advanced Analysis)

For detailed analysis, use Perfetto:

1. Go to https://ui.perfetto.dev
2. Drag and drop any trace JSON file
3. Explore the full timeline visualization

## Trace File Format

Traces use Chrome Trace Event Format (JSON):

```json
[
  {"name": "process_name", "ph": "M", "pid": 16, "args": {"name": "LPDDR5_BANK"}},
  {"name": "ACTIVATE", "cat": "LPDDR5_BANK", "ph": "X",
   "ts": 13125.0, "dur": 4375.0, "pid": 16, "tid": 0,
   "args": {"txn_id": 0, "cycle_issue": 42, "cycle_complete": 56}}
]
```

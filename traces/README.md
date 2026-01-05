# Trace Output Directory

Generated trace files from memory controller patterns and visualization tools.

## Directory Structure

```
traces/
├── lpddr5_swimlane.html    # Swimlane pipeline view
├── lpddr5_timing.html      # DRAM timing diagram view
├── lpddr5_blockdiagram.html # Animated block diagram
├── memory/
│   └── lpddr5/
│       ├── single-bank/    # Single bank pattern traces
│       ├── two-bank/       # Two bank pattern traces
│       ├── four-bank/      # Four bank pattern traces
│       └── complex/        # Complex pattern traces
```

## Visualization Tools

Three complementary viewers for understanding memory request flow:

### 1. Swimlane View (`lpddr5_swimlane.html`)

Horizontal pipeline visualization showing requests flowing through stages:

```
Request Queue → MC Scheduler → Cmd Bus → Bank (ACT) → Bank (CAS) → Data Bus → Return
```

**Best for:**
- Understanding request parallelism
- Seeing resource contention
- Tracking multiple concurrent requests
- Timeline-based analysis

**Features:**
- Horizontal lanes for each pipeline stage
- Color-coded request tokens
- Playback with cycle-by-cycle stepping
- Request filtering and highlighting
- Statistics panel

### 2. Timing Diagram (`lpddr5_timing.html`)

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
- Educational purposes

**Features:**
- Signal waveforms (CLK, CMD, Bank State, DQ)
- Timing constraint annotations with arrows
- Per-request or all-requests view
- Dark/light theme toggle

### 3. Block Diagram Animation (`lpddr5_blockdiagram.html`)

Animated architectural view with request tokens flowing through components:

```
┌──────────┐    ┌────────────────┐    ┌─────────────────────┐
│ Request  │───▶│ Memory         │───▶│ LPDDR5 Device       │
│ Queue    │    │ Controller     │    │ Banks, Sense Amps   │
└──────────┘    └────────────────┘    └─────────────────────┘
                        ▲                       │
                        └───────────────────────┘
                              Data Bus
```

**Best for:**
- Understanding physical architecture
- Visualizing data flow paths
- Educational/demo purposes
- Seeing bank state changes

**Features:**
- Animated request tokens
- Bank state coloring (idle/activating/reading/writing)
- Bus activity indicators
- Real-time statistics

## Keyboard Controls

All viewers support:
- **Space**: Play/Pause
- **Arrow Right**: Step forward
- **Arrow Left**: Step backward
- **Home**: Go to start
- **End**: Go to end (swimlane/timing)

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

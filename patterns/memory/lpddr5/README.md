# LPDDR5 Memory Controller Patterns

Multi-fidelity validation patterns for the LPDDR5 memory controller.

## Overview

These patterns validate LPDDR5 timing behavior through progressively complex memory access scenarios. Each pattern supports three simulation fidelity levels:

| Fidelity | Speed | Use Case |
|----------|-------|----------|
| **BEHAVIORAL** | ~100-1000x | Functional testing, CI/CD |
| **TRANSACTIONAL** | ~10-100x | Architecture exploration |
| **CYCLE_ACCURATE** | 1x (ref) | Timing validation, visualization |

## Pattern Organization

```
lpddr5/
├── common/                 # Shared infrastructure
│   ├── lpddr5_configs.hpp  # Timing configurations
│   ├── lpddr5_harness.hpp  # Test harness
│   ├── workloads.hpp       # Workload definitions
│   └── multi_fidelity.hpp  # Fidelity comparison framework
│
├── single-bank/            # Level 1: Single bank fundamentals
│   ├── page-hits/          # Sequential same-row reads
│   ├── page-conflicts/     # Different-row reads
│   └── mixed-rw/           # Read/write turnaround
│
├── two-bank/               # Level 2: Two bank operations
│   ├── same-group/         # tRRD_L constraint
│   └── diff-groups/        # tRRD_S (faster)
│
├── three-bank/             # Level 3: Three bank operations
├── four-bank/              # Level 4: Four bank + tFAW
├── dual-channel/           # Level 5: Multi-channel
└── complex/                # Level 6: Real-world patterns
```

## Key Timing Parameters (LPDDR5-6400 @ 3200 MHz)

| Parameter | Cycles | Description |
|-----------|--------|-------------|
| tRCD | 14 | Row address to column address delay |
| tRP | 14 | Row precharge time |
| tCL | 14 | CAS read latency |
| tRRD_L | 6 | ACT-to-ACT (same bank group) |
| tRRD_S | 4 | ACT-to-ACT (different bank group) |
| tFAW | 24 | Four activate window |
| tRTW | 14 | Read-to-write turnaround |
| tWTR_L | 10 | Write-to-read (same bank group) |

## Expected Access Latencies

| Scenario | Latency (cycles) | Formula |
|----------|------------------|---------|
| Page hit read | 22 | tCL + tBurst |
| Page empty read | 36 | tRCD + tCL + tBurst |
| Page conflict read | 50 | tRP + tRCD + tCL + tBurst |

## Running Patterns

```bash
# Build all patterns
cmake --preset release && cmake --build --preset release

# Run individual pattern
./build/patterns/memory/lpddr5/single-bank/page-hits

# With multi-fidelity comparison
./build/patterns/memory/lpddr5/single-bank/page-hits --fidelity

# Export trace for visualization
./build/patterns/memory/lpddr5/single-bank/page-hits --trace output.json
```

## Visualization

Traces export to Chrome Trace format for Perfetto:

1. Run pattern with `--trace` option
2. Open https://ui.perfetto.dev
3. Drag and drop the JSON file

The trace shows:
- Bank state timeline (IDLE, ACTIVATING, ACTIVE, READING, etc.)
- Data bus activity (READ_BURST, WRITE_BURST, TURNAROUND)
- Command bus activity (ACT, RD, WR, PRE)

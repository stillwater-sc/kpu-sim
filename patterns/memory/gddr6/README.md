# GDDR6 Memory Controller Patterns

Validation patterns for the cycle-accurate GDDR6 memory controller implementation.

## Overview

The GDDR6 memory controller implements the JEDEC JESD250 specification for Graphics DDR6 SGRAM.
It provides cycle-accurate simulation of GDDR6 timing constraints for calibrating transactional
and behavioral memory controller models.

## Architecture

### GDDR6 Key Characteristics
- **16 Banks** organized into **4 Bank Groups** (4 banks per group)
- **Dual 16-bit Channels** (x16 mode, two independent channels per chip)
- **16n Prefetch** architecture with Burst Length 16
- **Speed Grades**: 12-24 Gbps (configurations for 14, 16, 18, 20 Gbps)

### Clock Domains
- **CK (Command Clock)**: data_rate / 8 (e.g., 2.0 GHz for 16 Gbps)
- **WCK (Write Clock)**: data_rate / 2 (e.g., 8.0 GHz for 16 Gbps)

## Directory Structure

```
patterns/memory/gddr6/
├── README.md                    # This file
├── INVARIANTS.md               # Timing constraint documentation
├── common/
│   ├── gddr6_configs.hpp       # Standard configurations
│   ├── gddr6_harness.hpp       # Test harness
│   └── trace_validator.py      # Trace validation tool
└── single-bank/
    └── page_hits.cpp           # Page hit pattern
```

## Timing Parameters (GDDR6-16000)

| Parameter | Value (CK) | Description |
|-----------|------------|-------------|
| tRCDRD    | 18         | Row address to column address (read) |
| tRCDWR    | 18         | Row address to column address (write) |
| tRP       | 18         | Row precharge time |
| tRAS      | 28         | Row active time |
| tRC       | 46         | Row cycle time (tRAS + tRP) |
| tRL       | 20         | CAS read latency |
| tWL       | 8          | CAS write latency |
| tRRD_L/S  | 4          | ACT to ACT (same/diff bank group) |
| tCCD_L    | 3          | CAS to CAS (same bank group) |
| tCCD_S    | 2          | CAS to CAS (diff bank group) |
| tFAW      | 16         | Four activate window |

## Building

```bash
cmake --preset release
cmake --build --preset release
```

## Running Patterns

```bash
# Run page hits pattern
./build/patterns/memory/gddr6/gddr6_page_hits
```

## Trace Validation

```bash
# Validate generated traces
python3 patterns/memory/gddr6/common/trace_validator.py \
    traces/memory/gddr6/single-bank/page_hits_trace.json
```

## Visualization

Traces are exported in Chrome Trace JSON format:
1. Generate traces by running patterns
2. Open https://ui.perfetto.dev
3. Load the generated JSON file

## Speed Grade Configurations

| Speed Grade | CK Freq | WCK Freq | Peak BW/Device |
|-------------|---------|----------|----------------|
| GDDR6-14000 | 1.75 GHz | 7.0 GHz | 56 GB/s |
| GDDR6-16000 | 2.0 GHz  | 8.0 GHz | 64 GB/s |
| GDDR6-18000 | 2.25 GHz | 9.0 GHz | 72 GB/s |
| GDDR6-20000 | 2.5 GHz  | 10.0 GHz | 80 GB/s |

## Integration with Multi-Fidelity Framework

The cycle-accurate GDDR6 controller is used to calibrate:
1. **Transactional model**: Statistical timing based on page hit/empty/conflict rates
2. **Behavioral model**: Fixed latency approximation

Calibration data from pattern runs provides:
- Average latencies per access scenario
- Page hit/empty/conflict rates
- Bank utilization statistics

## References

- JEDEC JESD250D: GDDR6 SGRAM Standard (May 2023)
- See `INVARIANTS.md` for detailed timing constraint documentation

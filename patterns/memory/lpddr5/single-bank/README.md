# Single Bank Patterns

Fundamental LPDDR5 timing validation with single bank operations.

## Patterns

### page-hits
Sequential reads to the same row. Tests page hit behavior where the row buffer already contains the target data.

- **Expected latency**: tCL + tBurst = 22 cycles
- **Key behavior**: No ACTIVATE needed, direct CAS command

### page-conflicts
Reads to different rows in the same bank. Tests page conflict behavior requiring row precharge and new activation.

- **Expected latency**: tRP + tRCD + tCL + tBurst = 50 cycles
- **Key behavior**: PRECHARGE → ACTIVATE → CAS sequence

### mixed-rw
Alternating reads and writes to the same row. Tests bus turnaround timing.

- **Key timing**: tRTW (read-to-write) = 14 cycles, tWTR_L = 10 cycles
- **Key behavior**: Data bus direction change overhead

## Usage

```bash
# Run page hits pattern
./build/patterns/memory/lpddr5/single-bank/page-hits

# With multi-fidelity comparison
./build/patterns/memory/lpddr5/single-bank/page-hits --fidelity

# Export trace
./build/patterns/memory/lpddr5/single-bank/page-hits --trace my_trace.json
```

## Expected Statistics

| Pattern | Reads | Page Hits | Page Empty | Page Conflicts |
|---------|-------|-----------|------------|----------------|
| page-hits (8 reads) | 8 | 7 | 1 | 0 |
| page-conflicts (8 reads) | 8 | 0 | 1 | 7 |
| mixed-rw (4 R + 4 W) | 4 | 3-7 | 1 | 0 |

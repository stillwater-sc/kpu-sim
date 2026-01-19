# Benchmark Regression Testing

Performance regression detection for the KPU Simulator (v0.3.5+).

## Overview

The benchmark regression system automatically detects performance degradation >5% and fails CI builds when regressions are found. Baselines are managed via GitHub Actions artifacts.

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PR / Push      │────▶│  Run Benchmarks  │────▶│ Compare Results │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │ Download Baseline│◀─────────────┘
                        │   (from artifact)│
                        └──────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        No Baseline         No Regression      Regression >5%
              │                  │                  │
              ▼                  ▼                  ▼
        Generate New        Pass CI           Fail CI +
        Baseline            (update on main)  Post PR Comment
```

## CI Workflow

**File:** `.github/workflows/benchmark-regression.yml`

### Triggers

| Event | Behavior |
|-------|----------|
| Push to `main` | Run benchmarks, update baseline artifact |
| Pull Request | Run benchmarks, compare to baseline, post comment |
| Manual (`workflow_dispatch`) | Run with optional baseline update |

### Baseline Storage

Baselines are stored as GitHub Actions artifacts:
- **Artifact name:** `benchmark-baseline`
- **Retention:** 90 days
- **Updated:** On every push to `main`

No baseline files are committed to the repository.

## Local Usage

### Prerequisites

```bash
# Build the benchmark tool
cmake --preset release
cmake --build --preset release --target kpu-benchmark
```

### Commands

```bash
# Generate a new baseline
python scripts/benchmark_regression.py generate

# Check for regressions against baseline
python scripts/benchmark_regression.py check

# Update baseline (archives old one with timestamp)
python scripts/benchmark_regression.py update

# Show detailed comparison report
python scripts/benchmark_regression.py report

# Generate markdown report (for PR comments)
python scripts/benchmark_regression.py markdown

# Generate JSON report (for CI integration)
python scripts/benchmark_regression.py json-report
```

### Local Baselines

Local baselines are stored in `tests/benchmarks/baselines/`:
- `baseline.json` - Current baseline
- `baseline_YYYYMMDD_HHMMSS.json` - Archived baselines (created by `update`)

These are gitignored and for local development only.

## Regression Threshold

Default threshold: **5%** (configurable in `scripts/benchmark_regression.py`)

A regression is detected when:
- GFLOPS decreases by more than 5%
- Efficiency decreases by more than 5%

## Benchmark Metrics

Each benchmark result includes:

| Category | Metrics |
|----------|---------|
| **Timing** | cycles, compile_time_us, wall_time_us |
| **Compute** | flops, gflops, efficiency |
| **Memory** | external_bytes, arithmetic_intensity, bottleneck |
| **Tiling** | Ti, Tj, Tk, num_tiles |
| **Utilization** | dma, block_mover, streamer, compute |

## Example Baseline Entry

```json
{
  "name": "matmul",
  "config": "1024x1024x1024",
  "timing": {
    "cycles": 4370048,
    "compile_time_us": 6523.0,
    "wall_time_us": 12371.86
  },
  "compute": {
    "flops": 2147483648,
    "gflops": 491.409625,
    "efficiency": 0.479892
  }
}
```

## Handling Intentional Performance Changes

When making changes that intentionally affect performance:

1. **On PR:** CI will flag the regression
2. **After merge:** Baseline updates automatically on `main`
3. **Manual update:** Trigger workflow with `update_baseline: true`

## Troubleshooting

### "No baseline found"

First run on a new repo/branch. CI will generate initial baseline automatically.

### False positives

Benchmark variance can cause false positives. Consider:
- Running benchmarks multiple times locally
- Checking if changes are within normal variance
- Adjusting threshold if needed for specific cases

### Stale baseline

If baseline is >90 days old (artifact expired):
- Push to `main` to regenerate
- Or manually trigger workflow with `update_baseline: true`

## Related Files

| File | Purpose |
|------|---------|
| `.github/workflows/benchmark-regression.yml` | CI workflow |
| `scripts/benchmark_regression.py` | Regression testing script |
| `tools/benchmark/` | Benchmark CLI source |
| `tests/benchmarks/baselines/` | Local baseline storage (gitignored) |

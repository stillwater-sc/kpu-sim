# KPU Roofline Analysis Guide

**Version:** v0.3.2
**Part of:** Benchmarking & Observability (v0.3.x)

This guide covers the roofline analysis tools for understanding KPU performance characteristics.

---

## Overview

The roofline model is a visual performance analysis tool that shows the relationship between:
- **Arithmetic Intensity (AI)**: FLOP per byte of memory traffic
- **Achieved Performance**: GFLOPS
- **Hardware Limits**: Peak compute and memory bandwidth ceilings

The KPU simulator provides multi-level roofline analysis with three memory hierarchy levels:
- External memory (DRAM)
- L3 cache
- L2 cache

---

## Quick Start

### Using the CLI Tool

```bash
# Basic roofline analysis (text output)
kpu-benchmark roofline

# JSON output for programmatic analysis
kpu-benchmark roofline -f json -o roofline.json

# Extended sweep (up to 8K)
kpu-benchmark roofline --extended
```

### Using the Python Tool

```bash
# Run benchmarks and generate plot
python tools/roofline/roofline_plot.py --run-benchmark -o roofline.png

# Analyze existing benchmark results
python tools/roofline/roofline_plot.py results.json -o roofline.png

# Text summary only (no matplotlib required)
python tools/roofline/roofline_plot.py --run-benchmark --summary
```

---

## Understanding the Output

### Text Output Example

```
=== KPU Roofline Analysis (v0.3.2) ===

Hardware Specification:
  Peak Compute:        512 GFLOPS (at 1 GHz reference)
  External Bandwidth:  64 GB/s
  L3 Bandwidth:        128 GB/s
  L2 Bandwidth:        256 GB/s

Ridge Points (FLOP/byte):
  External Memory:     8.00
  L3 Cache:            4.00
  L2 Cache:            2.00

Config                          AI     GFLOPS     Eff%    BW GB/s Region
----------------------------------------------------------------------
64K_elements                  0.08       4.00    75.0%       48.0 memory
256x256x256                  42.67     460.80    90.0%          - compute
1024x1024x1024              170.67     501.76    98.0%          - compute

Analysis:
  Compute-bound points: 2
  Memory-bound points:  1
  ✓ Meets v0.3 target (>80% for large compute-bound)
  ✓ Meets v0.3.1 target (>70% BW utilization)
```

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Peak Compute** | Maximum theoretical GFLOPS (16×16 systolic array @ 1 GHz = 512 GFLOPS) |
| **Bandwidth** | Memory bandwidth at each hierarchy level |
| **Ridge Point** | AI where transition from memory-bound to compute-bound occurs |
| **Efficiency** | Achieved GFLOPS / Predicted GFLOPS |
| **BW GB/s** | Achieved memory bandwidth (for memory-bound operations) |

### Ridge Point Interpretation

```
        Performance
        (GFLOPS)
           ^
     512 --|-------------------- Peak Compute
           |            /
           |          /
           |        /   Compute-bound region
           |      /
           |    /   Memory-bound region
           |  /
           |/
           +-------------------------> Arithmetic Intensity
                 ^                     (FLOP/byte)
                 |
            Ridge Point (8 FLOP/byte)
```

- **AI < 8**: Memory-bound (limited by external bandwidth)
- **AI >= 8**: Compute-bound (limited by peak GFLOPS)

---

## JSON Output Format

```json
{
  "version": "0.3.2",
  "hardware": {
    "peak_gflops": 512.0,
    "external_bandwidth_gbs": 64.0,
    "l3_bandwidth_gbs": 128.0,
    "l2_bandwidth_gbs": 256.0,
    "ridge_point_external": 8.0,
    "ridge_point_l3": 4.0,
    "ridge_point_l2": 2.0
  },
  "points": [
    {
      "name": "elementwise_add",
      "config": "64K",
      "arithmetic_intensity": 0.083,
      "gflops": 3.984,
      "efficiency": 0.75,
      "predicted_gflops": 5.312,
      "bottleneck": "memory"
    },
    {
      "name": "matmul",
      "config": "1024x1024x1024",
      "arithmetic_intensity": 170.67,
      "gflops": 501.76,
      "efficiency": 0.98,
      "predicted_gflops": 512.0,
      "bottleneck": "compute"
    }
  ]
}
```

### Bottleneck Classification

| Value | Meaning |
|-------|---------|
| `"compute"` | AI >= ridge_external, limited by peak GFLOPS |
| `"external"` | ridge_l3 <= AI < ridge_external, limited by DRAM bandwidth |
| `"l3"` | ridge_l2 <= AI < ridge_l3, limited by L3 bandwidth |
| `"l2"` | AI < ridge_l2, limited by L2 bandwidth |

---

## Python Visualization Examples

### Basic Roofline Plot

```python
from tools.roofline.roofline_plot import (
    RooflineAnalysis, HardwareSpec, BenchmarkPoint, plot_roofline
)

# Create hardware spec
hw = HardwareSpec(
    peak_gflops=1024.0,
    external_bw=64.0,
    l3_bw=128.0,
    l2_bw=256.0
)

# Create analysis with benchmark points
analysis = RooflineAnalysis(hw=hw)
analysis.add_point(BenchmarkPoint(
    name="matmul", config="1024x1024",
    arithmetic_intensity=170.67, gflops=491.4, efficiency=0.48
))

# Generate plot
plot_roofline(analysis, output_path="roofline.png")
```

### Custom Hardware Configuration

```bash
# Model a different KPU configuration
python tools/roofline/roofline_plot.py results.json \
    --peak-gflops 2048 \
    --ext-bw 128 \
    --l3-bw 256 \
    --l2-bw 512 \
    -o custom_roofline.png
```

### Comparing Configurations

```python
import json
import subprocess

# Run benchmark with different tile sizes
configs = ["64x64x64", "128x128x128", "256x256x256"]
results = []

for config in configs:
    m, n, k = map(int, config.split('x'))
    output = subprocess.check_output([
        "kpu-benchmark", "single",
        "-m", str(m), "-n", str(n), "-k", str(k),
        "-f", "json"
    ])
    results.append(json.loads(output))

# Analyze results
for r in results:
    ai = r["memory"]["arithmetic_intensity"]
    gflops = r["compute"]["gflops"]
    eff = r["compute"]["efficiency"]
    print(f"{r['config']}: AI={ai:.1f}, {gflops:.1f} GFLOPS, {eff*100:.1f}% eff")
```

---

## Integration with CI/CD

### Regression Testing with Roofline

```yaml
# .github/workflows/benchmark-regression.yml
- name: Run roofline analysis
  run: |
    ./build/tools/benchmark/kpu-benchmark roofline -f json -o roofline.json
    python tools/roofline/roofline_plot.py roofline.json --summary

- name: Check efficiency threshold
  run: |
    python -c "
    import json
    with open('roofline.json') as f:
        data = json.load(f)
    avg_eff = data['analysis']['average_efficiency']
    assert avg_eff > 0.4, f'Efficiency {avg_eff:.1%} below 40% threshold'
    "
```

### Generating Reports

```bash
# Generate full benchmark report with roofline
kpu-benchmark all -f json -o benchmark_report.json

# Create roofline visualization
python tools/roofline/roofline_plot.py benchmark_report.json \
    --title "KPU Performance Report $(date +%Y-%m-%d)" \
    -o reports/roofline_$(date +%Y%m%d).png
```

---

## Interpreting Results

### Healthy Performance Profile

A well-optimized workload should show:
- **Compute-bound operations** achieving 40-50% of peak
- **Memory-bound operations** achieving 70%+ of bandwidth limit
- Points close to the roofline ceiling

### Common Issues

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Low efficiency in compute-bound region | Poor tile selection | Try different tile sizes |
| Points far below memory ceiling | Bank conflicts | Adjust data layout |
| Unexpected memory-bound classification | Small problem size | Increase batch size |

### Optimization Workflow

1. **Profile**: Run roofline analysis to identify bottleneck
2. **Classify**: Determine if compute-bound or memory-bound
3. **Optimize**:
   - Compute-bound: Improve tile utilization, reduce control overhead
   - Memory-bound: Increase arithmetic intensity via tiling, data reuse
4. **Validate**: Re-run roofline to confirm improvement

---

## API Reference

### C++ API

```cpp
#include <sw/benchmark/benchmark.hpp>

using namespace sw::benchmark;

// Create harness with default hardware spec (512 GFLOPS @ 1 GHz, 64 GB/s)
BenchmarkHarness harness;

// Or with custom hardware spec
HardwareSpec hw;
hw.peak_gflops = 512.0;       // 16x16 systolic @ 1 GHz
hw.external_bw_gbs = 64.0;    // 4 channels x 16 GB/s

BenchmarkHarness custom_harness(hw);

// Run benchmark suite
auto suite = harness.sweep_matmul_square(64, 4096, 2);

// Generate roofline outputs
std::string data = harness.generate_roofline_data(suite);    // Text
std::string json = harness.generate_roofline_json(suite);    // JSON
std::string gnuplot = harness.generate_roofline_gnuplot(suite); // Gnuplot

// Query hardware spec
double ridge = hw.ridge_point_external();
std::string level = hw.bottleneck_level(arithmetic_intensity);
```

### Python API

```python
from tools.roofline.roofline_plot import (
    HardwareSpec,
    BenchmarkPoint,
    RooflineAnalysis,
    plot_roofline,
    load_benchmark_results,
    run_benchmark_cli
)

# Load results from file
points = load_benchmark_results("results.json")

# Or run benchmarks directly
points = run_benchmark_cli()

# Create analysis
hw = HardwareSpec(peak_gflops=1024, external_bw=64)
analysis = RooflineAnalysis(hw=hw, points=points)

# Get outputs
print(analysis.summary())       # Text summary
print(analysis.to_json())       # JSON output
plot_roofline(analysis, "out.png")  # Plot
```

---

## References

- Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- [KPU Execution Model](./kpu-execution-model.md)
- [Benchmarking Framework](./ROADMAP.md#v030---benchmarking--observability)

---

*Document created: v0.3.2*
*Last updated: 2026-01-23*

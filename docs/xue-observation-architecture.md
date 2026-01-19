# XUE Observation Architecture

**Version:** v0.3.3
**Part of:** Benchmarking & Observability (v0.3.x)

XUE (throughput **X**, **U**tilization, **E**fficiency) provides a hierarchical event observation framework for operational analysis of the KPU simulator, based on the Observation Architecture methodology pioneered at Intel Corporation.

---

## Background: The Observation Architecture

The XUE methodology is based on the **Observation Architecture (OA)** developed by E. Theodore L. Omtzigt at Intel Corporation (US Patent 6,023,759, filed 1997, granted 2000). This architecture provides a systematic approach to real-time performance monitoring that enables:

- **Pre-silicon validation**: Performance analysis on simulators before tape-out
- **Post-silicon validation**: Identical methodology applied to real hardware
- **Workload characterization**: Understanding how well a processor executes specific workloads

The key insight is that performance can be characterized through three fundamental metrics derived from operational analysis:

| Metric | Symbol | Definition | Measurement |
|--------|--------|------------|-------------|
| **Throughput** | X | Completion rate | Completions / Time |
| **Utilization** | U | Resource busy fraction | Busy_time / Total_time |
| **Efficiency** | E | Work per resource-time | Useful_work / (Resource × Time) |

### Operational Analysis Foundation

The Observation Architecture uses **operational analysis** to derive performance metrics from event counts rather than probabilistic models:

```
Throughput:   X = C / T     (completions per unit time)
Utilization:  U = B / T     (busy time fraction)
Latency:      L = B / C     (time per completion)
Queue Depth:  Q = D / A     (outstanding requests)
```

Where:
- **A** = Arrivals (transaction start events)
- **C** = Completions (transaction end events)
- **B** = Busy time (time-integrated resource occupation)
- **D** = Queue depth (time-integrated outstanding requests)
- **T** = Observation interval

This approach is powerful because:
1. It requires only event counting, not detailed state tracking
2. The same measurements work on simulators and real hardware
3. Results are deterministic (not statistical/probabilistic)
4. Analysis can be performed without disturbing system operation

---

## XUE in the KPU Simulator

The KPU simulator implements XUE through a hierarchical event observation framework that captures:

1. **Arrival and completion events** for all transactions
2. **Resource state transitions** (idle → busy → idle)
3. **Data movement volumes** (bytes transferred)
4. **Compute operations** (FLOPs executed)

This enables calculation of XUE metrics at every level of the memory hierarchy:

```
                    ┌─────────────────────────────────────────┐
                    │           XUE Observation               │
                    │  X: Tile completions/second             │
                    │  U: Processor Array utilization         │
                    │  E: FLOP/cycle efficiency               │
                    └─────────────────────────────────────────┘
                                      ▲
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
    ┌────┴────┐                 ┌─────┴─────┐                ┌─────┴─────┐
    │  DRAM   │                 │    L3     │                │  Compute  │
    │ X,U,E   │                 │   X,U,E   │                │   X,U,E   │
    └─────────┘                 └───────────┘                └───────────┘
```

---

## Event Hierarchy

XUE organizes events in a tree structure mirroring the KPU architecture:

```
System
├── Compute
│   ├── Matmul
│   │   ├── MATMUL_16x16        (tile-level operation)
│   │   ├── MATMUL_ACCUMULATE   (partial product accumulation)
│   │   └── MATMUL_WRITEBACK    (result tile output)
│   ├── Elementwise
│   │   ├── ADD, MUL, DIV, SUB
│   │   └── RELU, SIGMOID, TANH, GELU
│   └── Reduction
│       ├── SUM, MAX, MEAN
│       └── SOFTMAX, LAYERNORM
├── Memory
│   ├── External (DRAM)
│   │   ├── DRAM_READ, DRAM_WRITE
│   │   └── DRAM_ACTIVATE, DRAM_PRECHARGE
│   ├── L3 Buffer
│   │   ├── L3_TILE_PUSH (arrival), L3_TILE_POP (completion)
│   │   └── L3_CREDIT_RETURN (capacity released)
│   ├── L2 Buffer
│   │   ├── L2_TILE_PUSH, L2_TILE_POP
│   │   └── L2_CREDIT_RETURN
│   └── L1 Stream
│       ├── L1_STREAM_FEED
│       └── L1_CREDIT_RETURN
├── DataMovement
│   ├── DMA
│   │   ├── DMA_TRANSFER_START (arrival)
│   │   └── DMA_TRANSFER_COMPLETE (completion)
│   ├── BlockMover
│   │   ├── BM_PUSH_L3_L2
│   │   └── BM_CREDIT_WAIT (stall event)
│   └── Streamer
│       ├── STR_FEED_L2_L1
│       └── STR_CREDIT_WAIT
└── Synchronization
    ├── SYNC_BARRIER
    ├── SYNC_CREDIT_STALL
    └── SYNC_DATA_DEPENDENCY
```

Each event type supports:
- **Count**: Number of occurrences
- **Payload**: Bytes transferred or FLOPs computed
- **Timing**: Cycles spent (for latency calculation)

---

## XUE Metrics Calculation

### Throughput (X)

Throughput measures the rate of completed operations:

```cpp
// Tile throughput
double tile_throughput = tile_completions / observation_time;

// Bandwidth (bytes/second)
double dram_bandwidth = dram_bytes_transferred / observation_time;

// GFLOPS
double gflops = total_flops / (observation_time * 1e9);
```

### Utilization (U)

Utilization measures resource occupation:

```cpp
// Compute utilization
double compute_util = compute_busy_cycles / total_cycles;

// Memory channel utilization
double dram_util = dram_busy_cycles / total_cycles;

// Buffer utilization
double l3_util = l3_occupied_slots / l3_total_slots;
```

### Efficiency (E)

Efficiency measures useful work per resource-time:

```cpp
// Compute efficiency (vs peak)
double compute_eff = achieved_gflops / peak_gflops;

// Memory efficiency (vs peak bandwidth)
double mem_eff = achieved_bandwidth / peak_bandwidth;

// Roofline efficiency (vs roofline prediction)
double roofline_eff = achieved_gflops / roofline_predicted_gflops;
```

---

## Quick Start

### C++ Integration

```cpp
#include <sw/xue/event_collector.hpp>

using namespace sw::xue;

// Get the global collector
auto& xue = EventCollector::instance();

// Set simulation cycle for timing
xue.set_cycle(current_cycle);

// Record arrival/completion events
xue.record_dram_read(bytes);           // DRAM read completion
xue.record_l3_push(tile_bytes);        // L3 tile arrival
xue.record_matmul(M, N, K);            // Matmul completion

// Use scoped recording for latency measurement
{
    XUE_KERNEL_SCOPE("matmul_1024x1024");
    // ... kernel execution ...
}  // Records KERNEL_END, calculates latency

// Get XUE metrics
std::cout << xue.summary() << std::endl;
```

### Python Analysis Tool

```bash
# Predict performance using operational analysis
python tools/xue/xue_analysis.py --predict -M 1024 -N 1024 -K 1024

# Analyze XUE event data
python tools/xue/xue_analysis.py events.json

# Validate simulator vs operational analysis prediction
python tools/xue/xue_analysis.py events.json --validate simulation.json
```

---

## Integration with Roofline Model

XUE extends the Observation Architecture with **roofline analysis** for performance prediction:

```
Performance = min(Peak_FLOPS, Arithmetic_Intensity × Bandwidth)
```

The roofline model provides:
- **Upper bound prediction** from arithmetic intensity
- **Bottleneck identification** (compute-bound vs memory-bound)
- **Efficiency target** for XUE validation

### Hardware Model

| Parameter | Default | XUE Metric |
|-----------|---------|------------|
| `peak_gflops` | 1024 | X_max (throughput ceiling) |
| `dram_bandwidth_gbs` | 64 | Memory X_max |
| `l3_bandwidth_gbs` | 128 | L3 X_max |
| `l2_bandwidth_gbs` | 256 | L2 X_max |

### Ridge Points (Crossover from Memory-bound to Compute-bound)

| Level | Ridge Point | Interpretation |
|-------|-------------|----------------|
| DRAM | 16 FLOP/byte | AI ≥ 16: compute-bound |
| L3 | 8 FLOP/byte | AI ≥ 8: DRAM-bound |
| L2 | 4 FLOP/byte | AI ≥ 4: L3-bound |

---

## I/O Complexity Analysis

XUE includes tools based on **Hong-Kung I/O complexity theory** for understanding fundamental data movement requirements:

### Hong-Kung Lower Bound

For matrix multiply C = A × B:
```
Q ≥ Ω(MNK / √M_fast)
```

This establishes the minimum I/O regardless of algorithm, enabling efficiency analysis:

```cpp
#include <sw/xue/operational_analysis.hpp>

// Calculate theoretical minimum I/O
uint64_t min_io = IOComplexityAnalyzer::matmul_io_lower_bound(
    1024, 1024, 1024,  // M, N, K
    256 * 1024,        // L3 size (bytes)
    4                  // element size
);

// Calculate I/O efficiency
double io_eff = IOComplexityAnalyzer::io_efficiency(
    total_flops, actual_io, L3_bytes, element_size);
```

---

## Pre/Post-Silicon Validation

The XUE methodology enables consistent performance validation across:

| Phase | Platform | XUE Application |
|-------|----------|-----------------|
| **Pre-silicon** | Behavioral simulator | Functional validation + XUE prediction |
| **Pre-silicon** | Cycle-accurate simulator | XUE validation (prediction vs simulation) |
| **Post-silicon** | FPGA prototype | XUE validation (simulation vs hardware) |
| **Post-silicon** | Production silicon | XUE validation (all phases aligned) |

### Validation Workflow

```cpp
OperationalAnalyzer analyzer;

// 1. Predict from operational analysis
auto prediction = analyzer.analyze(xue.counters());

// 2. Compare with simulation results
auto validation = analyzer.validate(
    xue.counters(),
    actual_gflops,    // From simulation
    actual_cycles     // From simulation
);

// 3. Check alignment (target: within 10%)
if (validation.within_10_percent) {
    // Operational analysis validated
    // Same prediction will work on hardware
}
```

### Success Criteria

Per v0.3.3 requirements:
- Simulation results should match operational analysis within **10% accuracy**
- This validates that XUE predictions will transfer to post-silicon

---

## JSON Output Format

```json
{
  "version": "0.3.3",
  "summary": {
    "total_flops": 2147483648,
    "dram_bytes": 12582912,
    "arithmetic_intensity": 170.67
  },
  "xue_metrics": {
    "throughput_gflops": 491.4,
    "utilization_percent": 48.0,
    "efficiency_percent": 48.0
  },
  "categories": {
    "COMPUTE": {
      "total_events": 4096,
      "total_flops": 2147483648,
      "events": {
        "MATMUL_16x16": 4096,
        "MATMUL_ACCUMULATE": 3072,
        "MATMUL_WRITEBACK": 1024
      }
    },
    "MEMORY": {
      "total_events": 3,
      "total_bytes": 12582912,
      "events": {
        "DRAM_READ": 2,
        "DRAM_WRITE": 1
      }
    }
  }
}
```

---

## API Reference

### EventCollector

```cpp
class EventCollector {
public:
    static EventCollector& instance();

    void set_enabled(bool enabled);
    void set_cycle(uint64_t cycle);
    void advance_cycle(uint64_t delta = 1);

    // Compute events (completion)
    void record_matmul(uint32_t M, uint32_t N, uint32_t K, uint64_t cycles = 0);
    void record_relu(uint64_t elements, uint64_t cycles = 0);
    void record_softmax(uint64_t elements, uint64_t cycles = 0);

    // Memory events (arrival/completion)
    void record_dram_read(uint64_t bytes, uint64_t cycles = 0);
    void record_dram_write(uint64_t bytes, uint64_t cycles = 0);
    void record_l3_push(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0);

    // Data movement events
    void record_dma_transfer(uint64_t bytes, uint64_t cycles = 0);
    void record_blockmover_push(uint64_t bytes, ...);
    void record_streamer_feed(uint64_t bytes, ...);

    // Synchronization/stall events
    void record_credit_stall(uint64_t cycles = 1);
    void record_dependency_stall(uint64_t cycles = 1);

    // Scoped recording (for latency measurement)
    void begin_kernel(const std::string& name);
    void end_kernel();

    // Output
    std::string to_json() const;
    std::string summary() const;
    void reset();
};
```

### OperationalAnalyzer

```cpp
class OperationalAnalyzer {
public:
    explicit OperationalAnalyzer(const HardwareModel& hw = HardwareModel{});

    // Operational analysis
    OperationalResult analyze(const EventCounter& events) const;

    // Validation (simulation vs prediction)
    ValidationResult validate(const EventCounter& events,
                             double actual_gflops,
                             uint64_t actual_cycles) const;

    std::string to_json(const OperationalResult& result) const;
    std::string summary(const OperationalResult& result) const;
};
```

### IOComplexityAnalyzer

```cpp
class IOComplexityAnalyzer {
public:
    // Hong-Kung I/O lower bound
    static uint64_t matmul_io_lower_bound(
        uint64_t M, uint64_t N, uint64_t K,
        uint64_t fast_memory_bytes,
        size_t element_size = 4);

    // Optimal tile size for given cache
    static uint64_t optimal_tile_size(
        uint64_t fast_memory_bytes,
        size_t element_size = 4);

    // Reuse factor (FLOPs per I/O element)
    static double reuse_factor(
        uint64_t total_flops, uint64_t actual_io_bytes,
        size_t element_size = 4);

    // Efficiency vs Hong-Kung optimal
    static double io_efficiency(
        uint64_t total_flops, uint64_t actual_io_bytes,
        uint64_t fast_memory_bytes,
        size_t element_size = 4);
};
```

---

## Files

| File | Description |
|------|-------------|
| `include/sw/xue/event_hierarchy.hpp` | Event type definitions and hierarchy |
| `include/sw/xue/event_counter.hpp` | Event counting infrastructure |
| `include/sw/xue/event_collector.hpp` | Simulation instrumentation interface |
| `include/sw/xue/operational_analysis.hpp` | Roofline and I/O complexity analysis |
| `tools/xue/xue_analysis.py` | Python analysis tool |
| `tests/xue/test_xue.cpp` | Unit tests (79 assertions) |

---

## References

1. **Omtzigt, E. T. L.** (2000). "Dynamic Processor Performance and Functional Analysis Using Observation Architecture Having Event Generation and Selection". US Patent 6,023,759. Intel Corporation.

2. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures". Communications of the ACM, 52(4), 65-76.

3. Hong, J. W., & Kung, H. T. (1981). "I/O Complexity: The Red-Blue Pebble Game". Proceedings of the 13th Annual ACM Symposium on Theory of Computing, 326-333.

4. Demmel, J., Grigori, L., Hoemmen, M., & Langou, J. (2012). "Communication-Optimal Parallel and Sequential QR and LU Factorizations". SIAM Journal on Scientific Computing, 34(1), A206-A239.

5. Denning, P. J. (1968). "The Working Set Model for Program Behavior". Communications of the ACM, 11(5), 323-333.

---

*Document created: v0.3.3*
*Last updated: 2026-01-18*
*XUE methodology based on Observation Architecture (Omtzigt, Intel, 1997)*

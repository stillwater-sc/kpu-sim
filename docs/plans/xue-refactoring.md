# XUE Observation Architecture Refactoring Plan

## Problem Statement

The Python KPU package has implemented XUE-style metrics in Python, which is fundamentally wrong:
1. XUE Observation Architecture is designed for pre/post-silicon validation
2. Python has no chip features to observe
3. The current Python implementation duplicates the C++ XUE infrastructure
4. The C++ XUE headers exist but are not integrated into the simulator

## Goal

Remove all Python-based XUE code and properly integrate the C++ XUE Observation Architecture into the simulator, modeling the actual hardware observation pipeline that aggregates events without logic operations on the critical path.

## Current State

### C++ XUE Infrastructure (EXISTS - Well Designed)

| File | Status | Description |
|------|--------|-------------|
| `include/sw/xue/event_hierarchy.hpp` | ✅ Complete | 45+ event types, 5 categories |
| `include/sw/xue/event_collector.hpp` | ✅ Complete | Singleton, recording methods, RAII scopes |
| `include/sw/xue/event_counter.hpp` | ✅ Complete | Atomic thread-safe aggregation |
| `include/sw/xue/operational_analysis.hpp` | ✅ Complete | Roofline model, I/O complexity |
| `tests/xue/test_xue.cpp` | ✅ Complete | 79 assertions |
| `docs/xue-observation-architecture.md` | ✅ Complete | Documentation |

### Python XUE Code (TO BE REMOVED)

| File | Lines | Code to Remove |
|------|-------|----------------|
| `python/kpu/runtime.py` | 121-145 | `LevelMemoryStats` class |
| `python/kpu/runtime.py` | 147-220 | XUE fields in `ExecutionStats` |
| `python/kpu/runtime.py` | 865-949 | XUE extraction/integration |
| `python/kpu/_native/kpu_native.cpp` | 53-75 | `LevelMemoryStats` struct |
| `python/kpu/_native/kpu_native.cpp` | 88-208 | XUE fields in `NativeExecutionStats` |
| `python/kpu/_native/kpu_native.cpp` | 1304-1717 | XUE metric computation |
| `python/examples/minimal_mlp.py` | 31-103 | XUE analysis section |
| `python/examples/mnist_mlp.py` | 166-222 | XUE performance analysis |

## Design Principles

### Hardware Observation Pipeline Model

The Observation Architecture (from Intel Patent 6,023,759) is designed so that:

1. **Event Recording**: Single-cycle increment of counters - NO conditional logic
2. **Event Aggregation**: Hierarchical summation via dedicated hardware paths
3. **Zero Datapath Impact**: Observation never stalls the observed pipeline
4. **Post-hoc Analysis**: All complex analysis happens after data collection

### Implementation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION (C++)                             │
│                                                                 │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│   │ Compute  │   │ Memory   │   │   DMA    │   │   NoC    │     │
│   │ Fabric   │   │Controller│   │  Engine  │   │          │     │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘     │
│        │              │              │              │           │
│        ▼              ▼              ▼              ▼           │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              EventCollector (Singleton)                │    │
│   │  - record_matmul(), record_dram_read(), etc.           │    │
│   │  - Thread-local accumulation → global counter          │    │
│   │  - NO LOGIC on recording path (just increment)         │    │
│   └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              EventCounter (Aggregation)                │    │
│   │  - Category/Subcategory rollup                         │    │
│   │  - total_flops(), total_bytes(), etc.                  │    │
│   └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              OperationalAnalyzer                       │    │
│   │  - Roofline model predictions                          │    │
│   │  - I/O complexity analysis                             │    │
│   │  - Pre/post-silicon validation                         │    │
│   └────────────────────────────────────────────────────────┘    │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │        Python/pybind11        │
               │        (Read-Only API)        │
               │  - get_event_counts()         │
               │  - get_operational_analysis() │
               │  - NO computation             │
               └───────────────────────────────┘
```

## Implementation Phases

### Phase 1: Integrate C++ XUE into Simulator Components

**Files to Modify:**

1. `src/models/behavioral/compute/compute_fabric.cpp`
   - Add: `#include <sw/xue/event_collector.hpp>`
   - In `submit_matmul()`: Call `EventCollector::instance().record_matmul(M, N, K, cycles)`
   - In `submit_conv2d()`: Call `EventCollector::instance().record_elementwise(...)`
   - Similarly for all other submit methods

2. `src/models/transactional/compute/compute_fabric.cpp`
   - Same XUE event recording as behavioral

3. `src/models/transactional/memory/memory_controller.cpp`
   - Add: `record_dram_read()`, `record_dram_write()`
   - Track page hits/misses via DRAM_ACTIVATE, DRAM_PRECHARGE events

4. `include/sw/kpu/models/interfaces/compute_fabric_interface.hpp`
   - Add `EventCounter* get_event_counter()` method to interface

### Phase 2: Add XUE pybind11 Bindings (Read-Only)

**File: `python/kpu/_native/kpu_native.cpp`**

Replace Python XUE metrics with C++ XUE bindings:

```cpp
#include <sw/xue/event_collector.hpp>
#include <sw/xue/event_counter.hpp>
#include <sw/xue/operational_analysis.hpp>

// Expose read-only access to event counts
m.def("get_event_summary", []() -> py::dict {
    const auto& counter = EventCollector::instance().counter();
    py::dict result;
    result["total_flops"] = counter.total_flops();
    result["total_bytes"] = counter.total_bytes_moved();
    result["dram_bytes"] = counter.dram_bytes();
    result["arithmetic_intensity"] = counter.arithmetic_intensity();
    // ... category-level breakdowns
    return result;
});

m.def("get_operational_analysis", [](double actual_gflops, uint64_t actual_cycles) -> py::dict {
    OperationalAnalyzer analyzer;
    auto result = analyzer.analyze(EventCollector::instance().counter());
    auto validation = analyzer.validate(EventCollector::instance().counter(), actual_gflops, actual_cycles);
    // Return structured result
});

m.def("reset_event_counters", []() {
    EventCollector::instance().reset();
});
```

### Phase 3: Remove Python XUE Code

**Files to Modify:**

1. `python/kpu/runtime.py`
   - Remove `LevelMemoryStats` class
   - Simplify `ExecutionStats` to basic timing only (cycles, compute_cycles, memory_cycles)
   - Remove service_rate, throughput, per-level hierarchy fields
   - Add `xue_summary: Optional[Dict]` field populated from C++ `get_event_summary()`

2. `python/kpu/_native/kpu_native.cpp`
   - Remove `LevelMemoryStats` struct
   - Remove XUE fields from `NativeExecutionStats`
   - Remove inline service_rate/throughput calculations
   - Call C++ EventCollector for event recording during simulation
   - After execution, call `get_event_summary()` to populate stats

3. `python/examples/minimal_mlp.py`
   - Replace XUE analysis section with call to C++ operational analyzer
   - Display results from `kpu.get_operational_analysis()`

4. `python/examples/mnist_mlp.py`
   - Same refactoring as minimal_mlp.py

### Phase 4: Validate Pre/Post-Silicon Comparison Workflow

**New File: `python/examples/xue_validation.py`**

```python
"""XUE Pre/Post-Silicon Validation Workflow

This example demonstrates how to:
1. Run workload on simulator (pre-silicon)
2. Export XUE event counts
3. Compare against post-silicon measurements
"""
import kpu

# Run simulation
result = model(input)

# Get XUE summary from C++ (identical format to hardware counters)
xue = kpu.get_xue_summary()

# Run operational analysis
analysis = kpu.get_operational_analysis(
    actual_gflops=xue['achieved_gflops'],
    actual_cycles=xue['total_cycles']
)

# Validate against roofline prediction
print(f"Roofline Prediction: {analysis['predicted_gflops']:.2f} GFLOPS")
print(f"Actual Achieved:     {analysis['actual_gflops']:.2f} GFLOPS")
print(f"Prediction Error:    {analysis['percent_error']:.1f}%")
print(f"Within 10% Target:   {analysis['within_10_percent']}")
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/models/behavioral/compute/compute_fabric.cpp` | MODIFY | Add EventCollector calls |
| `src/models/transactional/compute/compute_fabric.cpp` | MODIFY | Add EventCollector calls |
| `src/models/transactional/memory/memory_controller.cpp` | MODIFY | Add EventCollector calls |
| `python/kpu/_native/kpu_native.cpp` | MODIFY | Replace Python XUE with C++ bindings |
| `python/kpu/runtime.py` | MODIFY | Remove Python XUE classes |
| `python/examples/minimal_mlp.py` | MODIFY | Use C++ XUE API |
| `python/examples/mnist_mlp.py` | MODIFY | Use C++ XUE API |
| `python/examples/xue_validation.py` | CREATE | Pre/post-silicon validation example |
| `python/tests/test_xue_integration.py` | CREATE | XUE integration tests |

## Success Criteria

1. **No Python XUE Computation**: All metrics computed in C++
2. **Event Recording on Fast Path**: `record_*()` calls are single atomic increments
3. **C++ XUE Tests Pass**: All 79 existing assertions pass
4. **Operational Analysis Works**: Roofline predictions available from Python
5. **Pre/Post-Silicon Format**: Event counts match hardware counter format
6. **10% Accuracy Target**: Roofline predictions within 10% of simulation

## Verification

1. **C++ XUE Tests:**
```bash
cmake --build --preset release && ctest --preset release -R xue
```

2. **Python Integration Tests:**
```bash
cd python && ~/.local/bin/pytest tests/test_xue_integration.py -v
```

3. **Example Validation:**
```bash
cd python && python examples/xue_validation.py
```

## Event Recording Design (Zero-Logic Fast Path)

**Granularity: Tile-Level (Hardware-Accurate)**

Events are recorded at the 16x16 tile level to match hardware counter behavior. A 1024x1024 matmul broken into 64x64 tiles generates 64 MATMUL_16x16 events, not 1 operation-level event.

```cpp
// CORRECT: Tile-level recording with simple atomic increment
void BehavioralComputeFabric::execute_matmul_fp32(...) {
    // Tile the operation
    for (uint32_t i = 0; i < M; i += TILE_M) {
        for (uint32_t j = 0; j < N; j += TILE_N) {
            for (uint32_t k = 0; k < K; k += TILE_K) {
                // Execute tile computation
                execute_tile(i, j, k, ...);

                // Record tile event (simple atomic increment)
                EventCollector::instance().record_matmul(
                    TILE_M, TILE_N, TILE_K,
                    TILE_CYCLES  // Fixed cycles per tile
                );
            }
        }
    }
}

// The record_matmul is just atomic increments:
void EventCollector::record_matmul(uint32_t M, uint32_t N, uint32_t K, uint64_t cycles) {
    auto& stats = counter_.get_stats(EventType::MATMUL_16x16);
    stats.count.fetch_add(1, std::memory_order_relaxed);
    stats.total_flops.fetch_add(2ULL * M * N * K, std::memory_order_relaxed);
    stats.total_cycles.fetch_add(cycles, std::memory_order_relaxed);
}

// WRONG: Logic on recording path
void record_matmul_BAD(...) {
    if (is_large_matmul(M, N, K)) {  // NO! Conditional logic
        categorize_workload(...);     // NO! Complex analysis
        update_histogram(...);        // NO! Data structure mutation
    }
}
```

This tile-level recording enables:
1. Accurate comparison with hardware tile counters
2. Proper breakdown of memory access patterns (each tile has specific L1/L2/L3 traffic)
3. Cache behavior analysis based on tile reuse patterns

## Key Insight

The Observation Architecture patent (6,023,759) specifically designs the event collection hardware to:
1. Use dedicated counter increment paths (no muxing with datapath)
2. Aggregate via hierarchical adder trees (not software loops)
3. Enable comparison between pre-silicon simulation and post-silicon hardware

By implementing XUE in C++ with simple atomic increments, we can:
1. Validate the C++ behavioral/transactional models produce correct event counts
2. Compare those counts against future hardware counters
3. Use the same analysis tools for both simulation and hardware

Python's role is purely as a **consumer** of the C++ event counts, not a producer or processor.

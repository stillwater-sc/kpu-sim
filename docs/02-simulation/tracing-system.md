# KPU Simulator Tracing System

## Overview

The KPU simulator tracing system provides cycle-accurate transaction logging for all distributed data movement and compute operations. This is essential for debugging, performance analysis, and visualization of resource utilization in the KPU's explicit data orchestration architecture.

## Design Philosophy

Unlike CPUs and GPUs that have instruction processors generating implicit data requests (aggregated through caches/warps), the KPU uses **explicit, system-level data schedules**. Data movement operators (DMA engines, block movers, streamers) marshal large blocks of structured data to compute fabrics. Since these are distributed, collaborating processes, comprehensive tracing is critical for:

- **Debug**: Understanding data flow and identifying bottlenecks
- **Performance Analysis**: Resource utilization, bandwidth analysis, conflict detection
- **Visualization**: Timeline views of component activity
- **Replay**: Deterministic reproduction of simulation runs

## Architecture

### Core Components

1. **TraceEntry** (`include/sw/trace/trace_entry.hpp`)
   - Cycle-based timestamps (not wall-clock time)
   - Component identification (type + instance ID)
   - Transaction types (READ, WRITE, TRANSFER, COMPUTE, etc.)
   - Status tracking (ISSUED, IN_PROGRESS, COMPLETED, FAILED)
   - Typed payloads (DMA, Compute, Memory, Control)

2. **TraceLogger** (`include/sw/trace/trace_logger.hpp`)
   - Thread-safe singleton for global trace collection
   - Lock-free transaction ID generation
   - Query operations (by component, cycle range, transaction type)
   - Minimal performance overhead

3. **TraceExporter** (`include/sw/trace/trace_exporter.hpp`)
   - CSV export (for spreadsheet analysis)
   - JSON export (for programmatic analysis)
   - Chrome Trace format (for chrome://tracing visualization)

## Trace Entry Format

### Timing
- `cycle_issue`: Cycle when transaction was issued
- `cycle_complete`: Cycle when transaction completed
- `clock_freq_ghz`: Optional clock frequency for time conversion

### Identification
- `component_type`: DMA_ENGINE, BLOCK_MOVER, STREAMER, COMPUTE_FABRIC, etc.
- `component_id`: Instance ID (e.g., DMA engine 0, 1, 2...)
- `transaction_id`: Unique transaction identifier

### Payload Types

#### DMA Payload
```cpp
struct DMAPayload {
    MemoryLocation source;        // Source: address, size, bank_id, type
    MemoryLocation destination;   // Destination: address, size, bank_id, type
    uint64_t bytes_transferred;   // Actual data size
    double bandwidth_gb_s;        // Theoretical bandwidth
};
```

#### Compute Payload
```cpp
struct ComputePayload {
    uint64_t num_operations;   // MACs, FLOPs, etc.
    uint32_t m, n, k;          // Matrix dimensions (GEMM)
    std::string kernel_name;   // Compute kernel identifier
};
```

## Usage Examples

### 1. Enabling Tracing on a DMA Engine

```cpp
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/trace/trace_logger.hpp>

// Create DMA engine with clock freq and bandwidth
DMAEngine dma(0, 1.0, 100.0);  // Engine 0, 1 GHz, 100 GB/s

// Enable tracing
auto& logger = TraceLogger::instance();
logger.set_enabled(true);
dma.enable_tracing(true, &logger);

// Set current cycle
dma.set_current_cycle(1000);

// Enqueue transfer (automatically traced)
dma.enqueue_transfer(
    DMAEngine::MemoryType::EXTERNAL, 0, 0x1000,
    DMAEngine::MemoryType::SCRATCHPAD, 0, 0x0,
    4096
);

// Process transfer (completion automatically traced)
dma.process_transfers(memory_banks, scratchpads);
```

### 2. Querying Traces

```cpp
// Get all traces for a specific component
auto dma_traces = logger.get_component_traces(
    ComponentType::DMA_ENGINE, 0
);

// Get traces within a cycle range
auto early_traces = logger.get_traces_in_range(0, 10000);

// Get traces by transaction type
auto transfers = logger.get_transaction_type_traces(
    TransactionType::TRANSFER
);

// Custom filtering
auto filtered = logger.get_filtered_traces([](const TraceEntry& e) {
    return e.get_duration_cycles() > 100;
});
```

### 3. Exporting Traces

```cpp
#include <sw/trace/trace_exporter.hpp>

// Export to CSV (for spreadsheet analysis)
export_logger_traces("trace.csv", "csv", logger);

// Export to JSON (for programmatic analysis)
export_logger_traces("trace.json", "json", logger);

// Export to Chrome Trace format (for visualization)
export_logger_traces("trace.trace", "chrome", logger);
```

### 4. Analyzing Bandwidth

```cpp
auto dma_traces = logger.get_component_traces(ComponentType::DMA_ENGINE, 0);

for (const auto& trace : dma_traces) {
    if (trace.status == TransactionStatus::COMPLETED &&
        std::holds_alternative<DMAPayload>(trace.payload)) {

        const auto& payload = std::get<DMAPayload>(trace.payload);
        uint64_t duration = trace.get_duration_cycles();

        // Calculate effective bandwidth
        if (duration > 0 && trace.clock_freq_ghz.has_value()) {
            double duration_ns = duration / trace.clock_freq_ghz.value();
            double effective_bw = payload.bytes_transferred / duration_ns;

            std::cout << "Transfer: " << payload.bytes_transferred << " bytes"
                     << ", BW: " << effective_bw << " GB/s" << std::endl;
        }
    }
}
```

## Chrome Trace Visualization

The Chrome Trace format export creates files compatible with `chrome://tracing`:

1. Export traces: `export_logger_traces("trace.trace", "chrome", logger);`
2. Open Chrome browser
3. Navigate to `chrome://tracing`
4. Click "Load" and select `trace.trace`

This provides:
- Timeline view of all component activity
- Zoom/pan navigation
- Per-component tracks
- Transaction details on hover
- Duration analysis

## DMA Engine Instrumentation

The DMA engine is fully instrumented as a reference implementation:

### Issue Trace
- Logged when `enqueue_transfer()` is called
- Captures source and destination details
- Records cycle of issue
- Status: ISSUED

### Completion Trace
- Logged when `process_transfers()` completes
- Includes calculated transfer latency based on bandwidth
- Records cycle of completion
- Status: COMPLETED
- Includes full DMA payload

### Bandwidth Modeling
Transfer latency is calculated as:
```
bytes_per_cycle = bandwidth_gb_s / clock_freq_ghz
transfer_cycles = ceil(size / bytes_per_cycle)
```

## Testing

### Running Trace Tests
```bash
# Build the project
mkdir build && cd build
cmake .. -DKPU_BUILD_TESTS=ON
make

# Run all trace tests
make test_trace

# Or use ctest directly
ctest -L trace --output-on-failure
```

### Test Coverage
- Single DMA transfer tracing
- Multiple queued transfers
- CSV export
- JSON export
- Chrome trace export
- Cycle range queries
- Bandwidth analysis

## Next Steps: Instrumenting Other Components

The DMA engine serves as a template. To instrument other components:

### 1. Add Tracing Support to Component Header
```cpp
#include <sw/trace/trace_logger.hpp>

class BlockMover {
private:
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;
    double clock_freq_ghz_;
    trace::CycleCount current_cycle_;

public:
    void enable_tracing(bool enabled = true, trace::TraceLogger* logger = nullptr);
    void set_current_cycle(trace::CycleCount cycle);
};
```

### 2. Log Transaction Issue
```cpp
void BlockMover::start_move(...) {
    uint64_t txn_id = trace_logger_->next_transaction_id();

    if (tracing_enabled_ && trace_logger_) {
        TraceEntry entry(
            current_cycle_,
            ComponentType::BLOCK_MOVER,
            static_cast<uint32_t>(mover_id),
            TransactionType::TRANSFER,
            txn_id
        );

        entry.clock_freq_ghz = clock_freq_ghz_;
        // Set payload...
        trace_logger_->log(std::move(entry));
    }
}
```

### 3. Log Transaction Completion
```cpp
void BlockMover::complete_move(...) {
    if (tracing_enabled_ && trace_logger_) {
        TraceEntry entry(start_cycle, ...);
        entry.complete(current_cycle_, TransactionStatus::COMPLETED);
        // Set payload...
        trace_logger_->log(std::move(entry));
    }
}
```

## Performance Considerations

1. **Minimal Overhead**: Tracing uses atomic operations and lock-free paths where possible
2. **Optional**: Can be disabled at runtime with zero overhead when disabled
3. **Buffering**: Pre-allocate trace buffer for large simulations
4. **Selective**: Use filters to log only specific components/transactions

## File Locations

```
include/sw/trace/
├── trace_entry.hpp      # Core trace format and data structures
├── trace_logger.hpp     # Thread-safe trace collection
└── trace_exporter.hpp   # Export to CSV, JSON, Chrome formats

src/trace/
├── trace_entry.cpp      # String conversion utilities
└── CMakeLists.txt       # Build configuration

tests/trace/
├── test_dma_tracing.cpp # Comprehensive DMA tracing tests
└── CMakeLists.txt       # Test configuration
```

## Summary

The KPU tracing system provides:
- ✅ Cycle-accurate transaction logging
- ✅ Zero overhead when disabled
- ✅ Thread-safe operation
- ✅ Multiple export formats
- ✅ DMA engine fully instrumented
- ✅ Comprehensive test coverage
- ✅ Chrome trace visualization support

This foundation enables productive debug, performance analysis, and visualization of the KPU's distributed data orchestration architecture.

# Component Test Harnesses and Schedule Testing Infrastructure

## Goal

Create a comprehensive testing environment for KPU data movement schedules with:
- Individual component harnesses for DMA, BlockMover, and Streamer
- System-level pipeline harness for full DRAM→L3→L2→L1→Compute flow
- Support for multiple concurrent components (DMA engines per bank, multiple BlockMovers, multiple Streamers)
- Statistics collection, event tracing, and performance analysis
- CLI tooling for schedule experimentation

## Existing Infrastructure

**Components (multi-fidelity):**
- `include/sw/kpu/models/interfaces/dma_engine_interface.hpp` — IDMAEngine, DMAEngineStatistics
- `include/sw/kpu/models/behavioral/datamovement/dma_engine.hpp` — BehavioralDMAEngine
- `include/sw/kpu/models/dataflow/block_mover_flow_executor.hpp` — BlockMoverFlowExecutor
- `include/sw/kpu/models/dataflow/streamer_flow_executor.hpp` — StreamerFlowExecutor

**Statistics and Tracing:**
- `include/sw/kpu/stats/stats_collector.hpp` — Central stats with X/U/E metrics
- `include/sw/kpu/stats/cycle_breakdown.hpp` — Cycle categorization
- `include/sw/kpu/trace/trace_logger.hpp` — Event collection
- `include/sw/kpu/trace/trace_exporter.hpp` — CSV, JSON, Chrome trace formats

**Existing Patterns:**
- `patterns/memory/lpddr5/common/lpddr5_harness.hpp` — Reference harness pattern with invariant checking
- `tests/dma/`, `tests/block_mover/`, `tests/streamer/` — Component test files

## Architecture

### Class Hierarchy

```
PatternHarnessBase<ConfigT>           // Abstract base with run(), validate(), stats()
    ├── DMAHarness                    // Single/multi DMA engine testing
    ├── BlockMoverHarness             // L3→L2 tile movement
    ├── StreamerHarness               // L2→L1 streaming with compute
    └── DataMovementPipelineHarness   // Full integrated pipeline
```

### Configuration Structures

```cpp
namespace sw::kpu::harness {

// Base configuration
struct HarnessConfig {
    SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL;
    bool enable_tracing = true;
    bool enable_stats = true;
    std::string trace_output_path;
    size_t max_cycles = 1'000'000;
};

// DMA-specific config
struct DMAHarnessConfig : HarnessConfig {
    size_t num_dma_engines = 1;           // Up to concurrent banks
    size_t dram_size = 4_GB;
    size_t l3_size = 8_MB;
    size_t l3_buffer_count = 8;
    size_t tile_size = 16 * 16 * 2;       // 16x16 fp16
};

// BlockMover-specific config
struct BlockMoverHarnessConfig : HarnessConfig {
    size_t num_block_movers = 2;          // Ingress/egress
    size_t l3_buffers = 8;
    size_t l2_banks = 16;
    size_t l2_bank_size = 64_KB;
    bool enable_transforms = true;        // Transpose, etc.
};

// Streamer-specific config
struct StreamerHarnessConfig : HarnessConfig {
    size_t num_streamers = 2;             // West/North edges
    size_t l2_banks = 16;
    size_t l1_depth = 4;                  // Pipeline depth
    size_t systolic_size = 16;
    bool enable_vector_engine = true;
};

// Full pipeline config
struct PipelineHarnessConfig : HarnessConfig {
    DMAHarnessConfig dma_config;
    BlockMoverHarnessConfig block_mover_config;
    StreamerHarnessConfig streamer_config;
    size_t compute_tile_size = 16;
    bool double_buffering = true;
};

} // namespace sw::kpu::harness
```

## Component Harnesses

### 1. DMAHarness

Tests DMA engines with credit-based flow to L3.

```cpp
class DMAHarness : public PatternHarnessBase<DMAHarnessConfig> {
public:
    explicit DMAHarness(const DMAHarnessConfig& config);

    // Load a DMProgram and execute DMA operations
    void load_program(const DMProgram& program);
    void run() override;

    // Inject specific tile requests
    void request_tile(MatrixID matrix, TileCoord coord, Address dram_addr);

    // Query state
    size_t tiles_transferred() const;
    size_t credits_available(size_t engine_id) const;

    // Statistics
    struct DMAStats {
        size_t tiles_loaded;
        size_t tiles_stored;
        size_t bytes_transferred;
        double bandwidth_utilization;
        CycleBreakdown cycle_breakdown;
    };
    DMAStats get_stats() const;

private:
    std::vector<std::unique_ptr<IDMAEngine>> dma_engines_;
    std::unique_ptr<IMemoryController> memory_controller_;
    L3BufferPool l3_buffers_;
    CreditManager credit_manager_;
};
```

**Key Tests:**
- Single tile load/store
- Burst transfers
- Multi-engine concurrent access
- Bank conflict detection
- Credit stall cycles

### 2. BlockMoverHarness

Tests L3→L2 tile movement with transforms.

```cpp
class BlockMoverHarness : public PatternHarnessBase<BlockMoverHarnessConfig> {
public:
    explicit BlockMoverHarness(const BlockMoverHarnessConfig& config);

    // Preload L3 with tiles
    void preload_l3(MatrixID matrix, TileCoord coord, const TileData& data);

    // Execute block moves
    void move_tile(MatrixID matrix, TileCoord coord,
                   size_t l3_buffer, size_t l2_bank,
                   Transform transform = Transform::IDENTITY);
    void run() override;

    // Query L2 state
    TileData read_l2(size_t bank, size_t offset);
    bool tile_resident_l2(MatrixID matrix, TileCoord coord);

    // Statistics
    struct BlockMoverStats {
        size_t tiles_moved;
        size_t transform_ops;
        double l3_bandwidth_utilization;
        double l2_bandwidth_utilization;
        std::map<Transform, size_t> transform_counts;
    };
    BlockMoverStats get_stats() const;

private:
    std::vector<std::unique_ptr<BlockMoverFlowExecutor>> block_movers_;
    L3BufferPool l3_buffers_;
    L2BankArray l2_banks_;
    TagCAM tile_tag_cam_;
};
```

**Key Tests:**
- Tile movement correctness
- Transform operations (transpose, swizzle)
- Credit flow L3→L2
- Multi-BlockMover scheduling
- Bank conflict handling

### 3. StreamerHarness

Tests L2→L1 streaming and compute integration.

```cpp
class StreamerHarness : public PatternHarnessBase<StreamerHarnessConfig> {
public:
    explicit StreamerHarness(const StreamerHarnessConfig& config);

    // Preload L2 with tiles
    void preload_l2(size_t bank, const TileData& data);

    // Stream operations
    void stream_rows(MatrixID matrix, TileCoord coord);    // A-operand
    void stream_cols(MatrixID matrix, TileCoord coord);    // B-operand
    void broadcast(MatrixID matrix, TileCoord coord);      // Bias/scalar
    void drain(MatrixID matrix, TileCoord coord);          // Accumulator
    void run() override;

    // Read compute results
    TileData read_accumulator();
    TileData read_drain_buffer();

    // Statistics
    struct StreamerStats {
        size_t rows_streamed;
        size_t cols_streamed;
        size_t broadcasts;
        size_t drains;
        double l2_read_bandwidth;
        double l1_write_bandwidth;
        size_t compute_operations;
    };
    StreamerStats get_stats() const;

private:
    std::vector<std::unique_ptr<StreamerFlowExecutor>> streamers_;
    L2BankArray l2_banks_;
    L1StreamBuffer l1_buffers_;
    ComputeFabric* compute_fabric_;
};
```

**Key Tests:**
- Row/column streaming correctness
- Compute integration (matmul fire on data arrival)
- Drain and writeback
- Vector engine operations
- Double-buffering

### 4. DataMovementPipelineHarness

Full integrated pipeline testing.

```cpp
class DataMovementPipelineHarness : public PatternHarnessBase<PipelineHarnessConfig> {
public:
    explicit DataMovementPipelineHarness(const PipelineHarnessConfig& config);

    // Load complete schedule
    void load_schedule(const DMProgram& program);
    void load_schedule_from_file(const std::string& path);

    // Memory initialization
    void init_dram(MatrixID matrix, const MatrixData& data);
    MatrixData read_dram(MatrixID matrix);

    // Execution
    void run() override;
    void step(size_t cycles = 1);  // Single-step for debugging

    // Tile journey tracking
    struct TileJourney {
        TileID tile_id;
        Cycle dram_fetch_start;
        Cycle l3_arrival;
        Cycle l2_arrival;
        Cycle l1_arrival;
        Cycle compute_start;
        Cycle compute_end;
        Cycle drain_complete;
        Cycle dram_writeback;

        Cycle total_latency() const;
        Cycle data_movement_latency() const;
        Cycle compute_latency() const;
    };
    std::vector<TileJourney> get_tile_journeys() const;

    // Aggregate statistics
    struct PipelineStats {
        DMAHarness::DMAStats dma_stats;
        BlockMoverHarness::BlockMoverStats block_mover_stats;
        StreamerHarness::StreamerStats streamer_stats;

        // Pipeline metrics
        size_t total_cycles;
        size_t stall_cycles;
        double pipeline_efficiency;
        double compute_utilization;
        double memory_bandwidth_utilization;

        // Bottleneck analysis
        std::string bottleneck_component;
        double bottleneck_utilization;
    };
    PipelineStats get_stats() const;

    // Validation
    bool validate_output(MatrixID matrix, const MatrixData& expected,
                         double tolerance = 1e-5);

private:
    DMAHarness dma_harness_;
    BlockMoverHarness block_mover_harness_;
    StreamerHarness streamer_harness_;

    std::unique_ptr<IMemory> dram_;
    TileJourneyTracker journey_tracker_;
    EventQueue event_queue_;
};
```

## Statistics Infrastructure

### HarnessStatsCollector

Extends existing StatsCollector for harness-specific metrics.

```cpp
class HarnessStatsCollector : public StatsCollector {
public:
    // Component utilization
    void record_component_active(Component comp, Cycle start, Cycle end);
    void record_component_stall(Component comp, Cycle start, Cycle end, StallReason reason);

    // Data movement
    void record_tile_transfer(TileID tile, Component src, Component dst,
                              Cycle start, Cycle end, size_t bytes);

    // Credit events
    void record_credit_granted(Component comp, size_t count);
    void record_credit_stall(Component comp, Cycle duration);

    // Generate reports
    std::string generate_summary() const;
    std::string generate_bottleneck_analysis() const;
    void export_csv(const std::string& path) const;
    void export_chrome_trace(const std::string& path) const;
};
```

### Validation Infrastructure

```cpp
class ScheduleValidator {
public:
    struct ValidationResult {
        bool passed;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    // Static validation (before execution)
    ValidationResult validate_schedule(const DMProgram& program);

    // Runtime invariant checking
    void check_credit_invariant(Component comp);      // Credits never negative
    void check_ordering_invariant(TileID tile);       // L3 before L2 before L1
    void check_data_integrity(TileID tile);           // Data matches expected

    // Post-execution validation
    ValidationResult validate_output(const MatrixData& actual,
                                     const MatrixData& expected,
                                     double tolerance);
};
```

## CLI Tool: schedule-runner

```bash
# Basic usage
schedule-runner --schedule matmul_4096x1024x8192.kpuasm \
                --init-a data/matrix_a.bin \
                --init-b data/matrix_b.bin \
                --expected-c data/expected_c.bin

# With detailed statistics
schedule-runner --schedule conv2d.kpuasm \
                --stats detailed \
                --trace chrome \
                --trace-output trace.json

# Component isolation
schedule-runner --schedule load_tiles.kpuasm \
                --harness dma \
                --dma-engines 4 \
                --stats bandwidth

# Performance analysis
schedule-runner --schedule matmul.kpuasm \
                --analyze bottleneck \
                --report performance.md
```

## File Organization

```
include/sw/kpu/harness/
├── harness_config.hpp              # Configuration structures
├── pattern_harness_base.hpp        # Abstract base class
├── dma_harness.hpp                 # DMA engine harness
├── block_mover_harness.hpp         # BlockMover harness
├── streamer_harness.hpp            # Streamer harness
├── pipeline_harness.hpp            # Full pipeline harness
├── harness_stats_collector.hpp     # Statistics collection
├── tile_journey_tracker.hpp        # Per-tile timing tracking
└── schedule_validator.hpp          # Validation utilities

src/harness/
├── pattern_harness_base.cpp
├── dma_harness.cpp
├── block_mover_harness.cpp
├── streamer_harness.cpp
├── pipeline_harness.cpp
├── harness_stats_collector.cpp
├── tile_journey_tracker.cpp
└── schedule_validator.cpp

tools/harness/
└── schedule_runner.cpp             # CLI tool

tests/harness/
├── test_dma_harness.cpp
├── test_block_mover_harness.cpp
├── test_streamer_harness.cpp
├── test_pipeline_harness.cpp
└── test_schedule_validator.cpp
```

## Implementation Plan

### Phase 1: Foundation

| File | Action |
|------|--------|
| `include/sw/kpu/harness/harness_config.hpp` | CREATE — Config structures |
| `include/sw/kpu/harness/pattern_harness_base.hpp` | CREATE — Abstract base |
| `src/harness/pattern_harness_base.cpp` | CREATE — Base implementation |
| `include/sw/kpu/harness/tile_journey_tracker.hpp` | CREATE — Tile timing |
| `src/harness/tile_journey_tracker.cpp` | CREATE |
| CMakeLists.txt updates | MODIFY — Add harness library |

### Phase 2: Individual Component Harnesses

| File | Action |
|------|--------|
| `include/sw/kpu/harness/dma_harness.hpp` | CREATE |
| `src/harness/dma_harness.cpp` | CREATE |
| `tests/harness/test_dma_harness.cpp` | CREATE |
| `include/sw/kpu/harness/block_mover_harness.hpp` | CREATE |
| `src/harness/block_mover_harness.cpp` | CREATE |
| `tests/harness/test_block_mover_harness.cpp` | CREATE |
| `include/sw/kpu/harness/streamer_harness.hpp` | CREATE |
| `src/harness/streamer_harness.cpp` | CREATE |
| `tests/harness/test_streamer_harness.cpp` | CREATE |

### Phase 3: Pipeline Integration

| File | Action |
|------|--------|
| `include/sw/kpu/harness/pipeline_harness.hpp` | CREATE |
| `src/harness/pipeline_harness.cpp` | CREATE |
| `tests/harness/test_pipeline_harness.cpp` | CREATE |
| `include/sw/kpu/harness/harness_stats_collector.hpp` | CREATE |
| `src/harness/harness_stats_collector.cpp` | CREATE |
| `include/sw/kpu/harness/schedule_validator.hpp` | CREATE |
| `src/harness/schedule_validator.cpp` | CREATE |

### Phase 4: CLI and Validation

| File | Action |
|------|--------|
| `tools/harness/schedule_runner.cpp` | CREATE — CLI tool |
| `tests/harness/test_schedule_validator.cpp` | CREATE |
| Integration tests with example schedules | CREATE |

## Verification

```bash
# Build
cmake --preset release && cmake --build --preset release

# Run harness tests
ctest --preset release -R harness

# Test individual component harnesses
./build/tests/harness/test_dma_harness
./build/tests/harness/test_block_mover_harness
./build/tests/harness/test_streamer_harness

# Test full pipeline
./build/tests/harness/test_pipeline_harness

# Use CLI tool
./build/tools/harness/schedule_runner \
    --schedule kernels/asm/matmul_4096x1024x8192.kpuasm \
    --stats detailed \
    --trace chrome --trace-output /tmp/trace.json

# Validate against expected output
./build/tools/harness/schedule_runner \
    --schedule kernels/asm/simple_matmul.kpuasm \
    --init-a testdata/a_32x32.bin \
    --init-b testdata/b_32x32.bin \
    --expected-c testdata/c_32x32.bin
```

## Dependencies

The harnesses build on existing infrastructure:
- IDMAEngine interface and implementations
- BlockMoverFlowExecutor and StreamerFlowExecutor
- StatsCollector and TraceLogger
- DMProgram and BehavioralProgramExecutor
- Memory models (DRAM, L3, L2, L1)

No new external dependencies required.

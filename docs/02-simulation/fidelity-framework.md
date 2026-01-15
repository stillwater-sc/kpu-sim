# Simulation Fidelity Framework

## Overview

The KPU simulator supports multiple levels of simulation fidelity, allowing users to trade off between simulation speed and timing accuracy. This framework provides a unified approach to configuring fidelity levels across all simulator components.

## Design Goals

1. **Composable Fidelity**: Each component can operate at its own fidelity level
2. **Consistent Interface**: All fidelity levels expose the same API
3. **Runtime Configurable**: Fidelity can be selected at configuration time
4. **Observable**: All levels support tracing and statistics collection
5. **Verifiable**: Cycle-accurate models support formal invariant checking

---

## Fidelity Levels

### Level 0: BEHAVIORAL (Functional)

**Purpose**: Functional correctness verification, software bring-up

| Aspect | Behavior |
|--------|----------|
| Timing | Instant or fixed latency |
| State | Minimal (data storage only) |
| Queuing | None |
| Contention | None |
| Speed | ~100-1000x faster than cycle-accurate |

**Use Cases**:
- Software/firmware development
- Functional verification
- Unit testing
- CI/CD pipelines

### Level 1: TRANSACTIONAL (Approximate)

**Purpose**: Early architecture exploration, performance estimation

| Aspect | Behavior |
|--------|----------|
| Timing | Statistical (mean + variance) |
| State | Aggregate (busy/idle) |
| Queuing | Basic contention modeling |
| Contention | Queue depth limits |
| Speed | ~10-100x faster than cycle-accurate |

**Use Cases**:
- Architecture design space exploration
- Workload characterization
- Bottleneck identification
- Power/performance estimation

### Level 2: CYCLE_ACCURATE (Detailed)

**Purpose**: Precise performance analysis, timing validation

| Aspect | Behavior |
|--------|----------|
| Timing | Per-cycle protocol timing |
| State | Full state machine |
| Queuing | Realistic scheduling (FR-FCFS, etc.) |
| Contention | Bank conflicts, bus arbitration |
| Speed | Baseline (1x) |

**Use Cases**:
- Performance analysis
- Timing closure verification
- Hardware/software co-design
- Trace generation for validation

---

## Component Fidelity Matrix

Each KPU component supports the following fidelity levels:

| Component | BEHAVIORAL | TRANSACTIONAL | CYCLE_ACCURATE |
|-----------|------------|---------------|----------------|
| **Memory Controller** | ✓ Fixed latency | ✓ Queue model | ✓ Full DRAM FSM |
| **DMA Engine** | ✓ Instant transfer | ✓ Bandwidth model | ✓ Channel arbitration |
| **L3 Tile** | ✓ Direct access | ✓ Bank contention | ✓ Port arbitration |
| **L2 Bank** | ✓ Direct access | ✓ Access latency | ✓ Multi-port timing |
| **L1 Buffer** | ✓ Direct access | ✓ Streaming model | ✓ Double-buffer FSM |
| **Block Mover** | ✓ Instant move | ✓ Transfer time | ✓ NoC routing |
| **Streamer** | ✓ Instant stream | ✓ Bandwidth limit | ✓ Systolic timing |
| **Compute Fabric** | ✓ Instant compute | ✓ Throughput model | ✓ Pipeline stages |
| **NoC** | ✓ Zero latency | ✓ Hop count model | ✓ Wormhole routing |

---

## Technology Profiles

Orthogonal to fidelity, each component can be parameterized with technology-specific profiles:

### Memory Technologies

```cpp
enum class MemoryTechnology {
    IDEAL,           // Perfect memory (configurable)
    LPDDR5,          // JEDEC LPDDR5 (mobile, edge AI)
    LPDDR5X,         // JEDEC LPDDR5X (high-performance mobile)
    DDR5,            // JEDEC DDR5 (servers)
    HBM3,            // JEDEC HBM3 (AI accelerators)
    HBM3E,           // JEDEC HBM3E (next-gen AI)
    GDDR6,           // JEDEC GDDR6 (graphics)
    GDDR7            // JEDEC GDDR7 (next-gen graphics)
};
```

### Compute Technologies

```cpp
enum class ComputeTechnology {
    IDEAL,           // Perfect compute (configurable)
    INT8_SYSTOLIC,   // INT8 systolic array
    FP16_SYSTOLIC,   // FP16 systolic array
    BF16_SYSTOLIC,   // BF16 systolic array
    FP32_SIMD,       // FP32 SIMD vector unit
    MIXED_PRECISION  // Dynamic precision selection
};
```

### Interconnect Technologies

```cpp
enum class InterconnectTechnology {
    IDEAL,           // Zero-latency crossbar
    MESH_2D,         // 2D mesh NoC
    TORUS_2D,        // 2D torus NoC
    HIERARCHICAL     // Hierarchical ring + mesh
};
```

---

## Verification Levels

Each component can independently enable verification:

```cpp
enum class VerificationLevel {
    NONE,              // No runtime checks (maximum speed)
    ASSERTIONS,        // Basic sanity checks (assert)
    INVARIANTS,        // Full formal invariant checking
    PROTOCOL           // Protocol compliance checking
};
```

---

## Configuration Schema

### Per-Component Configuration

```cpp
namespace sw::kpu {

// Base configuration for all components
struct ComponentConfig {
    SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL;
    VerificationLevel verification = VerificationLevel::NONE;
    bool enable_tracing = false;
    bool enable_statistics = true;
};

// Memory controller specific
struct MemoryControllerConfig : ComponentConfig {
    MemoryTechnology technology = MemoryTechnology::IDEAL;
    uint32_t speed_mt_s = 6400;
    uint8_t num_channels = 1;
    uint8_t banks_per_channel = 16;
    uint8_t bank_groups = 4;
    uint32_t queue_depth = 32;
};

// DMA engine specific
struct DMAEngineConfig : ComponentConfig {
    uint32_t num_channels = 8;
    uint32_t max_burst_size = 256;
    uint32_t bandwidth_gbps = 100;
};

// L3 tile specific
struct L3TileConfig : ComponentConfig {
    uint32_t capacity_kb = 256;
    uint8_t num_banks = 8;
    uint8_t num_ports = 4;
    uint32_t bank_width_bytes = 64;
};

// ... similar for other components

} // namespace sw::kpu
```

### Top-Level Simulator Configuration

```cpp
namespace sw::kpu {

struct SimulatorConfig {
    // Global defaults (can be overridden per-component)
    SimulationFidelity default_fidelity = SimulationFidelity::BEHAVIORAL;
    VerificationLevel default_verification = VerificationLevel::NONE;
    bool default_tracing = false;

    // Component counts
    uint32_t num_memory_controllers = 1;
    uint32_t num_dma_engines = 4;
    uint32_t num_l3_tiles = 4;
    uint32_t num_l2_banks = 16;
    uint32_t num_compute_tiles = 16;

    // Per-component configurations (optional overrides)
    std::optional<MemoryControllerConfig> memory_controller;
    std::optional<DMAEngineConfig> dma_engine;
    std::optional<L3TileConfig> l3_tile;
    std::optional<L2BankConfig> l2_bank;
    std::optional<ComputeFabricConfig> compute_fabric;
    std::optional<NoCConfig> noc;

    // Convenience: set all components to same fidelity
    void set_fidelity(SimulationFidelity fidelity);

    // Convenience: set fidelity by component class
    void set_memory_fidelity(SimulationFidelity fidelity);
    void set_compute_fidelity(SimulationFidelity fidelity);
    void set_interconnect_fidelity(SimulationFidelity fidelity);
};

} // namespace sw::kpu
```

---

## Mixed-Fidelity Simulation

The framework supports running components at different fidelity levels simultaneously:

### Example: Memory-Focused Analysis

```cpp
SimulatorConfig config;

// Cycle-accurate memory subsystem for detailed analysis
config.memory_controller = MemoryControllerConfig{
    .fidelity = SimulationFidelity::CYCLE_ACCURATE,
    .technology = MemoryTechnology::LPDDR5,
    .verification = VerificationLevel::INVARIANTS,
    .enable_tracing = true
};

// Transactional DMA for reasonable accuracy
config.dma_engine = DMAEngineConfig{
    .fidelity = SimulationFidelity::TRANSACTIONAL
};

// Behavioral compute (not the focus)
config.compute_fabric = ComputeFabricConfig{
    .fidelity = SimulationFidelity::BEHAVIORAL
};
```

### Example: Compute-Focused Analysis

```cpp
SimulatorConfig config;

// Behavioral memory (just need data)
config.memory_controller = MemoryControllerConfig{
    .fidelity = SimulationFidelity::BEHAVIORAL
};

// Cycle-accurate compute for detailed analysis
config.compute_fabric = ComputeFabricConfig{
    .fidelity = SimulationFidelity::CYCLE_ACCURATE,
    .technology = ComputeTechnology::INT8_SYSTOLIC,
    .enable_tracing = true
};
```

### Example: Full System Analysis

```cpp
SimulatorConfig config;
config.set_fidelity(SimulationFidelity::CYCLE_ACCURATE);
config.default_verification = VerificationLevel::INVARIANTS;
config.default_tracing = true;
```

---

## Interface Abstraction

Each component type has an abstract interface that all fidelity levels implement:

### Memory Controller Interface

```cpp
namespace sw::kpu {

class IMemoryController {
public:
    virtual ~IMemoryController() = default;

    // === Request Interface ===
    virtual std::optional<uint64_t> submit_read(
        uint64_t address,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    virtual std::optional<uint64_t> submit_write(
        uint64_t address,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    // === Simulation Interface ===
    virtual void tick() = 0;
    virtual void drain() = 0;
    virtual void reset() = 0;

    // === State Queries ===
    virtual uint64_t current_cycle() const = 0;
    virtual bool has_pending() const = 0;
    virtual bool can_accept() const = 0;
    virtual size_t pending_count() const = 0;

    // === Configuration ===
    virtual SimulationFidelity fidelity() const = 0;
    virtual const ComponentConfig& config() const = 0;

    // === Observability ===
    virtual void enable_tracing(bool enable) = 0;
    virtual bool tracing_enabled() const = 0;

    // === Statistics ===
    struct Statistics {
        uint64_t reads = 0;
        uint64_t writes = 0;
        uint64_t total_latency = 0;
        double avg_latency() const;
    };
    virtual const Statistics& stats() const = 0;
};

} // namespace sw::kpu
```

### Factory Pattern

```cpp
namespace sw::kpu {

// Factory creates appropriate implementation based on config
std::unique_ptr<IMemoryController> create_memory_controller(
    const MemoryControllerConfig& config);

std::unique_ptr<IDMAEngine> create_dma_engine(
    const DMAEngineConfig& config);

std::unique_ptr<IComputeFabric> create_compute_fabric(
    const ComputeFabricConfig& config);

// ... etc for each component type

} // namespace sw::kpu
```

---

## Implementation Classes

### Memory Controller Implementations

```
IMemoryController (interface)
    │
    ├── BehavioralMemoryController
    │       └── Instant/fixed latency, no state machine
    │
    ├── TransactionalMemoryController
    │       └── Queue-based, statistical timing
    │
    └── CycleAccurateMemoryController (abstract)
            │
            ├── LPDDR5MemoryController
            │       └── Full LPDDR5 protocol FSM
            │
            ├── HBM3MemoryController
            │       └── Full HBM3 protocol FSM
            │
            └── DDR5MemoryController
                    └── Full DDR5 protocol FSM
```

### Compute Fabric Implementations

```
IComputeFabric (interface)
    │
    ├── BehavioralComputeFabric
    │       └── Instant compute, result = A × B
    │
    ├── TransactionalComputeFabric
    │       └── Throughput-based timing
    │
    └── CycleAccurateSystolicArray
            └── Full pipeline, streaming timing
```

---

## Timing Synchronization

When components run at different fidelities, timing must be synchronized:

### Synchronization Rules

1. **Cycle Counter**: All components share a global cycle counter
2. **Event Queue**: Higher-fidelity components schedule completion events
3. **Blocking Semantics**: Lower-fidelity components complete within the call
4. **Callback Ordering**: Callbacks execute in cycle order

### Example: Mixed Fidelity Interaction

```
┌─────────────────────┐     ┌─────────────────────┐
│  BEHAVIORAL         │     │  CYCLE_ACCURATE     │
│  ComputeFabric      │     │  MemoryController   │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          │  submit_read(addr)         │
          │ ─────────────────────────► │
          │                            │ (queued, FSM processes)
          │  returns request_id        │
          │ ◄───────────────────────── │
          │                            │
          │        tick() × N          │
          │ ─────────────────────────► │
          │                            │ (FSM advances)
          │                            │
          │      callback()            │
          │ ◄───────────────────────── │
          │                            │
```

---

## Statistics and Tracing

All fidelity levels report consistent statistics:

### Common Statistics Interface

```cpp
struct ComponentStatistics {
    // Throughput
    uint64_t operations = 0;
    uint64_t bytes_transferred = 0;

    // Latency
    uint64_t total_latency_cycles = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t max_latency = 0;

    // Utilization
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;
    uint64_t stall_cycles = 0;

    // Derived metrics
    double avg_latency() const;
    double utilization() const;
    double bandwidth_gbps(double clock_ghz) const;
};
```

### Tracing Integration

All fidelity levels integrate with the unified tracing framework:

```cpp
// Behavioral: traces request submit/complete
// Transactional: traces queue depth, contention events
// Cycle-accurate: traces state transitions, commands
```

---

## YAML Configuration File Format

```yaml
# kpu_config.yaml
simulation:
  default_fidelity: TRANSACTIONAL
  default_verification: ASSERTIONS
  enable_tracing: true

memory:
  fidelity: CYCLE_ACCURATE
  technology: LPDDR5
  speed_mt_s: 6400
  channels: 2
  banks_per_channel: 16
  verification: INVARIANTS

dma:
  fidelity: TRANSACTIONAL
  channels: 8
  bandwidth_gbps: 100

compute:
  fidelity: BEHAVIORAL
  tiles: 16
  array_size: [16, 16]
  technology: INT8_SYSTOLIC

noc:
  fidelity: TRANSACTIONAL
  topology: MESH_2D
  dimensions: [4, 4]
```

---

## Migration Path

### Phase 1: Interface Definition
- Define `IMemoryController`, `IDMAEngine`, etc.
- Create factory functions
- Estimated: 1-2 days

### Phase 2: Wrap Existing Implementations
- Wrap `MemoryController` → `BehavioralMemoryController`
- Wrap `LPDDR5MemoryController` → implements `IMemoryController`
- Estimated: 2-3 days

### Phase 3: Add Transactional Layer
- Create `TransactionalMemoryController`
- Create `TransactionalDMAEngine`
- Estimated: 3-5 days

### Phase 4: Extend to All Components
- Apply pattern to L3, L2, L1, Compute, NoC
- Estimated: 5-10 days

### Phase 5: Configuration System
- YAML parser for config files
- CLI for fidelity selection
- Estimated: 2-3 days

**Total Estimated Effort: 2-4 weeks**

---

## File Structure

```
include/sw/kpu/
├── fidelity/
│   ├── simulation_fidelity.hpp      # Enums and base types
│   ├── component_config.hpp         # Configuration structs
│   └── component_interface.hpp      # Base interface
├── components/
│   ├── memory/
│   │   ├── memory_controller_interface.hpp
│   │   ├── behavioral_memory_controller.hpp
│   │   ├── transactional_memory_controller.hpp
│   │   └── lpddr5_memory_controller.hpp
│   ├── dma/
│   │   ├── dma_engine_interface.hpp
│   │   ├── behavioral_dma_engine.hpp
│   │   └── ...
│   └── compute/
│       ├── compute_fabric_interface.hpp
│       └── ...
└── simulator/
    ├── simulator_config.hpp
    └── component_factory.hpp

src/components/
├── memory/
│   ├── behavioral_memory_controller.cpp
│   ├── transactional_memory_controller.cpp
│   └── lpddr5_memory_controller.cpp
└── ...
```

---

## References

- SystemC TLM-2.0: Transaction-level modeling standard
- gem5: Multi-level timing simulation
- DRAMSim3: DRAM timing simulation
- Ramulator: Memory system simulation

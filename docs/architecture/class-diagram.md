# KPU Simulator — UML Class Diagrams

Layered class diagrams of the KPU simulator as of the current `main` branch. Each
diagram is scoped to a single subsystem so it stays readable. Start with the
**Overview** to see how the layers fit together, then drill into any subsystem.

Diagram conventions (Mermaid):

- `*--` composition (owns by value or `unique_ptr`)
- `o--` aggregation (holds a reference / non-owning pointer)
- `<|--` inheritance
- `<|..` interface realization
- `..>` dependency / uses

---

## 1. Overview — Top-Level Architecture

```mermaid
classDiagram
    class KPUSimulator {
        +start_dma_transfer()
        +start_block_transfer()
        +start_row_stream()
        +start_matmul()
        +step()
        +run_until_idle()
    }

    class ConcurrentTimingExecutor {
        +schedule_load() / store()
        +schedule_move() / writeback()
        +schedule_feed() / drain()
        +schedule_compute()
        +run() / step()
        +export_chrome_trace()
    }

    class TimingProcesses {
        <<subsystem>>
        MemoryControllerProcess
        DMAEngineProcess
        BlockMoverProcess
        StreamerProcess
    }

    class CSPPrimitives {
        <<subsystem>>
        CreditPool
        PartitionedCreditPool
        TagCAM
        WorkQueue
        TileWorkQueue
    }

    class TemporalModels {
        <<subsystem>>
        L3Tile / L2Bank / L1Buffer
        DMAEngine / BlockMover / Streamer
        ComputeFabric / SystolicArray
    }

    class BehavioralModels {
        <<subsystem>>
        BehavioralComputeFabric
        BehavioralMemoryController
    }

    class Interfaces {
        <<subsystem>>
        IProcess
        IComputeFabric
        IMemoryController
    }

    class FidelityFramework {
        <<subsystem>>
        SimulationFidelity
        VerificationLevel
        MemoryTechnology
        ComponentConfig
    }

    KPUSimulator *-- TemporalModels : owns
    KPUSimulator *-- ConcurrentTimingExecutor : timing simulation
    ConcurrentTimingExecutor *-- TimingProcesses : orchestrates
    ConcurrentTimingExecutor *-- CSPPrimitives : shared resources
    TimingProcesses ..|> Interfaces : implements IProcess
    BehavioralModels ..|> Interfaces : implements I*
    TemporalModels ..|> Interfaces : implements I*
    Interfaces ..> FidelityFramework : returns fidelity()
```

---

## 2. Timing / CSP — Credit-Based Dataflow Core

The credit-based dataflow simulator. Per `docs/kpu-execution-model.md`: credits flow
**UP**, data flows **DOWN**. Each process implements `IProcess::tick()` and emits
`TimingEvent`s. Shared `CreditPool`s and `TagCAM`s are owned by the executor and
held by reference inside each process.

```mermaid
classDiagram
    class IProcess {
        <<interface>>
        +tick(cycle) vector~TimingEvent~
        +is_idle() bool
        +has_pending_work() bool
        +id() uint32
        +name() string
        +reset()
    }

    class ConcurrentTimingExecutor {
        -current_cycle_ Cycle
        -events_ vector~TimingEvent~
        -pending_computes_ vector~PendingCompute~
        -fed_tiles_ set~TileID~
        +schedule_load/store/move/feed/drain/compute()
        +run() / step()
        +get_statistics() Statistics
        +export_chrome_trace()
        +export_csv()
    }

    class MemoryControllerProcess {
        -bank_states_ vector~BankState~
        -pending_requests_ vector~PendingRequest~
        -in_flight_transfers_ vector~InFlightTransfer~
        -completed_transfers_ vector~CompletedTransfer~
        +submit_request(tile, is_load, dma_id)
        +get_completed_transfer(engine_id)
        +tick(cycle)
    }

    class DMAEngineProcess {
        -pending_requests_ vector~PendingRequest~
        -stall_cycles_credit_ Cycle
        -stall_cycles_tag_ Cycle
        +schedule_load(tile)
        +schedule_store(tile)
        +tick(cycle)
    }

    class BlockMoverProcess {
        -in_flight_ optional~InFlightTransfer~
        -move_queue_ WorkQueue
        -writeback_queue_ WorkQueue
        -transpose_flags_ vector~bool~
        +schedule_move(tile, transpose)
        +schedule_writeback(tile)
        +tick(cycle)
    }

    class StreamerProcess {
        -in_flight_ optional~InFlightTransfer~
        -feed_queue_ WorkQueue
        -drain_queue_ WorkQueue
        +schedule_feed(tile)
        +schedule_drain(tile)
        +tick(cycle)
    }

    class CreditPool {
        -capacity_ size_t
        -available_ size_t
        +acquire() bool
        +release()
        +has_credit() bool
    }

    class TagCAM {
        -entries_ map~TileID,TagCAMEntry~
        +insert(tile, slot, cycle)
        +lookup(tile) bool
        +match(tile) TagCAMEntry
        +invalidate(tile) bool
        +ref_count(tile)
    }

    class TimingEvent {
        +type EventType
        +cycle Cycle
        +component_id uint32
        +tile_id TileID
        +duration Cycle
        +to_chrome_trace_json()
    }

    class LivelockDetector {
        +observe(cycle, events)
        +is_livelock_suspected()
    }

    IProcess <|.. MemoryControllerProcess
    IProcess <|.. DMAEngineProcess
    IProcess <|.. BlockMoverProcess
    IProcess <|.. StreamerProcess

    ConcurrentTimingExecutor *-- "many" MemoryControllerProcess
    ConcurrentTimingExecutor *-- "many" DMAEngineProcess
    ConcurrentTimingExecutor *-- "many" BlockMoverProcess
    ConcurrentTimingExecutor *-- "many" StreamerProcess
    ConcurrentTimingExecutor *-- "1" LivelockDetector
    ConcurrentTimingExecutor *-- "l3_credits, l2_credits" CreditPool
    ConcurrentTimingExecutor *-- "l3, l2, compute_result" TagCAM
    ConcurrentTimingExecutor ..> TimingEvent : emits

    DMAEngineProcess o-- MemoryControllerProcess : submits to
    DMAEngineProcess o-- CreditPool : l3_credits&
    DMAEngineProcess o-- TagCAM : l3_tag_cam&

    BlockMoverProcess o-- CreditPool : l3 + l2 credits&
    BlockMoverProcess o-- TagCAM : l3 + l2 tag cams&

    StreamerProcess o-- CreditPool : l2_credits&
    StreamerProcess o-- TagCAM : l2 + compute_result&
```

**Tick order:** `MC.tick() → DMA.tick() → BlockMover.tick() → Streamer.tick()`
(MC first so completions are visible to DMA the same cycle).

---

## 3. CSP Primitives

Reusable credit-flow building blocks. `PartitionedCreditPool` and `TileWorkQueue`
partition resources across the A/B/C matrix dimensions to prevent livelock.

```mermaid
classDiagram
    class CreditPool {
        -capacity_ size_t
        -available_ size_t
        +acquire() bool
        +release()
        +has_credit() bool
        +outstanding() size_t
        +reset()
    }

    class PartitionedCreditPool {
        -capacity_a_/b_/c_ size_t
        -available_a_/b_/c_ size_t
        +acquire(Matrix) bool
        +release(Matrix)
        +available(Matrix)
        +total_available()
    }

    class TagCAM {
        -capacity_ size_t
        -entries_ unordered_map
        +insert(tile, slot, cycle) bool
        +lookup(tile) bool
        +match(tile) optional~TagCAMEntry~
        +invalidate(tile) bool
        +ref_count(tile) uint32
        +find_oldest() optional~TagCAMEntry~
        +find_by_matrix(MatrixID)
    }

    class TagCAMEntry {
        +tile_id TileID
        +slot_id uint32
        +arrival_cycle Cycle
        +valid bool
        +ref_count uint32
    }

    class WorkQueue~T~ {
        -queue_ deque~T~
        -max_size_ size_t
        +enqueue(item) bool
        +try_peek() optional~T~
        +try_dequeue() optional~T~
        +find_if(pred)
        +dequeue_if(pred)
    }

    class TileWorkQueue {
        -queue_a_/b_/c_ deque~TileDescriptor~
        -max_per_matrix_ size_t
        +enqueue(tile) bool
        +peek(MatrixID)
        +dequeue(MatrixID)
        +dequeue_oldest_ready(pred, cycle)
    }

    class TileID {
        +matrix MatrixID
        +ti uint32
        +tj uint32
        +tk uint32
        +to_string()
    }

    class TileDescriptor {
        +tile_id TileID
        +dram_address Address
        +size_bytes Size
        +height/width/element_size Size
        +enqueue_cycle Cycle
        +priority Size
        +age_priority(cycle)
    }

    TagCAM *-- "many" TagCAMEntry
    TagCAMEntry ..> TileID
    TileDescriptor *-- TileID
    WorkQueue <|-- TileWorkQueue : specialization
```

---

## 4. Interfaces & Fidelity Framework

Multi-fidelity is enforced via interfaces that every implementation reports its
fidelity through. Concrete implementations live in `behavioral/` (instant),
`temporal/` (cycle-accurate), and `timing/` (CSP processes).

```mermaid
classDiagram
    class SimulationFidelity {
        <<enum>>
        BEHAVIORAL
        TRANSACTIONAL
        CYCLE_ACCURATE
    }

    class VerificationLevel {
        <<enum>>
        NONE
        ASSERTIONS
        INVARIANTS
        PROTOCOL
    }

    class MemoryTechnology {
        <<enum>>
        IDEAL
        LPDDR5 / LPDDR5X
        DDR5
        HBM2 / HBM2E / HBM3
        GDDR6 / GDDR7
    }

    class ComponentConfig {
        +memory_fidelity SimulationFidelity
        +compute_fidelity SimulationFidelity
        +dma_fidelity SimulationFidelity
        +noc_fidelity SimulationFidelity
        +verification_level VerificationLevel
        +memory_technology MemoryTechnology
    }

    class IComputeFabric {
        <<interface>>
        +submit_matmul()
        +submit_conv2d()
        +submit_elementwise()
        +submit_pool2d()
        +submit_softmax()
        +submit_layernorm()
        +tick() bool
        +is_busy() bool
        +fidelity() SimulationFidelity
    }

    class IMemoryController {
        <<interface>>
        +submit_read()
        +submit_write()
        +can_accept() bool
        +tick()
        +drain()
        +current_cycle() uint64
        +fidelity() SimulationFidelity
        +technology() MemoryTechnology
        +get_bank_state(ch, bank) BankState
        +is_row_open(ch, bank, row) bool
    }

    class IProcess {
        <<interface>>
        +tick(cycle) vector~TimingEvent~
        +is_idle()
        +has_pending_work()
        +reset()
    }

    class BehavioralComputeFabric
    class BehavioralMemoryController
    class TemporalComputeFabric
    class TemporalMemoryController
    class MemoryControllerProcess
    class DMAEngineProcess
    class BlockMoverProcess
    class StreamerProcess

    IComputeFabric <|.. BehavioralComputeFabric
    IComputeFabric <|.. TemporalComputeFabric
    IMemoryController <|.. BehavioralMemoryController
    IMemoryController <|.. TemporalMemoryController
    IProcess <|.. MemoryControllerProcess
    IProcess <|.. DMAEngineProcess
    IProcess <|.. BlockMoverProcess
    IProcess <|.. StreamerProcess

    ComponentConfig *-- SimulationFidelity
    ComponentConfig *-- VerificationLevel
    ComponentConfig *-- MemoryTechnology
    IComputeFabric ..> SimulationFidelity
    IMemoryController ..> SimulationFidelity
    IMemoryController ..> MemoryTechnology
```

---

## 5. Temporal Tier — Functional / Cycle-Accurate Models

These are the components owned directly by `KPUSimulator` for the value-computing
behavioral execution path. They model actual memory contents and perform actual
matrix arithmetic.

```mermaid
classDiagram
    class KPUSimulator {
        +Config config
        +step()
        +run_until_idle()
    }

    class ExternalMemory {
        -storage_ vector~uint8_t~
        -base_address_ Address
        -capacity_ Size
        +read() / write()
    }

    class L3Tile {
        -memory_model vector~uint8_t~
        -capacity Size
        -tile_id size_t
        +read() / write()
        +read_block() / write_block()
        +is_ready() bool
    }

    class L2Bank {
        -storage vector~uint8_t~
        +read_block() / write_block()
    }

    class L1Buffer {
        -storage vector~uint8_t~
        +read_block() / write_block()
    }

    class PageBuffer {
        +coalesce()
    }

    class DMAEngine {
        -transfer_queue vector~Transfer~
        -engine_id size_t
        -clock_freq_ghz_ double
        -bandwidth_gb_s_ double
        +enqueue_transfer()
        +update(cycle)
    }

    class BlockMover {
        -transfer_queue vector~BlockTransfer~
        -associated_l3_tile_id size_t
        +enqueue_transfer()
        +update(cycle)
    }

    class Streamer {
        -stream_state StreamState
        +start_stream(config)
        +update(cycle)
    }

    class ComputeFabric {
        -current_op MatMulConfig
        -compute_type ComputeType
        -systolic_array unique_ptr~SystolicArray~
        +start_matmul(config)
        +update(cycle, l1_buffers)
        +is_busy()
    }

    class SystolicArray~T~ {
        -pe_array vector~vector~PE~~
        -a_stream / b_stream / c_stream
        -rows / cols Size
        +cycle()
    }

    class AddressDecoder {
        +decode(address)
    }

    KPUSimulator *-- "many" ExternalMemory : host + banks
    KPUSimulator *-- "many" L3Tile
    KPUSimulator *-- "many" L2Bank
    KPUSimulator *-- "many" L1Buffer
    KPUSimulator *-- "many" PageBuffer
    KPUSimulator *-- "many" DMAEngine
    KPUSimulator *-- "many" BlockMover
    KPUSimulator *-- "many" Streamer
    KPUSimulator *-- "many" ComputeFabric
    KPUSimulator *-- AddressDecoder

    ComputeFabric *-- SystolicArray
    DMAEngine ..> AddressDecoder
    DMAEngine ..> ExternalMemory : transfers
```

---

## 6. Behavioral Tier — Instant / Functional Models

The fast path used for software bring-up. Behavioral models complete operations
in zero or fixed cycles and compute actual numerical results.

```mermaid
classDiagram
    class IComputeFabric {
        <<interface>>
    }
    class IMemoryController {
        <<interface>>
    }

    class BehavioralComputeFabric {
        -config_ ComputeFabricConfig
        -tile_id_ uint32
        +submit_matmul()
        +submit_conv2d()
        +submit_elementwise()
        +submit_pool2d()
        +submit_softmax()
        +submit_layernorm()
        +submit_batchnorm()
        +tick() bool
        +fidelity() BEHAVIORAL
    }

    class BehavioralMemoryController {
        -config_ MemoryControllerConfig
        -current_cycle_ uint64
        -pending_callbacks_ queue
        +submit_read()
        +submit_write()
        +can_accept() true
        +tick()
        +drain()
        +fidelity() BEHAVIORAL
    }

    class Kernels {
        <<utility>>
        dispatch_matmul~T~()
        dispatch_conv2d~T~()
        dispatch_elementwise~T~()
        dispatch_softmax~T~()
        dispatch_layernorm~T~()
    }

    IComputeFabric <|.. BehavioralComputeFabric
    IMemoryController <|.. BehavioralMemoryController
    BehavioralComputeFabric ..> Kernels : dispatches by dtype
```

---

## 7. Trace / Visualization Output

Cycle-accurate runs export Chrome Trace JSON and CSV for visualization. The
`TimingEvent` is the unit of trace data emitted by every `IProcess` per tick.

```mermaid
classDiagram
    class TimingEvent {
        +type EventType
        +cycle Cycle
        +component_id uint32
        +tile_id TileID
        +duration Cycle
        +slot_id uint32
        +component_name string
        +matrix_base_address Address
        +dram_address Address
        +to_chrome_trace_json()
    }

    class EventType {
        <<enum>>
        DMA_LOAD_START / COMPLETE
        DMA_STALL_CREDIT / TAG
        MC_ACCESS_TYPE / BANK_CONFLICT
        BM_MOVE_START / COMPLETE
        BM_WRITEBACK_START / COMPLETE
        STR_FEED_START / COMPLETE
        STR_DRAIN_START / COMPLETE
        COMPUTE_START / COMPLETE
        CREDIT_ACQUIRED / RELEASED
        TILE_ARRIVED_L3 / L2 / L1
        TILE_FED_TO_COMPUTE / DRAINED / CONSUMED
    }

    class ChromeTraceExporter {
        <<utility>>
        +export(events, filename)
    }
    class CSVExporter {
        <<utility>>
        +export(events, filename)
    }
    class TraceValidator {
        <<python tool>>
        +validate(trace.json)
        +check INV-001..INV-101
    }

    TimingEvent *-- EventType
    ChromeTraceExporter ..> TimingEvent : serializes
    CSVExporter ..> TimingEvent : serializes
    TraceValidator ..> ChromeTraceExporter : consumes JSON
```

---

## File Map

| Diagram | Primary source files |
|---|---|
| Overview | `include/sw/kpu/kpu_simulator.hpp` |
| Timing/CSP | `include/sw/kpu/timing/{concurrent_timing_executor, *_process, process_interface}.hpp` |
| CSP Primitives | `include/sw/kpu/timing/{credit_pool, tag_cam, work_queue, tile_descriptor}.hpp` |
| Interfaces & Fidelity | `include/sw/kpu/fidelity/*.hpp`, `include/sw/kpu/models/interfaces/*.hpp` |
| Temporal | `include/sw/kpu/models/temporal/{memory, datamovement, compute}/*.hpp` |
| Behavioral | `include/sw/kpu/models/behavioral/{compute, memory}/*.hpp` |
| Trace | `include/sw/kpu/timing/process_interface.hpp` (TimingEvent), exporters in `ConcurrentTimingExecutor` |

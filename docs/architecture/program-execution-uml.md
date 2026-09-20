# UML — how the classes deliver a simulated domain-flow program execution

**Scope:** the classes that turn a domain-flow program into a simulated execution, and how
they collaborate. Companion to `docs/architecture/class-diagram.md` (hardware components)
and `docs/architecture/adr/0001-program-contract-and-transactional-engine.md` (which
engine owns which tier).

**How to read it.** Everything marked **[today]** exists and runs. Everything marked
**[planned]** is decided in ADR 0001 but not yet built, and is drawn dashed. The frozen
engines (the DMProgram executors, the OFG flow executors — ADR 0001 D6) are deliberately
absent: they are not part of the path forward.

---

## 1. The stack at a glance

Two program layers exist today, reaching two different execution tiers. ADR 0001 makes the
L0 program the single portable one, so the left column becomes the front end of the right.

```mermaid
flowchart TB
    subgraph FE["Front end"]
        DFG["DomainFlowGraph / .dfg<br/>(domain_flow, external)"]
        KG["KernelGraph<br/>operator DAG [today]"]
    end

    subgraph L0L1["Portable program — ADR 0001 D1"]
        TP["TileProgram (L0)<br/>tile sequences + tile I/O"]
        SP["StreamProgram (L1)<br/>stream signatures, wavefronts"]
    end

    subgraph TIERS["Execution tiers"]
        REF["TileProgramReference<br/>BEHAVIORAL [today]"]
        TTE["TileTransactionExecutor<br/>TRANSACTIONAL [planned]"]
        CSP["ConcurrentTimingExecutor<br/>CYCLE_ACCURATE [today]"]
    end

    subgraph ANA["Analysis"]
        DAG["TileDag + DeviceDescriptor<br/>DAG, list schedule, DOT [today]"]
        MET["Metrics / RunStats"]
    end

    DFG -.->|"planned front end"| TP
    KG -->|"GraphCspExecutor [today]"| CSP
    TP --> REF
    TP -.-> TTE
    TP --> DAG
    SP --> DAG
    SP -.-> TTE
    TP -.->|"driver JIT, planned"| CSP
    REF --> MET
    TTE -.-> MET
    CSP --> MET
    DAG --> MET
```

The one gap worth naming: today `KernelGraph` reaches the cycle-accurate tier directly,
while `TileProgram` reaches only the reference and the analysis harness. Closing that —
one program, three tiers — is what ADR 0001 decided and what #264 and #265 implement.

---

## 2. The portable program (L0/L1) and its drivers

```mermaid
classDiagram
    class TileProgram {
        -string name_
        -operands_ : name to TensorOperand
        -vector~string~ order_
        -vector~TileOp~ ops_
        +add_operand(TensorOperand) TensorOperand&
        +push(TileOp) void
        +operand(name) TensorOperand&
        +has_operand(name) bool
        +operand_order() vector~string~&
        +ops() vector~TileOp~&
        +disassemble() string
    }
    class TensorOperand {
        +string name
        +Dim rows, cols
        +Dim tile_rows, tile_cols
        +vector~float~ values
        +at(r, c) float
        +row_begin(ti) Dim
        +row_end(ti) Dim
    }
    class TileOp {
        +TileOpKind kind
        +vector~TileCoord~ inputs
        +vector~TileCoord~ outputs
        +float alpha
        +int pivot_slot
        +string port
    }
    class TileCoord {
        +string operand
        +Dim ti, tj
        +to_string() string
    }
    class TileOpKind {
        <<enumeration>>
        Feed
        Drain
        MatMulAccum
        LuDiagFactor
        PivotApply
        TrsmLowerLeft
        TrsmUpperRight
    }
    class TileKernelState {
        +pivots : slot to row-swap list
        +vector~Dim~ perm
        +size_t swaps_performed
        +ensure_perm(n)
        +clear()
    }
    class TileKernels {
        <<module>>
        +apply(TileProgram, TileOp, TileKernelState)
        +kernel_matmul_accum(...)
        +kernel_lu_diag_factor(...)
        +kernel_pivot_apply(...)
        +kernel_trsm_lower_left(...)
        +kernel_trsm_upper_right(...)
    }
    class TileProgramReference {
        -TileKernelState state_
        +run(TileProgram) RunSummary
    }
    class TileTransactionExecutor {
        -TileKernelState state_
        +run(TileExecutionRequest) RunResult
    }

    TileProgram *-- TensorOperand
    TileProgram *-- TileOp
    TileOp *-- TileCoord
    TileOp --> TileOpKind
    TileCoord ..> TensorOperand : names
    TileProgramReference ..> TileKernels : in program order
    TileTransactionExecutor ..> TileKernels : in dataflow order
    TileKernels ..> TileKernelState : carries pivots
    TileKernels ..> TileProgram : mutates operand buffers
```

**The load-bearing relationship** is the pair of dashed arrows into `TileKernels`. Both
drivers call **one** kernel implementation and differ only in the order they choose and
what they measure, which is what makes the transactional tier bit-identical to the oracle
by construction rather than by testing (ADR 0001 D3.1, delivered by #270).

`TileKernelState` is the other subtlety: it carries a pivot decision from `LuDiagFactor` to
the `PivotApply` ops that replay it. That is the data-dependent control GEMM does not have,
and the reason a driver may not reorder ops past their pivot-slot edges.

### L1 and the analysis layer

```mermaid
classDiagram
    class StreamProgram {
        +computes : op index to WavefrontTiming
        +signature(operand) StreamSignature
        +disassemble() string
    }
    class StreamSignature {
        +edge, lanes, role
        +element_stride, bubble
        +array flow direction
    }
    class SpaceTimeMap {
        +schedule tau
        +projection u
        +output_stationary() SpaceTimeMap
        +b_stationary() SpaceTimeMap
        +a_stationary() SpaceTimeMap
        +fully_streaming() SpaceTimeMap
    }
    class WavefrontTiming {
        +fill, reduce, drain
        +latency() Cycle
    }
    class TileDag {
        -vector~DagNode~ nodes_
        -DeviceDescriptor dev_
        +critical_path_cycles() double
        +list_schedule() Schedule
        +to_dot(TileProgram, title) string
        +nodes() vector~DagNode~
    }
    class DagNode {
        +size_t op_index
        +TileOpKind kind
        +TileWork work
        +double duration
        +vector preds, succs
        +double start, finish
        +int worker
    }
    class DeviceDescriptor {
        +Topology topology
        +Dim compute_tiles, move_lanes, l3_tiles
        +double fabric_macs_per_cycle
        +double bytes_per_cycle
        +label() string
    }

    SpaceTimeMap ..> StreamProgram : derives
    StreamProgram *-- StreamSignature
    StreamProgram *-- WavefrontTiming
    TileDag *-- DagNode
    TileDag --> DeviceDescriptor
    TileDag ..> StreamProgram : optional L1 timing
```

`TileDag` recovers the dependency DAG from each op's **declared tile I/O** — RAW/WAR/WAW
over tiles, plus the pivot-slot edges. That recovery is the shared foundation: the
analysis harness list-schedules it, `to_dot` draws it, and the planned executor fires
against it.

---

## 3. The cycle-accurate tier (CSP)

```mermaid
classDiagram
    class ConcurrentTimingExecutor {
        -Config config_
        -vector~IProcess~ processes_
        -payloads_ : TileID to StoredPayload
        -Cycle current_cycle_
        +execute(ScheduleResult) ExecutionResult
        +tiles_at(level) vector~TileID~
        +statistics() Statistics
    }
    class Config {
        +Cycle compute_latency
        +Cycle compute_cycles_per_k_slice
        +Dim l3_credits, l2_credits, l1_credits
        +bool partition_l3_credits
    }
    class MatMulComputeSpec {
        +M, N, K
        +bias, activation
    }
    class FunctionalComputeSpec {
        +function operation
        +vector resident_tiles
    }
    class IProcess {
        <<interface>>
        +tick(Cycle) vector~TimingEvent~
        +name() string
    }
    class MemoryControllerProcess
    class DMAEngineProcess
    class BlockMoverProcess
    class StreamerProcess
    class CreditPool {
        +acquire() bool
        +release()
        +available() Dim
    }
    class PartitionedCreditPool
    class TagCAM {
        +insert(TileID) bool
        +match(TileID) bool
        +set_capacity(n)
    }
    class TileDescriptor {
        +TileID id
        +TileLocation location
        +TilePayload payload
    }
    class TilePayload {
        +vector~float~ values
    }
    class LivelockDetector
    class TileTracker

    ConcurrentTimingExecutor *-- Config
    ConcurrentTimingExecutor o-- IProcess
    ConcurrentTimingExecutor ..> MatMulComputeSpec
    ConcurrentTimingExecutor ..> FunctionalComputeSpec
    ConcurrentTimingExecutor *-- CreditPool
    ConcurrentTimingExecutor *-- TagCAM
    ConcurrentTimingExecutor ..> LivelockDetector
    ConcurrentTimingExecutor ..> TileTracker
    IProcess <|-- MemoryControllerProcess
    IProcess <|-- DMAEngineProcess
    IProcess <|-- BlockMoverProcess
    IProcess <|-- StreamerProcess
    CreditPool <|-- PartitionedCreditPool
    TileDescriptor *-- TilePayload
    DMAEngineProcess ..> CreditPool : waits for L3 credit
    BlockMoverProcess ..> TagCAM : waits for tile arrival
    StreamerProcess ..> TagCAM : waits for tile arrival
```

This is the credit-based dataflow model in class form: a producer pushes only when it holds
a **credit** from downstream, and consumers return credits upstream. `TagCAM` is how a
component waits for a **tile arrival** out of order. No component fetches on demand.

### What drives it

```mermaid
classDiagram
    class IScheduleGenerator {
        <<interface>>
        +generate(config) ScheduleResult
    }
    class MatMulScheduleGenerator
    class Conv2DScheduleGenerator
    class PoolingScheduleGenerator
    class BatchNormScheduleGenerator
    class OnlineSoftmaxScheduleGenerator
    class OnlineReductionScheduleGenerator
    class ScheduleResult {
        +vector~ScheduleOperation~ operations
        +ScheduleMetadata metadata
    }
    class ScheduleOperation {
        +OpKind kind
        +tile, feed, resident
    }
    class ScheduleExecutor {
        +execute(ScheduleResult) ExecutionResult
    }
    class GraphCspExecutor {
        +run(KernelGraph, input, ...) Result
    }
    class Result {
        +vector~float~ output
        +RunStats stats
    }
    class FunctionalMLPExecutor {
        +add_layer(...)
        +run(input) vector~float~
    }
    class FunctionalSoftmaxExecutor
    class FunctionalElementwiseExecutor
    class FunctionalReductionExecutor
    class ScheduleValidator {
        +validate_livelock_safety(...)
    }

    IScheduleGenerator <|-- MatMulScheduleGenerator
    IScheduleGenerator <|-- Conv2DScheduleGenerator
    IScheduleGenerator <|-- PoolingScheduleGenerator
    IScheduleGenerator <|-- BatchNormScheduleGenerator
    IScheduleGenerator <|-- OnlineSoftmaxScheduleGenerator
    IScheduleGenerator <|-- OnlineReductionScheduleGenerator
    IScheduleGenerator ..> ScheduleResult : produces
    ScheduleResult *-- ScheduleOperation
    ScheduleExecutor ..> ScheduleResult : consumes
    ScheduleExecutor ..> ConcurrentTimingExecutor
    GraphCspExecutor ..> IScheduleGenerator : per node
    GraphCspExecutor ..> ConcurrentTimingExecutor
    GraphCspExecutor ..> Result : returns output + stats
    FunctionalMLPExecutor ..> ConcurrentTimingExecutor
    FunctionalSoftmaxExecutor ..> ConcurrentTimingExecutor
    FunctionalElementwiseExecutor ..> ConcurrentTimingExecutor
    FunctionalReductionExecutor ..> ConcurrentTimingExecutor
    ScheduleValidator ..> ScheduleResult : checks envelope
```

---

## 4. Sequence — a DNN graph on the cycle-accurate tier [today]

How `m2_resnet` actually executes. Each node runs on a fresh executor and the statistics
are summed, so cross-op residency is not modelled — a known limitation, not a drawing
simplification.

```mermaid
sequenceDiagram
    autonumber
    participant App as m2_resnet
    participant KG as KernelGraph
    participant GX as GraphCspExecutor
    participant Gen as ScheduleGenerator
    participant CTE as ConcurrentTimingExecutor
    participant P as Processes MC/DMA/BM/STR
    participant CP as CreditPool + TagCAM

    App->>KG: build_resnet18(spec)
    App->>GX: run(graph, input tensor)
    loop per node in topological order
        GX->>Gen: generate(op config)
        Gen-->>GX: ScheduleResult
        GX->>CTE: execute(schedule)
        loop every cycle until complete
            CTE->>P: tick(cycle) in order MC, DMA, BM, STR
            P->>CP: acquire credit / match tile arrival
            CP-->>P: credit granted or wait
            P-->>CTE: TimingEvents
            CTE->>CTE: fire ready computes, apply payload values
        end
        CTE-->>GX: ExecutionResult with values + cycles
    end
    GX-->>App: Result: output tensor + RunStats (cycles, MACs, utilization)
    App->>App: compare output against host reference, tol 5e-3
```

## 5. Sequence — an L0 tile program today [today]

What `tile_characterize` does, and where `--dot` and the shared kernels sit.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as tile_characterize
    participant Der as derive_lu_tile_program
    participant TP as TileProgram
    participant Ref as TileProgramReference
    participant K as TileKernels
    participant Dag as TileDag

    CLI->>Der: derive(N, T)
    Der-->>TP: ops: GETRF, LASWP, TRSM, GEMM
    CLI->>Ref: run(program)
    loop per op in program order
        Ref->>K: apply(program, op, state)
        K->>TP: mutate operand buffers
        K->>K: record pivots into state
    end
    Ref-->>CLI: RunSummary with permutation
    CLI->>CLI: validate P.A = L.U against oracle
    CLI->>Dag: TileDag(program, device, optional L1)
    Dag->>Dag: recover RAW/WAR/WAW + pivot edges
    CLI->>Dag: list_schedule()
    Dag-->>CLI: makespan, utilization, bound
    CLI->>Dag: to_dot(program, title)
    Dag-->>CLI: Graphviz DAG
```

## 6. Sequence — the target transactional execution [planned]

The path ADR 0001 decided and #264/#265 implement. Note where it differs from §5: the
program arrives **from a file**, and ops fire on credits and residency instead of in
program order.

```mermaid
sequenceDiagram
    autonumber
    participant Host as host / loader
    participant F as executor factory
    participant TTE as TileTransactionExecutor
    participant Dep as dependency model
    participant Cred as credits + residency
    participant K as TileKernels

    Host->>Host: deserialize TileProgram from file
    Host->>F: create(fidelity TRANSACTIONAL, request)
    Note over F: request carries program, placement,<br/>DeviceDescriptor, optional L1, seed
    F-->>TTE: executor
    Host->>TTE: run(request)
    TTE->>Dep: recover tile + pivot-slot edges
    loop until no ops remain
        TTE->>Cred: inputs resident? output credit? resource free?
        Cred-->>TTE: ready set
        TTE->>K: apply(program, op, state) for the fired op
        K-->>TTE: values mutated in place
        TTE->>TTE: schedule completion event, advance to next event time
        TTE->>Cred: release slot after last consumer
    end
    TTE-->>Host: RunResult: values, timeline, stats, provenance
    Host->>Host: assert bit-exact vs TileProgramReference
```

---

## 7. Where each class lives

| Class | Header |
|---|---|
| `TileProgram`, `TensorOperand`, `TileOp`, `TileCoord`, `TileOpKind` | `include/sw/kpu/program/tile_program.hpp` |
| `TileKernelState`, `apply`, `kernel_*` | `include/sw/kpu/program/tile_kernels.hpp` |
| `TileProgramReference` | `include/sw/kpu/program/tile_program_reference.hpp` |
| `StreamProgram`, `StreamSignature`, `SpaceTimeMap`, `WavefrontTiming` | `include/sw/kpu/program/stream/stream_signature.hpp` |
| `TileDag`, `DagNode` | `include/sw/kpu/program/characterize/tile_dag.hpp` |
| `DeviceDescriptor`, `Topology` | `include/sw/kpu/program/characterize/device_model.hpp` |
| `ConcurrentTimingExecutor`, `Config`, compute specs | `include/sw/kpu/timing/concurrent_timing_executor.hpp` |
| `IProcess` and the four processes | `include/sw/kpu/timing/*_process.hpp` |
| `CreditPool`, `PartitionedCreditPool`, `TagCAM`, work queues | `include/sw/kpu/timing/{credit_pool,tag_cam,work_queue}.hpp` |
| `TileDescriptor`, `TilePayload`, `TileID`, `Cycle` | `include/sw/kpu/timing/tile_descriptor.hpp` |
| `IScheduleGenerator`, `ScheduleResult`, `ScheduleOperation` | `include/sw/kpu/timing/schedule/schedule_generator_interface.hpp` |
| `ScheduleExecutor`, functional executors | `include/sw/kpu/timing/schedule/` |
| `GraphCspExecutor`, `RunStats`, model specs | `include/sw/kpu/timing/graph/` |
| `KernelGraph`, `KernelNode`, `KernelEdge` | `include/sw/kpu/kernel_graph.hpp` |
| `TileTransactionExecutor`, `Placement` **[planned]** | `docs/plans/tile-transaction-executor.md` |

## 8. Deliberate omissions

- **The DMProgram/ISA path** (`BehavioralProgramExecutor`, `TransactionalProgramExecutor`,
  `ProgramSerializer`, `kpu-loader`) is frozen under ADR 0001 D6. `.kpubin` remains the
  driver-JIT output, so it will reappear in a future diagram as the JIT's product, not as a
  program the user hands the simulator.
- **The OFG flow executors** (`include/sw/kpu/models/dataflow/`) are omitted: their
  `execute_operation` is a no-op, so they compute nothing.
- **The driver JIT and placement pass** (#230 increment 3) are shown only as the source of
  `Placement` in §6; their internal classes do not exist yet.

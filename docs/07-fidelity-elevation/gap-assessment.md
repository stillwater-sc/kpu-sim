# Gap Assessment: Elevating BEHAVIORAL and TRANSACTIONAL Fidelity

## Executive Summary

The KPU simulator has three well-designed but disconnected subsystems:

1. **Schedule DSL + Compiler** — produces correct `DMProgram` instruction streams
   that describe tile movement through DRAM→L3→L2→L1→Compute→L1→L2→L3→DRAM
2. **Behavioral Tier** — computes correct numerical results (C = A × B) but
   uses its own ad-hoc orchestration that bypasses the ISA entirely
3. **Temporal/Dataflow Tier** — models timing and concurrency with real hardware
   components (DMA, BlockMover, Streamer) that hold actual data, but the
   ConcurrentExecutor is an analytical timing model that doesn't drive them

The three subsystems don't talk to each other. The DSL produces programs nobody
executes functionally. The behavioral tier produces correct answers without
following any program. The temporal components can move real data but nobody
asks them to.

```
Current state:

  Schedule DSL ──compile──▶ DMProgram ──▶ ConcurrentExecutor (timing only, no data)
                                    ╲
                                     ╲──▶ ProgramExecutor (temporal HW, but not wired to behavioral)

  BehavioralOrchestrator ──ad-hoc──▶ memcpy() ──▶ triple-loop matmul ──▶ correct C = A × B
       (ignores ISA)

  Temporal DMA/BM/STR ──hold real data──▶ nobody drives them from DMProgram functionally
```

**Goal state**: A single execution path where the DSL schedule drives actual data
through actual buffers, produces correct numerical results, AND generates accurate
timing:

```
  Schedule DSL ──compile──▶ DMProgram
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             BehavioralExec           TransactionalExec
             (real data,              (timing overlay,
              instant ops)             queue/bandwidth)
                    │                       │
                    ▼                       ▼
             Correct C = A × B       Cycle-accurate
             verified numerically    Gantt chart
```

---

## Component-by-Component Gap Analysis

### 1. DMA Engine

| Aspect | BEHAVIORAL | TEMPORAL | Gap |
|--------|-----------|---------|-----|
| Exists? | `BehavioralDMAEngine` | `DMAEngine` (temporal) | Both exist |
| Holds data? | Via `BehavioralMemoryModel` | Via `ExternalMemory` + `L3Tile` | Both hold real bytes |
| Accepts DMProgram? | No | Via `ProgramExecutor` | Behavioral has no program interface |
| Computes addresses? | Orchestrator computes | `ProgramExecutor.resolve_external_address()` | Need DSL compiler addresses to match |
| Multi-channel? | No (single engine) | Yes (selects by buffer slot) | Behavioral needs channel awareness |

**Gap**: `BehavioralDMAEngine` does Host↔L3 copies but is driven by the orchestrator's
ad-hoc logic, not by `DMInstruction` opcodes. It doesn't know about `DMA_LOAD_TILE`
or tile coordinates.

### 2. BlockMover

| Aspect | BEHAVIORAL | TEMPORAL | Gap |
|--------|-----------|---------|-----|
| Exists? | `BehavioralBlockMover` | `BlockMover` (temporal) | Both exist |
| L3→L2? | `l3_to_l2(tile, offset, bank, ...)` | `enqueue_block_transfer(...)` | Both work |
| Transpose? | Yes | Yes | Parity |
| Accepts DMInstruction? | No | Via `ProgramExecutor` | Same gap as DMA |
| Uses TileLayout? | No (orchestrator picks IDs) | No (ProgramExecutor picks IDs) | Neither uses it |

**Gap**: Same pattern — the behavioral BlockMover works but is driven ad-hoc.
The temporal BlockMover works and is driven by `ProgramExecutor`, but the
`ProgramExecutor` itself is wired to temporal models, not behavioral ones.

### 3. Streamer

| Aspect | BEHAVIORAL | TEMPORAL | Gap |
|--------|-----------|---------|-----|
| Exists? | **NO** | `Streamer` (temporal) | **CRITICAL GAP** |
| L2→L1? | Skipped (compute reads L2 directly) | Yes, row/col streaming | Behavioral tier has no L2→L1 model |
| L1→L2 drain? | Skipped | Yes | Same |

**Gap**: The behavioral tier has **no Streamer model at all**. The orchestrator
bypasses L1 entirely — it gets pointers into L2 banks and passes them to the
compute fabric. This is the biggest architectural shortcut in the behavioral tier.

For the ISA to be testable, we need a behavioral Streamer that:
- Accepts `STR_FEED_ROWS` / `STR_FEED_COLS` / `STR_DRAIN_OUTPUT` from DMProgram
- Copies tile data from L2 bank to L1 buffer (memcpy, instant)
- Signals compute when both A and B tiles are in L1
- Drains accumulator results from L1 back to L2

### 4. Compute Fabric

| Aspect | BEHAVIORAL | TEMPORAL | Gap |
|--------|-----------|---------|-----|
| Exists? | `BehavioralComputeFabric` | `ComputeFabric` (temporal) | Both exist |
| Computes values? | **YES** (full type dispatch) | **YES** (basic matmul) | Both compute |
| Operations | matmul, conv2d, softmax, layernorm, elementwise, pool2d | matmul only | Behavioral is richer |
| Triggered by? | Direct call from orchestrator | `update(cycle, l1_buffers)` | Different trigger models |
| VE operations? | `BehavioralVectorEngine` (bias + activation) | Via SFU class | Both exist |

**Gap**: The behavioral compute fabric is comprehensive but is called directly.
The temporal compute fabric reacts to L1 data arrival. For ISA testing, the
behavioral compute needs to be callable from a program-driven streamer, triggered
when stream_rows(A) + stream_cols(B) both complete for a tile.

### 5. Memory Hierarchy

| Aspect | BEHAVIORAL | TEMPORAL | Gap |
|--------|-----------|---------|-----|
| External Memory | `BehavioralMemoryModel` (Host region) | `ExternalMemory` (dense/sparse) | Both hold data |
| L3 | Regions in `BehavioralMemoryModel` | `L3Tile` (standalone, `vector<uint8_t>`) | Both hold data |
| L2 | Regions in `BehavioralMemoryModel` | `L2Bank` (standalone, `vector<uint8_t>`) | Both hold data |
| L1 | Regions in `BehavioralMemoryModel` | `L1Buffer` (standalone, `vector<uint8_t>`) | Both hold data |
| Address space | Encoded (region+id+offset in 64-bit) | Physical (decoded by AddressDecoder) | Incompatible addressing |

**Gap**: The behavioral memory uses encoded addresses
(`REGION_L3 | tile_id << 40 | offset`), while the temporal components use
physical byte arrays indexed directly. A behavioral program executor needs
to use the temporal-style memory components (which already hold real data)
OR map between the two address spaces.

### 6. Program Execution

| Aspect | ProgramExecutor | ConcurrentExecutor | BehavioralOrchestrator |
|--------|----------------|-------------------|----------------------|
| Input | DMProgram | DMProgram | Ad-hoc API calls |
| Drives HW? | Yes (temporal DMA/BM/STR) | No (analytical) | Yes (behavioral DMA/BM) |
| Moves data? | Yes (via temporal components) | No | Yes (via memcpy) |
| Computes? | Yes (temporal ComputeFabric) | No | Yes (behavioral ComputeFabric) |
| Timing? | Cycle-accurate | Statistical | None |
| Uses ISA? | **Yes** | **Yes** | **No** |

**Gap**: `ProgramExecutor` is the right architecture but uses temporal components.
We need a behavioral equivalent that interprets `DMProgram` using behavioral
components (instant operations, real data).

### 7. Dataflow / OFG Executors

| Aspect | Current State | Needed |
|--------|--------------|--------|
| OFG structure | Complete graph representation | Fine as-is |
| DMAFlowExecutor | Event sequencing, no data | Wire to behavioral DMA |
| BlockMoverFlowExecutor | Event sequencing, no data | Wire to behavioral BM |
| StreamerFlowExecutor | Event sequencing, no compute | Wire to behavioral STR + compute |
| Chrome Trace | Working visualization | Leverage for timing validation |

**Gap**: The OFG executors are correct dataflow schedulers but are "trains without
tracks." They fire events in the right order but don't call any actual hardware
component. For transactional fidelity, these need callbacks that invoke the
behavioral components for functional correctness while collecting timing statistics.

---

## Summary of Gaps

### Critical (blocks ISA/DSL validation)

| # | Gap | Impact |
|---|-----|--------|
| G1 | **No BehavioralStreamer** | Cannot test L2→L1→Compute→L1→L2 path from ISA |
| G2 | **No behavioral DMProgram executor** | Cannot functionally validate DSL schedules |
| G3 | **BehavioralOrchestrator bypasses ISA** | Functional tests don't exercise the instruction set |

### Important (blocks transactional timing validation)

| # | Gap | Impact |
|---|-----|--------|
| G4 | **OFG executors disconnected from components** | Cannot get timing from real execution |
| G5 | **No transactional orchestrator** | No coordinator for transactional tier |
| G6 | **Memory address space mismatch** | Behavioral encoded addresses ≠ temporal physical addresses |

### Nice-to-have (completeness)

| # | Gap | Impact |
|---|-----|--------|
| G7 | **KPUSimulator hardwired to temporal** | No fidelity switching in top-level API |
| G8 | **Conv2D/Softmax compute only in behavioral** | Temporal compute fabric is matmul-only |
| G9 | **No end-to-end test that validates DSL→values** | Cannot prove the system works |

---

## Implementation Plan

### Phase 1: Behavioral Program Executor (Closes G1, G2, G3, G9)

Build a `BehavioralProgramExecutor` that interprets a `DMProgram` using
behavioral components with real data. This is the minimum viable path to
testing that the DSL schedule produces correct matmul results.

**Architecture:**

```
                    DMProgram
                       │
              BehavioralProgramExecutor
              ┌────────┼────────────┐
              ▼        ▼            ▼
         BehavDMA   BehavBM    BehavSTR (NEW)
              │        │            │
              ▼        ▼            ▼
         ExternalMem  L3Tiles    L2Banks    L1Buffers
         (vector<u8>) (vector<u8>) (vector<u8>) (vector<u8>)
                                              │
                                    BehavComputeFabric
                                       (matmul)
```

**Key design**: Use the temporal-tier memory components (`ExternalMemory`,
`L3Tile`, `L2Bank`, `L1Buffer`) as the backing store. They already hold real
bytes. The behavioral program executor interprets instructions as instant
operations (memcpy between these components), not cycle-by-cycle.

#### Step 1.1: BehavioralStreamer

Create `include/sw/kpu/models/behavioral/datamovement/streamer.hpp`:

```cpp
class BehavioralStreamer {
public:
    // L2 → L1: Copy tile data from L2 bank to L1 buffer
    void feed_rows(L2Bank& src, uint32_t src_offset,
                   L1Buffer& dst, uint32_t dst_offset,
                   Size height, Size width, Size element_size);

    void feed_cols(L2Bank& src, uint32_t src_offset,
                   L1Buffer& dst, uint32_t dst_offset,
                   Size height, Size width, Size element_size);

    // L1 → L2: Drain accumulator results from L1 to L2
    void drain(L1Buffer& src, uint32_t src_offset,
               L2Bank& dst, uint32_t dst_offset,
               Size height, Size width, Size element_size);

    // Broadcast: replicate scalar to all positions
    void broadcast_row(L2Bank& src, uint32_t src_offset,
                       L1Buffer& dst, Size width, Size element_size);
};
```

Implementation: each method is a `memcpy` (or element-wise copy for column
streaming which transposes).

#### Step 1.2: BehavioralProgramExecutor

Create `include/sw/kpu/isa/behavioral_program_executor.hpp`:

```cpp
class BehavioralProgramExecutor {
public:
    struct HardwareContext {
        ExternalMemory& external_memory;
        std::vector<L3Tile>& l3_tiles;
        std::vector<L2Bank>& l2_banks;
        std::vector<L1Buffer>& l1_buffers;
    };

    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base);
    bool run();  // Execute to completion, returns true if HALT reached

    // Access results
    ExternalMemory& memory() { return hw_.external_memory; }

private:
    // Dispatch each opcode to behavioral components
    void dispatch_dma_load(const DMAOperands& ops);    // ExternalMem → L3
    void dispatch_dma_store(const DMAOperands& ops);   // L3 → ExternalMem
    void dispatch_bm_move(const BlockMoverOperands& ops);   // L3 → L2
    void dispatch_bm_writeback(const BlockMoverOperands& ops); // L2 → L3
    void dispatch_str_feed_rows(const StreamerOperands& ops); // L2 → L1
    void dispatch_str_feed_cols(const StreamerOperands& ops); // L2 → L1
    void dispatch_str_drain(const StreamerOperands& ops);     // L1 → L2
    void dispatch_compute(const TileCoord& tile);    // matmul on L1 data
    void dispatch_barrier();  // no-op in behavioral (all ops are instant)
};
```

The key insight: each `dispatch_*` method is a single `memcpy` between the
appropriate memory components. The whole program executes in microseconds.
Barriers are no-ops because all behavioral operations complete instantly.

**Compute trigger**: After `stream_rows(A)` and `stream_cols(B)` both execute
for a given tile iteration, the executor calls `BehavioralComputeFabric::submit_matmul()`
with pointers into the L1 buffers. The result accumulates in L1. After the K-loop,
`drain()` copies the accumulated result from L1 to L2.

#### Step 1.3: End-to-End Test

```cpp
void test_dsl_matmul_correctness() {
    // 1. Create A and B with known values
    ExternalMemory mem(16 * 1024 * 1024);  // 16 MB
    std::vector<float> a_data(64*64, 1.0f);
    std::vector<float> b_data(64*64, 1.0f);
    mem.write(0x0000, a_data.data(), a_data.size() * 4);
    mem.write(0x4000, b_data.data(), b_data.size() * 4);

    // 2. Build schedule via DSL
    auto sched = matmul_output_stationary(64, 64, 64, 16, 16, 16);
    DMProgram prog = compile_schedule(sched);

    // 3. Execute behaviorally
    std::vector<L3Tile> l3(4, L3Tile(128*1024));
    std::vector<L2Bank> l2(8, L2Bank(64*1024));
    std::vector<L1Buffer> l1(2, L1Buffer(16*1024));

    BehavioralProgramExecutor::HardwareContext hw{mem, l3, l2, l1};
    BehavioralProgramExecutor exec(hw);
    exec.load_program(prog, 0x0000, 0x4000, 0x8000);
    exec.run();

    // 4. Read result and verify
    std::vector<float> c_data(64*64);
    mem.read(0x8000, c_data.data(), c_data.size() * 4);

    for (auto& val : c_data) {
        assert(std::abs(val - 64.0f) < 1e-4f);  // ones × ones = K
    }
}
```

This test proves: DSL schedule → compile → execute → correct C = A × B.

### Phase 2: Transactional Timing Overlay (Closes G4, G5)

Once the behavioral executor works, add a transactional layer that wraps
each behavioral operation with timing accounting.

**Architecture:**

```
  TransactionalProgramExecutor
       │
       ├── BehavioralProgramExecutor  (for actual data)
       │
       └── TimingModel               (for cycle counts)
            ├── DMA: bytes / bandwidth = cycles
            ├── BM:  block_size / cache_line = cycles
            ├── STR: elements × stagger = cycles
            └── Compute: 2*M*N*K / throughput = cycles
```

The transactional executor calls the behavioral executor for each instruction
(to move real data and compute real values), then overlays timing by computing
the cycle cost of each operation using bandwidth/latency models. It maintains
a timeline (Gantt chart) of when each resource is busy.

```cpp
class TransactionalProgramExecutor {
    BehavioralProgramExecutor behavioral_;  // does the work
    ConcurrentExecutor timing_;              // models the timing

    bool run() {
        // For each instruction:
        //   1. behavioral_.dispatch(instr)  → moves actual data
        //   2. timing_.schedule(instr)      → computes when it would finish
        // At end: behavioral_ has correct C = A × B
        //         timing_ has accurate cycle count and Gantt chart
    }
};
```

This leverages the existing `ConcurrentExecutor` for timing and the new
`BehavioralProgramExecutor` for functional correctness.

#### Step 2.1: Wire OFG Executors to Behavioral Components

Add callback hooks in the OFG executors' `execute_operation()` methods:

```cpp
// In DMAFlowExecutor::execute_operation():
if (node.operation == Operation::LOAD) {
    if (behavioral_callback_) {
        behavioral_callback_(node);  // Actually moves data!
    }
}
```

This gives us credit-based dataflow scheduling with real data movement.

#### Step 2.2: Chrome Trace with Real Timing

The OFG executors already produce Chrome Trace events. With the transactional
timing overlay, these events now carry accurate cycle timestamps derived from
bandwidth models. The visualization shows real Gantt charts of tile movement
through the hierarchy.

### Phase 3: Integration and Validation (Closes G6, G7, G9)

#### Step 3.1: Unified Memory Interface

Create a thin adapter that lets the behavioral program executor use the
temporal memory components with a consistent interface:

```cpp
class TileMemory {
    ExternalMemory& ext;
    std::vector<L3Tile>& l3;
    std::vector<L2Bank>& l2;
    std::vector<L1Buffer>& l1;

    void copy_ext_to_l3(Address src, uint8_t l3_id, Address offset, Size bytes);
    void copy_l3_to_l2(uint8_t l3_id, Address src, uint8_t l2_id, Address dst, Size bytes);
    void copy_l2_to_l1(uint8_t l2_id, Address src, uint8_t l1_id, Address dst, Size bytes);
    // ... and reverse directions
};
```

This eliminates the address space mismatch (G6) by routing all operations
through explicit component IDs + offsets.

#### Step 3.2: Kernel Verification Test Suite

For each DSL kernel schedule, test:

```
test_matmul_functional:     DSL schedule → behavioral exec → C = A × B correct
test_matmul_timing:         DSL schedule → transactional exec → cycle count reasonable
test_matmul_tile_coverage:  All output tiles visited exactly once
test_matmul_no_data_hazard: No tile read before it's written

test_conv2d_functional:     DSL schedule → behavioral exec → conv correct
test_softmax_functional:    DSL schedule → behavioral exec → softmax correct
```

#### Step 3.3: Fidelity Switching in KPUSimulator

Add factory pattern to `KPUSimulator`:

```cpp
auto exec = simulator.create_executor(SimulationFidelity::BEHAVIORAL);
exec->load_program(prog, a_base, b_base, c_base);
exec->run();
// Same program, same result, different speed/detail
```

---

## Implementation Order and Dependencies

```
Phase 1 (ISA Functional Validation)
  ├── 1.1 BehavioralStreamer
  ├── 1.2 BehavioralProgramExecutor
  │        (depends on 1.1 + existing L3Tile, L2Bank, L1Buffer, ExternalMemory)
  └── 1.3 End-to-end matmul correctness test
           (depends on 1.2 + existing DSL + compile_schedule)

Phase 2 (Timing Overlay)
  ├── 2.1 TransactionalProgramExecutor
  │        (depends on 1.2 + existing ConcurrentExecutor)
  └── 2.2 OFG executor wiring
           (depends on 2.1)

Phase 3 (Integration)
  ├── 3.1 TileMemory adapter
  ├── 3.2 Full kernel test suite
  └── 3.3 Fidelity switching
```

Phase 1 is the critical path. It produces the first proof that:
**DSL schedule → DMProgram → execution → correct numerical result**.

Phase 2 adds the timing dimension without changing any functional behavior.

Phase 3 is cleanup and completeness.

---

## Files to Create/Modify

### Phase 1

| File | Action | Purpose |
|------|--------|---------|
| `include/sw/kpu/models/behavioral/datamovement/streamer.hpp` | CREATE | BehavioralStreamer |
| `src/models/behavioral/datamovement/streamer.cpp` | CREATE | Implementation |
| `include/sw/kpu/isa/behavioral_program_executor.hpp` | CREATE | Behavioral DMProgram interpreter |
| `src/software/isa/behavioral_program_executor.cpp` | CREATE | Implementation |
| `tests/isa/test_behavioral_program_executor.cpp` | CREATE | End-to-end matmul test |

### Phase 2

| File | Action | Purpose |
|------|--------|---------|
| `include/sw/kpu/isa/transactional_program_executor.hpp` | CREATE | Timing overlay |
| `src/software/isa/transactional_program_executor.cpp` | CREATE | Implementation |
| `include/sw/kpu/models/dataflow/dma_flow_executor.hpp` | MODIFY | Add behavioral callback |
| `include/sw/kpu/models/dataflow/block_mover_flow_executor.hpp` | MODIFY | Add behavioral callback |
| `include/sw/kpu/models/dataflow/streamer_flow_executor.hpp` | MODIFY | Add behavioral callback |

### Phase 3

| File | Action | Purpose |
|------|--------|---------|
| `include/sw/kpu/isa/tile_memory.hpp` | CREATE | Unified memory adapter |
| `tests/dsl/test_dsl_kernels_e2e.cpp` | CREATE | All 3 kernels verified |
| `include/sw/kpu/kpu_simulator.hpp` | MODIFY | Add fidelity factory |

---

## Success Criteria

Phase 1 is complete when:
- `test_behavioral_program_executor` creates A=ones, B=ones
- Compiles DSL matmul schedule to DMProgram
- Executes DMProgram through BehavioralProgramExecutor
- Reads C from ExternalMemory
- Verifies all elements equal K (the reduction dimension)
- Same test works for identity matrix (C = I × A = A)

Phase 2 is complete when:
- TransactionalProgramExecutor produces identical numerical results to Phase 1
- Additionally produces a cycle timeline showing DMA/BM/STR/Compute overlap
- Chrome Trace output visualizable in Perfetto

Phase 3 is complete when:
- All three DSL kernel schedules (matmul, conv2d, softmax) produce correct results
- `KPUSimulator` can switch between behavioral and transactional execution

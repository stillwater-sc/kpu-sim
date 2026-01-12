# Distributed Program Execution Model for KPU

## Executive Summary

The KPU simulator currently has a **behavioral reference model** that validates functional transformations work correctly, but lacks a **program-based execution model** where sequencers (DMA, BlockMover, Streamer) execute explicit instruction streams with synchronization. This document analyzes the gap and proposes an implementation plan.

---

## 1. Current State Analysis

### What Exists

| Component | ISA | Executor | Status |
|-----------|-----|----------|--------|
| **BlockMover** | ✅ Complete (27 opcodes) | ✅ `StatefulBlockMover` | Production-ready |
| **DMA Engine** | ❌ None | ❌ Imperative API only | Needs ISA |
| **Streamer** | ❌ None | ❌ Imperative API only | Needs ISA |
| **Trigger Network** | ✅ 16 channels | ✅ Integrated | Works |
| **Data Movement ISA** | ✅ High-level (24 opcodes) | ⚠️ Skeleton only | Needs executor |

### BlockMover ISA (Complete)

```
Location: include/sw/kpu/models/temporal/datamovement/block_mover_isa.hpp

Opcodes:
- Data Movement: PUSH_TO_L2, PULL_FROM_L2, SEND_{EAST,WEST,NORTH,SOUTH}, SEND_TO, RECEIVE
- Synchronization: WAIT_TRIGGER, EMIT_TRIGGER, BARRIER, FENCE, WAIT_DELIVERY
- Control Flow: LOOP_START, LOOP_END, JUMP, JUMP_IF_TRIGGER
- Configuration: SET_L2_BANK, SET_BUFFER
- Debug: TRACE_MARKER, HALT

Executor: StatefulBlockMover
- 8-state FSM: IDLE → WAITING_START → RUNNING → WAITING_* → HALTED
- Cycle-accurate simulation
- Nested loop support
- Trigger channel management
```

### What's Missing

1. **DMA Engine ISA** - No instruction set, only `enqueue_transfer()` API
2. **Streamer ISA** - No instruction set, only direct method calls
3. **Program Coordinator** - No mechanism to load/start multiple programs atomically
4. **Functional (non-cycle-accurate) Executor** - For fast validation

---

## 2. Gap Analysis

### Gap 1: DMA Engine Has No ISA

**Current State:**
```cpp
// Imperative API - not a program
dma_engine.enqueue_transfer(src_addr, dst_addr, size);
dma_engine.process_transfers(...);
```

**Target State:**
```cpp
// Program-based execution
DMAProgram program;
program.add(DMAOp::LOAD_TILE, {.src=host_addr, .dst=l3_addr, .size=tile_size});
program.add(DMAOp::EMIT_TRIGGER, {.channel=Trigger::A_TILE_READY});
program.add(DMAOp::WAIT_TRIGGER, {.channel=Trigger::COMPUTE_DONE});
program.add(DMAOp::STORE_TILE, {...});

StatefulDMAEngine dma(config);
dma.load_program(program);
while (!dma.is_idle()) dma.step(cycle++);
```

### Gap 2: Streamer Has No ISA

**Current State:**
```cpp
// Direct method calls
streamer.feed_rows_to_systolic(l2_bank, tile_addr, rows, cols);
streamer.drain_output(l2_bank, output_addr);
```

**Target State:**
```cpp
StreamerProgram program;
program.add(StreamerOp::WAIT_TRIGGER, {.channel=Trigger::A_TILE_READY});
program.add(StreamerOp::FEED_A_ROWS, {.l2_bank=0, .offset=0, .rows=64});
program.add(StreamerOp::FEED_B_COLS, {.l2_bank=1, .offset=0, .cols=64});
program.add(StreamerOp::WAIT_COMPUTE_DONE);
program.add(StreamerOp::DRAIN_C, {.l2_bank=2, .offset=0, .apply_activation=GELU});
program.add(StreamerOp::EMIT_TRIGGER, {.channel=Trigger::C_TILE_READY});

StatefulStreamer streamer(config);
streamer.load_program(program);
```

### Gap 3: No Distributed Program Coordinator

**Current State:**
- Each sequencer operates independently
- Manual coordination via trigger channels
- No atomic multi-program loading

**Target State:**
```cpp
DistributedProgram mlp_kernel;

// DMA programs (2 engines: A-loader, B-loader)
mlp_kernel.add_dma_program(0, dma_program_a);
mlp_kernel.add_dma_program(1, dma_program_b);

// BlockMover programs (16 L3 tiles)
for (int i = 0; i < 16; i++) {
    mlp_kernel.add_block_mover_program(i, bm_programs[i]);
}

// Streamer programs (16 compute tiles)
for (int i = 0; i < 16; i++) {
    mlp_kernel.add_streamer_program(i, str_programs[i]);
}

// Execute atomically
FunctionalExecutor executor(kpu_config);
executor.load(mlp_kernel);
executor.run_to_completion();
```

### Gap 4: No Kernel-to-Programs Compiler

**Current State:**
- `BlockMoverCompiler` generates BlockMover programs from dataflow graphs
- No equivalent for DMA or Streamer
- No unified compilation pipeline

**Target State:**
```cpp
KernelCompiler compiler(kpu_config);
Kernel mlp = Kernel::create_mlp(M, N, K, GELU, has_bias);

DistributedProgram programs = compiler.compile(mlp);
// Returns: DMA programs, BlockMover programs, Streamer programs
// All with proper synchronization points
```

### Gap 5: No Functional (Fast) Executor

**Current State:**
- `StatefulBlockMover` is cycle-accurate (slow for validation)
- Behavioral models don't execute programs

**Target State:**
```cpp
// Functional executor - validates correctness without cycle accuracy
FunctionalExecutor executor(kpu_config);
executor.load(distributed_program);
bool success = executor.run_to_completion();
auto stats = executor.get_stats();  // Operations executed, not cycles
```

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Kernel (MLP, MatMul, etc.)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KernelCompiler                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐            │
│  │ DMA Program Gen  │ │ BM Program Gen   │ │ Streamer Prog Gen│            │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘            │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DistributedProgram                                    │
│  ┌─────────────┐  ┌─────────────────────┐  ┌────────────────────┐          │
│  │ DMAProgram  │  │ BlockMoverProgram   │  │ StreamerProgram    │          │
│  │ (per engine)│  │ (per L3 tile)       │  │ (per compute tile) │          │
│  └─────────────┘  └─────────────────────┘  └────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│    FunctionalExecutor        │     │    CycleAccurateExecutor     │
│  (fast, for validation)      │     │  (detailed timing analysis)  │
│                              │     │                              │
│  ┌──────────────────────┐   │     │  ┌──────────────────────┐   │
│  │ FunctionalDMAEngine  │   │     │  │ StatefulDMAEngine    │   │
│  │ FunctionalBlockMover │   │     │  │ StatefulBlockMover   │   │
│  │ FunctionalStreamer   │   │     │  │ StatefulStreamer     │   │
│  └──────────────────────┘   │     │  └──────────────────────┘   │
└──────────────────────────────┘     └──────────────────────────────┘
```

---

## 4. ISA Definitions

### 4.1 DMA Engine ISA

```cpp
// include/sw/kpu/models/isa/dma_isa.hpp

enum class DMAOp : uint8_t {
    // Data Movement
    LOAD_TILE,          // External → L3: load tile from host memory
    STORE_TILE,         // L3 → External: store tile to host memory
    PREFETCH_TILE,      // Prefetch (non-blocking load)

    // Synchronization
    WAIT_TRIGGER,       // Wait for trigger channel(s)
    EMIT_TRIGGER,       // Emit trigger to channel(s)
    BARRIER,            // Wait for all pending transfers
    FENCE,              // Memory ordering fence

    // Control Flow
    LOOP_START,         // Begin loop (count in operand)
    LOOP_END,           // End loop (jump back if count > 0)
    JUMP,               // Unconditional jump
    JUMP_IF_TRIGGER,    // Conditional jump on trigger

    // Configuration
    SET_CHANNEL,        // Select DMA channel (for multi-channel engines)
    SET_BURST_SIZE,     // Configure burst size

    // Debug
    TRACE_MARKER,       // Emit trace event
    NOP,
    HALT
};

struct DMACommand {
    DMAOp op;

    // Transfer operands (for LOAD/STORE)
    uint64_t external_addr;     // Host memory address
    uint64_t l3_addr;           // L3 tile address
    uint32_t size;              // Bytes to transfer
    uint8_t l3_tile_id;         // Which L3 tile

    // 2D transfer support
    uint16_t rows, cols;
    uint32_t external_stride;
    uint32_t l3_stride;

    // Synchronization operands
    uint16_t trigger_mask;      // For WAIT/EMIT_TRIGGER
    uint16_t trigger_dest_mask; // Destination mask for EMIT

    // Control flow operands
    uint16_t loop_count;        // For LOOP_START
    int16_t jump_offset;        // For JUMP/LOOP_END

    // Metadata
    TileDescriptor tile;        // Tile identity for tracing
};

struct DMAProgram {
    std::vector<DMACommand> commands;
    uint8_t engine_id;
    std::string name;
};
```

### 4.2 Streamer ISA

```cpp
// include/sw/kpu/models/isa/streamer_isa.hpp

enum class StreamerOp : uint8_t {
    // L2 → L1 Feeding
    FEED_A_ROWS,        // Feed A matrix rows to systolic array west edge
    FEED_B_COLS,        // Feed B matrix columns to systolic array north edge
    FEED_BIAS,          // Feed bias vector for MLP fusion

    // L1 → L2 Draining
    DRAIN_C,            // Drain C accumulator to L2
    DRAIN_C_WITH_ACT,   // Drain with fused activation
    DRAIN_C_WITH_BIAS_ACT, // Drain with fused bias + activation (MLP)

    // Synchronization
    WAIT_TRIGGER,       // Wait for trigger(s)
    EMIT_TRIGGER,       // Emit trigger(s)
    WAIT_L2_READY,      // Wait for L2 bank data ready
    WAIT_COMPUTE_DONE,  // Wait for systolic array computation complete
    BARRIER,            // Wait for all pending streams

    // Control Flow
    LOOP_START,
    LOOP_END,
    JUMP,

    // Configuration
    SET_L2_BANK,        // Select source/dest L2 bank
    SET_ACTIVATION,     // Configure activation function
    SET_VECTOR_ENGINE,  // Configure vector engine parameters

    // Debug
    TRACE_MARKER,
    NOP,
    HALT
};

struct StreamerCommand {
    StreamerOp op;

    // L2 operands
    uint8_t l2_bank;
    uint32_t l2_offset;

    // Dimensions
    uint16_t rows, cols;
    uint32_t stride;

    // Activation (for DRAIN_WITH_*)
    ActivationType activation;
    bool has_bias;
    uint32_t bias_offset;

    // Synchronization
    uint16_t trigger_mask;
    uint16_t trigger_dest_mask;

    // Control flow
    uint16_t loop_count;
    int16_t jump_offset;

    // Tile metadata
    TileDescriptor tile;
};

struct StreamerProgram {
    std::vector<StreamerCommand> commands;
    uint8_t streamer_id;
    std::string name;
};
```

### 4.3 Distributed Program Bundle

```cpp
// include/sw/kpu/models/isa/distributed_program.hpp

struct DistributedProgram {
    // Programs for each sequencer type
    std::vector<DMAProgram> dma_programs;           // One per DMA engine
    std::vector<BlockMoverProgram> bm_programs;     // One per L3 tile
    std::vector<StreamerProgram> str_programs;      // One per compute tile

    // Global synchronization points
    struct SyncPoint {
        std::string name;
        std::set<std::pair<SequencerType, uint8_t>> participants;
    };
    std::vector<SyncPoint> sync_points;

    // Metadata
    std::string kernel_name;
    KernelOpType op_type;
    uint32_t m, n, k;  // Problem dimensions

    // Validation
    bool validate() const;
    std::string summary() const;
};
```

---

## 5. Synchronization Model

### 5.1 Trigger Channels (Existing)

```
Channel 0: DMA_LOAD_DONE       - DMA signals tile loaded to L3
Channel 1: A_TILE_READY        - BlockMover signals A tile in L2
Channel 2: B_TILE_READY        - BlockMover signals B tile in L2
Channel 3: C_TILE_READY        - Streamer signals C tile ready for writeback
Channel 4: COMPUTE_START       - Streamer signals systolic can begin
Channel 5: COMPUTE_DONE        - Systolic signals computation complete
Channel 6: DRAIN_DONE          - Streamer signals drain complete
Channel 7: TRANSFER_DONE       - L3↔L3 transfer complete
Channels 8-15: USER_0..USER_7  - Application-defined
```

### 5.2 Synchronization Patterns

**Pattern 1: DMA → BlockMover Handoff**
```
DMA Engine:                    BlockMover[i]:
  LOAD_TILE (A to L3[i])         WAIT_TRIGGER(DMA_LOAD_DONE)
  EMIT_TRIGGER(DMA_LOAD_DONE)    PUSH_TO_L2
                                 EMIT_TRIGGER(A_TILE_READY)
```

**Pattern 2: BlockMover → Streamer Handoff**
```
BlockMover[i]:                 Streamer[i]:
  PUSH_TO_L2 (A)                 WAIT_TRIGGER(A_TILE_READY & B_TILE_READY)
  EMIT_TRIGGER(A_TILE_READY)     FEED_A_ROWS
  PUSH_TO_L2 (B)                 FEED_B_COLS
  EMIT_TRIGGER(B_TILE_READY)     WAIT_COMPUTE_DONE
                                 DRAIN_C_WITH_ACT
                                 EMIT_TRIGGER(C_TILE_READY)
```

**Pattern 3: Systolic Wavefront (L3 → L3 Forwarding)**
```
BlockMover[0,0]:               BlockMover[0,1]:
  RECEIVE (A from DMA)           WAIT_TRIGGER(A_TILE_READY, from [0,0])
  PUSH_TO_L2                     RECEIVE (A forwarded from [0,0])
  SEND_EAST (A)                  PUSH_TO_L2
  EMIT_TRIGGER(A_TILE_READY)     SEND_EAST (A)
                                 EMIT_TRIGGER(A_TILE_READY)
```

---

## 6. Implementation Plan

### Phase 1: DMA Engine ISA and Executor

**Files to Create:**
```
include/sw/kpu/models/isa/dma_isa.hpp           # DMA ISA definition
include/sw/kpu/models/functional/dma_engine.hpp # Functional executor
src/models/functional/datamovement/dma_engine.cpp
```

**Deliverables:**
1. `DMAOp` enum with 15 opcodes
2. `DMACommand` struct with all operands
3. `DMAProgram` container with validation
4. `FunctionalDMAEngine` - program executor without cycle accuracy
5. Unit tests for DMA program execution

**Effort:** Medium (existing DMAEngine as reference)

### Phase 2: Streamer ISA and Executor

**Files to Create:**
```
include/sw/kpu/models/isa/streamer_isa.hpp
include/sw/kpu/models/functional/streamer.hpp
src/models/functional/datamovement/streamer.cpp
```

**Deliverables:**
1. `StreamerOp` enum with 15 opcodes
2. `StreamerCommand` struct with feed/drain operands
3. `StreamerProgram` container
4. `FunctionalStreamer` - executor with compute fabric integration
5. Unit tests

**Effort:** Medium

### Phase 3: Functional BlockMover (Simplified from Temporal)

**Files to Create:**
```
include/sw/kpu/models/functional/block_mover.hpp
src/models/functional/datamovement/block_mover.cpp
```

**Deliverables:**
1. Reuse existing `BlockMoverProgram` and `BlockMoverCommand`
2. `FunctionalBlockMover` - non-cycle-accurate executor
3. Executes same ISA as `StatefulBlockMover` but faster

**Effort:** Small (adapts existing code)

### Phase 4: Distributed Program Bundle and Coordinator

**Files to Create:**
```
include/sw/kpu/models/isa/distributed_program.hpp
include/sw/kpu/models/functional/program_coordinator.hpp
src/models/functional/program_coordinator.cpp
```

**Deliverables:**
1. `DistributedProgram` - bundles all sequencer programs
2. `ProgramCoordinator` - loads and steps all sequencers together
3. Global trigger network integration
4. Deadlock detection (optional)

**Effort:** Medium

### Phase 5: Kernel-to-Programs Compiler

**Files to Create:**
```
include/sw/kpu/compiler/distributed_compiler.hpp
src/compiler/distributed_compiler.cpp
```

**Deliverables:**
1. `DistributedCompiler` - generates `DistributedProgram` from `Kernel`
2. DMA program generation for input/output staging
3. Streamer program generation with activation fusion
4. Integration with existing `BlockMoverCompiler`
5. Output-stationary matmul program generation
6. MLP layer program generation

**Effort:** Large

### Phase 6: Examples and Validation

**Files to Create:**
```
examples/functional/matmul_program.cpp      # MatMul as distributed program
examples/functional/mlp_program.cpp         # MLP as distributed program
tests/functional/test_dma_isa.cpp
tests/functional/test_streamer_isa.cpp
tests/functional/test_distributed_execution.cpp
```

**Deliverables:**
1. Working matmul executing as DMA + BM + Streamer programs
2. Working MLP with bias + activation fusion
3. Comparison against behavioral reference

**Effort:** Medium

---

## 7. File Structure

```
include/sw/kpu/models/
├── isa/                              # ISA definitions (NEW)
│   ├── dma_isa.hpp                   # DMA instruction set
│   ├── streamer_isa.hpp              # Streamer instruction set
│   ├── distributed_program.hpp       # Multi-program bundle
│   └── trigger_protocol.hpp          # Trigger channel conventions
├── functional/                       # Functional executors (NEW)
│   ├── dma_engine.hpp                # Program-based DMA
│   ├── block_mover.hpp               # Program-based BlockMover
│   ├── streamer.hpp                  # Program-based Streamer
│   ├── program_coordinator.hpp       # Multi-sequencer coordinator
│   └── compute_fabric.hpp            # (exists - reuse behavioral)

src/models/functional/
├── datamovement/
│   ├── dma_engine.cpp
│   ├── block_mover.cpp
│   └── streamer.cpp
├── compute/
│   └── (reuse behavioral)
└── program_coordinator.cpp

include/sw/kpu/compiler/
└── distributed_compiler.hpp

src/compiler/
└── distributed_compiler.cpp
```

---

## 8. Example: MLP Layer as Distributed Program

```cpp
// Conceptual example of MLP: Y = activation(X @ W + bias)

// DMA Engine 0: Load X tiles to L3 west edge
DMAProgram dma_x;
for (int ti = 0; ti < M_tiles; ti++) {
    dma_x.add(LOAD_TILE, {.src=x_addr + ti*tile_bytes, .dst_l3=ti % 4, ...});
    dma_x.add(EMIT_TRIGGER, {.channel=DMA_LOAD_DONE, .dest_mask=1<<(ti%4)});
}
dma_x.add(HALT);

// DMA Engine 1: Load W tiles to L3 north edge
DMAProgram dma_w;
for (int tj = 0; tj < N_tiles; tj++) {
    dma_w.add(LOAD_TILE, {.src=w_addr + tj*tile_bytes, .dst_l3=tj, ...});
    dma_w.add(EMIT_TRIGGER, {.channel=DMA_LOAD_DONE, .dest_mask=1<<tj});
}
dma_w.add(HALT);

// BlockMover[i]: Forward tiles through mesh + push to L2
BlockMoverProgram bm[16];
for (int i = 0; i < 16; i++) {
    bm[i].add(WAIT_TRIGGER, {.mask=DMA_LOAD_DONE});
    bm[i].add(RECEIVE);           // From DMA or neighbor
    bm[i].add(PUSH_TO_L2);
    bm[i].add(EMIT_TRIGGER, {.channel=A_TILE_READY});
    // ... similar for B ...
    bm[i].add(WAIT_TRIGGER, {.mask=C_TILE_READY});
    bm[i].add(PULL_FROM_L2);      // Writeback
    bm[i].add(HALT);
}

// Streamer[i]: Feed systolic, drain with activation
StreamerProgram str[16];
for (int i = 0; i < 16; i++) {
    str[i].add(LOOP_START, {.count=K_tiles});
      str[i].add(WAIT_TRIGGER, {.mask=A_TILE_READY | B_TILE_READY});
      str[i].add(FEED_A_ROWS, {...});
      str[i].add(FEED_B_COLS, {...});
      str[i].add(WAIT_COMPUTE_DONE);
    str[i].add(LOOP_END);
    str[i].add(DRAIN_C_WITH_BIAS_ACT, {.activation=GELU, .bias_offset=...});
    str[i].add(EMIT_TRIGGER, {.channel=C_TILE_READY});
    str[i].add(HALT);
}

// Bundle and execute
DistributedProgram mlp;
mlp.add_dma_program(0, dma_x);
mlp.add_dma_program(1, dma_w);
for (int i = 0; i < 16; i++) {
    mlp.add_bm_program(i, bm[i]);
    mlp.add_str_program(i, str[i]);
}

FunctionalExecutor executor(kpu_config);
executor.load(mlp);
executor.run_to_completion();
```

---

## 9. Success Criteria

1. **ISA Completeness**: DMA and Streamer ISAs cover all operations needed for MatMul and MLP
2. **Functional Validation**: Program-based execution produces same results as behavioral model
3. **Synchronization Correctness**: Triggers properly coordinate DMA → BM → Streamer → Writeback
4. **MLP End-to-End**: Can express and execute `Y = GELU(X @ W + bias)` as distributed program
5. **Performance**: Functional executor runs 10x faster than cycle-accurate for validation

---

## 10. Implementation Order Recommendation

| Phase | Component | Dependencies | Effort | Priority |
|-------|-----------|--------------|--------|----------|
| 1 | DMA ISA + Executor | None | Medium | **High** |
| 2 | Streamer ISA + Executor | Phase 1 | Medium | **High** |
| 3 | Functional BlockMover | None | Small | Medium |
| 4 | Program Coordinator | Phases 1-3 | Medium | **High** |
| 5 | Distributed Compiler | Phase 4 | Large | Medium |
| 6 | Examples + Validation | Phase 5 | Medium | High |

**Recommended Start:** Phase 1 (DMA ISA) in parallel with Phase 3 (Functional BlockMover)

---

## 11. Open Questions for Review

1. **Trigger Granularity**: Should triggers be per-tile or per-operation?
2. **Error Handling**: How should sequencers report errors to coordinator?
3. **Dynamic Dispatch**: Should programs support runtime-determined jump targets?
4. **Multi-Kernel**: Should coordinator support pipelining multiple kernels?
5. **Naming**: "Functional" vs "Behavioral" vs "Programmatic" executor?

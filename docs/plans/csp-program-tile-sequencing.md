# The CSP Program as the Center of Tile Sequencing

**Date:** 2026-10-09
**Status:** Q1-Q6 decided 2026-10-09 (all as recommended; §7); steps 1-2 done; step 2b (`kpu-run`) next
**Supersedes:** step 3 and later of `docs/plans/system-schedule-debugger.md`. Steps 1-2 of that
plan (round-robin arbitration, the compute fabric, `kpu_s1`) stand: they are resource models
inside an interpreter, valid whatever program it runs. Step 3, the dispatcher (#340, on hold),
is withdrawn. Its finding is kept (§1.2).
**Related:**
- ADR 0002 (the CSP program is derived from the DFP; every level interprets it; §4 and §5.1 name
  this gap).
- #281 (`CspProgram`), #283 (L-T2 and L-CA run the program), #230 (placement).

## 1. Problem Statement

### 1.1 Sequencing enters the machine from outside the program

ADR 0002 makes the CSP program the thing every level executes. It is processes, channels, and
the ordered tile sequence each process moves, derived from the Domain Flow Program. The
cycle-level executor never sees one:

| What the ADR says | What the code does |
|---|---|
| The CSP program is derived from the DFP | `derive_matmul_tile_program` / `derive_lu_tile_program` hand-code loops into L0. Nothing lowers a DFG, `KernelGraph` or `ComputationalGraph` into a program. |
| The CSP program has explicit processes and channels | `TileProgram` is one flat `std::vector<TileOp>` with logical ports. There is no `Process` or `Channel` type anywhere in `program/`. `CspProgram` (#281) "does not exist" (`virtual_platform.hpp:25`). |
| Every level interprets the same program | L-B and L-T1 run L0. L-CA (`ConcurrentTimingExecutor`) runs a `ScheduleResult` from separate C++ generators (`MatMulScheduleGenerator`, ...), a second source of sequencing. `kpu-run` refuses L-CA for that reason (`execution_level.hpp:99`). |
| Reuse is the program's decision | L-T1 holds a tile until its last user (`unfinished_users`). L-CA's generators re-emit a LOAD for every use, and reuse happens only when the L3 tag CAM finds the tile still resident. |

### 1.2 What the dispatcher measured, and why it was the wrong fix

On `kpu_s1`, the generator's matmul 256³ with 32³ tiles has 1,024 LOADs of 128 tiles.
- **Released all at once,** all 128 fit the L3, every repeat is a CAM hit, and DRAM sees 128
  loads.
- **Paced by an outside dispatcher,** credits returned between uses: P = 1 gave 768 loads and a
  3.3 times longer run.

The reuse, the residency and the lookahead were never the program's. They fell out of how a flat
list met a cache-like tag match. Pacing that list from outside adds a third source of sequencing.
The fix is to make the CSP program the only one.

## 2. Architecture

### 2.1 The hypothesis: KPU tile sequencing is an MPI decomposition

**Level 1: the machine as one flat L3 and one fat compute fabric.**
- The program addresses one L3 and one compute fabric (CF), as an MPI program addresses one
  rank.
- Its outer loops walk the tile index space.
- Its **inner loops collapse into a tile function call.** A compute tile is a multi-dimensional
  execution engine, a whole matmul engine for example. The program never sequences matrix
  elements; it calls `gemm(A[i,k], B[k,j], C[i,j])` as a rank calls `dgemm` on its local block.
- L0 today is very nearly this program. It is flat, has logical ports and no engines, and its
  ops are tile functions (`MatMulAccum`, `LuDiagFactor`, `TrsmLowerLeft`, ...).

**Level 2: divide and conquer over distributed L3 and CF tiles.**
- When the machine has many L3 tiles and compute tiles, the level-1 program is partitioned as a
  distributed-memory program is:
  - the outer loops split into the loops over partitions (who owns which block of the index
    space) and the loops within a partition (the level-1 program on the owner's block);
  - operands that cross partitions move by explicit communication.
- For matmul on a 2D grid of compute tiles this is **SUMMA**: each tile owns a block of C, and
  the A row-panels and B column-panels are broadcast along grid rows and columns, each step a
  local tile-function call.
- For LU it is the 2D block-cyclic decomposition of ScaLAPACK.
- The NoC carries the communication: L3 to L3 transfers and multicasts.

**Below L3: the tile function's own decomposition.**
- The L2 and L1 CSP processes (BlockMovers and Streamers) take a tile apart into the row and
  column segments that become the compute fabric's edge streams.
- This is the inside of the tile function call, and it is mechanical given the function's
  space-time map. The L1 `StreamProgram` (stream signatures, wavefront timing) already describes
  it per tile shape.

### 2.2 The layers, and what executes them

```
  Domain Flow Program (domain_flow)      -- authored; the source of truth
          │  derive (today: derive_* hand-coded loops; later: lower_to_csp from the DFG, #281)
          ▼
  L0 TileProgram                         -- level 1 projected flat: ordered tile functions
          │  lower (§3.2): residency, processes, channels, decided against a machine
          ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ CspProgram  (#281)                                                           │
  │                                                                              │
  │   level 1   one logical DMA ─► L3 store ─► BlockMover ─► Streamer ─► CF      │
  │             ordered actions per process; channel capacities; residency       │
  │             (each tile: loaded once / reloaded where capacity forces it)     │
  │                                                                              │
  │   level 2   Distribution: partitions = (L3 tile, CF tile) owners;            │
  │             per-partition level-1 programs + communication actions (NoC)     │
  │                                                                              │
  │   below L3  per tile function: its L2/L1 segment and stream decomposition    │
  │             (StreamProgram), referenced, not re-sequenced                    │
  └───────────────┬──────────────────────┬───────────────────────┬───────────────┘
                  │                      │                       │
                  ▼                      ▼                       ▼
            L-B behavioral        L-T1 block-sequential    L-CA cycle-level
            actions atomic        tile moves on lanes      ConcurrentTimingExecutor:
            (values)              under capacity           each process's action list
                                  (values)                 fed to that process; channel
                                                           capacity = credit pool; bursts
                                                           begin when the DMA process's
                                                           action begins (values)
```

CREDITS UP, DATA DOWN holds by construction:
- a process action begins only when its input channel holds the tile (tag match) and its output
  channel has a credit;
- a consumer's last read of a tile returns the credit.

No list is released from outside, and no dispatcher exists.

## 3. Design

### 3.1 The IR: `CspProgram`

```cpp
namespace sw::kpu::program::csp {

enum class ProcessKind : uint8_t { Dma, BlockMover, Streamer, Compute };

// One step of one process. A movement action moves a tile from one channel to another; a call
// runs a tile function on tiles its input channels hold, writing its output tile.
struct Action {
    enum class Kind : uint8_t { Load, Store, Move, Feed, Drain, Call, Send, Receive };
    Kind kind;
    TileCoord tile;                 // the tile moved, or the call's output
    std::vector<TileCoord> inputs;  // Call: the tile function's operands
    TileOpKind function{};          // Call: MatMulAccum, LuDiagFactor, ...
    ChannelId from, to;             // movement: source and destination channels
    std::size_t l0_op = kNone;      // the L0 op it implements (traceability, values)
};

struct Process {
    ProcessKind kind;
    std::string name;               // "dma", "bm", "str.row", "cf"   (level 1: one of each)
    std::vector<Action> actions;    // in order: the process's tile sequence
};

// A channel is a bounded buffer between processes: its capacity is its credit count. The L3
// store is a channel whose entries carry a consumer count: a tile's credit returns when its
// last consumer in the program has read it -- residency is the program's, not a cache's.
struct Channel {
    ChannelId id;
    std::string name;               // "dram", "l3", "l2", "l1", "cf.in", ...
    std::size_t capacity;           // in tiles; derived by the lowering (Q2)
};

struct Partition {                  // level 2: one owner of a block of the index space
    std::size_t l3_tile, cf_tile;
    std::vector<Process> processes; // the level-1 program on this block, plus Send/Receive
};

class CspProgram {
public:
    const TileProgram& source() const;            // the L0 it was lowered from
    std::vector<Process> processes;               // level 1
    std::vector<Channel> channels;
    std::vector<Partition> partitions;            // level 2 (empty = level 1 on one site)
    ReuseReport reuse() const;                    // loads per tile, DRAM bytes vs minimum
    std::string disassemble() const;              // ADR 0002 §6: a disassembler, no syntax
};

}  // namespace sw::kpu::program::csp
```

### 3.2 Lowering L0 to level 1: residency is decided here

WRONG (the generators today): a LOAD is emitted for every use, and the hardware's tag match
decides at run time whether it costs a DRAM read.

CORRECT: the lowering walks L0 in order with the target's L3 capacity, and decides each tile's
residency.
- **A tile is loaded** when it is first needed and not resident.
- **It stays resident until its last use,** unless capacity forces it out. In that case the
  victim is the resident tile whose next use is furthest away (Belady). That is the minimum
  number of reloads possible *for this order*.
- **Every Feed of a resident tile is a Move from L3, not a Load.**
- **The L3 channel's consumer count for each residency** is the number of reads the program
  makes of it before it is freed.

```cpp
struct LoweringOptions {
    std::size_t l3_capacity_tiles;      // from the device spec (a machine parameter)
    // How the program walks the index space is the schedule's: L0's order today; a reuse-aware
    // order (blocked panels) is the next plan, and lowers through this same function.
};
CspProgram lower(const TileProgram& l0, const LoweringOptions&);
```

On `kpu_s1` (128 slots), matmul 256³ with 32³ tiles then lowers to **128 Loads**, each tile once,
by the program's decision. With a smaller L3 the reloads are counted in the program itself, before
any simulation runs.

### 3.3 Level 2: the MPI-style distribution

```cpp
struct Distribution {
    // Owner-computes over the output tiles: a 2D block (or block-cyclic) map of C's tile grid
    // onto the compute-tile grid; each owner's L3 tile is its home (placement, #230).
    enum class Map : uint8_t { Block2D, BlockCyclic2D };
    Map map;
    std::size_t grid_rows, grid_cols;   // from the device's array layout
};
CspProgram distribute(const CspProgram& level1, const Distribution&, const ArrayLayout&);
```

- **Partitions:** each partition's processes are the level-1 program restricted to the C tiles
  it owns.
- **Communication:** an operand tile another partition needs travels by Send/Receive actions on
  the NoC.
  - SUMMA's panel broadcast for matmul: an A panel along a grid row, a B panel along a column.
  - The pivot and panel broadcasts for LU.
- **Binding:** partitions bind to (L3 tile, CF tile) pairs, with DMA engines by the floorplan's
  port attachment.

### 3.4 Interpreting the program: L-CA without a `ScheduleResult`

WRONG: a generator emits a global op list; a loop (or a dispatcher) hands it to the executor.

CORRECT: a `CspDriver` gives each executor process its own action list.
- DMA engine e gets exactly its Loads and Stores, in program order. The BlockMover gets its
  Moves, the streamers their Feeds and Drains, and each compute tile its Calls.
- Channel capacities become credit pools.

The executor's existing semantics then *are* CSP:
- a process takes its next action when its input is present (tag CAM) and its output has credit;
- a Load decomposes into bursts as it begins, through the DMA window and the round-robin arbiter;
- a Call runs on its compute tile for the tile function's MAC time.

Values flow as they do today: payloads, `schedule_matmul_compute`, and C read back from DRAM.

```cpp
class CspDriver {
public:
    CspDriver(ConcurrentTimingExecutor&, const CspProgram&);
    bool run();                         // until every process has finished its actions
};
```

- **`ScheduleResult` and its generators stay** for the tests that use them, but no new work
  drives L-CA through them.
- `kpu-run --level cycle-accurate` runs (#283, the L-CA half). The same program then runs at
  L-B, L-T1 and L-CA, and `kpu-run` compares their values, as ADR 0001 D5 requires.

### 3.5 What the system-schedule view becomes

The `.sflow` record (system-schedule plan §3.4) records the CspProgram's execution:
- each process's actions with their begin and end;
- each channel's occupancy against its capacity;
- each residency, with its consumer count;
- each action linked to its bursts and DRAM commands.

The schedule view is then the program running. A stall reads "compute 17 waited for Load
A[2,0] on the dma process, which waited on the arbiter".

## 4. Implementation Steps

Each step is one PR and ends green.

1. **The IR and level-1 lowering.**
   - Files: `program/csp/csp_program.hpp`, `csp/lower.hpp`, the disassembler.
   - Matmul and LU lower from L0 with Belady residency against the L3 capacity.
   - An L-B interpreter runs the CspProgram's actions atomically.
   - Tests:
     - L-B over the CspProgram is bit-identical to the L0 reference;
     - matmul 256³/32³ against 128 slots lowers to 128 Loads, and against fewer slots to exactly
       Belady's reload count (checked by brute force on small cases);
     - every Feed of a resident tile is a Move;
     - consumer counts match the program's reads.
   - (Done.)
   - **As built,** `include/sw/kpu/program/csp/`:
     - **`csp_program.hpp`, the IR.**
       - Four level-1 processes: dma, bm, str and cf.
       - Four channels: dram, l3 (capacity = the L3's slots), l2 and cf (transit; level 1 makes
         no residency decision there).
       - Every action is in program order, and each process holds its own ordered action list.
       - Action kinds: Load, Store, Move, Writeback, Feed, Drain, Call (a whole L0 tile
         function), and Release (the L3 credit returns).
       - Residencies record their opening action, consumer count, release and dirty state.
       - Also `reuse()` and the disassembler.
     - **`lower.hpp`, the lowering** from L0 on the flat machine, in two styles. Explicit
       (matmul) follows L0's Feeds and Drains, and the fabric accumulates from zero. Implicit
       (LU) brings every kernel's operands in place and writes its outputs back, dirty.
       - **Residency:** a tile stays until its last use. When the L3 is full, Belady evicts
         (furthest next use; stored first if dirty).
       - **Refusal:** an op whose working set exceeds the L3 is refused by name.
     - **`behavioral.hpp`, L-B over the program.**
       - Every action is atomic and moves real tile values between channels.
       - Calls run the shared L0 kernels.
       - A read of a tile that is not where the program says it is (a released one, for
         example) is an error.
   - **Measured:**
     - **Matmul 256³/32³ against S1's 128 slots:** 128 Loads (each tile once, by the program's
       decision), 1,024 Moves, 64 Stores, and 8 consumers per input residency. The
       generator's 1,024 LOADs had this reuse only by tag-CAM accident.
     - **Matmul 48³/16³, L3 capacity 1 to 18:**

       | Slots | 1 | 2 | 3 | 4 | 6 | 10 | 18 |
       |---|---|---|---|---|---|---|---|
       | Loads | 54 | 48 | 42 | 36 | 30 | 21 | 18 |

       Each equals the brute-force optimum, by dynamic programming over (position, resident
       set).
     - **LU 64²/16²:** 16 loads unbounded; 51 with 3 slots, the smallest working set a
       trailing update allows.
   - **Values:** L-B over the CSP program is bit-identical to the L0 reference for matmul and
     LU, at unbounded and tight capacities.
2. **L-CA from the program, one site** (`kpu_s1`).
   - `CspDriver` feeds `ConcurrentTimingExecutor` per process.
   - `kpu-run --level cycle-accurate` is enabled for single-site machines.
   - Tests:
     - values are bit-identical across L-B, L-T1 and L-CA;
     - each DMA engine's issue order is the program's;
     - DRAM loads equal the program's Loads (128 on S1).
   - Measured: the S1 makespan against the generator path's 36,668 cycles.
   - (Done, except `kpu-run`, which is split out as step 2b below.)
   - **As built:** `timing/csp_driver.hpp`, `CspDriver`.
     - **Actions to executor queues:** each program action goes, in program order, to its
       executor process's queue.
     - **The dma process takes credits in order:** a Load is handed over only once the one
       before it holds its L3 credit, so credits go in program order (no hold-and-wait).
     - **Engine binding:** which engine (a lane of the dma process) is the executor's. A fixed
       round-robin binding starved stores behind loads on the whole-tile path, where an engine
       posts one tile at a time.
     - **Residency rides on the descriptor** (`TileDescriptor::l3_consumers`). The arrival seeds
       the L3 entry with the program's consumer count. A program Load never hits the tag CAM:
       a copy left by the previous residency is waited out, so DRAM reads equal the program's
       Loads.
     - **Calls are accumulating matmul computes** (`MatMulComputeSpec::accumulate`). One
       k-slice per call; C stays in the fabric; each call is ordered after the previous one on
       its tile. The result reaches DRAIN only after the last call scheduled on the tile, and
       the chain fills on its first call and drains on its last.
     - **Refusals, by name** (Q2): a program lowered for more L3 than the machine holds (or an
       unbounded one), and a machine with several L3 tiles (level 2's).
     - **Scope:** explicit (matmul) programs. LU at L-CA, which needs its kernels as functional
       computes and in-place residencies, is the next increment.
     - `ConcurrentTimingExecutor::livelock_detected()` is the progress check `run()` makes,
       for callers that step the executor themselves (first written for #340).
   - **Tests** (`test_csp_driver`):
     - **S1 256³/32³:** C is bit-identical to the L0 reference and to L-B; DRAM loads = the
       program's Loads = 128; 64 stores; every L3 credit returns.
     - **Tighter programs** (L3 4, 8, 16 slots, against 128^3): DRAM reads equal the program's
       Loads, reloads included, and values hold.
     - **Order:** the dma process's credits are taken in program order.
     - **Refusals:** an oversized or unbounded program, and a multi-L3 machine.
   - **Measured** on S1, matmul 256³/32³:

     | | CSP program | generator |
     |---|---|---|
     | cycles (whole-tile DMA) | 43,968 | 36,668 |
     | cycles (DMA window 32) | 38,089 | 35,813 |
     | first compute | **1,465** | 10,077 |
     | DRAM loads | 128 | 128 |
     | compute busy | 6,144 | 6,144 |

     - **The program gets compute going about 7 times sooner.** Its calls take one k-slice at
       a time instead of waiting for all eight.
     - **It ends 6-20% later.** The tail after the last compute is longer: stores drain behind
       the remaining loads and moves on the one BlockMover.
     - **Attributing the gap** (which store waited on what) is what the step-5 record is for.
       It is recorded here, not tuned blind.
1b. **The CSP language** (ADR 0004; `docs/plans/csp-language.md`): the written, structured
   form of the CSP program. It has a parser, a validator, a lowering to the IR, and a printer.
   Its drivers are matmul, LU, and the linear operator (matmul, bias and activation), the last
   for fusion with tile contexts on block-move ingress and egress. It comes before 2b.
2b. **`kpu-run --level cycle-accurate`.**
   - `run_at` takes a device model, not the deployment spec L-CA needs (DRAM, compute fabric).
   - The L0 corpus and serialization tests iterate every implemented level, so flipping
     `level_implemented` would run LU and spec-less devices at L-CA.
   - Step 2b gives `run_at` the spec and a per-program, per-machine `level_supported()`. The
     corpus then runs L-CA where a program and machine allow it, and says why where they don't.
3. **L-T1 from the program.** `TileTransactionExecutor` reads the CspProgram's residency instead
   of re-deriving it. Equivalence test: identical makespan and records on L0 programs it already
   runs.
4. **Level 2: the distribution.**
   - `distribute()` with Block2D, SUMMA's broadcasts as NoC Send/Receive, and the binding to
     (L3, CF) pairs.
   - Runs on T4 and T16 with the NoC wired.
   - Tests:
     - values are bit-identical;
     - each partition computes only the C tiles it owns;
     - the broadcast volume equals SUMMA's analytic count;
     - partitioning plus NoC is measured against `kpu_s1` on the same problem.
5. **The record, checker and viewer from the program** (the system-schedule plan's steps 4-6,
   rebased): `.sflow` from CspProgram execution, `sflow_check.py`, and the sysflow page.
6. **Below L3, made explicit.** Streamer and BlockMover action lists come from the tile
   function's StreamProgram (row and column segments), so L-T2 (#283) has the process structure
   it needs.

## 5. Verification

- **Values:** bit-exact against the L0 reference at every level, on every machine, after every
  step.
- **Reuse is the program's:** the Loads in the CspProgram equal the DRAM loads in the L-CA run.
  An interpreter never adds or removes one.
- **One source of sequencing:** a test confirms no L-CA path in `kpu-run` or `kpu-sysim` builds
  a `ScheduleResult`.
- **The MPI correspondence, checked:** on T16, level-2 matmul's NoC traffic equals SUMMA's
  analytic broadcast volume for the same grid and tile size.

## 6. Key Invariants

- **The CSP program is the only source of tile sequencing.** Interpreters decompose its actions;
  they never reorder or invent them.
- **Residency is decided in the lowering, not at run time.** A tile's L3 credit returns when the
  program's last consumer reads it; no tag-CAM match creates reuse the program did not decide.
- **Credits up, data down.** An action begins only with its input present and its output's
  credit.
- **Inner loops are tile functions.** No CSP process sequences matrix elements. Below L3, the
  tile function's stream decomposition is the only element-level structure, derived from its
  space-time map.
- **Values never move** across levels, machines, distributions or orders.
- **Schedule parameters stay out of the machine:** order, distribution map and grid mapping are
  the program's; capacities and rates are the spec's.

## 7. Decided on Review (2026-10-09)

| Q | Decision |
|---|---|
| Q1 | Level 1: one process per kind (the flat machine). Level 2 instantiates and binds them per partition. |
| Q2 | The lowering sets channel capacities from the spec and records them in the program. Interpreters refuse a program that exceeds the machine. |
| Q3 | Derive from L0 now; DFG lowering (#281) later produces the same L0. |
| Q4 | Belady residency (the furthest next use) for the given order. |
| Q5 | Keep the generators for existing tests, freeze them for new work, and retire them later. |
| Q6 | #340 closed unmerged (2026-10-09). Its finding is §1.2, and its session log is kept, marked withdrawn. |

## 8. Questions (as posed)

- **Q1. Process granularity.** Recommended: level 1 has one process per kind (one logical DMA,
  one L3, one BlockMover, one streamer pair, one CF), the flat machine of the hypothesis. Level 2
  instantiates them per partition and binds them to engines, as MPI ranks bind to nodes. The
  alternative is per-instance processes at level 1, which would put machine size into the
  machine-independent program.
- **Q2. Channel capacities.** Recommended: the lowering decides them from the spec (L3 slots,
  L2 banks) and records them in the program. Interpreters refuse a program whose capacities
  exceed the machine. Capacity is a machine parameter, so the program depends on its target, as
  an MPI decomposition depends on the rank count.
- **Q3. Derivation source.** Recommended: L0 now. `derive_*` already produces the level-1
  ordering. DFG lowering (`lower_to_csp`, #281) later produces the same L0 and needs no change
  here.
- **Q4. Residency policy.** Recommended: Belady, the furthest next use, for the given order. It is
  optimal for a fixed order and makes the reload count the order's property. Choosing a better
  order (blocking) is the next plan.
- **Q5. The generators and `ScheduleResult`.** Recommended: keep them for the existing tests,
  freeze them for new work, and retire them once every L-CA user runs a CspProgram.
- **Q6. PR #340.** Recommended: close it. Its finding is recorded here (§1.2) and in its session
  log.

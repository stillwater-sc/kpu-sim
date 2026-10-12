# L3 Placement Is Behavior: L-B Models Where Every Tile Lives

**Date:** 2026-10-11
**Status:** Approved 2026-10-11 (Q1-Q6 as recommended, §7). Steps 1-2 merged (#364, #365); step 3 (L-B executes placement) in review
**Raised by:** the architect, 2026-10-11, reviewing `docs/plans/domain-flow-debugger.md` (#363):
> The L-B behavioral model has to model L3 placement otherwise the CSP sequencing doesn't do
> anything. The tile placement in L3 is a key component of the behavior of the machine, in the
> same way that loading an operand into the registers of a CPU is key to the behavior of a CPU.

**Amends:**
- `docs/plans/csp-program-tile-sequencing.md` §2.1 ("level 1: the machine as one flat L3");
- ADR 0004 decision 3 (placement explicit only at level 2);
- the tile-flow debugger's Q2 ("L3 pooled first");
- the honesty rule in `docs/plans/domain-flow-debugger.md` §3.7 ("L-B models a capacity, not
  placement"), which was wrong. #363 is held until this plan is decided.

## Decided on review (2026-10-11)

| Question | Decision |
|---|---|
| Q1 Who allocates | The walker, by a named policy written in the program; default `fifo` |
| Q2 Explicit placement | Yes: `resident X @ slots(...)` overrides the policy; a collision is refused |
| Q3 L3 transactions in the trace | Yes: opset 1.2 adds the L3 ops with their slots; the replay validates placement |
| Q4 Banks | `bank = slot % l3.banks` (interleaved) for now |
| Q5 L2 and L1 | L3 first; L2 and L1 in a later plan |
| Q6 Order | L-B and the trace first (steps 2-4), then L-T1, then L-CA |

## 1. Problem Statement

A CSP program's `resident` and `release` statements are its register allocation. They decide
which domains occupy the L3, and when a slot is given back for the next one. Today no level gives
those decisions a place to land.

**What each level does with the L3 today** (read from the code, 2026-10-11):

| Level / artifact | The L3 is | Where | Consequence |
|---|---|---|---|
| **L-B** | a map from tile name to values, per channel; a count against capacity | `csp/behavioral.hpp`: `store_`, `held()` | No tile has a slot. A release frees "one"; a load takes "one". |
| **L-T1** | anonymous credits: a priority queue of the times each free slot frees | `csp/transactional.hpp`: `credits_`; `Slot` has a tile and an interval, no index | A load takes *any* credit that is free first. |
| **L-CA** | one credit pool and one Tag CAM per L3 tile; the home L3 tile from `TileDescriptor::l3_tile` | `concurrent_timing_executor.hpp`, `dma_engine_process.hpp` | The `slot_id` is each DMA engine's **own round-robin counter**, modulo the CAM's capacity (`allocate_slot`). It is a label, unrelated to which slot was freed. Two engines (T4 has eight) can give two live tiles the same `slot_id`. |
| **L0 trace** | absent | `TileOpKind`: Feed, Drain and the compute ops only | The trace records no L3 transaction: no load, no release, no slot. |
| **CspDriver (L-CA)** | one L3 tile only | `csp_driver.hpp`: refuses `l3_tiles != 1` | Level 1 runs on one L3 tile; level 2 is not built. |
| **Deployment spec** | `l3.tiles`, `l3.banks` (marked "L-T2"), `capacity_tiles` | `deployment_spec.hpp` | Banks are declared, and no level models them. |

**Why this matters.**
- **Behavior that depends on placement is invisible to the oracle.** Examples:
  - which slot and bank a domain lands in;
  - that slot's reuse after a release (a write-after-read on the slot);
  - two live tiles in one slot;
  - a panel landing in one bank.
- **The precedent is #279:** a slot-release defect that every count-based invariant passed. The
  tile-flow checker caught it only with a ninth invariant, "every L3 residency is filled"
  (`docs/plans/tile-flow-debugger.md` §5).
- **Each level that gets placement would invent its own.** L-CA already has: a per-engine counter.
  A level's placement would then differ from another's by construction, so placement could never
  be compared across levels. That is the second-oracle failure this repo removed for values
  (`docs/plans/retire-l0-derivations.md`).

**How it happened.** The level-1 hypothesis (2026-10-09) addressed the machine "as one flat L3,
as an MPI rank addresses its memory". ADR 0004 then made placement explicit only at level 2,
meaning the owner of a partition (which L3 tile, which compute tile). Placement *within* an L3 was
assigned to no level. L-B was built as a values oracle, and a count was enough for values.

## 2. Architecture

Placement is made **once**, by the walker that every level already shares, and carried on the
action. Every level then executes the same placement.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ .csp program   machine flat(l3 = 17, place = fifo);    resident B[:, j];  release ...  │
└──────────────────────────────┬─────────────────────────────────────────────────────────┘
                               │ walk (csp/lang/walk.hpp): the one source of action order
                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Walker + L3Allocator (NEW)        open residency  ──►  slot = allocator.take(policy)   │
│   free list per L3 tile             release        ──►  allocator.give(slot)            │
│   bank = bank_of(slot)              retain         ──►  slot held past the program      │
│                                                                                          │
│  Action{kind, tile, residency, ..., place = L3Address{l3_tile, slot, bank}}  (NEW)       │
└──────────┬──────────────────────────────┬──────────────────────────────┬────────────────┘
           │ the same actions              │                              │
           ▼                               ▼                              ▼
┌─────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
│ L-B  (the oracle)   │   │ L-T1                         │   │ L-CA  (CspDriver)            │
│  L3 = slots[h][s]   │   │  a Load waits for ITS slot's │   │  TileDescriptor carries the  │
│  write: slot free   │   │  Release (no anonymous       │   │  slot; the DMA inserts at it │
│  read: tag == tile  │   │  credit)                     │   │  (no allocate_slot counter)  │
│  release: frees it  │   │  bank occupancy recorded     │   │  per-slot credit; banks at   │
│  refuses violations │   │                              │   │  L-T2 / L-CA timing          │
└─────────┬───────────┘   └──────────────────────────────┘   └──────────────────────────────┘
          │ the trace (opset 1.2): L3 transactions with their slots
          ▼
   L0 trace  ──►  TileProgramReference replays it bit-exact (now including placement)
   .tflow / .dflow  ──►  slots drawn where they are, for real
```

## 3. Design

### 3.1 An L3 address

```cpp
// csp/csp_program.hpp
struct L3Address {
    std::uint32_t l3_tile = 0;      // which L3 module (always 0 at level 1; level 2's owner)
    std::uint32_t slot = 0;         // which tile-sized buffer in that module
    std::uint32_t bank = 0;         // bank_of(slot): derived, recorded for the record and L-T2
};

struct Action {
    // ... kind, tile, process, residency, context (unchanged)
    std::optional<L3Address> place;     // NEW: the L3 slot this action writes, reads or frees
};

struct Residency {
    // ... (unchanged)
    L3Address place;                    // NEW: fixed when it opens; freed when it releases
};
```

- A module's slots are `capacity_tiles / l3.tiles`, the split L-CA already makes (the remainder
  goes to the low modules).
- The bank is a function of the slot, `bank = slot % l3.banks` (interleaved), until the spec says
  otherwise (Q4).

### 3.2 Who decides the slot: the walker, by a named policy; the program may override

```
WRONG — each level allocates its own slots
        (L-CA's per-engine round-robin counter today; an L-T1 "any free credit")
        → three placements of one program, none of them the program's, none comparable

WRONG — the viewer or the recorder infers a slot from the order of releases
        → a placement that no level executed

CORRECT — the walker allocates once, as a compiler's register allocator does, and the slot rides
          on the action; every level executes that slot, and the trace records it
```

```cpp
// csp/lang/place.hpp (NEW)
enum class PlacePolicy : std::uint8_t {
    Fifo,       // the slot freed longest ago: fewest write-after-read waits on reuse
    Lowest,     // the lowest free slot: compact, and reuses hot slots soonest
    BankSpread, // the free slot in the least-occupied bank: spreads a panel across banks
};

class L3Allocator {
public:
    L3Allocator(std::uint32_t l3_tiles, std::uint32_t slots, std::uint32_t banks, PlacePolicy p);
    L3Address take(std::uint32_t l3_tile);      // throws when the module is full (the validator
                                                //   has already proven it never is)
    void give(const L3Address& a);              // a release; a retained slot is never given back
    std::size_t held(std::uint32_t l3_tile) const;
};
```

- **The policy is a schedule choice, not a machine parameter**, as prefetch depth is (a
  `schedule` option, never a deployment field). So it is written in the program,
  `machine flat(l3 = 17, place = fifo)`, defaulting to `fifo`. It is part of the program's
  identity and digest.
- **The program may place a residency explicitly**, as a hand-allocated register:
  `resident B[:, j] @ slots(0..7);`. The walker refuses a placement that collides with a live
  residency, naming both tiles (Q2).
- **Validation stays symbolic** (ADR 0004 §4.3). It proves capacity per iteration, as it does
  today. The slots are assigned as the walker emits, in O(capacity) state, so a 1M x 1M program
  is never unrolled to place it.

### 3.3 L-B executes the placement: the L3 is addressed storage

```cpp
// csp/behavioral.hpp
struct L3Slot { std::optional<TileCoord> tag; std::vector<float> values; bool retained = false; };
std::vector<std::vector<L3Slot>> l3_;       // [l3_tile][slot], replacing store_[Chan::L3]
```

| Action | L-B does | L-B refuses (`BehavioralError`, naming the slot and both tiles) |
|---|---|---|
| Load, Writeback, Inherit | writes the tile into `place` and tags it | the slot holds a live tile |
| Move, Store (read) | reads `place`, checking that its tag is the tile | the tag is another tile, or the slot is empty |
| Release | frees `place` | the slot is free, or holds another tile |
| Retain | marks `place` held past the program's end | — |

`held(L3)` becomes the number of tagged slots. Values do not change: the same tiles are read and
written in the same order. The corpus's *values* stay byte-identical; its traces gain the
placement (§3.5).

### 3.4 The finer levels execute the same placement

- **L-T1.** A Load waits for **its** slot's Release, not for the first free credit. This is a real
  behavioral change: a program, or a policy, that reuses a slot too early now serializes on it,
  exactly as a register's write-after-read hazard does. `Slot` gains its `L3Address`, so `.tflow`
  can draw slots where they are (the tile-flow debugger's step 6).
- **L-CA.**
  - `TileDescriptor` carries the program's `l3_tile` and `slot`. The DMA inserts at that slot, and
    `allocate_slot` is removed for program-driven runs.
  - Credits become per slot within each module: a credit is "slot s is free", not "some slot is
    free".
  - Banks enter the L3 timing at L-T2 and L-CA (`l3.banks`), so conflicts within a bank show.
- **Comparison across levels gains placement.** Every level must report the same slot for every
  residency. A difference is a defect of that level, as a value difference is.

### 3.5 The trace records the L3 transactions (opset 1.2)

The L0 trace has no L3 transaction today: it holds feeds, drains and tile functions. A trace of
"transactions and operations" without its L3 transactions cannot carry placement, and
`TileProgramReference`'s replay cannot validate it.

- New ops: `L3_LOAD`, `L3_WRITEBACK`, `L3_RELEASE` and `L3_RETAIN`, each with `l3=`, `slot=` and
  `tile=`. A Feed names the slot it reads (`from=l3:slot`).
- `TileProgramReference` replays them on an addressed L3. It must reproduce L-B's placement and
  values bit for bit, and it refuses an occupied write or a mistagged read.
- **The corpus is regenerated** with `kpu-run --emit-l0` (the one source of an L0). The diff is the
  added ops; the recorded values are unchanged. `.kpuld` loadables carry `.csp`, so they need no
  change.

### 3.6 Retain and inherit carry their slots

A retained tile stays in its slot across launches (`.kpuld`, `KpuDevice`). An `inherit` must open
the residency **in that slot**, so the walker's allocator starts with the inherited slots taken.
The orchestrator hands the next operator its slot map along with the retained tiles. A mismatch
is refused by name.

### 3.7 Level 2 is the same address with `l3_tile` set

Level 2's partition map chooses the owning L3 module, `l3_tile`. Within the module, the same
allocator places the slot. So level 2 adds owners and NoC communication, not a second placement
model. Level 2 is not built in this plan; the address leaves room for it.

## 4. Implementation Steps

Each step is one PR, reviewed and merged before the next. The suite stays green after each.

1. **Decide and record.**
   - This plan, after review.
   - An ADR 0004 amendment: placement within an L3 is behavior; the walker allocates it by a named
     policy; it is explicit at every level.
   - Amend `csp-program-tile-sequencing.md` §2.1, and mark the tile-flow debugger's Q2 superseded.
   - `CLAUDE.md`'s fidelity table: L-B "computes values **and placement**".
2. **The address and the allocator.**
   - `csp/csp_program.hpp`: `L3Address`, `Action::place`, `Residency::place`.
   - New `csp/lang/place.hpp`: `L3Allocator` and `PlacePolicy`.
   - `csp/lang/walk.hpp`: allocate on open, give on release, hold on retain.
   - The language: `place = fifo | lowest | bank-spread` in `machine flat(...)` (`parse.hpp`,
     `print.hpp`, `format.hpp`).
   - Tests (`tests/program/test_csp_place.cpp`, new):
     - each policy's slot sequence on matmul 128/32 and LU 64/16;
     - no two live residencies share a slot, at any step;
     - a release frees exactly its slot;
     - the explicit `@ slots(...)` placement, and its collision refusal.
3. **L-B executes placement.**
   - `csp/behavioral.hpp`: the addressed L3 and the refusals of §3.3.
   - `driver/csp_step.hpp`: `CspStep::place`; `kpu-run --step` prints the slot.
   - Tests:
     - every refusal, triggered by a hand-edited action list;
     - values bit-identical to before for the corpus programs (the same computation);
     - `held` equals the number of tagged slots.
4. **The trace carries placement** (opset 1.2).
   - `tile_program.hpp`: the L3 ops; `serialize/l0_format.hpp` reads and writes them.
   - `TileProgramReference` runs on an addressed L3.
   - `csp_reference`'s replay compares placement as well as values.
   - The corpus is regenerated; `tests/program/corpus/README.md` records why.
   - Tests: `test_l0_serialization` (round trip), `test_csp_run` (the replay reproduces placement),
     and the corpus tests.
5. **L-T1 executes placement.**
   - `csp/transactional.hpp`: a Load waits on its slot; `Slot` gains its address.
   - `.tflow` binds slots (the tile-flow debugger's step 6).
   - Tests: placement equals L-B's; a policy that reuses early stalls, measured on matmul (`lowest`
     against `fifo`).
6. **L-CA executes placement.**
   - `TileDescriptor` gains the slot; `dma_engine_process.hpp` inserts at the program's slot.
   - Per-slot credits in `ConcurrentTimingExecutor`; `CspDriver` passes the slot.
   - L3 banks enter the timing.
   - The program's `banks` is checked against the deployment's `l3.banks` (`csp_level_supported`),
     as `l3` is against `capacity_tiles`. Deferred here from step 3 (decided 2026-10-12): the
     deployments declare 4 banks and every existing program defaults to 1, so the check waits
     until banks change the timing, with the programs updated to match.
   - Tests (`test_csp_driver`, `test_csp_spec_oracle`): placement equals L-B's; bank conflicts
     measured.
7. **Retain and inherit carry slots** (§3.6): the orchestrator and `KpuDevice`.
   - Tests: `test_orchestrator`, `test_csp_inherit_retain`.

#363 (the domain-flow debugger) resumes after step 3, with slots in its machine view.

## 5. Verification

```bash
nice -n 19 cmake --build build -j 2 -- -k 0
cd build && nice -n 19 ctest -j 4 --output-on-failure

# Placement is visible and the same at every level
build/tools/kpu-run/kpu-run --program tests/program/corpus/matmul_48x48x48_t16.csp \
    --deploy tests/program/deploy/kpu_s1.json --step --step-limit 20      # each step names its slot
```

Expected:
- **No two live residencies share a slot**, at any step of any corpus program, at every level.
- **The corpus values are unchanged** (the same computation). The traces differ only by the added
  L3 ops.
- **The replay reproduces L-B's placement and values** bit for bit.
- **L-T1 and L-CA report L-B's slot for every residency.**
- **Reintroducing #279's slot-release defect**, in L-B or in the replay, is refused at the step it
  occurs, naming the slot and both tiles.
- **`lowest` against `fifo`** on matmul at L-T1: `lowest` shows more slot waits. The measured
  difference is recorded in the session log.

## 6. Key Invariants

1. **One placement.** The walker assigns every residency's L3 address once. Every level executes
   that address, and the trace records it. No level allocates a slot of its own.
2. **A slot holds at most one live tile.** A write into a live slot is refused.
3. **A read finds its tile.** A Move or Store reads the slot its residency was placed in, and the
   slot's tag is that tile.
4. **A release frees exactly its slot.** A retained slot is held past the program's end, and the
   next operator's `inherit` opens it there.
5. **Placement does not change values.** The same tiles are read and written in the same order;
   placement changes *where* and, at the timed levels, *when*.
6. **The policy is the program's.** It is written in the program and part of its digest, never a
   deployment field.
7. **Occupancy is the tagged slots:** `held(L3)` equals the number of tagged slots, at every step.

## 7. Questions for review

- **Q1. Who allocates by default?**
  - **Recommended:** the walker, by a named policy, with `fifo` as the default (the slot freed
    longest ago, which fewest reuses wait on).
  - The alternative is that every program places every residency explicitly. That is exact, but
    burdensome for a written program, and kpu-csp-gen would have to emit it.
- **Q2. Explicit placement in the language** (`resident X @ slots(...)`).
  - **Recommended:** yes, as an override, refused on collision.
  - The alternative is policy only, adding explicit placement when a program needs it.
- **Q3. The L3 transactions in the L0 trace** (opset 1.2, §3.5).
  - **Recommended:** yes. The trace should hold the machine's L3 behavior, and the replay should
    validate it.
  - The alternative is to keep placement in the CSP record (`.dflow`, `.tflow`) only. The trace
    would stay smaller, but would not validate placement.
- **Q4. Banks.**
  - **Recommended:** `bank = slot % l3.banks` (interleaved) for now, recorded at L-B and timed at
    L-T2 and L-CA.
  - The alternative is a bank map in the deployment spec (a contiguous range per bank), which
    is closer to a real floorplan.
- **Q5. L2 and L1.**
  - **Recommended:** L3 first, as raised. L2 and L1 slots follow the same pattern once the stream
    layer places tiles in L2 banks. That is a later plan.
- **Q6. Order of the levels.**
  - **Recommended:** L-B and the trace first (steps 2-4), then L-T1, then L-CA. The oracle leads,
    and each finer level refines a placement that already exists.

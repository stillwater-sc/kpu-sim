# ADR 0002 — CSP is a program layer, not a fidelity; and the virtual-platform driver

| | |
|---|---|
| **Status** | **Accepted** (2026-09-22) |
| **Date** | proposed and accepted 2026-09-22 |
| **Supersedes** | the tier/engine naming in ADR 0001 D2 (not its choice of portable program) |
| **Context docs** | `docs/plans/kpu-program-model.md` (D6), `docs/architecture/adr/0001-program-contract-and-transactional-engine.md` |

---

## 1. The error being corrected

ADR 0001 and everything downstream of it (the UML doc, the executor design note) used
**"CSP tier"** as a synonym for **CYCLE_ACCURATE**, and treated the three fidelities as
three different engines consuming different inputs.

That is a category error. **CSP — communicating sequential processes — is the sequencing
mechanism of domain flow.** It is a *program layer*: the processes (DMA, BlockMover,
Streamer, compute tile) and the tile sequences they exchange, which get a tile to the
compute fabric in the right order. Cycle-accurate is not the CSP layer; it is one
**interpretation** of a CSP program. The transactional levels are other interpretations of
**the same program**.

Concretely: a block linear-algebra operator has a CSP program that moves matrix/tensor
tiles. You can validate that program functionally by treating each block move as atomic.
The DMA and L3/L2/L1 scratchpad layers, however, operate at finer granularity, so a
*further* interpretation articulates each block move into resource transactions — a DMA
block read, an L2 streamer read. Every model executes the same Domain Flow Program; they
differ in **how finely they decompose its transactions**, and therefore in what they can
tell you.

## 2. The corrected layer model

### 2.1 Program layers (what gets executed)

| Layer | What it is | Where it lives today |
|---|---|---|
| **DFP — Domain Flow Program** | the recurrence / SURE over an index space with an affine schedule; the source of truth | `domain_flow` (external) |
| **CSP program** | the sequencing lowering: processes + channels + the ordered tile sequences each process moves. This is what a "block linear-algebra operator" *is* at the machine level | `TileProgram` (L0) is today's block-sequential form; its explicit process/channel structure is the gap (§5.1) |
| **Stream signatures** | how a tile becomes an element stream at an array edge, with wavefront timing | `StreamProgram` (L1) |
| **Resource programs** | *derived per interpretation*: each block move decomposed into DMA bursts, L3/L2/L1 reads and writes, compute-tile pushes | the decomposition is the interpreter's job (§2.2); the vocabulary exists in `DMOpcode` |

The DFP is authored; **the CSP program is derived from it** and never hand-written; the
resource transactions are derived from the CSP program by whichever interpreter is running.
**Nothing below the DFP is a separate program to maintain** — which is why the CSP layer
needs a disassembler for inspection, but no source syntax or parser.

### 2.2 Execution levels (how the same program is interpreted)

| Level | Transaction granularity | Time model | The question it answers |
|---|---|---|---|
| **L-B** behavioral | whole block move, atomic | none | is the CSP program functionally correct? |
| **L-T1** block-sequential transactional | one tile / block move | per-move duration, tile-granularity credits and capacity | does the sequencing deliver the right tiles, in the right order, under finite buffers? |
| **L-T2** resource transactional | the fixed per-resource vocabulary of §3.3: `read`/`write` on every resource, `push` on the compute tile | per-transaction, queueing per engine / port / bank | where is the bandwidth or occupancy bottleneck? |
| **L-CA** cycle-accurate | protocol events, per cycle | FSMs, arbitration, DRAM timing | protocol compliance; the calibration ground truth |

**The process structure is identical at every level.** A `BlockMover` is the same CSP
process in all four; what changes is whether its move is one atomic event, one timed
transfer, a sequence of bank-level transactions, or a per-cycle protocol exchange.

This replaces "fidelity = which engine you pick" with **fidelity = how deeply the
interpreter decomposes a CSP transaction**. It also explains why values must be identical
across levels: decomposition changes *when* things happen, never *what* is computed.

### 2.3 Cross-level contracts

These are what make the levels worth having, rather than four unrelated simulators:

1. **Values are level-invariant.** Every level drives the same tile kernels. L-B, L-T1 and
   L-T2 are **bit-exact** to each other (same accumulation order, fixed by the program's
   WAW ordering). L-CA matches within a stated tolerance where its order legitimately
   differs.
2. **Each finer level calibrates the coarser one.** L-T1 is calibrated against L-T2, and
   L-T2 against L-CA. An uncalibrated level reports that in its provenance rather than
   being quoted as measured.
3. **Timing is monotone in detail, not in value.** A finer level may be faster or slower;
   what it may not do is contradict the coarser level's *ordering* of the program.
4. **If a level needs information the CSP program does not carry, that is a program-model
   gap** — fix the program layer, never special-case the interpreter.

## 3. The virtual platform

The simulator is the foundation of a virtual platform that behaves like the physical KPU
SoC. That demands one object you load and drive, not a collection of executors.

```
                    ┌────────────────────────────────────────────┐
   deployment spec  │            VirtualPlatform                 │
   (device + engines)──▶ EngineDeployment: processes, channels,  │
                    │                     resource pools         │
   program (DFP/CSP)──▶ ProgramImage                             │
   test state space ┼──▶ Backdoor (simulation-only, NOT a bus):  │
   (pre/post cond.) │      poses or collects the state of ANY    │
                    │      resource — HOST, DRAM, L3, L2, L1,    │
                    │      CF register files. No physical        │
                    │      manifestation in the SUT (§3.4)       │
                    │                                            │
                    │  run(level) ─▶ IInterpreter ─▶ RunResult   │
                    │  step()     ─▶ one transaction at `level`  │
                    └────────────────────────────────────────────┘
```

### 3.1 The lowering boundary: who turns a DFP into a CSP program

The platform accepts **only** a `CspProgram`. Lowering is a separate, testable step that
happens before load:

```cpp
// owned by the compiler side (domain_flow; relocating per #232), not by the platform
CspProgram lower_to_csp(const DomainFlowProgram&, const LoweringOptions&);
```

`kpu-run --program X` decides by what X is:

```
X is a .dfg / DFP        ──▶ lower_to_csp(dfp, opts) ──▶ CspProgram ──▶ load_program()
X is a serialized CSP IR ──▶ deserialize            ──▶ CspProgram ──▶ load_program()
```

Keeping the boundary sharp buys two things. The platform never depends on the compiler, so
it can execute a CSP program whose DFP is unavailable — the case every regression corpus
needs. And lowering stays independently testable: the same DFP lowered twice must produce
an identical CSP program, which is a cheap and strong compiler test.

`LoweringOptions` carries the tiling and schedule choices; those are compiler decisions, not
platform ones, and they belong in the program's provenance.

### 3.2 API sketch

```cpp
enum class ExecutionLevel { Behavioral, BlockSequential, ResourceTransactional, CycleAccurate };

class VirtualPlatform {
public:
    explicit VirtualPlatform(const DeploymentSpec&);       // engines + capacities + bandwidths

    // --- load: program and data arrive through the backdoor, not through the datapath,
    // so staging weights never distorts the timing of the run that follows.
    ProgramHandle load_program(const CspProgram&);
    Backdoor&     backdoor();   // any addressable resource (§3.4), CF register files included

    // --- state: a run starts from an immutable snapshot, never from whatever the
    // previous run left behind, or the "pure function" claim below is false.
    StateSnapshot snapshot() const;                        // capture the staged situation
    void          restore(const StateSnapshot&);           // reset to a known situation

    // --- execute: the snapshot is an EXPLICIT ARGUMENT, not an implied ambient state.
    // run() restores it first, and its digest is part of the run identity and the
    // provenance. Passing it makes "which state did this run start from?" answerable
    // from the call site rather than from execution history.
    RunResult run(ProgramHandle, ExecutionLevel, const StateSnapshot& initial_state);
    StepCursor step_begin(ProgramHandle, ExecutionLevel);  // single-step at that granularity
    bool       step(StepCursor&);                          // one transaction per call

    // --- observe
    const Timeline& timeline() const;                      // per transaction, at the run's level
    const Stats&    stats() const;
    const Provenance& provenance() const;                  // deployment, level, calibration state
};
```

`Backdoor` is a **separate, simulation-only interface** (§3.4) between any resource's state
and the conceptual test state space. It is *not* the resources' own `load`/`store`, which
are physical and local. `ResourceManager::{write,read,copy,memset}` is the closest existing
primitive, but it is a starting point to build on, not the contract — the backdoor must stay
distinguishable in the code from anything the SUT can physically do. It is **timing-free by
construction** and flagged in the run's provenance, so nobody mistakes a backdoor-staged
tensor for one that was DMA'd.

Its purpose is to make the simulator and its tests tractable: put the model into a state
whose response is known, run one transaction, read the response back.

`step()` is what makes the platform usable for writing and testing Domain Flow Programs:
at L-T1 you step block moves, at L-T2 you step DMA bursts and L2 reads. The unit of
stepping *is* the level.

### 3.3 The resource model: a fixed transaction vocabulary

Every modelled resource exposes **two small, fixed surfaces**. This is what keeps L-T2
calibratable and keeps the interpreters from growing per-device special cases.

**Data transactions** — fixed per resource type:

| Resource | Data transactions |
|---|---|
| Memory device | `read`, `write` |
| Memory controller | `read`, `write` |
| DMA controller | `read`, `write` |
| Block mover | `read`, `write` |
| Streamer | `read`, `write` |
| NoC | `read`, `write` |
| **Compute tile** | **`push` only** |

The compute tile is the exception because it is a reactive fabric, not a memory: operands
are pushed into it and results are pushed out of it. Nothing on the **data path** addresses
a value inside the array and pulls it.

**The compute fabric talks only to L1** (clarified 2026-09-23). Its sole data interface is
the L1 layer: operands are pushed in from L1 and results are pushed back out to L1, and no
transaction connects it to L2, L3 or DRAM. Movement therefore *ends at L1* — a hop chain
runs DRAM→L3→L2→L1 and stops, and any model whose movement terminates "at the fabric" has
mislabelled its last hop.

**Nor may the chain be collapsed** (corrected 2026-09-23). Each leg is governed by its own
CSP process — DMA for DRAM↔L3, BlockMover for L3↔L2 (and L3→L3 across the NoC), Streamer
for L2↔L1 — and the pathways between them are physically distinct. Modelling two legs as one
stage describes a machine that cannot be built, so **a span always contains all of its
hops**. An earlier revision of this paragraph said collapsing L3→L2 and L2→L1 was allowed
and cited #264 increment 5 as doing it; both the permission and the implementation were
wrong, and #292 removes the collapsed mode rather than deprecating it. Residency may change
where a chain *starts* — a tile already in L3 begins at the BlockMover — but never which
hops it contains.

**The compute fabric is a domain flow compute engine** (clarified 2026-09-23). This is a
containment, and the direction matters:

> A domain flow compute engine **can be** a systolic array.
> A systolic array **is not** a domain flow compute engine.

So "systolic array" is one realization of the fabric, never a definition of it, and the two
terms are not interchangeable. The practical consequences for this repo:

- A **timing model** may legitimately be systolic — the L1 stream timing is, and says so.
  That is a statement about one realization's latencies, not about what the fabric is.
- A **structural or semantic** claim must not assume systolic behaviour: wavefront shape,
  fixed operand skew, a rigid two-dimensional array, or a schedule that only a systolic
  array admits. A domain flow engine is not obliged to have any of them.
- Naming that equates the two (`ComputeFabric / SystolicArray`) describes today's
  implementation, not the architecture, and should not be read as the definition.

**The two surfaces are not the same surface.** `push` is the compute tile's entire *data
transaction* vocabulary. `load`/`store` are *state transactions*, available on every
resource including the compute tile's register files, and they are how the backdoor reads a
result out (§3.4). So the state-in/state-out test below is consistent with "push only": the
tile is never read **by the datapath**, while the backdoor reads its registers out of band,
exactly as it reads an L2 bank. A model that let a streamer *read* a compute tile would be
the violation; a test that inspects its registers is not.

**Settled (2026-09-22): the compute tile pushes results out. There is no drain that pulls
from it.** `push` is therefore the tile's complete data vocabulary in both directions, and
nothing — streamer, block mover or backdoor datapath — reaches into the array to fetch a
value.

One naming consequence worth flagging, because it will mislead otherwise: the L0 program
already has a **`Drain` op**, and "drain" sounds like a pull. It is not one. It is the
program-level *extraction sequence*, and the code already works this way — a completed
compute pushes its result into the result tag CAM (`compute_result_tag_cam_`, "tracks
result tiles ready for DRAIN") and the extraction then matches on it. So at L-T2 a `Drain`
op decomposes into **a push out of the compute tile, followed by writes down the
hierarchy** — never a read of the fabric. The op keeps its name; the decomposition is what
matters.

**State-management transactions** — the same on every resource, and **physical**:

```
clear      reset      single-step      load      store
```

These are operations the System Under Test can actually perform, between resources that are
actually connected. A resource `load`/`store` references **local** memory state over a real
port: the DMA engine executes a load from DRAM and stores a block into L3 because a bus
exists between them. They are modelled, they cost time, and a program can cause them.

This uniform control surface is what makes deployment automatable: a harness can bring any
resource to a known state, advance it, and snapshot it without knowing which kind of
resource it is holding.

**The backdoor is NOT these transactions** (§3.4). Conflating the two would let the
simulator model a machine that cannot exist.

### 3.4 The backdoor: a global operator with no physical manifestation

**The backdoor is not a re-use of the resources' own state transactions. It is a separate,
simulation-only mechanism, and it is deliberately unphysical.**

A resource's `load`/`store` (§3.3) is *local*: it moves state between resources a bus
connects. The backdoor is *global*: it connects **any** resource's state to a **conceptual
test state space** that holds pre-conditions and post-conditions. A backdoor load of L1
drops a row or column straight into an L1 streaming buffer from that space. **No physical
bus or port could ever connect L1 to it** — the test state space is a conceptual device,
not a component of the System Under Test.

That is the whole point: it is *truly* a back door. It follows that:

| | Resource state transactions (§3.3) | Backdoor |
|---|---|---|
| Exists in the SUT | **yes** — real ports and buses | **no** — conceptual |
| Scope | local, between connected resources | global, any resource ↔ test state space |
| Costs time | yes, modelled | no, out of band |
| A program can cause one | yes | **never** |
| Implemented by | the modelled resource | the simulation harness |

**Invariants this imposes:**

1. **The backdoor gets its own interface.** It must not be expressed in terms of, or routed
   through, any state-management functionality that has a physical manifestation in the
   SUT. Reusing the physical surface would make the unphysical path indistinguishable from
   the physical one at the point where it matters — in the model's own code.
2. **No CSP program can emit a backdoor transaction.** If a program could, the simulator
   would be executing a machine that cannot be built. The backdoor belongs to the harness
   and to tests, never to the program layer.
3. **Backdoor use is recorded in provenance**, because a state established this way may be
   *unreachable by any program*. That is exactly its value for testing — you can pose a
   situation the datapath would take a million cycles to construct — and exactly its risk:
   a model validated only from backdoor-established states has never exercised the paths
   that would reach them physically.

**Addressability.** Posing and reading a situation requires every resource's state to be
*nameable*: not just DRAM, but each L3 tile, L2 bank, L1 vector and compute-tile register
file. The platform therefore carries a **global naming map** over `(device, resource-kind,
instance, offset)`.

This map is a **simulation-side naming scheme**, not a physical address space the hardware
implements. The SUT keeps its own real address spaces for the transactions that actually
travel its buses; the backdoor map exists so a test can say *which state* it means. Once
the map spans resources rather than a flat memory, spanning *devices* costs nothing extra
structurally — which is why this ADR adopts multi-device from the start rather than
retrofitting it. A single-device deployment is then just a map with one device in it.

Two consequences worth stating:

- **Tests become state-in / state-out.** Inject a known L2 bank content **through the
  backdoor**, push one tile down the datapath, then read the compute tile's output register
  **through the backdoor**: a unit test of the fabric with no program at all. Only the
  middle step touches the SUT; both ends come from the conceptual test state space, which is
  why this is consistent with the compute tile being push-only on the datapath (§3.3) — the
  backdoor is not a datapath (§3.4). That is a materially different test style from "run a
  program and compare the answer", and it is the one that finds protocol bugs.
- **Every level shares the map.** Addressing is a property of the platform, not of an
  interpreter, so a backdoor setup written for L-T2 works unchanged at L-CA.

### 3.5 Engine deployment — automate, coordinate, scale

A deployment is **data, not code**:

```jsonc
{
  "topology": "checkerboard", "compute_tiles": 16,
  "dma":      { "engines": 4, "bytes_per_cycle": 64, "burst": 256 },
  "l3":       { "tiles": 8, "banks": 8, "capacity_tiles": 32 },
  "l2":       { "banks_per_tile": 8 }, "l1": { "vectors": 4 },
  "movers":   { "block_movers": 4, "streamers": 4 }
}
```

- **Automate:** `EngineDeployment::from_spec()` instantiates the processes, channels and
  resource pools. No engine is constructed by hand in a test or a demo.
- **Coordinate:** one `VirtualPlatform` owns the deployment; every driver goes through it.
  The characterization harness becomes a *consumer* of the platform rather than a parallel
  path — which is the G9 "must connect, not add a fourth path" discipline applied to
  drivers.
- **Scale:** a run is a pure function of **`(program, initial_state, deployment, level)`**.
  Sweeps are lists of deployments; results are cacheable and parallelizable; CI runs the
  same function with a small deployment that a design-space exploration runs with hundreds.

  **`initial_state` is the whole input, not just "data".** Backdoor writes mutate resource
  state, so a run that reads state a previous run left behind is not reproducible and must
  not be cached. The platform therefore takes an **immutable state snapshot** as part of the
  run identity: `run()` restores it first, its digest goes into the cache key and the
  provenance, and two runs with the same four inputs are guaranteed to agree.

  **The snapshot is passed, not implied.** The flow is explicit:

  ```cpp
  platform.backdoor().stage(...);                        // pose the situation
  const StateSnapshot initial = platform.snapshot();     // capture it
  auto result = platform.run(prog, level, initial);      // restore it, then execute
  ```

  Taking `initial_state` as an argument rather than reading whatever the platform happens
  to hold is what makes the identity checkable at the call site: a reviewer can see which
  state a run started from without reconstructing the history of backdoor writes that
  preceded it, and a cache key cannot silently disagree with the state that was actually
  restored.

## 4. Driver architecture

One interface, four implementations, all fed by the platform:

```cpp
class IInterpreter {
public:
    virtual RunResult run(const CspProgram&, EngineDeployment&, const RunOptions&) = 0;
    virtual bool      step(StepCursor&) = 0;               // one transaction at this level
    virtual ExecutionLevel level() const = 0;
};
```

| Interpreter | Level | Status |
|---|---|---|
| `BehavioralInterpreter` | L-B | **exists** — `TileProgramReference` |
| `BlockSequentialInterpreter` | L-T1 | **exists** — `TileTransactionExecutor` (#264 increments 1–3), to be reframed under this name |
| `ResourceTransactionalInterpreter` | L-T2 | **missing** — the main new work |
| `CycleAccurateInterpreter` | L-CA | **exists in substance** — `ConcurrentTimingExecutor` + the four processes, but consumes `ScheduleResult` rather than the CSP program |

The driver application is then thin and level-agnostic:

```console
kpu-run --program lu.csp --deploy checkerboard16.json --level resource-transactional \
        --data A=a.bin --step --timeline out.json
```

Because the level is a flag, the same command validates a program functionally, then
measures it at two transaction granularities, then checks it against cycle-accurate — which
is exactly "write and test Domain Flow Programs at different abstraction levels".

## 5. What this changes in flight

1. **§5.1 — the CSP program needs explicit processes and channels, derived from the DFP.**
   Today `TileProgram` is a flat tile sequence with logical ports: the block-sequential
   *projection* of a CSP program, with the process/channel structure implicit. L-T2 needs
   it explicit, because a DMA burst belongs to the DMA process and an L3 write belongs to
   the block mover. This is derivation work, not language design (§6.1).
2. **ADR 0001 D2's tier table is renamed, not reversed.** Its decisions stand: L0 is the
   portable program, values answer to the reference, one factory selects the engine. What
   changes is that the engines are *interpreters of one program at different transaction
   granularities*, and "CSP" stops being used as a fidelity name.
3. **`TileTransactionExecutor` is L-T1** — correctly built, wrongly named. Increment 4
   (credits and capacity) is still exactly the right next step, because tile-granularity
   capacity is what L-T1 is *for*.
4. **PR #275 should be held.** Its `fidelity-framework.md` edit frames the tiers as
   value/timing fidelity; under this ADR the framing is transaction granularity. The
   value-correction half stays right either way.
5. **The UML doc §1/§3 headings need the same correction**, and gain L-T2.

## 6. Answers recorded (2026-09-22)

1. **The CSP program is derived from the DFP.** It is never hand-authored, so it needs no
   source language or parser — an in-memory IR with a disassembler is sufficient, and
   `--program` names a DFP or a serialized CSP IR, not a file anyone writes by hand. §5.1's
   work is therefore *deriving* explicit processes and channels, not designing a syntax.
2. **The L-T2 transaction set is fixed per resource type** — see §3.3. Every resource
   understands read and write, except the compute tile, which only understands push. That
   makes the vocabulary small, uniform and calibratable.
3. **The backdoor exists to simplify the simulator and its tests**, by setting up a state
   whose response is known and reading the result back. It is timing-free and out of band:
   writing poses a situation, reading collects it.
   **Correction (2026-09-22):** an earlier draft described it as the out-of-band use of each
   resource's own `load`/`store`. That was wrong and is now §3.4. Resource `load`/`store` is
   *physical and local* — the DMA engine loads from DRAM and stores into L3 because a bus
   exists. The backdoor is *global and unphysical*: it connects any resource's state to a
   conceptual test state space, and no bus could ever connect an L1 streaming buffer to
   that space. It therefore gets its own interface and may never be expressed through
   state-management functionality that the SUT physically implements.
4. **Multi-device from the start**, and for a specific reason: the backdoor requires every
   resource to be reachable, which means **every resource must be addressable** (§3.4).
   Once the address map spans resources, spanning devices is the same mechanism.
5. **The compute tile pushes results out; there is no drain that pulls.** `push` is its
   complete data vocabulary in both directions (§3.3). The L0 `Drain` op is an extraction
   *sequence*, not a pull, and decomposes at L-T2 into a push out of the tile followed by
   writes down the hierarchy.
6. **Question withdrawn — it was malformed.** I asked whether L-T2 "subsumes the `.kpubin`
   path", which presupposed that a fidelity level might execute a *different program*.
   It does not: **every level executes the same program articulation**; what differs is
   that L-T2's models articulate lower-level state transactions — a DMA read, an L3 write
   by the block mover. `.kpubin` is a driver-JIT artifact for one device, not a program
   layer belonging to a fidelity.

## 7. On acceptance

**Correct what already encodes the old framing:**

- [x] PR #275 — `fidelity-framework.md` framed the tiers as value/timing fidelity; it now
      leads with transaction granularity (#278). The value correction stayed right either way.
- [x] `docs/architecture/program-execution-uml.md` — the "CSP tier" headings are gone and the
      level set carries L-T2 (#278).
- [x] ADR 0001 — carries the forward pointer for the tier/engine naming.
- [ ] `TileTransactionExecutor` is the **L-T1 block-sequential interpreter**; naming to
      follow, behaviour unchanged. **Still outstanding** — no `BlockSequentialInterpreter`
      exists yet, and the rename is cheapest once #283 introduces the sibling it is named
      against.

**New work this opens — filed 2026-09-23:**

- [x] **#281** — derive explicit **CSP processes and channels** from the DFP; today
      `TileProgram` is the block-sequential projection with the structure implicit (§5.1).
- [x] **#283** — **L-T2 resource-transactional interpreter** over the fixed vocabulary of
      §3.3. Depends on #281 and #282.
- [x] **#282** — **VirtualPlatform**: deployment spec, global naming map, `run(level)`,
      `step()`, snapshot/restore, provenance. Blocks #283, #284 and #285.
- [x] **#284** — **Backdoor** as its own interface, with the §3.4 invariants enforced — not
      routed through any physically-manifested state path. Depends on #282.
- [x] **#285** — **`kpu-run`** driver: `--program`, `--deploy`, `--level`, `--step`.
      Depends on #282, #283 and #265.

**Filed alongside, not from this ADR:**

- [x] **#286** — transaction visualization: single-step plus **station occupancy** at
      L3/L2/L1 and the compute-fabric PEs. Not an ADR 0002 decision, but it is the
      observability counterpart to the credit model this ADR's levels are built on, and it
      is what would have caught the #279 release bug from a trace rather than from review.

Dependency order for the five: **#281 and #282 first** (neither is blocked), then **#283**,
with **#284** and **#285** following #282. #286 is usable for L3 at L-T1 before any of them.

**Unchanged:** #264 increment 4 (tile-granularity credits and capacity) is exactly what
L-T1 is for.

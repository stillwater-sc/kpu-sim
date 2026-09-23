# TileTransactionExecutor — design note

**Status:** design, for review before implementation (#264)
**Decided by:** ADR 0001 (`docs/architecture/adr/0001-program-contract-and-transactional-engine.md`)
**Companions:** `docs/plans/kpu-program-model.md` (D6), `docs/architecture/program-execution-assessment.md`

---

## 1. What this is

`TileTransactionExecutor` (namespace `sw::kpu::program`) is the **TRANSACTIONAL tier**: it
executes an L0 `TileProgram` and returns **exact values and tile-granularity timing** from
one run.

It is the engine the BLAS → MLP → DNN ladder runs on, because it is the only tier that is
both numerically trustworthy and fast enough for real problem sizes. The cycle-accurate CSP
tier stays the timing authority and the calibration source; the L0 reference stays the value
authority.

### Non-goals

- **Not element-level.** No wavefronts, no per-element streams, no per-cycle stepping. That
  is the CSP tier's job.
- **Not a scheduler.** It does not choose an order to minimize makespan. Ordering freedom
  belongs to the driver JIT's placement pass (D6 §4a). The executor *reacts*.
- **Not a data mover.** It models movement; it does not copy tile data (see §3).
- **Not multi-device.** One device descriptor per run.

## 2. Inputs and outputs

```
   L0 TileProgram ─┐
   placement       ├─→ TileTransactionExecutor ─→ RunResult { values (in operand buffers),
   DeviceDescriptor│                                           timeline, stats, provenance }
   L1 StreamProgram┘ (optional)
```

| Input | Source | Notes |
|---|---|---|
| `TileProgram` | `include/sw/kpu/program/tile_program.hpp` | the portable program (ADR D1) |
| Placement | driver JIT (#230 incr. 3); until then a **real `Placement` object** from `Placement::single(device)` | logical tile → L3 tile, compute op → CF tile. Decided: an explicit object from day one, so the JIT later replaces a *value* rather than forcing a signature change |
| `DeviceDescriptor` | `include/sw/kpu/program/characterize/device_model.hpp` | extended per §5 |
| `StreamProgram` (optional) | `include/sw/kpu/program/stream/stream_signature.hpp` | ADR §7.4: systolic latencies when present, lumped model otherwise |

## 3. Values: shared kernels, and why residency is bookkeeping

`TensorOperand::values` is **one row-major buffer per operand**
(`tile_program.hpp:44`), and tiles are views into it (`row_begin`/`row_end`, clamped for
trailing tiles). The kernels mutate that buffer in place.

Two consequences:

1. **The executor never copies tile data.** "Tile resident in L3" is *metadata* used for
   timing and credits. Movement is modeled, not performed. This is what makes the tier
   fast, and it is sound because the program declares complete tile I/O, so residency
   cannot change a result.
2. **Bit-exactness must come from sharing the kernels, not reimplementing them.**

### 3.1 Prerequisite refactor: extract the tile kernels

The per-op kernels live in `TileProgramReference`'s private section
(`tile_program_reference.hpp:85-231`), together with transient state: `pivots_`
(slot → ordered row swaps), `perm_` (row permutation), `swaps_performed_`.

Extract them into a shared header, e.g. `include/sw/kpu/program/tile_kernels.hpp`:

```cpp
struct TileKernelState {                 // was TileProgramReference's private state
    std::map<int, std::vector<std::pair<Dim,Dim>>> pivots;
    std::vector<Dim> perm;
    std::size_t swaps_performed = 0;
};

void apply(TileProgram&, const TileOp&, TileKernelState&);   // dispatch on op.kind
```

`TileProgramReference` becomes a thin in-order driver over `apply`; the transactional
executor is a dataflow-order driver over the same `apply`. **One implementation, so
bit-exactness is structural rather than a thing we test for and hope.** The existing
reference tests must pass unchanged after the refactor — that is the refactor's acceptance
criterion.

### 3.2 Why dataflow order gives the same numbers

- **Accumulation order.** `MatMulAccum` is a read-modify-write on its output tile
  (`tile_dependencies.hpp` records it that way, emitting `TileWaw` between successive
  accumulations). Successive K-slices into the same
  `C[ti,tj]` are WAW-ordered, so they retain **program order**, and float summation order
  is unchanged.
- **Disjointness.** Independent ops write disjoint tiles, by declared tile I/O (D6 §3a).
- **Single-threaded firing.** Concurrency is modeled *in time*, not executed in parallel.
  At each event the kernel runs to completion on the host before the next event is
  processed, so there is no interleaving inside a tile op and no data race on
  `TileKernelState`.
- **Determinism.** Events are ordered by `(time, op_index)`; the op index breaks every
  tie, so a run is reproducible and independent of container iteration order.

### 3.3 One correctness gap to close: pivot-slot anti-dependencies

`TileDag` records the pivot slot's producer→consumer edge only: `LuDiagFactor` writes
`slot_writer[slot]`, and `PivotApply` takes a dependency on that writer
(`tile_dag.hpp:305-312`). There is **no edge stopping a later `LuDiagFactor` from
overwriting a slot while earlier `PivotApply`s still have to read it** — and
`exec_lu_diag_factor` begins with `pivots_[slot].clear()`
(`tile_program_reference.hpp:129-130`).

Today's derivation is safe by luck: `slot = k`, unique per panel
(`derive/lu_tile_program.hpp:48`). In dataflow order that safety must be explicit.

**Closed in increment 2** (`include/sw/kpu/program/tile_dependencies.hpp`): pivot slots
now carry `PivotRaw`, **`PivotWar`** and **`PivotWaw`** edges, modelled exactly as tile
hazards are. Slot reuse is therefore *correct* rather than lucky, which supersedes the
interim plan to refuse a program that reuses a slot — refusing would reject a program the
model now orders properly. Reuse is still **reported** (`reused_pivot_slots()`), because it
serializes panels that would otherwise be independent, so a generator doing it by accident
should find out.

## 4. The firing rule

An op is **ready** when all three hold (ADR D3.2):

1. **Dependencies satisfied** — every RAW/WAR/WAW predecessor over tiles *and* pivot slots
   has completed.
2. **Inputs resident** — every declared input tile is resident at the buffer level its
   consumer reads from, on the CF tile the placement assigned.
3. **Credit available** — a free slot exists for each output tile that this op
   **materializes**, plus a free resource of the required kind (CF tile for compute, hop
   lane for movement).

   **An already-resident output acquires no second credit.** `MatMulAccum` writes the same
   `C[ti,tj]` on every K-slice; only the first accumulation materializes that tile and takes
   a slot, and the rest reuse it. Requiring a free slot per output *op* would refuse valid
   programs whenever capacity is full — a blocked accumulation that already owns its
   destination. The slot is returned only after the tile's last consumer completes (§5),
   which for an accumulated output is its `Drain` or final reader, not the last
   accumulation.

A ready op **fires immediately**; there is no priority search and no lookahead. When more
than one op is ready for the same free resource, the tie-break is **lowest op index**,
which is the program order the compiler emitted. That keeps the executor faithful to the
program rather than optimizing it.

> **Why not reuse `TileDag::list_schedule`?** It is a greedy critical-path-first list
> scheduler (`tile_dag.hpp:94-151`) — an optimizer that *chooses* an order, which is the
> placement pass's job, and it rescans every node per pick (O(n²)). It stays in the
> harness as the analytical lower bound and design-of-experiments tool. Since increment 2
> the **dependency construction is shared** (`tile_dependencies.hpp`, which `TileDag` now
> delegates to); what the executor does not borrow is the *scheduling*.

### Stall and refusal

- The run **stops and refuses** only when **all three** hold: no op is ready, **the event
  queue is empty**, and no resource is in flight. An empty ready set on its own is the
  normal state between pipeline stages — a transfer or a compute is outstanding and its
  completion event will release successors — so refusing on that alone would reject valid
  runs. The refusal carries a diagnosis: which ops are waiting, on which tiles or credits,
  and which level is exhausted. A wedged run must never be reported as a slow one.
- A cheap **pre-execution check** reuses the harness's static feasibility test
  (`characterization.hpp:143`, peak live tiles versus L3 capacity) to refuse an
  over-committed program before executing anything.
- An op kind the executor cannot perform **aborts with the op name** (ADR D3.7). Never
  skipped, which is the failure mode that silently dropped bias and activation on the
  DMProgram path.

## 5. Credits and capacity

Capacity is modeled **in tiles**, per level, per placement unit — the natural granularity
for this tier, and it matches how the program talks about the machine.

| Level | Counted in | Scope |
|---|---|---|
| L3 | tile slots | per L3 tile in the topology |
| L2 | tile slots | per CF tile's staging buffers |
| L1 | tile slots (operand vectors) | per CF tile |

Rules:

- **Credit acquire** happens when a tile's transfer into a level begins; **credit return**
  happens when the tile's **last consumer at that level** completes. Last-use is computed
  statically from the DAG, so returns are deterministic.
- **A `Drain` is a consumer, not a deallocator.** It reads its tile, so it does **not**
  release the slot on its own. The slot is released when the last consumer at that level
  completes — which is often, but not always, that `Drain`: a tile drained and then read
  again by a later op keeps its slot until that later read finishes. This is the exact rule,
  not an implementation choice, because differing release timing would produce different
  capacity stalls and different refusals from the same program.
- **Residency is checked before movement.** A `Feed` of a tile already resident at the
  target level costs nothing — this is how on-chip reuse appears (D6 §3, "tile residency +
  re-injection"), and it is the main thing the tier must get right for tiled GEMM, where
  one `B[tk,tj]` serves a column of output tiles.
- **Credits gate, they never reorder.** Blocking on a credit stalls that op only; other
  ready ops continue. That is buffer back-pressure in the credit-based dataflow model
  (`CLAUDE.md`, "Credits UP, Data DOWN") at tile granularity.
- Capacity `0` in the descriptor means **unbounded** for that level, which preserves the
  harness's current behavior for design-space sweeps.

`DeviceDescriptor` gains: per-level tile capacities, per-hop lane counts and bandwidths
(§6), and optional fixed per-op overheads (§7). Existing fields keep their meaning, so
harness experiments keep working.

## 6. Movement, per hop

Movement is a chain of **CSP processes**, each reading one physical memory and writing the
next. There is no aggregate "movement" resource, and — the point this section previously got
wrong — **there is no opportunity to collapse hops, because the physical pathways are not
there.** A span always contains all of its hops.

### 6.1 The path, as the machine has it

For a byte in DRAM to reach an input stream port on a row or column edge of the compute
fabric:

| # | Hop | Governing CSP process | Notes |
|---|---|---|---|
| 1 | DRAM → L3 | **DMA** | picks the data up out of DRAM |
| 2 | L3 → L2 | **BlockMover** | may **restructure/reshape** the data |
| 3 | L2 → L1 | **Streamer** | writes an L1 stream buffer |
| 4 | L1 → fabric | the **L1 stream buffers** push elements in | not a mover: see below |

Inside the fabric the domain flow program keeps pushing data toward a result, and that
result is **pushed back out to the L1 stream buffers**. The return path is the same
processes in reverse:

| # | Hop | Governing CSP process | Notes |
|---|---|---|---|
| 5 | L1 → L2 | **Streamer** | reads an L1 buffer holding results |
| 6 | L2 → L3 | **BlockMover** | |
| 7 | L3 → DRAM | **DMA** | only if the result needs to reach DRAM |

**Reuse** does not shortcut the chain, it re-enters it. Either a **BlockMover** moves the
data from one L3 to another L3 **via the NoC**, or a **BlockMover** reads the L3 and writes
an L2, from where it is pushed down in the regular fashion (Streamer → L1 → fabric) to
participate in another computation.

### 6.2 What this forbids

- **No collapsed hop.** L3→L2 and L2→L1 are distinct processes over distinct pathways;
  modelling them as one stage models a machine that cannot be built. An earlier revision of
  this section offered a "minimum viable scope" that collapsed them and called the collapse
  a descriptor setting — that was wrong, and #264 increment 5 implemented it before the
  error was caught.
- **No hop terminating at the fabric.** The fabric's only data interface is L1 (ADR 0002
  §3.3). Hop 4 is the L1 buffers pushing elements in, not a mover carrying a tile, so it
  owns no lanes and no bandwidth.
- **No skipped middle.** Residency changes where a chain *starts*, never which hops it
  contains: a tile already in L3 begins at hop 2, and a result consumed again from L3
  re-enters at hop 2 or crosses the NoC. Dropping a *prefix* the data has already traversed
  is reuse; dropping an *interior* hop is a pathway that does not exist.

### 6.3 Lanes and bandwidth (normative)

Lanes belong to the **process**, because that is the physical resource: DMA engines,
BlockMovers, Streamers, NoC links. Inbound and outbound legs of the same process share its
pool — there is one set of BlockMovers, not one per direction.

Left ambiguous, the same descriptor yields different makespans in different
implementations, which would make calibration meaningless. So:

- **Every `*_bytes_per_cycle` is per lane, not aggregate.** A process's peak throughput is
  `lanes × bytes_per_cycle`.
- **One transfer occupies exactly one lane** for its whole duration. Transfers are not
  striped across lanes.
- **Lanes give concurrency, never speed-up.** A single transfer's duration is independent of
  how many lanes are idle: `duration = ceil(bytes / bytes_per_cycle)` on its one lane. More
  lanes let more transfers overlap; they never shorten one transfer.
- A transfer occupies its lane from start to completion, with no preemption and no
  re-ordering once started.

Hops for one tile are **pipelined, not serialized end-to-end**: hop *n+1* may start as soon
as hop *n* completes for that tile, and different tiles occupy different hops concurrently,
bounded by each process's lanes. Pipelining is how the chain stays affordable — it is not an
excuse to shorten it.

L3↔L3 NoC moves exist only for topologies that declare them, and stay unused until
multi-compute-tile execution lands (#244). Modelling them keeps the descriptor honest about
what the topology can do.

## 7. Durations and the event engine

**Compute duration**, in order of preference:

1. From L1, when a `StreamProgram` is present: the systolic wavefront latency, with drains
   stretched by the C-stream bubble. `TileDag::l1_duration_` already does exactly this,
   including clamped trailing-tile extents (`tile_dag.hpp:163-190`) — reuse it.
2. Otherwise the lumped model: `macs / fabric_macs_per_cycle`, plus an optional fixed
   per-op-kind overhead from calibration.

**Movement duration:** `bytes / bandwidth` for the hop, plus optional fixed setup cost.

**Queueing** is emergent: an op waits for a free resource, and that wait *is* the queueing
delay. There are no explicit queue-length distributions in the first version (ADR §7.3:
deterministic calibrated means first; variance is #268).

**Engine:** a min-heap of `(time, op_index, event_kind)` events with a ready-set, O(E log E)
in events, no idle-cycle stepping. Target scale: **10⁶ tile ops** without pathological
memory growth, since a real ResNet or a large GEMM at T=16 produces very many tile ops. The
harness's O(n²) ready-scan is explicitly not the model here.

### 7.1 Cycle quantization and zero-work ops (normative)

`Cycle` is `uint64_t` (`include/sw/kpu/timing/tile_descriptor.hpp:20`), while the cost
models produce doubles, so the conversion has to be pinned down:

- **Round up:** `cycles = ceil(duration_double)`. Truncation would let sub-cycle work become
  0, silently contradicting §9: aggregate compute cycles must be positive whenever the run
  contains a non-zero-work compute op.
- **Positive floor:** any op with **non-zero work** costs **at least 1 cycle**, even after
  rounding. A 1-element tile compute is cheap, not free.
- **Genuinely zero-work ops cost 0 cycles** and complete at the current time. The two cases
  are a `Feed`/`Drain` of a tile already resident at the target level (§5), and a
  degenerate tile with an empty extent. They still appear in the timeline, marked
  `zero_work`, so a reader can see the reuse rather than wonder where the transfer went.
- **Zero-duration events cannot stall the engine.** Each op fires exactly once, so a
  zero-duration completion is popped at the current time, may enqueue successors at that
  same time, and the heap drains monotonically: time never moves backwards, and the total
  number of events is bounded by the op count. Ties at equal time are broken by op index, so
  a chain of zero-work ops resolves in program order within one timestamp.

## 8. Calibration against the cycle-accurate tier

Transactional timing is only as good as its calibration (ADR D3.5, D5).

**What gets fitted:** effective `fabric_macs_per_cycle` per CF tile, per-hop bandwidths,
and per-op-kind fixed overheads.

**Procedure:**

1. Sweep GEMM over sizes, tile sizes and (where the CSP path supports it) dataflow
   strategies, recording CSP cycles.
2. Fit the coefficients by least squares on those points.
3. Report **relative makespan error** — median and p95 — over a held-out set of sizes, so
   the fit is not scored on its own training points.
4. Commit the coefficients as a named device profile, with the fit report next to it.

**The band (decided):** **median ≤ 10%, p95 ≤ 25%** relative makespan error against the CSP
tier. A model that clears that is calibrated for GEMM; one that does not is reported as
uncalibrated rather than quietly shipped.

**CI guard:** a test asserts the band on a small fixed sweep, so calibration drift fails a
build rather than degrading quietly. The band may be tightened as the model improves; it is
not to be loosened without a recorded reason in the fit report.

**Known limitation, to state in the fit report:** the CSP tier is single-compute-tile
today, so multi-CF coefficients are **extrapolation** until #244 lands. Any result with
`compute_tiles > 1` is reported as uncalibrated.

## 9. Results and provenance

`RunResult` carries, besides the mutated operand buffers:

- **timeline**: per op — kind, start, finish, resource, waited-on reason, and
  **`zero_work`** (bool), set for the zero-duration cases §7.1 enumerates. Without that
  flag a reader cannot tell deliberate reuse — a feed of an already-resident tile — from
  work that went missing, which is exactly the distinction the reuse model needs to show.
- **stats**: run-level aggregates — makespan; **aggregate compute cycles**, summed over
  compute ops only, which is `0` exactly when the run contains no non-zero-work compute op
  and positive otherwise; per-hop busy cycles and utilization; peak residency per level;
  credit stalls per level; MACs and bytes. Zero-work ops (a resident-tile `Feed`/`Drain`, an
  empty extent — §7.1) contribute nothing here and are never counted as compute work. The
  per-op view lives in `timeline`; `stats` never repeats it per op.
- **bounds**: the analytical lower bound, computed against the **executor's own** resource
  model, not the harness's aggregate one:

  ```
  lower_bound = max( critical_path,
                     compute_work / compute_tiles,
                     max over hops h of ( work(h) / (lanes(h) × bytes_per_cycle(h)) ) )
  ```

  The harness's `lower_bound` (`tile_dag.hpp:147`) divides total movement work by one
  aggregate `move_lanes`, which is a different — and weaker — bound now that movement is
  per hop (§6). Both may be reported, but the executor's bound is labelled as such, and a
  harness-derived figure is labelled `harness_bound` so the two are never compared as if
  they were the same quantity.
- **provenance**: device profile name, placement used, whether L1 was present, the seed,
  and whether any coefficient was extrapolated

Provenance matters because timing depends on placement (ADR §5 risk): a number without its
placement and profile cannot be compared with another number.

## 10. Increments

1. **Extract the tile kernels** (§3.1). Reference tests pass unchanged. No behavior change.
2. **Dependency model** — **done, for the model itself.** Tile RAW/WAR/WAW, feed
   availability, and pivot-slot RAW/WAR/WAW, with typed edges, reuse reporting and
   blocked-op diagnosis (`build_tile_dependencies`,
   `TileDependencies::explain_blocked`). The only consumer wired to it so far is
   **`TileDag`**, which now delegates instead of carrying a second copy. Executor reuse —
   firing against these edges and using `explain_blocked` for stall diagnosis — is
   increment 3; nothing calls it from an executor yet, because no executor exists.
3. **Executor skeleton** — **done.** `TileTransactionExecutor`
   (`include/sw/kpu/program/tile_transaction_executor.hpp`): event engine over
   `(time, op index)`, the §4 firing rule against the shared dependency model, a real
   `Placement` (unpinned or JIT-style pinned), compute-tile and movement-lane resources,
   §7.1 quantization, stall refusal with `explain_blocked` diagnosis, and a
   `RunResult` carrying timeline, stats, bounds and provenance. Acceptance met: **bit-exact**
   GEMM (37x29x23 on a 16-tile, so every trailing tile is clamped) and tile LU — values,
   permutation and swap count — versus the reference; compute cycles non-zero and scaling
   with M, N, K and tile size; identical cycles and timeline across repeated runs. No
   capacity limits yet, so the refusal path is unreachable until increment 4.
4. **Credits and capacity** (§5) — **done for L3.** Tile-granularity L3 slots with
   residency-based reuse (a feed of a resident tile costs nothing), release when the
   *last user completes*, and refusal with a capacity diagnosis. The release is a
   credit-return refcount, not a statically chosen highest-index user: readers of one
   tile are deliberately unordered, so the highest-indexed reader can finish first, and
   releasing on it frees a slot an earlier reader still holds — which undercounts
   residency and lets a run exceed the capacity being enforced. §5 already specified the
   completion rule; the refcount is what implements it. **Slots are acquired in PROGRAM ORDER**, which
   is the invariant that makes the model deadlock-free: the weaker "don't promote a ready
   op past an earlier ready one" rule wedges at every finite positive capacity below the
   unbounded run's peak residency — 29 tiles here, so it fails at 25 despite the live set
   being 21; `l3_tiles == 0` denotes unbounded, not zero capacity — because
   feeds are ready immediately and take every slot while the computes that would retire
   those tiles are not ready yet. At 29 or above it matches unbounded and cannot deadlock,
   so what the weaker rule really lacks is the ability to run between the true live set and
   its own greedy peak. Program-order acquisition bounds the live set by the program-order
   live set, so any budget at or above `peak_live_tiles` is **sufficient** — it is
   guaranteed to complete. It is not always **necessary**: residency reuse and ops that
   need no new slot can reorder enough that a smaller budget still runs (the shared-reader
   regression program has a static peak of 6 and completes in 5). For the derived matmul
   program the two coincide exactly, 21 completing and 20 refusing, and that equality is
   asserted — two independent implementations of the same question agreeing.
   **L2/L1 capacity is NOT here**: §5 counts it per compute tile, so it waits for the
   placement pass to bind tiles to compute tiles (increment 5).
5. **Per-hop movement** (§6) — **done, and then corrected.** Movement is a chain of CSP
   processes with per-process lane pools: DMA (DRAM↔L3), BlockMover (L3↔L2, and L3→L3 via
   the NoC), Streamer (L2↔L1). A `Feed` traverses DMA→BlockMover→Streamer and a `Drain` the
   reverse; hops are **pipelined**, so hop *n+1* starts when hop *n* completes, and §6.3's
   semantics are asserted rather than assumed — one transfer holds one lane for its whole
   duration, and **lanes give concurrency, never speed-up**.

   The first implementation of this increment was **wrong in its central premise**: it
   offered a collapsed single-hop mode and treated the collapse as a descriptor setting,
   following §6's own earlier wording. **Hops cannot collapse** — L3→L2 and L2→L1 are
   distinct processes over distinct pathways, and a span always contains all its hops.
   Corrected in #292: the collapsed mode is gone, not deprecated, because a mode that models
   an unbuildable machine has no valid use. Residency now changes only where a chain
   *starts* (a tile in L3 begins at the BlockMover), never which hops it contains.

   **Still not here:** per-CF-tile lane attribution and L2/L1 capacity, both of which need a
   static binding of compute ops to compute tiles (increment 5b / the placement pass).
6. **Calibration** (§8) with the CI band, plus the device profile and fit report.
7. **Wire to the ADR D2 factory** so `SimulationFidelity::TRANSACTIONAL` reaches it, and
   only through there.

Increments 1–3 are what #265 needs to demonstrate "a program file executes at transactional
fidelity with correct values". Increments 4–6 are what make the timing worth quoting.

## 11. Open questions

Questions 1 and 4 were answered on #269 and are recorded in the sections they affect;
they are kept here, struck through, so the decision trail stays readable.

1. ~~**Initial error band.**~~ **Answered (2026-09-20): median ≤ 10%, p95 ≤ 25%** relative
   makespan error against CSP. Recorded in §8 and asserted in CI.
2. ~~**L2/L1 collapse.**~~ **Answered (2026-09-23): the question was malformed.** It asked
   whether one on-chip hop is acceptable for a first calibrated version. It is not
   acceptable at any version: BlockMover (L3↔L2) and Streamer (L2↔L1) are distinct CSP
   processes over distinct physical pathways, so a collapsed hop models a machine that
   cannot be built. They are separate from the start, and a span always contains all its
   hops (§6.2).
3. ~~**Drain semantics.**~~ **Resolved in §5 (2026-09-20):** a `Drain` is a consumer, not a
   deallocator — the slot is released when the last consumer at that level completes, which
   may or may not be the `Drain` itself.
4. ~~**Placement interface.**~~ **Answered (2026-09-20): a real `Placement` object**, which
   the JIT later replaces. `Placement::single(device)` supplies the default, so no retrofit
   is needed when #230 increment 3 lands. Recorded in §2.
5. **Where DNN tile kinds land** (ADR D7): in `tile_kernels.hpp` beside the linear-algebra
   kernels, or in a separate `dnn_tile_kernels.hpp`? This decides whether one header grows
   without bound.
6. **Program-level parallel operands.** Feeds of the *same* tile by different consumers are
   independent today (`tile_dag.hpp:290-297`). Confirm that stays true once capacity is
   modeled, so a shared operand tile doesn't serialize independent output tiles through one
   credit.

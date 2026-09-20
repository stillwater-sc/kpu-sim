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
  (`tile_dag.hpp:299-302` already records it that way). Successive K-slices into the same
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
(`derive/lu_tile_program.hpp:48`). In dataflow order that safety must be explicit. The
executor will:

- add **WAR/WAW edges on pivot slots**, exactly as it does for tiles; and
- **validate single-assignment** of each slot at load time, and refuse a program that
  reuses one, until the anti-dependency is modeled.

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
> harness as the analytical lower bound and design-of-experiments tool. The executor
> borrows its **dependency construction**, not its scheduling.

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

Today the harness has a single aggregate `move_lanes` pool. That cannot express the real
bottleneck, which is usually DRAM bandwidth rather than on-chip movement. The executor
models each hop separately (ADR D3.4):

| Hop | Realized by | Resource | Duration |
|---|---|---|---|
| DRAM → L3 | DMA | DRAM lanes, DRAM bandwidth | `bytes / dram_bytes_per_cycle` |
| L3 → L2 | BlockMover | per-CF-tile mover lanes | `bytes / bm_bytes_per_cycle` |
| L2 → L1 | Streamer | per-CF-tile streamer lanes | `bytes / str_bytes_per_cycle` |
| L3 → L3 | neighbor moves (checkerboard) | link lanes | `bytes / link_bytes_per_cycle` |

- An L0 `Feed` expands into the hop chain needed to make its tile resident where the
  consumer reads it, **skipping hops already satisfied by residency**. A `Drain` is the
  reverse chain.
- Hops for one tile are **pipelined, not serialized end-to-end**: hop *n+1* may start as
  soon as hop *n* completes for that tile, and different tiles occupy different hops
  concurrently, bounded by each hop's lanes.
- L3↔L3 links exist only for topologies that declare them, and stay unused until
  multi-compute-tile execution lands (#244). Modeling them now keeps the descriptor honest
  about what the topology can do.

### 6.1 Lane and bandwidth semantics (normative)

Left ambiguous, the same descriptor yields different makespans in different
implementations, which would make calibration meaningless. So:

- **Every `*_bytes_per_cycle` is per lane, not aggregate.** A hop's peak throughput is
  `lanes × bytes_per_cycle`.
- **One transfer occupies exactly one lane** for its whole duration. Transfers are not
  striped across lanes.
- **Lanes give concurrency, never speed-up.** A single transfer's duration is independent of
  how many lanes are idle: `duration = ceil(bytes / bytes_per_cycle)` on its one lane. More
  lanes let more transfers overlap; they never shorten one transfer.
- A transfer occupies its lane from start to completion, with no preemption and no
  re-ordering once started.

**Minimum viable scope:** DRAM→L3 and L3→CF (L2 and L1 collapsed into one hop). The table
is the target; the first increment may collapse the on-chip hops, provided the collapse is
a descriptor setting rather than a hardcoded assumption.

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
2. **Dependency model**: tiles plus pivot-slot RAW/WAR/WAW, slot single-assignment
   validation, stall diagnosis. Reuses `TileDag`'s construction.
3. **Executor skeleton**: event engine, firing rule, compute resources only, no capacity
   limits. Acceptance: **bit-exact** GEMM and tile LU versus the reference, including
   ragged trailing tiles; non-zero compute cycles scaling with M, N, K and tile size;
   identical cycles across repeated runs.
4. **Credits and capacity** (§5), including residency-based reuse and refusal of
   over-committed programs.
5. **Per-hop movement** (§6), starting from DRAM→L3 plus a collapsed on-chip hop.
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
2. **L2/L1 collapse.** Is one on-chip hop acceptable for the first calibrated version, or
   should BlockMover and Streamer be separate from the start?
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

# E8: Online Softmax CSP Movement Pattern

**Status:** DRAFT (softmax-T1, issue #154) — awaiting review
**Epic:** #77 (pattern class P3, row-streaming reduction)
**Depends on:** E3 (#72, online reduction) — complete
**Date:** 2026-07-14
**Will unlock (once T2–T5 land):** the flash-attention rescaling
primitive (E12), the M4 attention milestone's `softmax.functional` gate,
and — via its new schedule-tier resident-dependency mechanism — the
LayerNorm/RMSNorm apply phase (E9) and the reduction ROW_NORMALIZE apply
phase that E3-T4 deferred.

---

## 1. The pattern, characterized

### Safe online softmax

`softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)`, with `m = max_j x_j`
subtracted for numerical stability. The **online** (single-pass-stats)
form computes `m` and the normalizer `l = sum_j exp(x_j - m)` in ONE pass
over the row using a running max with sum rescaling, instead of the naive
two stat passes (one for max, one for sum):

```text
m = -inf ; l = 0
for each tile X_t in the row:              # stats pass (one pass)
    m_t = max(X_t)
    if m_t == -inf: continue               # all-(-inf) tile contributes nothing;
                                           # skipping avoids exp(-inf - -inf) = NaN
    m_new = max(m, m_t)
    l     = l * exp(m - m_new) + sum_j exp(X_t[j] - m_new)   # rescale on new max
    m     = m_new
for each tile X_t in the row:              # apply pass
    Y_t = (l > 0) ? exp(X_t - m) / l : 1/N # nonempty all-(-inf) row -> uniform
```

The `m_t == -inf` guard is load-bearing: on the first tile `m` is still
`-inf`, so without it `exp(m - m_new) = exp(-inf - -inf) = NaN` would
poison `l` for an all-`-inf` prefix. With it, `m` stays `-inf` until a
finite element arrives (then `exp(-inf - finite) = 0`, no NaN), and a
whole-row-`-inf` case leaves `l = 0`, handled by the apply fallback.

The load-bearing move is `l * exp(m - m_new)`: when a later tile raises
the max, every exp already accumulated into `l` was taken against the old
max and must be down-scaled by `exp(m_old - m_new)`. This is the same
rescale flash attention applies to its partial outputs — E8 is that
primitive in its standalone (materialized-row) form, and E12 fuses it.

### Movement signature

The stats pass is a P3 streaming reduction whose accumulator is the pair
`(m, l)` — a two-lane compute-resident state, exactly the E3 accumulator
shape (`VE_REDUCE` moment-triplet machinery generalizes: E8 needs lanes
`[m, l]`). The stats pass therefore inherits E3's constant working set of
2 for its own streaming.

The apply pass reads each row tile a **second** time and needs the
finalized `(m, l)`. Two realizations, chosen a priori by the envelope
(the #67 discipline), exactly as E3's ROW_NORMALIZE:

1. **Row-resident** (`reduction_tiles + 2 <= per_matrix_burst_share`):
   each row tile is loaded from DRAM **once** (one LOAD + one MOVE) and
   retained in L2 across both phases via `consumer_count = 2` (the E2
   1:1:k discipline, k=2), which seeds the L2 TagCAM ref-count to 2 at
   MOVE time. Two FEEDs then consume the same resident L2 tile — the
   first for stats, the second for apply — and the L2 credit is released
   only after the second FEED. So the movement per row tile is
   `LOAD, MOVE, FEED(stats), FEED(apply)`: **one** DRAM read, **two**
   feeds. `(m, l)` reaches the apply computes as a **compute-resident
   dependency** (Section 2). DRAM traffic: 1 read + 1 write per element,
   the online-softmax payoff over the 4-pass generator's extra scratch
   round-trips.
2. **Re-streamed** (otherwise): the row is re-read from DRAM for the apply
   pass; `(m, l)` still rides the resident path. Constant working set 3.

This removes the existing multi-pass generator's `reduction_tiles + 2`
residency floor (its exp-scratch tiles pinned from pass 2 to pass 4) and
its #139 DRAIN-without-COMPUTE defect — the online generator supersedes
`SoftmaxScheduleGenerator`.

## 2. The schedule-tier resident-dependency mechanism (E8's core capability)

E3-T4 deferred ROW_NORMALIZE's apply phase because the stat reached the
apply computes only via a DRAM round-trip (store then reload), which is
timing-correct but **races in the value plane** — nothing orders
`load(stat)` after `store(stat)`. The correct delivery is a
**compute-resident dependency**: the `(m, l)` tile produced by the stats
compute stays in the compute fabric and feeds the apply computes without a
drain/reload. The executor already models this for matmul
(`MatMulComputeSpec::resident_tiles`) and functional computes
(`FunctionalComputeSpec::resident_tiles`, the #66 per-instance
`completed_compute_counts` accounting); what is missing is a way to
express it at the **schedule tier**. E8-T2 adds it:

- `ScheduleOperation` gains `std::vector<TileID> resident_tiles` — inputs
  a COMPUTE consumes from compute storage (produced by a prior COMPUTE)
  rather than from a fresh FEED.
- `ConcurrentTimingExecutor` gains
  `schedule_compute(tile, feed_deps, resident_deps)` — recording
  `resident_dependencies` on the `PendingCompute` alongside the feed
  dependencies (the field already exists and is honored by
  `dependencies_satisfied`). Timing-only computes thus gain resident
  ordering with no value plane.
- `ScheduleExecutor` routes a COMPUTE carrying `resident_tiles` to that
  overload; the `FunctionalComputeBinder` maps them straight onto
  `FunctionalComputeSpec::resident_tiles`.

This is a small, general addition — it is the capability the norm apply
phase (E9) and the deferred reduction ROW_NORMALIZE both need, so it is
built once here and reused.

**Ordering guarantee it provides:** an apply COMPUTE with
`resident_tiles = {ml_state}` cannot start until the stats COMPUTE that
produced `ml_state` has completed (its `completed_compute_counts` reaches
the required count), so the running state is final before any normalize
reads it. No DRAM round-trip, no race.

## 3. Rescale-on-new-max value semantics (T2)

The stats COMPUTE folds a tile into `(m, l)` with the rescale. On the
value tiers:

- **CSP functional:** the stats op is a chained-accumulator compute (the
  E3 model) whose state is `[m, l]`; each tile applies
  `m_new = max(m, tile_max)`, `l = l*exp(m-m_new) + Σexp(tile-m_new)`.
  The apply op reads the row tile and resident `[m, l]` and emits
  `exp(x - m)/l`.
- **Behavioral ISA:** softmax on the behavioral tier is a *program* of
  the existing `VE_REDUCE` (running max/sum) and `VE_ELEMENTWISE`
  (sub/exp/div) ops — the `KernelCompiler` already emits one — not a new
  single op. (This is the T2 refinement of the T1 design: the genuinely
  new capability is the schedule-tier resident dependency, Section 2, not
  an ISA op.) The `[m, l]` running state lives in the CSP functional
  binder above, which is where the online movement pattern is validated.

**Edge cases (defined, pinned in T4/T5), distinct from each other:**

- **Empty row** (`reduction_elems == 0`): rejected at generation
  (dimensions must be non-zero) — softmax over zero elements is undefined,
  so `1/N` is never evaluated with `N == 0`.
- **Nonempty all-`-inf` row** (every element `-inf`, e.g. a fully masked
  attention row): the stats guard leaves `l = 0` and `m = -inf`; the
  apply emits the **uniform** distribution `1/N` over the row's `N`
  elements, matching a host safe-softmax reference that special-cases the
  degenerate normalizer. This is a real case in masked attention, not a
  theoretical one.

## 4. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 #155 (ISA/executor closure) | `ScheduleOperation::resident_tiles`; `ConcurrentTimingExecutor::schedule_compute(tile, feed, resident)`; `ScheduleExecutor` routing + binder mapping; resident-ordering validation test. (No new behavioral ISA op: softmax on the behavioral tier composes existing `VE_REDUCE` + `VE_ELEMENTWISE`, per Section 3; the `[m, l]` rescale semantics live in the CSP binder, T3/T4.) |
| T3 #156 (generator) | `OnlineSoftmaxScheduleGenerator` (row-resident / re-streamed realization, executable COMPUTEs, `(m,l)` resident hand-off); supersedes the 4-pass `SoftmaxScheduleGenerator`; resolves #139 for softmax |
| T4 #157 (functional + oracle) | Value-producing online softmax on the CSP executor vs a host safe-softmax oracle (max-subtracted), default + constrained envelopes; behavioral ISA path |
| T5 #158 (regression) | shape × envelope matrix, credit/stall invariants, and a single-pass-vs-multi-pass DRAM-traffic comparison (the payoff); characterization on the epic; closes the epic |

## 5. Risks

- **Resident-dep executor change** touches the #66 compute accounting: the
  new `schedule_compute` overload must record `resident_dependencies`
  exactly as matmul/functional already do, and the existing matmul/
  reduction paths must be unaffected (T2 keeps them green).
- **Numerical stability**: the rescale must use the *updated* max, and the
  `m_t == -inf` guard (Section 1) must precede it to avoid
  `exp(-inf - -inf) = NaN`. The oracle compares against a host
  safe-softmax with relative tolerance (exp/division amplify fp error).
  The two degenerate rows are handled distinctly (Section 3): empty rows
  are rejected at generation, nonempty all-`-inf` rows emit uniform.
  Both, plus an all-`-inf` *prefix* followed by finite values, are pinned
  in T4/T5.
- **Realization selection** reuses E3's envelope formula; the row-resident
  `consumer_count = 2` delivery holds an L2 credit across both phases —
  T5 asserts credit conservation on the realization boundary.
- Superseding `SoftmaxScheduleGenerator` must keep any current callers
  working: T3 leaves the old generator in place (deprecated) until the
  softmax coverage row is fully online, then removes or redirects it.

## 6. Projected coverage-matrix effect (conditional on T2–T5 landing)

Nothing is claimed done by this design (the #93 contract: cells flip in
the delivering PR). WHEN T2–T5 land, the `softmax` row goes to done across
its stages, the schedule-tier resident-dependency mechanism unblocks E9
(norm apply) and the E3 ROW_NORMALIZE apply phase, and the M4
`softmax.functional` gate is satisfied.

# E3: Streaming / Online Reduction Pattern

**Status:** DRAFT (reduce-T1, issue #104) — awaiting review
**Epic:** #72 (pattern class P3, row-streaming reduction)
**Date:** 2026-07-13
**Will unlock (once T2-T5 land):** online softmax (E8),
LayerNorm/RMSNorm/BatchNorm statistics (E9), global pooling (E7),
flash-attention rescaling (E12) — and the remaining half of `#18`
(VE_REDUCE numerical semantics) closes with T5.

---

## 1. The pattern, characterized

### P3: row-streaming reduction

A stream of tiles is consumed by a **running statistic** — max, min, sum,
mean, variance — held in an accumulator that never leaves the compute
fabric while the stream flows past it. The defining movement property is
the inverse of P6: instead of N inputs producing N outputs with no reuse,
N inputs produce ONE output (the stat), and the accumulator is the only
state carried between tiles.

As a system of recurrence equations (the SURE/domain-flow view):

```text
s_0     = init(op)
s_k     = combine(op, s_{k-1}, local_reduce(op, X_k))    k = 1..N
stat    = finalize(op, s_N)
```

The dependence vector of the combine chain is (1) — a 1-D linear
recurrence. Two consequences fall out directly:

1. **Within a tile, reduction is fully parallel** (`local_reduce` is a
   tree over `tile_elems` lanes); **across tiles, the combine is
   sequential** — s_k cannot start before s_{k-1}. The CSP latency model
   is therefore matmul-shaped: the accumulation chain scales with the
   stream length exactly as the K dimension scales a matmul
   (`compute_cycles_per_k_slice` per chain step), and the executor's
   existing K-scaled latency applies unchanged.
2. **The accumulator is a resident tile**, not a streamed one. It
   consumes compute-fabric storage, never an L2 bank credit, and drains
   exactly once at the end. Credit analysis: the stats pass has a
   **constant working set of 2** (one streaming tile in flight + the
   output stat burst), independent of stream length. Streaming reduction
   can never be refused by the envelope on stream length — only degenerate
   pools refuse it.

### The two-phase problem (stats are half the story)

Every P3 consumer applies the stat back to the data: softmax normalizes
by max and sum, layernorm by mean and variance. The data must be seen
**twice** — once to compute the stat, once to apply it. Three movement
realizations, in order of preference:

1. **Row-resident two-phase** (when the row fits the L2 share): row tiles
   are delivered once with `consumer_count = 2` — the E2 1:1:k discipline,
   with k=2 — fed first to accumulate, fed again to apply the finalized
   stat. DRAM traffic: 1 read + 1 write per element (optimal).
   Working set: `reduction_tiles + 2` — precisely the formula the #90
   softmax/layernorm envelope analysis already enforces.
2. **Re-streamed two-pass** (when the row does not fit): stats pass
   (constant working set 2), then a P6-style apply pass re-reading the
   row from DRAM with the stat as a P5 broadcast operand. DRAM traffic:
   2 reads + 1 write. Working set: constant 3. **This is what makes big
   rows executable at all** — today's softmax/layernorm generators refuse
   them a priori.
3. **Online rescaling** (flash-style): outputs kept resident and rescaled
   as the running stat evolves — this fuses the two phases into one pass
   but requires the P8 chained-fusion machinery and is **explicitly E12
   scope**. E3 designs the stat state so E12 can rescale (max and sum
   exposed separately, not pre-divided), and stops there.

**Envelope-derived selection formula** (the #67 constructive-safety
discipline):

```text
row_resident_ok := reduction_tiles + 2 <= per_matrix_burst_share(l3, l2)
realization     := row_resident_ok ? ROW_RESIDENT : RESTREAMED
```

The generator picks the realization a priori and stamps it in the
metadata; there is no empirical fallback. Both realizations are
livelock-safe by construction (constant or envelope-checked working set,
paired emission for the apply pass).

## 2. VE_REDUCE operand encoding (the #18 numerical half, E3's share)

Today `VE_REDUCE` carries `std::monostate`; the assembler consumes the op
token and discards it. E3-T2 gives it real operands:

```cpp
enum class VEReduceOp : uint8_t {
    MAX, MIN, SUM,      // scalar accumulator state
    MEAN, VAR           // moment state: [count, sum, sumsq]
};

struct VEReduceOperands {
    VEReduceOp op = VEReduceOp::SUM;
    uint8_t phase = 0x3;    // bit 0: INIT (write the op identity),
                            // bit 1: ACCUMULATE (combine src into state),
                            // bit 2: FINALIZE (state -> stat values)
    uint8_t l1_src = 0;     // input L1 buffer (tile being consumed)
    uint8_t l1_acc = 0;     // accumulator L1 buffer (read-modify-write)
    // element count rides the SET_TILE_DIM auto state, like VE_ELEMENTWISE
};
```

Design decisions:

- **Accumulator ABI (fixed, shared by assembler / serializer / behavioral
  executor / CSP binder).** The accumulator lives at **offset 0 of
  `l1_acc` as three contiguous fp32 lanes**, regardless of op (unused
  lanes are zero). `SET_TILE_DIM`'s element size applies to `l1_src`
  only; the accumulator is always fp32.

  | op | raw state (during ACCUMULATE) | after FINALIZE |
  |---|---|---|
  | MAX / MIN / SUM | `acc[0]` = running stat, identity -inf / +inf / 0 | `acc[0]` unchanged (already the stat) |
  | MEAN | `acc[0]`=count, `acc[1]`=sum, `acc[2]`=sumsq | `acc[0]`=mean, `acc[1]`=count, `acc[2]`=sumsq |
  | VAR | `acc[0]`=count, `acc[1]`=sum, `acc[2]`=sumsq | `acc[0]`=var, `acc[1]`=mean, `acc[2]`=count |

  VAR's FINALIZE exposing the mean in `acc[1]` is deliberate: layernorm
  needs both, and one VAR reduction supplies them. **Raw (pre-FINALIZE)
  moment lanes are op-private**: consumers read stats only after
  FINALIZE, which is what permits a later Welford upgrade (state lanes
  reinterpreted, ABI unchanged) without touching any consumer. E12's
  flash rescaling reads running MAX and SUM — scalar accumulators whose
  raw state IS the stat — so it is unaffected by moment-lane privacy.
- **The moment triplet is a pure combine** (mergeable across blocks — the
  parallel-combine property the SURE derivation gives us); accumulation
  precision is fp32 with the Welford upgrade path above if long-stream
  error bounds demand it.
- **VAR/MEAN edge semantics are defined, not emergent**:
  `count == 0` at FINALIZE -> mean = var = NaN (empty reduction has no
  answer; NaN propagates honestly). `count == 1` -> var = 0 exactly.
  Variance uses the **population divisor (N)** — matching
  layernorm/batchnorm semantics — and FINALIZE clamps
  `var = max(0, sumsq/count - mean²)` so floating-point cancellation can
  never produce a negative variance (and downstream `sqrt` never NaNs on
  rounding noise).
- **Phase is a flag set, not an enum**: INIT is explicit (writes the op
  identity into the state lanes), so the ISA level never depends on
  "first touch" detection - a single-tile reduction is
  `INIT|ACCUMULATE|FINALIZE` in one instruction, a streamed reduction is
  `INIT|ACCUMULATE`, then N-1 x `ACCUMULATE`, then `FINALIZE` (alone or
  fused with the last ACCUMULATE).
- **Serializer impact:** the `DMInstruction` variant widens 17 → 18
  (`VEReduceOperands` at index 17); static_asserts and the write/read
  branches update accordingly (same drill as VEOperands in #100).
  Existing `.kpubin` v2 files remain readable (new alternative appended).
- The assembler's `parse_ve_reduce` parses
  `VE_REDUCE op, src, acc [, INIT] [, FINALIZE]` (ACCUMULATE implied,
  flags optional); the disassembler mirrors it.

Semantics land in both value-producing executors, exactly as E2 did it:
- `BehavioralProgramExecutor`: replace the no-op case with the running
  combine over the addressed L1 buffers.
- CSP tier: reductions ride `FunctionalComputeSpec` — **no executor core
  change**. The accumulator is the *target tile* of a chain of computes:
  `COMPUTE_k(stat_tile, inputs = {X_k feed dep, stat_tile resident dep})`
  for k ≥ 1, `COMPUTE_0` listing only `X_0`. The per-instance resident
  accounting from #66 (`completed_compute_counts`) already enforces the
  chain order: at schedule time of COMPUTE_k the stat tile has k prior
  scheduled computes, which is exactly the required count. This was
  verified against the machinery while designing — the first compute must
  NOT list the accumulator (required = max(1, 0) = 1 would deadlock).
  The CSP first compute therefore carries `INIT|ACCUMULATE` semantics in
  its lambda — the structural mirror of the explicit ISA INIT flag.

## 3. OnlineReductionScheduleGenerator (T3)

Forms:

| Form | Meaning | Consumers |
|---|---|---|
| `FULL_REDUCE` | one stat over the whole stream (N tiles -> 1) | global pooling, loss reductions |
| `ROW_STATS` | per-row stats over the reduction dim, rows batched | softmax max/sum, norm mean/var |
| `ROW_NORMALIZE` | ROW_STATS + apply phase (two-phase, realization per the envelope formula) | softmax, layernorm substrate |

All forms emit **executable COMPUTEs** with full dependency sets (the
`#101` discipline). `ROW_NORMALIZE` in ROW_RESIDENT realization uses
`emit_broadcast_tile`-style seeded delivery with `consumer_count = 2` per
row tile; in RESTREAMED realization it emits two passes, the second a
paired P6 apply stream with the stat as a P5 resident operand — both
mechanisms exist from E2 unchanged. The multi-pass softmax/layernorm
generators are NOT touched in E3 (their rewrite on top of this generator
is E8/E9 scope, where #139's remaining families get resolved).

## 4. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 #105 (ISA/executor closure) | `VEReduceOperands` + assembler/disassembler/serializer round-trip (variant 18, static_asserts); behavioral running-combine semantics for MAX/MIN/SUM/MEAN/VAR incl. phase flags; CSP chained-accumulator validation test (resident-dep ordering) |
| T3 #106 (generator) | `OnlineReductionScheduleGenerator` (FULL_REDUCE / ROW_STATS / ROW_NORMALIZE), envelope-derived realization selection, executable COMPUTEs, metadata stamping |
| T4 #107 (functional + oracle) | Value-producing CSP execution via `FunctionalComputeBinder` for all five ops x forms, verified against independent host oracles (incl. Kahan-reference for SUM on long streams); behavioral ISA path for streamed ACCUMULATE/FINALIZE sequences |
| T5 #108 (regression) | op x stream-length x envelope matrix (incl. realization boundary cases: row exactly fits / exceeds by one; empty and single-sample VAR/MEAN edge cases), credit conservation + accumulator-residency invariants, characterization on the epic; coverage row flips + #18 closure happen HERE, not before |

## 5. Risks

- **Chain serialization vs. wall-clock**: the combine chain makes the
  stats pass latency-linear in stream length; the pipeline still overlaps
  load/move/feed with compute (as the E2-T5 characterization showed
  stalls absorbing), but T5 must characterize the chain-bound regime so
  E8/E12 know when fusion pays.
- **VAR numerical robustness**: `sumsq/count - mean²` cancels
  catastrophically for large means. The Section 2 semantics bound the
  failure modes (clamp at 0, defined NaN for empty, population divisor),
  and the raw-lane privacy rule permits a Welford upgrade without ABI or
  consumer change. T4's oracle bounds must use relative tolerance, not
  exact match, for MEAN/VAR on long streams, and T5 pins the edge cases.
- **consumer_count = 2 residency** holds an L2 credit for the whole
  two-phase span of a row; the envelope formula accounts for it, but T5
  must assert the credit-conservation invariant on the realization
  boundary (row exactly filling the share).
- Serializer variant widening is the same compatibility drill as #100;
  #144 (operand indices 7-15 round-trip) remains open and untouched.

## 6. Projected coverage-matrix effect (conditional on T2-T5 landing)

Nothing in this section is claimed done by this design: each stage cell
flips only in the PR that delivers it, per the #93 contract. WHEN T2-T5
land: the `online_reduction` row will go to done across all five stages,
`#18` will close entirely (its elementwise half already closed with E2),
and E8 (softmax), E9 (norms), and E12 (flash rescaling) unblock — the M4
attention-path gate requires `online_reduction.functional`. E7 (pooling)
will reuse FULL_REDUCE for global pooling.

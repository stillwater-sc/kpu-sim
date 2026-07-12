# E2: Broadcast + Two-Operand Aligned Streaming Patterns

**Status:** APPROVED 2026-07-12 (bcast-T1, issue #99)
**Epic:** #71 (pattern classes P5 broadcast, P6 two-operand aligned streaming)
**Date:** 2026-07-12
**Unlocks:** bias application, residual add, SwiGLU gating, RoPE tables,
BatchNorm scale/shift — the elementwise substrate of milestone M2 (ResNet)
and the numerical half of #18 for VE_ELEMENTWISE.

---

## 1. The two patterns, characterized

### P6: two-operand aligned streaming (`C = op(A, B)` elementwise)

Two tile streams must arrive **rate-matched** at compute: the operation for
output tile `i` fires only when `A[i]` and `B[i]` have both been fed. The
movement signature differs from matmul's A×B in a crucial way: there is no
reuse — every input tile is consumed exactly once by exactly one output
tile, so the working set is constant (one A tile + one B tile + one C tile
in flight per pipeline stage) regardless of tensor size.

**Credit analysis.** Both streams draw from the same L3/L2 pools. Without
pairing discipline, a greedy prefetcher can fill the pool with far-ahead
A tiles while `B[0]` cannot enter — the single-matrix monopolization
pattern that per-matrix credit partitioning (#89) blocks structurally
(A and B partitions are independent), and that interleaved emission
(`load A[i], load B[i]` pairs, the #67 discipline) prevents by
construction even in shared-pool mode. **Envelope requirement: 3 tiles**
(one per stream + output), identical to conv2d's streaming bound.

**Alignment mechanism.** Nothing new is needed in the executor: the
per-instance feed accounting from #66 already expresses "compute fires
when both operands' required feed counts are met" — a two-operand
elementwise compute is a `FunctionalComputeSpec` with
`input_tiles = {A[i], B[i]}`. The pattern work is in the *schedule*
(paired emission) and the *ISA* (operand encoding), not the executor core.

### P5: broadcast (one small tile → many consumers)

A bias vector, BN scale/shift pair, or RoPE cos/sin table is loaded once
and consumed by every output tile of the operation. Two legitimate
movement realizations:

1. **Resident-operand broadcast** (chosen for E2): the broadcast tile is
   fed once and pinned as a *resident* input across all consuming computes
   — exactly what `FunctionalComputeSpec::resident_tiles` and
   `MatMulComputeSpec::bias` already express, and what the layernorm
   working-set analysis (#90) already accounts for (resident params count
   against the envelope share for their whole lifetime).
2. **Replicated delivery** (`STR_BROADCAST_ROW/COL`): the streamer
   physically re-feeds the tile to each consumer edge. This is the
   hardware-faithful realization for the systolic fabric and becomes
   timing-relevant when consumers are spatially distributed. E2 gives the
   opcode faithful *timing* (one feed transaction per consumer, tile
   remains L2-resident, credit released only after the LAST consumer),
   with ref-count semantics: broadcast(k consumers) = insert with
   ref_count k, each consuming feed decrements.

**Credit analysis for broadcast.** The broadcast tile holds one L2 credit
for its entire consumer span (the #90 layernorm formula already models
this). The ref-count realization reuses TagCAM semantics unchanged: a
broadcast is *scheduled* as k feeds of the same tile — which the existing
work-conserving streamer already executes correctly; what is missing is
only the single-load-many-feeds emission discipline (today every feed
requires a matching prior move, per the #61 1:1 invariant). **Design
choice: broadcast relaxes LOAD:MOVE:FEED from 1:1:1 to 1:1:k, expressed
explicitly** — the schedule marks the tile's expected consumer count so
TagCAM ref-counts are seeded correctly at MOVE time rather than
accumulated by duplicate moves. This preserves credit conservation:
1 credit per entry, released at ref 0 after k feeds.

## 2. VE_ELEMENTWISE operand encoding (the #18 numerical half, E2's share)

Today `VE_ELEMENTWISE` carries `std::monostate` — a structural marker
(#142). E2-T2 gives it real operands:

```cpp
struct VEOperands {
    VEOp op;              // ADD, SUB, MUL, DIV, MAX, MIN,
                          // NEG, ABS, SQRT, EXP, LOG,      (unary)
                          // ADD_S, MUL_S, POW_S            (scalar-broadcast)
    uint8_t num_inputs;   // 1 (unary), 2 (binary)
    float scalar;         // for *_S forms
    uint8_t l1_src_a;     // L1 buffer of operand A
    uint8_t l1_src_b;     // L1 buffer of operand B (binary forms)
    uint8_t l1_dst;       // L1 buffer of result
};
```

Semantics land in **both value-producing executors**:
- `BehavioralProgramExecutor`: replace the no-op case with elementwise
  application over the addressed L1 buffers (reusing the typed kernels in
  `quantization/kernels.hpp` for dtype coverage).
- CSP tier: elementwise computes ride `FunctionalComputeSpec` (already
  functional); the VE encoding is what lets *ISA programs* (the #142
  kernel-factory output) execute rather than only hand-built specs.

`VE_REDUCE` encoding is deliberately **out of E2's scope** — running
max/sum/mean/var reduction state is E3's core design (#104) and shares
the struct layout but not the semantics.

## 3. Deliverables mapped to the epic tasks

| Task | Content |
|---|---|
| T2 #100 (ISA/executor closure) | `VEOperands` + assembler/disassembler/serializer round-trip; behavioral executor semantics for unary/binary/scalar forms; CSP broadcast ref-count seeding (MOVE with consumer count); `STR_BROADCAST_*` timing in both executors |
| T3 #101 (generator) | `ElementwiseScheduleGenerator` (paired two-stream emission, envelope 3-tile working set) + broadcast emission helper (1:1:k discipline) usable by the norm/epilogue generators; resolves #139 for the elementwise family by emitting COMPUTE with both-operand deps |
| T4 #102 (functional + oracle) | End-to-end CSP execution of `C = op(A, B)` and bias-broadcast cases with elementwise host-oracle comparison (the #66 payload machinery); ISA-program path validated via BehavioralProgramExecutor |
| T5 #103 (regression) | Execution matrix (op × shape × envelope, incl. constrained envelopes + partitioned credits), stall/credit invariants, coverage-matrix row updates (elementwise + broadcast → functional/regression done; #18 elementwise half closed) |

## 4. Risks

- **Ref-count seeding** touches the #61-hardened TagCAM invariants: the
  1:1:k discipline must keep "one credit per entry, one release at zero"
  exactly; T5's conservation checks guard it (and the A-flood test from
  #89 must stay green).
- **Rate-mismatch livelock**: if one operand stream stalls (e.g., its
  partition exhausted), the other must not consume unbounded buffers —
  bounded by paired emission + the envelope bound; asserted under
  constrained envelopes in T5.
- `VEOperands` widens the `DMInstruction` variant — serializer format
  version bump required (`.kpubin` compatibility note in T2).

## 5. Coverage-matrix effect (on epic completion)

`elementwise` and `broadcast` rows: all five stages → done. `batchnorm`
functional unblocks (E9 work but its elementwise substrate exists).
Milestone M2's `elementwise.functional` gate requirement satisfied.

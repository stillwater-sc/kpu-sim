# A blocked linear-algebra test set: the GEMM schedule space and blocked decompositions

**Status:** planned 2026-11-18, at the user's request. No step has started.
**Epic:** #479. **Steps:** #480–#492, in §5's order. **Milestones:** v1.1.0 (#480–#486) and
v1.2.0 (#487–#492, the epic).
**Related:** ADR 0002 §3.1 (the lowering boundary), #281 (derive CSP from the Domain Flow
Program), `docs/plans/kpu-run-csp-programs.md` (why `kpu-csp-gen` exists, and its Q4),
`docs/04-compute-fabric/matrix-tiling-strategies.md` (an earlier tiling study),
`tools/csp-gen/README.md`.

## 1. Why

Domain Flow's premise is that most compute-intensive operations are linear-algebra expressions,
and that **localization, reuse and batching** make them efficient. Every fast BLAS does three
things:
- it blocks;
- it schedules to maximize a block's reuse in local memory;
- it batches rank updates.

The right schedule depends on the shape. A tall-skinny GEMM is scheduled differently from a wide
one, and a deep one differently from both.

Blocked decompositions (LU, QR, Cholesky, Schur, …) reorganize their CSP programs in the same way:
a panel factorization, then a trailing update that is mostly GEMM. The blocking decides both the
program's shape and its reuse.

The test set has two jobs:
1. **Exercise schedule variability** on GEMM, the kernel every other operator reduces to.
2. **Validate the stack**: the CSP language, the compiler's lowering, the runtime and the
   hardware data path. Each operator is picked for what it forces them to handle, not for
   coverage.

## 2. What exists

| Piece | Today |
|---|---|
| Problem shape | Square only. `--size N` means M = N = K. |
| Tile shape | Square only. `--tile T` means T × T tiles. |
| Schedule knobs | `--orient` (which panel is outer), `--stage-a` (row schedule), `--dataflow` (`os`, `ws`, `is`). No blocking hierarchy, no loop-order choice, no `kc` blocking of the reduction. |
| Operators | `matmul`, `linear` and `lu`. LU is right-looking and keeps A resident whole, so it never runs a matrix larger than L3. |
| Language | An index is one tile or `:` for a whole row or column of tiles (`csp-language.md:119`). A sub-panel block can only be written tile by tile in a loop. The grammar allows rectangular tensors and tiles, but no generator has written them, so the levels never ran them. |
| Analysis | `matrix-tiling-strategies.md` (January 2026, before the CSP stack): a three-level tiling hierarchy, tile-size selection and reuse factors, aimed at the earlier schedule generators. |

## 3. The GEMM schedule space

### 3.1 The language and the levels (#480)

- **Shapes.** Audit and test M ≠ N ≠ K and non-square tiles through the walker, placement,
  L-B, L-T1 and L-CA. The tiles are A `tm × tk`, B `tk × tn`, C `tm × tn`.
- **A range index**, e.g. `B[k0..k1, j]`, so a block is something you can make resident, release
  or stage as one unit. It walks to the same actions as the equivalent loop of single tiles, and
  capacity is counted per tile.

### 3.2 A schedule is data (#481)

A `Schedule` description, in a library (`include/sw/kpu/program/csp/sched/`):

| Field | Meaning |
|---|---|
| shapes | M, N, K; each operand's tile shape |
| L3 block | `mc × kc × nc`, in tiles: what is resident in L3 |
| L2 block | what is staged in L2 |
| fabric | what is stationary in the array: the dataflow (`os`, `ws`, `is`) |
| loop order | the block loops' order, as in GotoBLAS/BLIS's loop nest. It decides which operand is reused at which level. |
| partial C | how C is carried across `kc` blocks: in the fabric, in an L2 accumulator merged by the Combiner, or written back and accumulated (`init accumulate`) |

`kpu-csp-gen` writes CSP from a `Schedule`. Today's `--orient`, `--stage-a` and `--dataflow` become
named presets that reproduce today's programs byte for byte, so the corpus is unchanged.

**The library is shared with the compiler.** `lower_to_csp` (ADR 0002 §3.1, #281) needs exactly
this as its `LoweringOptions`. So the work isn't thrown away when the lowering arrives:
`kpu-csp-gen` becomes a command-line front end to the same library.

### 3.3 Choosing by shape (#482)

- **A reuse model** gives the default: DRAM and L3 traffic per operand for a schedule, starting
  from `matrix-tiling-strategies.md` §3.2–3.3. It must reproduce the standard rules:

  | Shape | Schedule |
  |---|---|
  | tall-skinny (M ≫ N, K) | keep B resident; stream A's row blocks past it |
  | wide (N ≫ M, K) | the mirror: keep A resident; stream B |
  | deep (K ≫ M, N) | split K; carry partial C in L2 through the Combiner; batch the rank-`kc` updates |
  | square | the blocked schedule the model's traffic favours |

- **`--search`** lists the valid schedules in a bounded space and ranks them by running them:
  L-T1 for speed, then L-CA to confirm the best few. The bubble report's reuse and idle numbers
  are the evidence. This is the schedule search that `kpu-run-csp-programs.md` Q4 deferred to
  the generator.

**Acceptance:** a shape sweep (tall-skinny, wide, deep and square, at three sizes):
- every schedule computes L-B's values at every level;
- for each non-square shape, the chosen schedule beats the fixed canonical one on DRAM traffic
  and on L-CA makespan.

The sweep's table is recorded here when #482 lands.

## 4. The operators

| # | Operator | Issue | What it validates |
|---|---|---|---|
| 1 | GEMM over shapes, with transposes (Aᵀ·B, A·Bᵀ) | #480–#483 | The schedule space (§3). Transpose on a move or a stream is a data path feature: backprop needs it, and attention needs QKᵀ. |
| 2 | TRSM, many right-hand sides | #484 | Triangular iteration spaces (bounds on an outer index) and a new tile function. It is the panel step of LU and Cholesky; `trsm_ll` exists only inside LU today. |
| 3 | SYRK | #485 | Only one triangle of C is computed (a triangular output space), and rank-k batching. It is Cholesky's trailing update. |
| 4 | Cholesky, blocked: in-core, out-of-core, left- and right-looking | #486 | The simplest blocked decomposition, with no pivoting: blocking's effect on a CSP program, isolated from pivoting. |
| 5 | LU, blocked, out-of-core | #487 | Panel streaming and look-ahead for a matrix larger than L3, keeping the value-dependent pivot flowing between calls. |
| 6 | QR, blocked Householder (compact WY), and TSQR | #490 | A tall-skinny, BLAS-2-heavy panel, then a GEMM-heavy trailing update with block reflectors. TSQR adds a reduction tree across compute tiles (#403). |
| 7 | GEMV and GER | #488 | Bandwidth-bound and with little reuse. The simulator must show the bandwidth limit as bubbles, not hide it: the "reuse is deliberate" rule. |
| 8 | Batched and strided-batched GEMM | #489 | Many small problems: scheduling across compute tiles, distribute and broadcast (#403), the runtime's batching. |
| 9 | SYMM and TRMM | #491 | An operand stored as one triangle and read whole: transposed tile access on a move. |
| 10 | Hessenberg reduction, then the Schur form | #492 | Two-sided updates, with reuse from both sides. The Schur form's QR iteration runs until convergence, so its sweep count depends on the data: the host sequences the sweeps (v0.13's host orchestration). The tridiagonal and bidiagonal reductions follow the same pattern. |

Two neighbours are tracked by their own plans, not here:
- **Attention:** QKᵀ, online softmax, then ·V, with FlashAttention-style tiling. It chains
  transposes, fused epilogues across operators and L2 staging (`e8_online_softmax_pattern.md`,
  `e10_fused_epilogue_pattern.md`).
- **Conv2d as implicit GEMM:** layout transforms on a move (`e4_layout_transform_pattern.md`,
  `e6_conv2d_pattern.md`).

## 5. Order

1. **The GEMM schedule space** (#480, #481, #482). It closes the variability gap, and everything
   below reuses its blocking.
2. **Transposes** (#483).
3. **TRSM, then SYRK, then Cholesky** (#484, #485, #486): the first blocked decomposition, with
   no pivoting.
4. **GEMV/GER and batched GEMM** (#488, #489): the bandwidth-bound contrast, and multi-tile.
5. **Out-of-core LU, then QR and TSQR** (#487, #490).
6. **SYMM/TRMM, then Hessenberg and Schur** (#491, #492), with host orchestration.

## 6. Rules for every step

- **Values at every level.** Every program computes L-B's values at L-B, L-T1 and L-CA, checked
  per tile and per operand. Timing claims rest on L-CA, never on L-T1 alone, since L-T1 is
  uncalibrated.
- **Reuse is stated.** A schedule's reuse is in its data movement (`resident`, `stage`, an L2
  accumulator). No level may supply reuse the program does not state.
- **A reference schedule, kept golden.** Each operator lands as a reference program in the corpus
  (`tests/program/corpus/`), written by `kpu-run` once every level agrees. That makes the test set
  the acceptance suite for the DFP → CSP lowering (#281): lowering an operator's DFP must
  reproduce its reference program, or explain the difference.
- **No second source of programs.** The operators extend the shared schedule library, which the
  lowering will call. They are not a separate hand-written operator library.

## 7. Open questions

- **Q1. Where a transpose happens** (#483): in the BlockMover, in the Streamer, or at the array's
  edge. This is an architecture decision for kpu-sim to make; the RTL then implements it.
- **Q2. The search's budget** (#482): how large a space, and how many candidates go on to L-CA.
- **Q3. Milestones** (decided 2026-11-18): **v1.1.0** takes phases 1–3, the GEMM schedule space,
  transposes, and TRSM, SYRK and Cholesky (#480–#486). **v1.2.0** takes phases 4–6, GEMV and
  GER, batched GEMM, out-of-core LU, QR and TSQR, SYMM and TRMM, and Hessenberg and Schur
  (#487–#492), with the epic (#479).

# PLASMA Tile Algorithms — decomposition, KPU hardware requirements, test plan

**Status:** reference + assessment (companion to `kpu-program-model.md`, epic #229).
Motivated by the L0 `TileProgram` work (#230): the L0 layer expresses a
linear-algebra operator as an explicit DAG of **tile kernels** (§3a of the program
model), and we are aligning that kernel set to the **PLASMA** tile-algorithm API.
This document (1) summarizes how PLASMA decomposes dense linear algebra into tile
kernels, (2) maps each kernel's requirements onto KPU hardware and asks whether the
simulator/fabric already supports it, and (3) gives a staged test plan to validate
that we have the required functionality.

---

## 1. The PLASMA tile-algorithm model

PLASMA (Parallel Linear Algebra Software for Multicore Architectures) reorganizes
LAPACK's blocked algorithms into **tile algorithms**: the matrix is stored as a grid
of small square **tiles** (`T×T`), and each algorithm is expressed as a **DAG of
tile tasks**, where each task is a **tile kernel** operating on a *small, fixed*
number of tiles (1–4) that fit in fast memory. A runtime schedules the DAG,
respecting data dependencies inferred from each kernel's tile read/write sets.

This is exactly the L0 model:
- **tiles** ↔ `TensorOperand` tiles;
- **tile kernels** ↔ `TileOpKind` (`MATMUL_ACCUM`=GEMM, `LU_DIAG_FACTOR`=GETRF, …);
- **DAG from tile I/O** ↔ §3a "every op declares its full tile I/O", from which the
  driver-JIT placement pass recovers the dependency DAG (§4a).

Two properties matter for a dataflow fabric:
1. **Locality** — every kernel touches only a few tiles, so it maps to tile-resident
   compute (L3/L2/L1 + fabric) with no global data motion.
2. **Nearest-neighbor factorization** — where pivoting/orthogonalization would
   otherwise need global communication, PLASMA uses *pairwise* (tile-by-tile) kernels
   (`TSTRF`/`TSQRT`) so all coupling is between **adjacent tiles** — the property the
   KPU's neighbor-pivoting requirement is really about.

---

## 2. Tile-kernel catalog

The kernels below are the union needed to cover Cholesky, LU, QR, and triangular
solve. "Shape" = tiles read/written. "Core math" is what the tile does.

| Kernel | Op | Shape (in → out) | Core math | Reducible to GEMM? |
|---|---|---|---|---|
| **GEMM** | matmul / trailing update | `A,B,C → C` | `C ± A·B` | is GEMM |
| **SYRK** | symmetric rank-k | `A,C → C` | `C − A·Aᵀ` (lower) | GEMM-class |
| **POTRF** | Cholesky diagonal factor | `A → A` | `A = L·Lᵀ` | no — needs **sqrt**, divide |
| **TRSM** | triangular solve (tile) | `T,B → B` | `B := T^{-1}·B` or `B·T^{-1}` | no — **substitution + divide** |
| **GETRF** | LU diagonal factor (+pivot) | `A → A, ipiv` | tile LU, partial pivot | no — **divide + pivot (argmax+swap)** |
| **LASWP** | apply row interchanges | `A, ipiv → A` | permute rows | no — **data-dependent permute** |
| **GESSM** | apply L + pivots (incr. LU) | `L,ipiv,A → A` | `A := L^{-1}·P·A` | no — TRSM + LASWP |
| **TSTRF** | pairwise LU of a tile pair | `U,A → U,A, ipiv` | LU of `[U;A]`, pairwise pivot | no — divide + pairwise pivot |
| **SSSSM** | apply pairwise transform | `A_ik,ipiv,A_kj,A_ij → A_kj,A_ij` | update tile pair | GEMM + LASWP |
| **GEQRT** | QR diagonal factor | `A → A, T` | Householder `A = Q·R` | no — **norm(sqrt), divide** |
| **TSQRT** | pairwise QR of a tile pair | `R,A → R,A, T` | QR of `[R;A]`, pairwise | no — norm, divide |
| **ORMQR/UNMQR** | apply Q (diagonal) | `A,T,C → C` | `C := Qᵀ·C` | GEMM-class (rank-b updates) |
| **TSMQR** | apply pairwise Q | `A,T,C_k,C_i → C_k,C_i` | apply pairwise reflectors | GEMM-class |

**Primitive families the catalog needs (this is the crux for HW):**
- **P1 — MAC / GEMM** (dense multiply-accumulate). Used by GEMM, SYRK, ORMQR, TSMQR,
  SSSSM, and the update parts of every factor kernel.
- **P2 — Triangular substitution** (forward/back solve): a dependent-MAC recurrence
  with a **divide** at each step. Used by TRSM, GESSM, and inside POTRF/GETRF.
- **P3 — Scalar functions**: **divide/reciprocal** (LU, TRSM), **sqrt** (Cholesky,
  QR norms), sign.
- **P4 — Reductions**: sum (norms need sum-of-squares → P3 sqrt), and
  **argmax-with-index** (partial pivoting).
- **P5 — Data-dependent permute**: **row exchange** driven by P4's argmax (pivoting).
- **P6 — On-chip tile residency + reuse moves**: a diagonal factor is reused across
  its whole block-row and block-column; results move tile→tile without a DRAM round
  trip (L3↔L3 / L2 staging).
- **P7 — In-place tile read-modify-write** and a **DAG runtime** that schedules on
  tile dependencies.

---

## 3. Per-operator decomposition

Notation: `nt` block-columns; `k` is the panel index; the trailing submatrix shrinks
as `k` advances.

### 3.1 Cholesky (SPD, no pivoting) — {POTRF, TRSM, SYRK, GEMM}
```
for k:
  POTRF A[k,k]                              # L_kk L_kkᵀ = A[k,k]
  for i>k: TRSM  A[i,k] <- A[k,k]           # L[i,k] = A[i,k] L_kk^{-ᵀ}
  for i>k: SYRK  A[i,i] -= A[i,k]·A[i,k]ᵀ   # symmetric trailing update
  for i>k, j in (k,i): GEMM A[i,j] -= A[i,k]·A[j,k]ᵀ
```
Simplest factorization: no pivoting (P5 not needed), but needs **sqrt** (P3).

### 3.2 LU — two pivoting modes
**(a) Confined (block) pivoting — {GETRF, LASWP, TRSM, GEMM}** *(implemented in L0)*
```
for k:
  GETRF A[k,k]                              # factor diagonal tile, within-tile partial pivot
  for g<k: LASWP A[k,g] (ipiv_k)            # replay swaps onto already-computed L (left)
  for j>k: LASWP A[k,j] (ipiv_k)            # replay swaps onto trailing row-block
  for j>k: TRSM  A[k,j] <- A[k,k]  (lower)  # U[k,j] = L_kk^{-1} A[k,j]
  for i>k: TRSM  A[i,k] <- A[k,k]  (upper)  # L[i,k] = A[i,k] U_kk^{-1}
  for i>k,j>k: GEMM A[i,j] -= A[i,k]·A[k,j]
```
Pivot search is confined to the diagonal tile (P5 local to one tile). Numerically
adequate for well-conditioned / diagonally-dominant systems; can fail if a diagonal
tile is ill-conditioned though the whole matrix is fine.

**(b) Incremental / pairwise (neighbor) pivoting — {GETRF, GESSM, TSTRF, SSSSM}**
```
for k:
  GETRF A[k,k]                              # factor diagonal tile
  for j>k: GESSM A[k,j] <- A[k,k], ipiv_kk  # apply L_kk^{-1} + pivots to the row-block
  for i>k:
    TSTRF A[k,k], A[i,k]                     # pairwise-factor diagonal + sub-diagonal tile
    for j>k: SSSSM A[k,j], A[i,j] <- A[i,k], ipiv_ik   # apply the pairwise transform
```
Pivoting is **between adjacent tiles** (`TSTRF` exchanges rows across the tile
boundary) — the dataflow-faithful "neighbor pivoting". Numerically stronger than
(a); this is the target variant for ill-conditioned inputs. **Not yet implemented.**

### 3.3 QR — {GEQRT, TSQRT, ORMQR/UNMQR, TSMQR}
```
for k:
  GEQRT A[k,k]                              # Householder QR of the diagonal tile -> R_kk, reflectors T
  for j>k: ORMQR A[k,j] <- A[k,k], T        # apply Qᵀ to the row-block
  for i>k:
    TSQRT A[k,k], A[i,k]                     # pairwise QR of the tile pair -> updated R, reflectors
    for j>k: TSMQR A[k,j], A[i,j] <- A[i,k]  # apply the pairwise Qᵀ
```
Same pairwise structure as LU-(b). Adds **norm/sqrt** (P3) for the Householder
reflectors. No pivoting for basic QR (column pivoting is an add-on).

### 3.4 Triangular solve (`A·X = B`, A triangular) — {TRSM, GEMM}
```
for k:                    # forward (lower) or backward (upper) sweep
  TRSM  X[k] <- A[k,k]
  for i≠k: GEMM B[i] -= A[i,k]·X[k]
```
Pure P1+P2; this is also the "solve" phase after LU/Cholesky/QR factorization.

**Observation:** every operator is `{diagonal factor} + {pairwise/panel propagation}
+ {TRSM apply} + {GEMM trailing update}`. Cover that shape and you cover dense direct
linear algebra. GEMM/SYRK/ORMQR/TSMQR/SSSSM are the "wide" GEMM-class kernels; the
factor kernels (POTRF/GETRF/TSTRF/GEQRT/TSQRT) carry the hard primitives (P2–P5).

---

## 4. KPU hardware-requirements mapping and gap analysis

Do we have what the PLASMA methodology needs? Per primitive family:

| # | Primitive | Needed by | KPU status | Gap / action |
|---|---|---|---|---|
| **P1** | MAC / GEMM array | all | **Have** — the compute fabric is a systolic MAC array; L0 `MATMUL_ACCUM` proven | none |
| **P2** | Triangular substitution (dependent MAC + divide) | TRSM, GESSM, POTRF, GETRF | **Partial** — functional reference has TRSM; fabric support for a substitution *dataflow* (vs. GEMM) is unmodeled | Model a TRSM fabric configuration / decide it runs on the VE; needs P3 divide |
| **P3** | Scalar divide / reciprocal / **sqrt** | GETRF, TRSM (divide); POTRF, QR (sqrt) | **Gap** — the fabric is MAC-only; no divide/sqrt unit is modeled | Add a scalar/special-function capability (fabric or VE); required before Cholesky/QR |
| **P4** | Reductions: sum-of-squares, **argmax-with-index** | QR norms (sum); pivoting (argmax) | **Partial** — `VE_REDUCE` exists (sum/max); argmax **with index** for pivoting is unconfirmed | Confirm/add index-returning argmax reduction |
| **P5** | Data-dependent row permute (pivot swap) | GETRF, LASWP, TSTRF | **Gap** — no row-exchange primitive; movers do tile moves, not value-selected row swaps | Decide: BlockMover-driven row swap vs. VE permute; model it |
| **P6** | On-chip tile residency + reuse moves (L3↔L3 / L2) | all factor kernels (diagonal reused across row/col) | **Gap** — timing tier is single-compute-tile; L3/L2 are credit pools, no inter-tile reuse move | The separate multi-compute-tile executor issue (§4a of the program model) |
| **P7** | In-place tile RMW + DAG scheduling on tile deps | all | **Partial** — L0 reference does in-place RMW; DAG recovery/placement is the driver-JIT pass (§4a), not yet built | Increment 3 (placement pass) |

**Bottom line.** GEMM-class work (P1) is fully supported and the confined-pivoting
tile LU (§3.2a) runs today in the L0 functional reference. To support the **full
PLASMA methodology** the KPU/simulator is missing four capabilities, in rough order
of leverage:
1. **P3 scalar special functions** (divide + sqrt) — gates Cholesky, QR, and honest
   TRSM/GETRF.
2. **P5 pivoting** (argmax + row swap) — gates partial-pivot and pairwise-pivot LU.
3. **P6 multi-compute-tile residency + reuse moves** — gates *executing* any
   multi-tile factorization on more than one CF tile (already tracked separately).
4. **P2 triangular-substitution fabric mode** — gates hardware-faithful TRSM timing.

These are hardware-capability questions the simulator must answer (which fidelity
tier models each; whether the fabric or a vector engine owns P3/P4/P5). The L0
functional layer can *represent and validate the math* of every kernel ahead of the
hardware model — which is what the test plan below drives.

---

## 5. Test plan — validating we have the required functionality

Four levels, each a concrete gate. Levels 0–1 are functional (L0, no timing) and can
land now; levels 2–3 probe/require the hardware model.

### Level 0 — tile-kernel functional references (unit)
For each kernel, a functional reference in `TileProgramReference` + a unit test
validating it in isolation by local reconstruction/identity:

| Kernel | Status | Validation |
|---|---|---|
| GEMM (`MATMUL_ACCUM`) | ✅ done | vs. naive matmul, exact |
| GETRF (`LU_DIAG_FACTOR`) | ✅ done | single-tile `P·A=L·U` |
| LASWP (`PIVOT_APPLY`) | ✅ done | permutation replay matches ipiv |
| TRSM lower/upper | ✅ done | `T·X == B` residual |
| SYRK | ▫ todo | `C == C0 − A·Aᵀ` |
| POTRF | ▫ todo (needs P3 sqrt) | `L·Lᵀ == A` |
| GESSM | ▫ todo | `== L^{-1}·P·A` |
| TSTRF | ▫ todo (needs P5) | pair `P·[U;A] = L·R` |
| SSSSM | ▫ todo | matches applying TSTRF transform |
| GEQRT / TSQRT | ▫ todo (needs P3) | `Qᵀ·[.] == R`, `QᵀQ=I` |
| ORMQR / TSMQR | ▫ todo | matches dense `Qᵀ·C` |

### Level 1 — operator functional (integration)
Each operator as a `TileProgram`, validated end-to-end:

| Operator | Kernels | Validation | Status |
|---|---|---|---|
| matmul | GEMM | vs. naive, exact | ✅ done |
| **LU (confined pivot)** | GETRF,LASWP,TRSM,GEMM | `P·A=L·U` reconstruction | ✅ done |
| LU (pairwise/neighbor) | GETRF,GESSM,TSTRF,SSSSM | **solve residual** `‖A·x−b‖` (incremental pivoting has no simple L,U) | ▫ todo |
| triangular solve | TRSM,GEMM | `‖A·x−b‖` | ▫ todo |
| Cholesky | POTRF,TRSM,SYRK,GEMM | `‖L·Lᵀ−A‖` | ▫ todo |
| QR | GEQRT,TSQRT,ORMQR,TSMQR | `‖Q·R−A‖`, `‖QᵀQ−I‖` | ▫ todo |

Grow the operator-coverage matrix (`tests/coverage/pattern_coverage.json`) with these
rows so a milestone can't over-claim linear-algebra support.

### Level 2 — hardware-capability probes (simulator)
A capability matrix asserting the simulator/ISA actually provides each primitive a
kernel needs — this is the "do we have the HW requirements" gate. For each
(kernel × primitive) cell, a probe test that either exercises the primitive on the
compute fabric / VE / mover, or is marked an explicit **gap** with an issue:

- **P1 MAC:** already covered by the matmul timing tests.
- **P2 TRSM mode:** probe that a triangular-substitution schedule runs on the fabric
  (or is declared VE-owned).
- **P3 divide/sqrt:** probe the compute ISA for a reciprocal/sqrt op; **currently a
  gap** → file an issue for a scalar/special-function capability.
- **P4 argmax-with-index:** probe `VE_REDUCE` for index-returning max; confirm or gap.
- **P5 row-swap:** probe a value-selected row exchange (BlockMover or VE); **gap**.
- **P6 multi-CF residency/reuse:** blocked on the multi-compute-tile executor issue.
- **P7 DAG placement:** covered once the driver-JIT placement pass (increment 3) lands.

Deliverable: a `plasma_capability.json` (kernel × primitive × {have|gap|planned})
plus a `test_plasma_capability` gate, mirroring the pattern-coverage gate.

### Level 3 — timing / L1 (once streams exist)
For each kernel, an L1 stream signature and a cycle-accurate timing validation on each
target topology (single / NEWS / checkerboard, §4a). Validates that the tile kernels
not only compute correctly but are *schedulable and deadlock-free* under the device's
credits and mover capability.

### Sequencing
1. **Now:** Level 0/1 rows that need no new HW primitive — SYRK, GESSM, triangular-
   solve operator, and the pairwise-pivot LU (TSTRF/SSSSM) validated by solve residual.
2. **Next:** Level 2 capability matrix — turns the P3/P4/P5 gaps into filed issues.
3. **Then:** POTRF/Cholesky and QR once P3 (sqrt/divide) is modeled; multi-CF (P6);
   and the L1/timing level.

---

*Companion documents:* `kpu-program-model.md` (L0/L1 program + driver JIT + §3a/§4a),
`model-ingestion-compilation-epic.md` (epic #229), `dfg-kpu-versioning.md`.

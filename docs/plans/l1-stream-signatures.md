# L1 Stream Signatures — the spatial/temporal layer over L0

**Status:** design + first increment (Phase 0 increment 2, #230; companion to
`kpu-program-model.md` §3/§6). L1 adds the **spatial/temporal (timing)** layer on top
of the L0 `TileProgram`: it says *how* each L0 tile becomes an **element stream** into
or out of the reactive fabric, and *how long* each tile-compute takes as a systolic
wavefront. **L1 never changes values** — it is derived from L0 + the array mapping, so
`L0` alone stays the functional reference and `L0+L1` is the timing model
(BEHAVIORAL vs. TRANSACTIONAL/CYCLE_ACCURATE, per `SIMULATION_FIDELITY_FRAMEWORK.md`).

---

## 1. What L1 attaches to L0

For each L0 op:
- **Feed / Drain** (a tile ↔ fabric-port injection) → a **`StreamSignature`**: which
  array **edge**, how many **lanes**, the per-element **lane + wavefront skew**, and the
  **rate** (elements/cycle/lane).
- **compute** (`MatMulAccum`, …) → a **`WavefrontTiming`**: the systolic latency of
  that tile-compute on the array.

L1 is a thin layer keyed by L0 op index — it does not copy or mutate the L0 program.

## 2. The systolic schedule (output-stationary matmul)

Tile matmul `C[i,j] += Σ_k A[i,k]·B[k,j]`, with `i∈[0,R)`, `j∈[0,S)`, `k∈[0,K)`, mapped
to an `R×S` array where **PE(i,j) accumulates C[i,j]** (output-stationary). The classic
space-time schedule is the linear wavefront

```
σ(i,j,k) = i + j + k
```

which gives, for each operand, a single skewed stream at one array edge:

| Stream | Edge | Lane | Injection time of element | Propagation |
|---|---|---|---|---|
| **A** (`A[i,k]`) | **West** | row `i` | `t = i + k` | east, +1 cycle/column |
| **B** (`B[k,j]`) | **North** | col `j` | `t = j + k` | south, +1 cycle/row |
| **C** (`C[i,j]`) | **South** (drain) | col `j` | `t = i + j + (K−1)` | out the bottom |

Check: `A[i,k]` injected at West-PE(i,0) at `i+k` reaches PE(i,j) at `i+j+k`; `B[k,j]`
injected at North-PE(0,j) at `j+k` reaches PE(i,j) at `i+j+k` — they meet exactly when
PE(i,j) processes product `k`. ✔

**Latency of one tile-compute** (`R×S` array, depth `K`):
```
latency = (R−1) + (S−1) + (K−1) + 1        # fill + reduce + drain
```
e.g. a 16×16×16 tile-compute = 46 cycles (not a single lumped MAC cost) — the fill and
drain ramps are now explicit, which is the point of L1.

## 3. `StreamSignature` (per feed/drain)

A tile's `rows×cols` elements become a stream where element `(r,c)`:
- injects on **lane** `= r` (row axis) or `= c` (col axis),
- at **time** `t0 + skew_row·r + skew_col·c`,
- at **rate** elements/cycle/lane.

For the schedule above every matmul stream has `skew_row = skew_col = 1`; they differ
only by **edge** and **lane axis**:

| L0 op | operand | edge | lane axis | lanes |
|---|---|---|---|---|
| `Feed → West` | A tile (`Ti×Tk`) | West | Row | `Ti` |
| `Feed → North` | B tile (`Tk×Tj`) | North | Col | `Tj` |
| `Drain → South` | C tile (`Ti×Tj`) | South (output) | Col | `Tj` |

`time_span = skew_row·(rows−1) + skew_col·(cols−1)` is the stream's first→last element
delay; `element_count = rows·cols`.

## 4. Derivation from L0

`derive_matmul_streams(L0)` walks the L0 ops and emits the table above from each tile's
element extent (clamped trailing tiles handled), inferring the array dims from the tile
shape. **Increment-1 assumption:** each tile fits the physical array (`R×S = Ti×Tj`,
`K = Tk`). Larger tiles sub-blocking onto a fixed physical array (multiple passes) is a
follow-on. This is the systolic specialization of the general derivation in
`kpu-program-model.md` §4 (`stream_port(t) = { A[f_A(x)] : σ(x)=t }`); domain_flow's
`IndexSpace`/`AffineMap`/`schedule`/`wavefront` are the general engine for non-matmul
operators.

## 5. Integration path (what L1 unlocks next)

1. **Physically-shaped compute cost.** The characterization DAG (`tile_dag.hpp`)
   currently lumps a compute op at `MACs / macs_per_cycle`. Swapping in
   `WavefrontTiming::latency()` makes the fill/drain ramps and the true wavefront depth
   visible in the makespan — a follow-on that plugs into the existing harness with no
   interface change.
2. **Drive the cycle-accurate CSP model.** L1 stream signatures are the input the
   `ConcurrentTimingExecutor` needs (injection order + rate + skew per port), replacing
   the current inline schedule generation (Phase 0 §6.2).
3. **Other operators.** LU/Cholesky/QR tile kernels get their own schedules (TRSM is a
   substitution wavefront; GETRF/panel factor a triangular front) — derived via the
   domain_flow polyhedral engine rather than the hand-specialized matmul schedule.

## 6. Scope of this first increment

Deliver the L1 representation (`StreamSignature`, `WavefrontTiming`, `StreamProgram`),
the **matmul** derivation, and validation that the signatures + latency are correct and
that values are untouched. Wiring L1 latency into the timing model (§5.1) and the
non-matmul schedules (§5.3) are the next increments.

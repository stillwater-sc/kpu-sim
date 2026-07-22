# L1 Stream Signatures — the spatial/temporal layer over L0

**Status:** design + first increment (Phase 0 increment 2, #230; companion to
`kpu-program-model.md` §3/§6). L1 adds the **spatial/temporal (timing)** layer on top
of the L0 `TileProgram`: it says *how* each L0 tile becomes an **element stream** into
or out of the reactive fabric, and *how long* each tile-compute takes as a systolic
wavefront. **L1 never changes values** — it is derived from L0 + a **space-time
mapping**, so `L0` alone stays the functional reference and `L0+L1` is the timing model
(BEHAVIORAL vs. TRANSACTIONAL/CYCLE_ACCURATE, per `SIMULATION_FIDELITY_FRAMEWORK.md`).

The central fact: **the streams are not fixed — they are a function of the space-time
mapping** (which operand is held stationary). L1 must therefore be parameterized by
that mapping and must also state the **network** the resulting streams require (a 2-D
mesh vs. a hexagonal overlay).

---

## 1. The space-time mapping

A tiled matmul is a 3-D uniform recurrence over the iteration domain `(i,j,k)`:
```
c(i,j,k) = c(i,j,k-1) + a(i,j,k)·b(i,j,k)      # C accumulates along +k
a(i,j,k) = a(i,j-1,k)                          # A is invariant along j  → e_A = [0,1,0]
b(i,j,k) = b(i-1,j,k)                          # B is invariant along i  → e_B = [1,0,0]
                                               # C is invariant along nothing but +k → e_C = [0,0,1]
```
Each variable `V` has a **propagation direction** `e_V` — the iteration axis along which
its value is reused (constant).

A **space-time mapping** is two integer vectors:
- **schedule** `τ` — time of iteration `x` is `σ(x) = τ·x` (validity: `τ·e_V > 0` ∀V).
- **projection** `u` — the direction collapsed to form the 2-D array; iterations that
  differ by a multiple of `u` share one PE (`τ·u ≠ 0` so they run at different times).

A variable is **stationary** iff `u ∥ e_V` (projecting along its reuse axis pins it in a
PE); otherwise it **streams**. This one relation generates the whole dataflow taxonomy:

| `u` (project out) | Stationary | Streams | Analogy |
|---|---|---|---|
| `[0,0,1]` (k) | **C** | A, B | **output-stationary** |
| `[1,0,0]` (i) | **B** | A, C | **weight(B)-stationary** (TPU MXU) |
| `[0,1,0]` (j) | **A** | B, C | **A-stationary** |
| `[1,1,1]` | *none* | A, B, C | **fully-streaming (hexagonal)** |

`τ = [1,1,1]` (`σ = i+j+k`) throughout below.

## 2. Deriving a stream signature from the mapping

For a **streaming** variable, its physical flow in the array is `e_V` projected onto the
2-D PE plane; it enters at the edge opposite the flow (or, for the result, exits). Two
timing quantities come out of `τ` and the geometry:
- **lane skew** — the per-lane start offset (successive lanes light up later).
- **element stride** — cycles between consecutive elements on one lane. `stride = 1` is a
  *dense* stream; `stride > 1` means a **bubble** of `stride-1` empty cycles per element.

**Bubbles come from result evacuation.** Input streams enter the boundary densely
(stride 1). A result that must **traverse the filled array** to reach an edge picks up a
bubble, because each successive result is produced one cycle later *and* one lattice
point further from the exit — so its exit time advances by two, not one.

### 2.1 Output-stationary (`u=[0,0,1]`) — the worked case
PE`(i,j)` holds `C[i,j]`. With `σ=i+j+k`:
- **A** → **West** edge, lane = row `i`, injected at `t=i+k` → dense (stride 1), skew `i`.
- **B** → **North** edge, lane = col `j`, injected at `t=j+k` → dense, skew `j`.
- **C** is stationary, ready at `t=i+j+K−1`, and must be **evacuated**. Draining it
  **South** would collide with B's own southbound flow *and* with sibling C's — the
  southbound links are occupied. So C drains **North**, over the free northbound links.
  `C[i,j]` travels `i` rows up, exiting the North edge at lane `j` at
  `t = (i+j+K−1) + i = 2i+j+K−1`. Along a column, consecutive rows exit **two** cycles
  apart → **stride 2, bubble 1**. (The bubble is exactly what keeps successive results
  from colliding en route.)
- **Network:** flows are `A:+col`, `B:+row (S)`, `C:−row (N)` — all along the two mesh
  axes (north/south links are distinct directions) → a **2-D mesh** suffices.

### 2.2 Weight(B)-stationary (`u=[1,0,0]`)
`B[k,j]` is preloaded across the `(j,k)` array. **A** streams down each column
(`t`-indexed by `i`), **C** accumulates along `+col` and exits the **East** edge — and,
because C exits at the edge where its accumulation finishes (no orthogonal traversal),
the C readout is **dense (no bubble)**. Two axes → **2-D mesh**. This is the TPU-style
weight-stationary dataflow. `u=[0,1,0]` (A-stationary) is the mirror image.

### 2.3 Fully-streaming / hexagonal (`u=[1,1,1]`)
No variable is stationary. Because **`u ∥ τ`** (`s=τ=[1,1,1]`), space and time are
*aligned*: every iteration on the plane `i+j+k=t` fires at time `t` and maps to a
distinct PE, giving **perfect concurrency with no contention and no bubbles**. The three
variables flow along the three projections of `e_A,e_B,e_C` onto the plane ⊥`[1,1,1]` —
**three directions at 60°** → a **hexagonal** connectivity (Kung–Leiserson).
- **Network:** a hex array needs three link directions. On a rectangular 2-D mesh this
  is a **network overlay** (the third/diagonal direction added on top of the two mesh
  axes); alternatively a **native hexagonal fabric** can be built. L1 records this
  requirement so the driver-JIT placement (§4a of the program model) can refuse or
  overlay.

## 3. The L1 objects

- **`SpaceTimeMap`** — `τ`, `u`, with the presets above; reports the stationary operand.
- **`StreamSignature`** (per variable) — `role` (Stationary / StreamIn / StreamOut),
  physical `flow` direction, entry/exit `edge`, `lane_skew`, `element_stride`
  (`bubble = stride−1`), `lanes`, tile `rows×cols`, `rate`.
- **`WavefrontTiming`** (per compute) — the systolic latency of one tile-compute
  (`(R−1)+(S−1)+(K−1)+1` fill+reduce+drain for `τ=[1,1,1]`).
- **`NetworkOverlay`** — required `FabricTopology` (Mesh2D / Hexagonal), whether it needs
  an overlay on a 2-D mesh, and the distinct stream directions.
- **`StreamProgram`** — the `SpaceTimeMap`, the per-variable signatures, the
  `NetworkOverlay`, and per-op wavefront timings; `disassemble()` renders it.

## 4. Derivation from L0

`derive_matmul_streams(L0, SpaceTimeMap)` produces the signatures + network for the
chosen mapping (the four canonical projections above), reading tile extents from L0
(clamped trailing tiles handled) and inferring the array dims from the tile shape.
Increment-1 assumption: each tile fits the physical array. This is the systolic
specialization of the general rule in `kpu-program-model.md` §4
(`stream_port(t) = { V[f_V(x)] : σ(x)=t }`); domain_flow's `IndexSpace`/`AffineMap`/
`schedule`/`wavefront` are the general engine for arbitrary operators and mappings.

## 5. What L1 unlocks next

1. **Physically-shaped compute + movement cost.** The characterization DAG
   (`tile_dag.hpp`) can swap its lumped `MACs/macs_per_cycle` for `WavefrontTiming` and
   the stream bubbles, exposing fill/drain ramps and the readout stalls in the makespan.
2. **Network-aware placement.** The `NetworkOverlay` tells the driver-JIT placement pass
   (§4a) whether the chosen schedule fits the device's fabric (mesh) or needs a hex
   overlay — a feasibility input alongside capacity and mover capability.
3. **Schedule search.** With the mapping parameterized, the characterization harness can
   sweep *dataflow* (output- vs. weight- vs. A-stationary vs. hex) the way it sweeps
   size/tiling — discovering which stationary operand and network a given problem+device
   wants (the domain-flow analogue of choosing a dataflow in CUDA/CUTLASS).
4. **Drive the cycle-accurate CSP model** from the stream signatures (Phase 0 §6.2), and
   derive non-matmul schedules (TRSM/GETRF/QR) via the polyhedral engine.

## 6. Scope of this increment

The parameterized L1 representation (`SpaceTimeMap`, `StreamSignature` with role/flow/
bubble, `NetworkOverlay`, `StreamProgram`) and the matmul derivation for the four
canonical mappings, with validation of the signatures, the output-stationary evacuation
bubble, and the mesh-vs-hex network classification. Wiring L1 into the timing model
(§5.1), network-aware placement (§5.2), and non-matmul schedules are next increments.

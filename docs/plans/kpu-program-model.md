# KPU Program Model — layered tile/stream program + driver JIT

**Status:** concept design (companion to `model-ingestion-compilation-epic.md`, epic
#229). Supersedes the earlier "serialize `ScheduleResult`" framing of Phase 0 —
`DMProgram` and `ScheduleResult` both mismodel the program (see §5).

**Motivation:** the KPU is a **reactive dataflow fabric** — it computes when streams
arrive and *bubbles* a wavefront when its inputs aren't there yet. The program should
therefore **articulate the streams** into/out of the fabric and the fabric's own
computation — *not* sequence the DMA/BlockMover/Streamer engines. Those engines are
machinery that *realizes* the streams, and configuring them for a specific KPU is a
**JIT step in the driver**, not part of the portable program.

---

## 1. Three roles (NVIDIA-shaped)

| Role | Produces | Analog | Where it lives |
|---|---|---|---|
| **Compiler** (`domain_flow`) | portable KPU program (device-independent) | **PTX** | domain_flow |
| **Driver JIT** | this device's data-path config (DMA/L3/BM/L2/Streamer/L1 settings) | **SASS** | **the driver** — *modeled in kpu-sim today* as part of the virtual platform; migrates to the real driver as the platform matures |
| **Hardware** | execution (values + timing) | GPU | kpu-sim executor |

kpu-sim currently plays **both** driver-JIT and hardware. The portable program is the
stable contract; the data-path config is a device-specific, cacheable JIT artifact.

## 2. The physical structure the program targets

The ideal is **Memory ↔ Streamer ↔ Compute Fabric**. Because large memories are wide
and slow, a hierarchy realizes it:

```
DRAM ─DMA→ L3(on-chip mem tiles) ─BlockMover→ L2(banks) ─Streamer→ L1(vectors) → Compute Fabric
                                                                                  (reactive array)
```

The program should configure: **(a) the on-chip memory tiles (L3 working set),
(b) the compute fabric (the domain-flow / recurrence program), (c) how streams are
pushed into and extracted from the fabric.** The `L3↔BM↔L2↔Streamer↔L1` path is just
the machinery that turns tile sequences into element streams — its configuration is
JIT (§4).

## 3. The layered program (Candidate C, prototyped via B)

Two levels, deliberately separable by **fidelity**:

### L0 — tile sequences (the "outer loop") — *functional*
The ordered sequence of **tiles** pushed into / pulled from each fabric port over the
operator's tiled iteration. For matmul `C = A·B`, tiled `Ti×Tj×Tk`:

```
for (ti, tj) over output tiles:          # outer loop
  for tk over K-slices:
    feed  A[ti,tk] → West-port           # tile sequence per port
    feed  B[tk,tj] → North-port
    compute-accumulate into C[ti,tj]     # tile-level compute (recurrence, per tile)
  drain C[ti,tj] → South-port
```

**Key property (why this is first):** processing the tile sequence with the operator's
**tile-level compute** produces the *correct result* — no streams, no wavefront, no
timing needed. So **L0 is a pure functional reference calculation**. It is
device-independent (parameterized by tiling; no engine/bank ids). On-chip **reuse**
(a `B[tk,tj]` reused across a column of output tiles, or shared across compute tiles)
appears here as tile *residency + re-injection* in the sequence — the "on-chip block
moves to reuse tiles among compute tiles" become tile-lifetime facts, not engine ops.

### L1 — stream signatures — *spatial/temporal (timing)*
How each L0 tile becomes an **element stream** into the array: injection order (which
row/col edge, in what element order), rate, and the wavefront timing. Derived from L0
+ the affine **schedule** and the array mapping. L1 adds the spatial/temporal behavior
the cycle-accurate model needs; it does **not** change the functional result.

### compute recurrence — the domain-flow program
The fabric's computation (the SURE / recurrence + array dims). Device-independent
(parameterized by array size). This is domain_flow's native form.

**Fidelity mapping:** L0 tile-sequences → BEHAVIORAL/functional validation;
L0+L1 → TRANSACTIONAL/CYCLE_ACCURATE timing. This is exactly the multi-fidelity tier
split (`docs/SIMULATION_FIDELITY_FRAMEWORK.md`).

## 3a. Explicit tile propagation — every op declares its full tile I/O

**Rule:** every L0 `TileOp` declares the *complete* set of tiles it reads and
writes, and its kernel touches **no tile it did not declare**. The program stays
a flat, valid serial order, but because tile I/O is complete, the driver JIT can
recover the **tile-dependency DAG** (RAW/WAR/WAW over tiles) and thus which ops
are independent — the prerequisite for placing work on more than one compute
tile (§4a). An op that reaches beyond its declared tiles hides dataflow and
defeats placement.

Matmul already obeys this (one `MatMulAccum` per `(ti,tj,tk)`, reading exactly
`A[ti,tk]`,`B[tk,tj]`, writing `C[ti,tj]`), and so does the LU **trailing
update**. The one place that must be decomposed the same way is the LU **panel
factorization**: the down-the-tile-column propagation must be *in the program*,
not collapsed into a single op that implicitly spans the column. The **tile-LU**
(PLASMA-style) form makes it explicit:

```
for k:
  A[k,k]                 = GETRF(A[k,k])              # factor the diagonal tile (in-tile pivoting)
  for i>k: A[i,k],A[k,k] = TS_PIV(A[i,k], A[k,k])     # pairwise-eliminate each sub-diagonal tile
                                                       #   against the diagonal — rows exchanged
                                                       #   ONLY between the two neighboring tiles
  for j>k: A[k,j]        = TRSM(A[k,k], A[k,j])        # U row-panel
  for i>k,j>k: A[i,j]   -= A[i,k] . A[k,j]             # trailing update = matmul (alpha=-1)
```

**Where "neighbor pivoting" actually lives:** it is the *tile-pairwise* exchange
in `TS_PIV` — a nearest-neighbor operation between adjacent tiles in a column,
chosen precisely because a global column pivot search would break tile locality
on a dataflow fabric. `P·A = L·U` with `P` a product of these local tile-pair
exchanges. (The first L0 cut — issue #230, PR #238 — took a shortcut: a single
`LU_PANEL_FACTOR A[k,k]` op whose kernel reaches the whole column through the
shared matrix buffer. Correct numbers, but the multi-tile propagation is
invisible. This section is the corrected model to implement.)

## 4. Derivation from the DFG, and the driver JIT

**Streams are derivable from the DFG operators** (domain_flow already has the math):
for an operator with domain of computation `D`, affine operand access `f_A: D→idx(A)`,
and affine schedule `σ: D→t`, the **tile sequence** (L0) is `f_A` projected to tile
granularity over the iteration nest, and the **stream** (L1) is
`stream_port(t) = { A[f_A(x)] : σ(x)=t }`. domain_flow's `IndexSpace`, `AffineMap`,
`schedule`, `wavefront`, and `RecurrenceVariable`/SURE are exactly these primitives.

**Driver JIT — problem 2 (is the stream program always transformable into a data-path
config?): no, it is a realizability analysis.** The reactive fabric gives one freedom
and leaves three hard constraints:
- **Free:** stream-*rate* under-provisioning just **bubbles** the wavefront (correct,
  slower) — bandwidth is a *performance* axis, not a correctness gate.
- **Hard (else re-tile → a different stream, or refuse):** (1) **capacity** — working
  set (resident L3 tiles + L2 staging + L1 vectors) ≤ device L3/L2/L1 (the existing
  envelope / livelock-safety check); (2) **mover capability** — the L3→L2 tiling and
  L2→L1 gather/reformat and fan-out the stream needs are producible by this device's
  BM/Streamer; (3) **deadlock-free schedulability** within the device's credit/buffer
  counts.

So the driver JIT is: **portable program → feasibility check → data-path config**;
on failure it re-tiles/re-schedules (which changes the stream) or refuses. It may
**cache** the lowered config per (program, device) in the program cache.

## 4a. Device descriptor + placement (the driver JIT's target)

**The spatial layout is not in L0 — by design.** One device-independent
`TileProgram` runs on a single L3/CF set, on a NEWS arrangement, or on a
checkerboard; only the **driver-JIT mapping** differs, never the program. The
placement is specified by two things the JIT owns (neither exists yet — this is
increment 3):

**(1) Device descriptor** — the target KPU's spatial resources:
- **topology** of the L3/CF fabric: `single` (one L3 + one CF), `NEWS` (four L3
  tiles on the N/E/W/S edges feeding one central CF), or `checkerboard` (a 2-D
  grid of interleaved L3/CF tiles with nearest-neighbor L3↔L3 links);
- per-tile **capacities** (L3/L2/L1 bytes), **mover capability** (BM/Streamer
  reshape / gather / fan-out), and credit/buffer counts.

**(2) Placement pass** (in the driver JIT) — maps the L0 program onto that device:
- recover the tile-dependency DAG from each op's declared tile I/O (§3a);
- assign L0 logical tiles → physical L3 tiles and L0 compute ops → physical CF
  tiles over space **and** time (a space-time schedule);
- check feasibility (capacity / mover-capability / deadlock, §4) — infeasible ⇒
  re-tile (a different L0) or refuse;
- emit the data-path config (`.kpubin`, the SASS-analog).

**The same LU/matmul program on each topology:**
| Topology | How the placement pass maps it |
|---|---|
| **single L3/CF** | the whole DAG is time-multiplexed through one L3 + one CF; independent tile ops serialize |
| **NEWS + 1 central CF** | operand tiles staged on the surrounding N/E/W/S L3 tiles feed the central CF's West/North input ports; results drain South — matching the port names L0 already uses |
| **checkerboard** | independent tile ops (e.g. the trailing-update GEMMs for different `(i,j)`) run on **different CF tiles concurrently**; inter-tile reuse becomes on-chip **L3→L3 block moves** between neighbors |

**Simulator coupling:** the checkerboard mapping is only *executable/measurable*
once the timing executor models **multiple CF tiles** (today it is single-compute-
tile; L3/L2 are credit pools). That is the separate multi-compute-tile
simulator-capability issue — the descriptor/placement design here says *what* to
target; the executor work is *what runs it*.

## 5. Why `DMProgram` and `ScheduleResult` both mismodel this

| IR | What it is | Why it's not the portable program |
|---|---|---|
| `DMProgram` / `.kpubin` | linear DMA/BM/Streamer instruction stream + config + loops + barriers | it is the **driver-JIT output** (SASS-analog) — device-bound, over-sequenced |
| `ScheduleResult` | flat movement ops (LOAD/MOVE/FEED/…/COMPUTE) with engine ids | movement-centric + device-scalar + single-compute-tile; over-specifies compute sequencing the reactive fabric doesn't need |

Neither is the **portable tile/stream program**. `ScheduleResult` is the closest
starting point for **L0** (it already enumerates tile transfers into ports) but must
be reframed as *tile sequences into fabric ports* with engine/bank ids **removed**
(they belong to the JIT output), and grown to carry compute-tile placement + inter-
tile reuse (the multi-tile gaps).

## 6. Reshaped Phase 0 (supersedes "serialize ScheduleResult")

1. **L0 tile-sequence representation + functional reference** — a device-independent
   tile-sequence program (a fresh `TileProgram` type — see §7) with **explicit tile
   propagation (§3a)**: every op declares its full tile I/O, and multi-tile
   operators (LU panel factorization) are decomposed into tile-LU ops, not
   monolithic kernels. A functional reference processes it to the correct result
   (matmul validated against the ResNet oracle; LU validated by `P·A=L·U`).
   *Status:* first cut landed (PR #238, matmul + neighbor-pivot LU); **remaining:
   refactor the LU panel to the explicit tile-LU form of §3a** (the shortcut noted
   there).
2. **L1 stream signatures** — add the spatial/temporal stream layer for timing;
   drive the cycle-accurate CSP model from it (replacing inline generation).
3. **Driver JIT** (modeled in kpu-sim) — the **device descriptor + placement pass
   (§4a)**: lower L0/L1 → device data-path config for a chosen topology
   (single / NEWS / checkerboard) with the feasibility analysis (§4); + the
   **`.kpubin` disassembler** showing the per-engine DMA / BlockMover / Streamer
   programs (the JIT output).
4. **Serialize the portable program** (versioned per `dfg-kpu-versioning.md`);
   round-trip → identical cycles (`resnet_regression`).

**Separate, larger issue (simulator capability):** multi-compute-tile execution +
inter-tile reuse moves + resource orchestration in the timing executor (today it is
single-compute-tile; L3/L2 are credit pools) — needed to *exploit* the multi-tile
program, tracked apart from the format/serializer.

## 7. Open questions

- ~~L0 tile-sequence concrete schema: extend/reframe `ScheduleResult`, or a fresh
  `TileProgram` type?~~ **Resolved:** a fresh `TileProgram` type (PR #238),
  leaving `ScheduleResult` as an internal CSP artifact.
- `TS_PIV` numerics: tile-pairwise (neighbor) pivoting is stable enough for the
  dataflow model but weaker than global partial pivoting — decide the acceptance
  bar (reconstruction tolerance / conditioning) for the operator conformance
  corpus.
- Device-descriptor scope (§4a): the minimal fields the placement pass needs
  (topology + capacities + mover capability + credits) vs. what defers to the
  timing model.
- Where the feasibility analysis (§4) and the placement pass (§4a) are authored so
  they are shared by the driver-JIT model here and the eventual real driver.

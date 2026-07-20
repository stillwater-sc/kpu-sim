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
   tile-sequence program derived from the DFG (matmul first), and a functional
   reference calculation that processes it to the correct result (validated against
   the ResNet oracle). *This is the new first increment* (per the "define the outer
   loop first, it's enough for functional validation" decision).
2. **L1 stream signatures** — add the spatial/temporal stream layer for timing;
   drive the cycle-accurate CSP model from it (replacing inline generation).
3. **Driver JIT** (modeled in kpu-sim) — lower L0/L1 → device data-path config with
   the feasibility analysis (§4); + the **`.kpubin` disassembler** showing the
   per-engine DMA / BlockMover / Streamer programs (the JIT output).
4. **Serialize the portable program** (versioned per `dfg-kpu-versioning.md`);
   round-trip → identical cycles (`resnet_regression`).

**Separate, larger issue (simulator capability):** multi-compute-tile execution +
inter-tile reuse moves + resource orchestration in the timing executor (today it is
single-compute-tile; L3/L2 are credit pools) — needed to *exploit* the multi-tile
program, tracked apart from the format/serializer.

## 7. Open questions

- L0 tile-sequence concrete schema: extend/reframe `ScheduleResult`, or a fresh
  `TileProgram` type (leaving `ScheduleResult` as an internal CSP artifact)?
- Where the feasibility analysis (§4) is authored so it is shared by the driver-JIT
  model here and the eventual real driver.

# `kpu-run` — the driver that tests the simulation models against each other

**Status:** design note, for review before implementation
**Issue:** #285 (driver), with #286 (visualization) consuming the same step machinery
**Authority:** ADR 0002 §4 (driver architecture), §3.5 (one platform, not a fourth path)

## 1. The problem

There is no way to execute a Domain Flow Program from a command line and see what the
simulation models do with it.

`examples/characterize/tile_characterize.cpp` is the closest thing, and it does not drive
the executor: it includes `characterization.hpp` (the analytical harness) and
`tile_program_reference.hpp` (L-B), and **not** `tile_transaction_executor.hpp`. So the
five increments of L-T1 work — credits, capacity, residency reuse, per-hop movement — are
reachable only from unit tests. Nobody can run a program against them, vary a device, and
look at the result.

## 2. What the driver is *for*

Not "a CLI for convenience". The reason it earns its place is **differential testing**:

> Values are level-invariant (ADR 0002 §2). Decomposition changes *when* things happen,
> never *what* is computed.

That makes disagreement between two levels a **bug signal by construction**, and the
driver is what turns it into a check anybody can run. Bit-exactness between L-B and L-T1
is currently asserted inside one test file over two derived programs; a driver generalises
it to any program and any device the user can express, which is where the interesting
counterexamples live.

The corollary is what makes this urgent rather than nice-to-have: a level that returns
plausible *timing* while computing the wrong *values* is the failure mode this repo is
most exposed to, because timing is what everyone looks at. #279 is the precedent — a
capacity model that reported a healthy peak while exceeding its own budget, caught by
review rather than by anything runnable.

## 3. Levels, honestly

| Level | Interpreter | Status today |
|---|---|---|
| **L-B** behavioral | `TileProgramReference` | **exists** |
| **L-T1** block-sequential | `TileTransactionExecutor` | **exists** (#264 increments 1–5) |
| **L-T2** resource-transactional | — | **missing**, #283 |
| **L-CA** cycle-accurate | `ConcurrentTimingExecutor` + the four processes | exists **in substance**, but consumes a `ScheduleResult` rather than the program |

The driver ships against the two that exist and is shaped so the other two slot in without
changing its interface. It must not pretend a level exists: `--level resource-transactional`
fails with "not implemented (#283)", never silently falls back to a level that works.
A driver that quietly answers a different question than it was asked is worse than one
that refuses.

## 4. Decisions

### D1 — A new binary, sharing the derivation with the characterizer

`kpu-run` *executes*; `tile_characterize` *sweeps and analyses*. Merging them would give
one tool two jobs and two output formats. But they must **share** program derivation and
`DeviceDescriptor` construction rather than each parsing its own `--sizes`/`--tiles`, or
the two will drift and a bug will reproduce in one and not the other.

**So:** extract the shared argument→program and argument→device mapping into a small
header both include. No new derivation logic.

### D2 — Not through `VirtualPlatform` yet, but behind one seam

ADR 0002 §3.5 warns against adding a parallel path. Building the driver directly against
both executors is exactly how that happens.

**So:** all level selection goes through **one** function —

```cpp
RunOutcome run_at(ExecutionLevel, TileProgram&, const DeviceDescriptor&,
                  const Placement&, const stream::StreamProgram*);
```

— which is the seam `VirtualPlatform` (#282) takes over, and the only place #283 adds a
case. The driver itself never names an executor type. That is a deliberately small
commitment: it does not build the platform early, and it does not let the driver grow
executor knowledge it would later have to shed.

### D3 — Value comparison is bit-exact, and is the default

`--compare` runs every available level and diffs results against **L-B as the authority**
(ADR 0001 D5). Bit-exact via `memcmp`, not a tolerance, for L-B/L-T1/L-T2; tolerance only
for L-CA, with the band from ADR 0001 §7.5.

A disagreement exits **non-zero**. The driver is a test, so it must be usable in CI
without a human reading its output.

### D4 — `--step` granularity is the level's own

Stepping means "advance one transaction *at this level*", which is the only definition that
survives contact with four levels:

| Level | One step is |
|---|---|
| L-B | one `TileOp` applied |
| L-T1 | one **event**: a hop completing, or an op firing |
| L-T2 | one resource transaction (`read`/`write`/`push`) |
| L-CA | one cycle |

L-T1's timeline already records hop intervals, so its step cursor is a projection of data
the executor produces today rather than new machinery. This is the hook #286 needs, which
is why it is in the plan and not deferred.

### D5 — Reuse the existing trace **format**, and write the mapping into it

`--timeline out.json` emits Chrome Trace Event Format via
`include/sw/trace/trace_exporter.hpp`, so `tools/trace/` and `tools/visualization/` read it
unchanged. **No new format** is a constraint on the output, not a claim that the plumbing
exists — and the distinction is the whole of increment 2's cost:

- `ChromeTraceExporter::export_traces` consumes `std::vector<TraceEntry>` (component +
  transaction records with DMA/compute/memory payloads), and
  `ResourceTrackerExporter::export_to_chrome_trace` consumes
  `std::map<ResourceId, ResourceTrack>` (occupancy tracks). **Neither consumes L-T1's
  `TileOpRecord` or `HopRecord`**, and nothing in the repo serializes them today.
- Nothing serializes the occupancy stats either — `hop_busy_cycles`, `hop_utilization`,
  `peak_l3_residency` exist only in `TileRunStats`, read by tests.

So `--timeline` is a **mapping** to write: `TileOpRecord`/`HopRecord` → `TraceEntry`, one
event per hop so a transfer's legs appear separately rather than as one span. Station
occupancy aims at `ResourceTrack` instead, which is the type designed for it — but that
also needs the residency *series* the executor does not emit, which is the first gap #286
lists. #286 renders this; it does not come for free with it.

## 5. Increments

1. **Run and compare.** — **done.** `kpu-run --algo matmul|lu --size --tile --level`,
   diffing values bit-exactly against L-B and exiting non-zero on disagreement. The shared
   derivation is extracted to `driver/program_spec.hpp` and `tile_characterize` now uses it
   (D1), and `run_at` is the only function that names a level (D2). A level with no
   interpreter is refused with its issue number, and the header prints which levels were
   *not* run so a clean report is not mistaken for full coverage.

   LU is compared on its **pivot permutation and swap count** as well as its values, since
   a value diff alone would miss reordered pivoting.

   Registering the CLI tests turned up a latent bug: this project never includes CTest, so
   `BUILD_TESTING` is empty and `examples/characterize`'s smoke test had **never
   registered**. The guard is `KPU_BUILD_TESTS`; both are fixed, which is why the suite
   went from 150 to 155 tests.
2. **Device knobs and the timeline.** — **done.** The knobs are **per CSP process**, since
   #292 established that movement is a chain and lanes belong to the process:
   `--dma-engines`, `--block-movers`, `--streamers`, `--noc-links` with a
   `--*-bytes-per-cycle` each, plus `--compute-tiles`, `--l3-tiles`, `--macs-per-cycle`
   and `--streams <dataflow>`. (This entry previously listed `--dram-lanes` and
   `--onchip-lanes`, which were the collapsed model's knobs and no longer exist.)

   `--timeline out.json` writes Chrome Trace Event Format through the existing exporter,
   **one event per hop** — a transfer's legs run on different processes at different
   times, and collapsing them into one span would hide exactly what the chain exists to
   show. As D5 warned, this was a mapping to write: `TileOpRecord`/`HopRecord` →
   `TraceEntry` in `driver/timeline_trace.hpp`, since nothing serialized those records.
   Each hop lands on the trace component matching its process (`DMA_ENGINE`,
   `BLOCK_MOVER`, `STREAMER`), carries the bytes it moved, and its interval is asserted
   to equal the leg's interval exactly.

   `--timeline` is **refused** when only L-B was asked for, because L-B models no
   intervals; and `--streams` is refused for LU, which has no stream derivation, rather
   than silently ignoring the flag.
3. **`--step`** with the cursor of D4, at L-B and L-T1. — **done.** `--step` prints one
   line per transaction, `--step-limit n` windows it.

   **The two levels step by different mechanisms, and the difference is honest rather
   than incidental.** L-B *executes*: `apply()` is public, so the cursor holds the kernel
   state and applies one op per step, and values are observably incomplete partway
   through — which is what makes it useful for debugging arithmetic. L-T1 *replays* the
   recorded timeline, because the executor is an event engine whose schedule depends on
   the whole program; pausing it mid-flight would be a different executor. §D4 called the
   L-T1 cursor a projection of data the executor already produces, and that is exactly
   what it is. The consequence worth stating: values cannot be inspected mid-replay.

   Ordering is the part that makes a replay readable, and it must **mirror the executor**
   rather than merely be self-consistent: at one cycle the executor processes completions
   first and only then fires ready ops, so within a cycle every *release* precedes every
   *acquire* — across ops, not just within one. Getting that wrong made the replay report
   occupancy **above capacity** (2 BlockMovers busy on a one-mover device), because a lower-
   indexed op's acquire sorted ahead of a higher-indexed op's release. Asserted now, along
   with a fire preceding its first leg, a leg's end preceding the next leg's start, every leg
   closing before its op completes, cycles never running backwards, and lane occupancy
   conserved for **every** process.

   **Station occupancy is deliberately absent.** How many tiles sit in L3, L2 or L1 at a
   given cycle needs the residency *series*, which the executor does not emit — the first
   gap #286 lists. The stepper reports lane occupancy and says so; calling lane occupancy
   "station occupancy" would be the wrong kind of helpful.
4. **Program from a file**, once #265 lands, so `--program foo.l0` is literal rather than
   a derivation spec.
5. **Platform + L-T2**: `run_at` delegates to `VirtualPlatform` (#282) and gains the L-T2
   case (#283). The characterizer becomes a consumer of the same path.

Increment 1 is what makes "test the simulation models" true; 2 is what makes it
*informative*; 3–5 follow the issues they depend on.

## 6. What this note does not decide

- **The deployment spec format** (#282). The driver takes flags now; a `--deploy file.json`
  replaces them later without changing `run_at`.
- **Backdoor access** (#284). Out of scope: no CSP program may emit one, so a program
  driver is the wrong place for it.
- **Multi-device.** Follows the naming map in #282.

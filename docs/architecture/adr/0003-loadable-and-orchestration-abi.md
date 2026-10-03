# ADR 0003 — The loadable is the unit of deployment, and the KPU is called through a payload-free ABI

| | |
|---|---|
| **Status** | **Accepted.** Decided 2026-09-25 (#305 §10 closed); recorded 2026-10-03, once increments 1–3 had made the format and the ABI real rather than proposed |
| **Date** | 2026-10-03 |
| **Amends** | #229 §4a (the compiler/hardware boundary); `docs/plans/dfg-kpu-versioning.md` §4 (the binary format) |
| **Preserves** | ADR 0001 D1 (L0 `TileProgram` is the portable program); ADR 0002 (four levels, one `VirtualPlatform`) |
| **Context docs** | `docs/plans/program-encapsulation-and-orchestration.md` (the #305 plan, §1–§11), `docs/plans/kpuld-increment3-mmio-abi-and-reserve-then-launch.md` |
| **Implemented in** | PR #307 (container), #308 (deciding orchestrator), #310 (MMIO ABI, reserve-then-launch) |

---

## 1. Context

#265 gave one operator a durable form: the L0 `TileProgram`, serialized. #282 gave the
simulator an object that owns a deployment and runs one program against it. Between them,
one operator was loadable and one program runnable. The thing that would make a *model*
runnable was missing, and it was missing in three places at once:

| exists | missing |
|---|---|
| one operator as a portable program | a **unit of deployment**: N operators, their order, and the resource-management decisions between them |
| a platform that runs one program | the **orchestration semantics** that sequence operators and the state changes around them |
| test cases with tensor values inline | tensors that **cannot** be inlined: tens to hundreds of gigabytes |

The orchestration lived in test code. A test derived programs, decided what to load, called
`run_at`, and read the results back. That works as a test, but it is not an artifact you can
hand to a machine and say *run this model*. #283 (L-T2), #284 (backdoor) and #286 (event
record) are all features of executing *something*, and that something had no form. That is
why #305 was marked as blocking all three.

## 2. Decision drivers

1. **Program and data must separate.** A 70B-parameter model is not a file section you load.
   It is a region you map. Peak host memory must track the working set, not the model.
2. **The KPU datapath is credit-based dataflow** (`CLAUDE.md`, `docs/01-architecture/kpu-execution-model.md`).
   Whatever sequences operators must not become a second data path into the buffers.
3. **The KPU is stateful.** If every call re-placed every tile, the storage hierarchy would be
   decoration. Residency has to be part of the calling convention.
4. **Hand-rolled binary formats rot in this repo.** `ProgramSerializer`/`.kpubin` is the
   evidence: `kernels/bin/*.kpubin` abort with `std::get: wrong index for variant` after
   opcodes were renumbered with no version bump.
5. **Runs are pure functions of their inputs** (ADR 0002 §3.5). Anything that makes decisions
   at run time must keep that true.
6. **A simulator hang produces no evidence.** Every failure mode has to be a refusal that has a
   cause.

## 3. Options considered

| concern | options | chosen |
|---|---|---|
| package | ONNX + `external_data`; a custom binary; **FlatBuffers with our own schema** | FlatBuffers. ONNX gets data separation right, but it has no place for a RISC-V image, a call ABI or residency directives, and protobuf needs a heap on the consumer side. A custom binary is rejected on the evidence of driver 4 |
| orchestration code | a static descriptor stream; a bespoke bytecode; **an ELF** | ELF. It is what an ISS loads and what a debugger reads. A static stream is a *record* of decisions, not a program that makes them (D6) |
| orchestrator ISA | RV32IMAC bare-metal; **RV64GC** | RV64GC. A runtime allocator is real code maintained for two targets, and that is worth more than a small core |
| Renode bridge | a C# reimplementation of the KPU; **IPC to the C++ `VirtualPlatform`** | IPC. A C# KPU would be a fifth execution engine in a fifth language |
| runtime decisions | replay a baked schedule; **decide at run time** | decide. The consequences are D5–D7 |
| allocation | program-order acquisition; **reserve-then-launch**; a static partition; detect and recover | program-order first (increment 2), then reserve-then-launch (increment 3). A static partition wastes capacity, and detect-and-recover needs a rollback that does not exist |

## 4. Decision

### D1 — Three layers, each doing what it is good at

| concern | format |
|---|---|
| orchestration **code** | an **ELF** |
| the **package** | **FlatBuffers**, schema `schemas/kpu_loadable.fbs`, file identifier `KPLD`, extension **`.kpuld`** (after NVDLA's "loadable") |
| tensor **data** | **external blobs**, referenced by uri + offset + length, never contained |

A tensor table entry carries what the **DMA** needs: name, dtype, shape, tiling, a 64-bit
device address, the source extent, and a **declared** content digest. The digest is declared
by the producer and verified lazily or not at all, and provenance must say which. Hashing
100 GB at load time defeats mapping it, and claiming a digest is verified when nobody checked
it is worse than making no claim.

**Device addresses are 64-bit regardless of the orchestrator's XLEN.** Getting this wrong
would mean a format change later, so it is decided now.

### D2 — The compiler/hardware boundary is the loadable

```
  .onnx  ──►  DomainFlowGraph  ──►  ═══ compiler / hardware boundary ═══  ──►  .kpuld  ──►  VirtualPlatform
              (canonical IR)                                                 (this ADR)    (ADR 0002, any level)
```

- **#229 §4a's boundary moves from `.kpubin` to `.kpuld`.** The argument's shape is unchanged:
  one hard boundary, the compiler to its left, the simulator to its right, and kpu-sim never
  lowers. Only the artifact's identity changes, in the direction ADR 0001 already pushed.
- **ADR 0001 D1 is preserved.** The loadable *contains* L0 programs, one per operator, in
  #265's format.
- **The versioning plan's §4 is answered rather than bypassed.** It asked for "one binary format,
  one source format". That is now `.dfg` as source and `.kpuld` as binary, with L0 inside it.
  This supplies the binary format it asked for; it does not add a third path.
- ONNX stays the *ingestion* format upstream. `.kpubin`/`DMProgram` is at most an internal JIT
  artifact for one device.

### D3 — The container is verified before it is trusted, and versioned like an ABI

- **Verify, then read.** The generated FlatBuffers accessors do not bounds-check. The reader
  runs the verifier first and then its own structural checks, and refuses with a **cause**:
  `NotALoadable`, `Truncated`, `MalformedContainer`, `UnsupportedVersion`,
  `MissingRequiredField`, `InconsistentRecord`, `CapabilityMismatch`.
- **Canonical bytes.** The same `Loadable` always writes the same bytes, so a digest can key a
  cache and a checked-in fixture can be compared byte for byte.
- **#265's version machinery, reused.** It has a format version, a `min_consumer` derived from
  the records present (never hand-maintained), and a `producer` so that `bad_producers` has
  something to match. Version gates are checked **before** the orchestration image runs: an
  ELF from a producer this build does not understand is refused, not executed.
- **Absent means not declared.** Optional fields are tables, not structs, so a report of what a
  level ignored does not turn into noise.
- **A capability profile (R6) with three outcomes.** A loadable declares the compute-tile
  kinds, dtypes and minimum resources it needs. Against a deployment, each requirement is
  *satisfied*, *mismatched* (refused), or *unverifiable* (reported, because the deployment does
  not declare it). Folding the third outcome into either of the other two is how a capability
  check starts lying.
- **A golden corpus, executed in CI**, with `*.kpuld -text -diff` in `.gitattributes`. That includes a
  fixture that demands a newer reader (`tests/program/loadable/needs_a_newer_reader.kpuld`).

### D4 — The orchestrator is a stored-program machine; the KPU is not; and the line between them carries no payload

| | orchestrator (RV64GC) | KPU datapath |
|---|---|---|
| model | stored-program, sequential | credit-based dataflow |
| decides | which operator runs next; which tiles to place, keep and release | nothing: it pushes when it holds a credit |
| touches | descriptors, completions, status | tiles |

Resource management is a *dependency set*, and organizing one is what sequential code is good
at. A manager core does not violate the dataflow rule, because that rule governs the datapath.

> **The rule: no descriptor, completion, register or ring entry carries payload.**

An MMIO window into buffer contents *is* a backdoor. #284 owns the backdoor so that it stays
unphysical **and flagged in provenance**. A second, unflagged one arriving through this ABI
would silently invalidate every timing result that used it.

**The guest memory map is part of the ABI.** In an ordinary platform, RAM is visible to the
guest by default, so the hole would open without anyone writing it:

| region | orchestrator | KPU |
|---|---|---|
| orchestrator RAM (code, stack, allocator; the ELF is loaded here) | RW | — |
| the loadable's **metadata tables only** | R, read in place | — |
| descriptor ring, completion ring, diagnosis area | RW | RW |
| **tensor DRAM** | **no access: faults** | DMA only |

`PLACE` is the only way a tensor byte moves, and the memory map is what makes that true
rather than merely intended.

### D5 — The call ABI

**Vocabulary.** None of these carries payload.

| descriptor | meaning |
|---|---|
| `RESERVE` | claim L3 credits for one operator, **all or none** (D7) |
| `PLACE` | move one tile along **one** leg, under the named operator's reservation. Hops never collapse: DRAM→L1 is three descriptors, because it is three moves |
| `RELEASE` | return a tile's credit. With `kReleaseAtLastRead` it travels *before* the `LAUNCH` it governs and completes *after* it, so completion order is not issue order |
| `CONFIGURE` | load a domain-flow program into a programmable compute tile (increment 5) |
| `LAUNCH` | run an operator, given where its operands already are |
| `FENCE` | order a completion against a later descriptor |

**The ABI can say no.** For a static schedule, blocking on credit is fine, because the
compiler proved the schedule fits. For a runtime allocator, a block is a **hang** and a refusal
is a **decision point**. Every descriptor gets exactly one completion: `Done`, or a refusal
carrying a `RefusalCause` (`InsufficientCredit`, `ReservationOutOfOrder`, `NoReservation`,
`ReservationExceeded`, `Unsupported`), the arithmetic behind it (needed, available, capacity,
blocking operator), and readable text that names the operator.

**Status surface.** The orchestrator may read inventory (the #282 naming map), occupancy
(whether a tile is resident), credits (free, reserved, held), per-operator manifests (which
tiles an operator reads, and its peak live set), and completions. It may not read tile
contents or any tensor element. The manifest is derived by the **device**, so the orchestrator
never parses L0.

**Wire format.** The authoritative layout is `include/sw/kpu/orchestration/abi.hpp`. In
summary:

- ABI version 1.0, magic `KPULD` in the `ID` register.
- 64-bit registers in a 4 KiB window, accessed as aligned 8-byte words.
- 64-byte descriptor and completion records, little-endian and packed by hand.
- Every field is an index, a count, an id or a flag. Names cross as indices into tables both
  ends already hold (the loadable's tensors and operators, the deployment's devices), so no
  record has a variable-length field.
- Writing `DRING_TAIL` is the doorbell.
- Diagnosis text lives in a device-written DIAG area that the completion locates.
- A completion that releases several tiles continues in further records rather than
  growing.

**Residency is a fact the device owns.** The device keeps the authoritative credit ledger. The
orchestrator keeps a memory of its own decisions and checks it against `ST_HELD`/`ST_COMPLETED`
before each operator. One fact has one spelling, and disagreement is a refusal.

### D6 — Decisions are made at run time, and must be deterministic

The orchestration program makes real placement decisions; it does not replay a baked
schedule. A run then stays a pure function of its inputs **provided** the decisions derive
only from those inputs: no wall clock, no uninitialized reads, no address-derived hashing, no
iteration order that depends on allocation. The check is to **record** what was decided:

- The **descriptor trace** is evidence, not a program. Two runs of one loadable must produce
  byte-identical traces, and so must two transports (direct and MMIO), and in increment 4 two
  compilations of the orchestrator (host and RV64).
- `RunIdentity` gains the **orchestrator image digest** and the **declared tensor-data digest
  with its verification state**, because the decisions come from that image and the values
  from that data. *(Owed by increment 4, when an image exists to digest.)*

### D7 — Allocation is reserve-then-launch, enforced by the device

The L-T1 executor is deadlock-free because new slots are acquired in program order. A weaker
rule, "do not promote a ready op past an earlier ready one", wedges at every finite capacity
below the greedy peak. It was measured at 29 tiles for the derived matmul, failing at 25 with
a true live set of 21. A runtime allocator is exactly the thing that can reproduce that
wedge. So:

- **R1.** A `RESERVE` for operator *j* is granted only if every uncompleted *i < j* already
  holds one.
- **R2.** A `RESERVE` is granted only if it fits in the free credits. It is all or none, and
  refused rather than queued.
- **R3.** A `PLACE` must name an operator holding a reservation. A placement made ahead of an
  earlier operator's launch must fit inside that reservation.
- **R4.** Operators launch in order, and only within what they reserved. The bound,
  `peak_live_tiles + retained`, is checked by the device at launch, not trusted from the
  orchestrator.

**Why this cannot deadlock.** By R1, the earliest uncompleted operator either holds a
reservation or no later one exists. If it holds one, that reservation is sufficient, so it
completes and returns credits. If no reservation exists, then either its own reservation fits,
or the machine is too small for it given what is held, which is a refusal with a diagnosis.
Nothing ever waits on an unbounded condition, so the only outcomes are progress or a refusal.
Hold-and-wait is broken because no operator holds slots an earlier one still needs.

**What this buys.** Tiles for operator *k+1* may be reserved and placed before *k* launches,
which is an order program-order acquisition forbids. A placement never runs ahead of the
tile's producer: tensors the current operator writes are excluded.

**What it is checked against.** A capacity sweep asserts that no executor refusal ever follows
a granted reservation. With the bound reduced by one, that sweep fails. The greedy allocator
without reservations is kept as a test-only ablation; the device must refuse it, naming the
later operator holding the credits, and never hang.

### D8 — One orchestrator source, two targets, and an honest clock

- **One source**, compiled for the host (the reference, and CI) and cross-compiled for RV64GC
  (Renode). Both reach the machine only through a `KpuPort`. The source must compile
  **freestanding**: no exceptions, no RTTI, an arena instead of a general heap. The host build
  follows the same subset, or the two stop being the same program.
- **The Renode bridge is IPC to the C++ `VirtualPlatform`**: shared memory for the DRAM region
  and a socket for descriptors and completions, imitating the shape of Renode's Verilator
  channel. The C# side stays a thin MMIO shim with no model in it.
- **L-B models no time.** A KPU call at L-B costs a declared constant, and provenance records
  "timing not modelled (L-B); orchestration time only". A timing-free level must not advance a
  virtual clock and print the result next to real measurements.
- **At L-T1 a `PLACE` is not timed.** The executor schedules each leg inside the run, so a
  `PLACE` completion reports `timed = false`, never `cycles = 0`. Placing ahead therefore buys
  allocation freedom at L-T1, not speed. #283 (L-T2) is where a `PLACE` becomes a timed
  transaction.

## 5. Consequences

### Positive

- There is an artifact you can hand to a machine: one `.kpuld` runs a model at any level,
  through either transport, with values bit-exact against the in-process path.
- **Statefulness is measured.** Two GEMMs sharing a weight tensor save exactly that tensor's four
  DMA transfers. This is a transfer count, not a flag.
- The "no payload" rule is checked three ways: by the **types** (structured-binding tripwires
  over `Descriptor` and `Completion`), by the **wire** (fixed 64-byte records with
  `static_assert`ed sizes), and by the **run**. On the run, tensor DRAM faults on the
  orchestrator's bus, and two runs whose inputs differ in every element produce byte-identical
  bus logs. Each check was confirmed to fail when its property is broken.
- The minimum L3 is now the machine's, not the orchestrator's guess. For the three-GEMM chain
  it is 6 slots cold and 10 warm; increment 2's `|tiles to place|` check had reported 8 and 12.

### Negative and costs

- The reservation bound is conservative by construction (`peak_live_tiles` plus retained
  tiles). A tighter, searched bound is possible and deliberately deferred until a test shows
  the extra slots cost something real.
- FlatBuffers and `flatc` are build dependencies, kept private to `kpu_program`, so nothing
  outside it sees `<flatbuffers/...>`.
- The orchestrator now has a protocol to get wrong. Every protocol bug becomes a refusal, but a
  refusal still stops the run.

### Risks

- **Freestanding drift.** `descriptor.hpp` currently includes executor and naming-map headers
  whose inline functions `throw`, so the decider cannot yet compile under `-fno-exceptions`.
  Increment 4 must split that header first. Until then, "one source, two targets" is a
  requirement, not a property.
- **Renode is new technology here.** This repo has no RISC-V and no .NET surface. Increment 4
  carries that risk on purpose, after the semantics it must reproduce are pinned by the host
  reference.

## 6. Implementation status

| #305 increment | state | where |
|---|---|---|
| 1. the container | done | PR #307: `include/sw/kpu/loadable/`, `schemas/kpu_loadable.fbs`, `tests/program/loadable/` |
| 2. deciding orchestrator, program-order acquisition | done | PR #308: `include/sw/kpu/orchestration/{descriptor,status,orchestrator}.hpp` |
| 3. MMIO ABI, reserve-then-launch | done | PR #310: `include/sw/kpu/orchestration/{port,kpu_device,abi,mmio}.hpp` |
| 4. RISC-V under Renode | next | split `descriptor.hpp`; freestanding decider; RV64 differential trace test; tensor-DRAM fault test in the guest; `RunIdentity` gains the image and data digests (D6) |
| 5. heterogeneous compute tiles | planned | compute-tile kinds in `DeviceSpecification`; `CONFIGURE` becomes real |
| 6. scale | planned | ONNX in (#229 [A]) → `.kpuld` with external weights, mapped not loaded |

## 7. Deferred, with the reason

- **Multi-device orchestration.** The naming map spans devices, but execution refuses more than
  one. The loadable carries an unused device axis, because adding one later would be a format
  change.
- **L-T2 and L-CA under Renode.** Time reconciliation gets harder as the level gets finer, so
  this starts at L-B and L-T1, where the answer is known.
- **Preemption and multi-tenancy.** Nothing in the ABI forbids them, and nothing yet needs
  them.
- **The backdoor (#284) stays the only unphysical path to tensor contents.** D4 is what keeps
  this ADR from quietly adding a second one.

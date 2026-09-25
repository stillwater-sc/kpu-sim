# Encapsulating the program, separating the data, and orchestrating from RISC-V

**Status:** reviewed; two decisions recorded below. Not yet implemented.
**Prompted by:** the #265/#282 work, which exposed that there is no *unit of deployment*
**Relates to:** #229 (model ingestion, §4a's compiler/hardware boundary), ADR 0001 D1 (L0 is
the portable program), ADR 0002 (four levels, `VirtualPlatform`),
`docs/plans/dfg-kpu-versioning.md` (R1–R9), `docs/09-virtual-platform/virtual_platform_analysis.md`
(NVDLA loadable reference), `docs/09-virtual-platform/qemu-vs-userspace-runtime.md`
**Likely outcome:** ADR 0003 on acceptance, because it moves a boundary #229 already named

## Decided on review (2026-09-25)

- **The orchestration program makes real placement decisions at runtime**, rather than replaying
  a schedule the compiler baked. Consequences are folded into §6.4 (a status surface of metadata,
  never payload; an ABI that can refuse; determinism as a requirement), §6.5 (deadlock-freedom
  becomes the orchestrator's problem) and §7.1 (one orchestrator source, two targets).
- **Allocation: program-order acquisition first, then reserve-then-launch** (§6.5). The existing
  deadlock proof applies verbatim in increment 2, and out-of-order placement is bought
  deliberately in increment 3 rather than assumed.

Everything else in §10 is still open.

---

## 1. What the program work exposed

#265 gave the L0 `TileProgram` a durable serialized form, and #282 gave the simulator an
object that owns a deployment and executes a program against it. Between them they made one
thing loadable and runnable. They also made the missing piece unmistakable:

| exists | missing |
|---|---|
| **one operator** as a portable program (L0, #265) | a **unit of deployment**: N operators, their order, and the resource-management decisions between them |
| a platform that runs **one program** (#282) | the **orchestration semantics** that sequence operators and the state changes around them |
| a test case with `VALUES_ROW` inline (#265 increment 2) | tensors that **cannot** be inlined — tens to hundreds of gigabytes |

Today the orchestration lives in test code and demos: a test derives programs, decides what to
load, calls `run_at`, and reads results. That is fine for a test and is not a deliverable. There
is no artifact you can hand to a machine and say *run this model*.

**So the encapsulation has to be solved before more of the simulator is worth building.** Every
further fidelity level (#283 L-T2), observability facility (#286) and backdoor (#284) is a
feature of executing *something*, and right now the something has no form.

## 2. Three layers, each doing what it is good at

The headline recommendation, because it answers the framing question (COFF? ELF? TFLM?)
directly:

| concern | format | why |
|---|---|---|
| orchestration **code** | **ELF** | it is what a RISC-V ISS loads, what a toolchain emits, and what a debugger understands. Nothing is gained by inventing a code container |
| the **package** | **FlatBuffers** (our own schema) | zero-copy: the orchestrator reads its own tables *in place*, with no parse step and no heap — the property that matters on a resource-constrained manager core |
| tensor **data** | **external blobs**, referenced | a 70-billion-parameter model is not a section of a file you load; it is a region you map |

TFLM is the right precedent for the *shape* of the package and the wrong precedent for its
*contents*: TFLM is self-contained because an MCU has one flat address space and a model that
fits in it. The moment tensors are hundreds of gigabytes, "self-contained" stops being a
simplification and becomes an impossibility. ONNX with `external_data` gets the separation
right and is still the wrong container here (§5).

### 2.1 Where this sits

```
  .onnx                            exchange format, external
    │  domain_flow front-end                                    (#229 [A])
    ▼
  DomainFlowGraph                  canonical SOURCE IR          (#229 D3)
    │  domain_flow passes + KPU backend                         (#229 [B])
    ▼
══════════════ compiler / hardware boundary ══════════════
  KPU loadable  (.kpuld)           ← THIS PLAN: the unit of deployment
    ├── manifest: versions, min_consumer, producer, required machine profile
    ├── orchestration program: an ELF that DECIDES placement at runtime (§6.4, §7.1)
    ├── operator table → an L0 TileProgram each              (#265 format)
    ├── domain-flow programs for programmable compute tiles
    ├── tensor table → EXTERNAL blobs (uri, offset, extent, dtype, tiling, digest)
    └── deployment requirements, checked against a DeploymentSpec (#282)
    │
    ▼  Renode: RISC-V ISS runs the orchestration program;
       the KPU is a callable, STATEFUL hardware subroutine behind MMIO
  VirtualPlatform at a chosen level                             (#282, ADR 0002)
```

### 2.2 What this does *not* replace, stated so nothing is reversed by implication

- **L0 stays the portable program.** ADR 0001 D1 is preserved, not overturned: the loadable
  *contains* L0 programs. One operator is still an L0 `TileProgram`, and #265's format is still
  how one is written down.
- **`.dfg` stays compiler-internal**, exactly as the versioning plan has it.
- **ONNX stays the ingestion format**, upstream of the boundary, per #229 [A].
- **`.kpubin`/`DMProgram` becomes an internal JIT artifact for one device, or is retired.** ADR
  0001 already demoted it; this plan removes the last reason to keep it at the boundary.

### 2.3 Two amendments this plan makes, on purpose

1. **#229 §4a said the compiler/hardware boundary is `.kpubin`.** It becomes the **loadable**,
   which carries L0 rather than DMProgram. The *shape* of that section's argument is unchanged —
   one hard boundary, compiler to its left, simulator to its right, kpu-sim never lowers — only
   the artifact's identity changes, and it changes in the direction ADR 0001 already pushed.
2. **`dfg-kpu-versioning.md` §4 asked for "one binary format, one source format".** This
   *supplies* the binary format it wanted rather than adding a third path: `.dfg` source,
   `.kpuld` binary, and L0 as the portable program inside it. The section's own note already
   records that ADR 0001 unseated `.kpubin` from that role and left the slot open.

## 3. The orchestrator is a stored-program machine. The KPU is not.

This is the part most likely to be mis-implemented later, so it is stated as a rule rather than
as background.

`CLAUDE.md` is emphatic that the KPU is **not** a stored-program machine: credits flow up, data
flows down, and a component pushes only when it holds a credit. That rule is about the
**datapath** — DMA, BlockMover, Streamer, compute fabric. It is not violated by a manager core,
and the reason a manager core is the right answer is exactly the reason given: resource
management is a **dependency set**, and a dependency set is what sequential code is good at.

| | orchestrator (RISC-V) | KPU datapath |
|---|---|---|
| model | stored-program, sequential | credit-based dataflow |
| decides | which operator next; which tiles to place and release; when a span may start | nothing — it pushes when it has a credit |
| touches | descriptors and completions | tiles |
| may block | yes, on a completion | never on the orchestrator |

**The orchestrator must have no data path into L3/L2/L1.** It issues descriptors and reads
completions; it does not read or write tile contents. Two reasons, and the second is the one
that bites:

- a datapath through the orchestrator would make the KPU von Neumann in practice while the
  documentation said otherwise, which is how an architecture rots;
- an MMIO window into buffer contents *is* a backdoor. #284 owns the backdoor deliberately:
  it is simulation-only, unphysical, and **flagged in a run's provenance** so nobody mistakes a
  staged tensor for a moved one. A second, unflagged one arriving through the orchestration ABI
  would silently invalidate every timing result that used it.

A reviewer should be able to check both properties by reading the descriptor vocabulary (§6.2)
and finding no descriptor that carries payload.

## 4. Separating program from data, concretely

A tensor table entry carries what the **DMA** needs and nothing the orchestrator needs:

| field | why |
|---|---|
| logical name, dtype, shape, tiling | the operator's operands are tiles of it |
| device address (64-bit) | where it lives in the simulated DRAM space |
| source: uri + offset + length | where the bytes really are |
| content digest, and a `verified` flag | reproducibility — with an honest caveat, below |

**Nothing loads the tensor.** The DMA reads it from simulated DRAM, and DRAM is backed by the
file. In Renode a memory region can be file-backed, so a 100 GB weight file is *mapped*, and the
peak host memory is the working set rather than the model.

**The digest is declared by the producer and verified lazily or not at all**, and the provenance
must say which. Hashing 100 GB at load time defeats the purpose of mapping it; claiming a
verified digest that was not computed is worse than claiming nothing. This is the same rule the
platform work arrived at for `unmodelled_fields` and for `StateSnapshot`'s coverage tag: a field
that says "checked" when nothing checked it is a lie with a long half-life. So `RunIdentity`
gains a data component that is the **declared** digest plus a verification state.

**64-bit device addresses regardless of the orchestrator's XLEN.** A 32-bit manager core is
attractive (§10) and cannot address a 100 GB tensor space. The descriptor fields are therefore
64-bit by construction, and the orchestrator manipulates them as opaque pairs if it is RV32.
Getting this wrong is a format change later, so it is a decision now.

## 5. The container: options, and why FlatBuffers

| option | verdict |
|---|---|
| **ONNX (protobuf) + `external_data`** | gets data separation right, wrong container: there is no place for a RISC-V image, a call ABI, tile schedules or residency directives without abusing `metadata_props`; protobuf needs a heap and a runtime on the **consumer** side, which is the side we are trying to keep small; and we would inherit its three orthogonal version axes. Stays the **ingestion** format, upstream. |
| **FlatBuffers, our own schema** | **recommended.** Zero-copy reads — the orchestrator reads its operator and tensor tables in place, no parse, no heap. Add-only schema evolution matches R8 by construction. Proven by TFLM for precisely this job. Costs a new dependency and a schema compiler in the build, and requires the verifier to be used rather than trusted. |
| **a custom binary** | rejected, with evidence rather than taste: `ProgramSerializer`/`.kpubin` is the hand-rolled binary this repo already has, and `kernels/bin/*.kpubin` **rotted** — opcodes renumbered with no version bump, and those files now abort with `std::get: wrong index for variant`. Reinventing offsets, alignment, verification and evolution is how that happened. |

Two things the schema must get right on day one, because both are expensive later:

- **Verification is not optional.** A malformed loadable must be **refused with a cause**, in
  the shape #265 established (`NotAnL0File`, `MalformedPreamble`, `UnsupportedVersion`, …), not
  trusted because FlatBuffers accessors do not bounds-check by default. The reader runs the
  generated verifier first, then its own structural checks.
- **A "declared" field must be distinguishable from a default.** #282 increment 1 learned this
  the hard way: absent has to mean *not declared*, or a report of what a level ignored becomes
  noise. FlatBuffers gives this naturally — an absent field is absent — provided the schema uses
  tables rather than structs for anything optional.

## 6. The KPU as a callable, stateful hardware subroutine

### 6.1 Statefulness is the whole point, so the ABI must express residency

If the call were "do this GEMM on these DRAM pointers", the KPU would re-place every tile on
every call and its storage hierarchy would be decoration. Efficiency demands that a tile placed
for operator *k* still be resident for operator *k+1*, which means **residency is part of the
calling convention**, not an implementation detail behind it.

So a `LAUNCH` names *where its operands already are*, and the orchestrator is what knows. That
is the resource-management dependency set the manager core exists to track.

### 6.2 The descriptor vocabulary

A descriptor ring in simulated memory, a doorbell register, a completion ring, an interrupt.
Descriptor kinds — note that none of them carries payload (§3):

| descriptor | meaning | mover |
|---|---|---|
| `PLACE` | bring a tensor tile into a named resource | DMA (DRAM→L3), BlockMover (L3→L2, L3→L3), Streamer (L2→L1) |
| `RELEASE` | return the credit for a tile | — |
| `CONFIGURE` | load a domain-flow program into a programmable compute tile | Streamer |
| `LAUNCH` | run an operator on a compute tile, given its operands' residency | — |
| `FENCE` / `WAIT` | order a completion against a later descriptor | — |

**Resources are named with #282's naming map** — `dev0/l3[3]`, `dev0/cf[2]/l1[1]`. That map was
built for the backdoor and the event record; this is its third consumer, which is the evidence
that building it as *identity* before *state* was right.

**No collapsed hops.** `PLACE` names one leg. A tile reaching L1 from DRAM is three descriptors,
because it is three CSP processes over three physical pathways — the span always contains all of
its hops, and a descriptor that claimed to move DRAM→L1 would describe a machine that cannot be
built.

**`RELEASE`, never "evict".** The vocabulary is credits and buffers.

### 6.3 Heterogeneous compute tiles

The fabric is not uniform: some tiles have a fixed ISA (a VIO tile, an FFT tile), others are
programmable and execute a domain flow program as data is pushed into them.

- `DeviceSpecification` (#282) gains a **compute-tile kind** list — additive, R8:
  `{fixed:vio, fixed:fft, programmable}`.
- The loadable's operator table says which kind each operator **requires**.
- Loading a loadable against a deployment that lacks the kind is **refused**. This is the R6
  capability dimension the versioning plan already asked for, finally with a home: *"an int8
  program on an fp32-only fabric must be rejected, not mis-run."* Compute-tile kind is the same
  argument in a different dimension.
- A programmable tile's behaviour comes from a domain-flow program `CONFIGURE`d into it, and it
  **executes as data arrives** — which is the credit-based datapath, not a stored-program one.
  A fixed tile takes no program and its `LAUNCH` names an operator it implements.

### 6.4 Runtime decisions need a status surface — metadata, never payload

**Decided (2026-09-25): the orchestration program makes real placement decisions at runtime.**
That settles §10 Q4 and has consequences worth spelling out, because a decision-making
orchestrator needs to *see* the machine, and §3 forbids it a data path.

The resolution is the distinction #282 increment 3 already drew for the naming map: *"does this
resource exist?"* is answerable from the deployment, *"what is in it?"* needs state. The ABI
draws the same line one notch further along:

| the orchestrator may read | it may not read |
|---|---|
| resource inventory — what exists, from the naming map | tile **contents** |
| occupancy — which tensor tile is resident in which resource | any element of any tensor |
| credit availability per process and per level | |
| completion records, with what each one released | |

All of the left column is **metadata about placement**; none of it is payload. That keeps §3's
rule intact — no descriptor and no status read carries data — while giving the allocator
everything it needs. It also keeps the backdoor (#284) the only unphysical path to contents.

**The ABI must be able to say no.** This is the part that makes runtime decisions safe rather
than merely possible. A `PLACE` or a reservation the machine cannot currently satisfy must
complete with a **refusal** — *insufficient credit* — not block until it can. For a static
schedule, blocking is fine: the compiler proved the schedule fits. For a runtime allocator, a
block is a hang, and a refusal is a **decision point**. The L-T1 executor already behaves this
way internally — it "stalls and then refuses with a diagnosis rather than wedging silently" —
and that property has to survive being exposed through the ABI rather than being an internal
courtesy.

**The orchestrator must be deterministic.** ADR 0002 §3.5 says a run is a pure function of
`(program, initial_state, deployment, level)`. Runtime decisions do not break that *provided*
they are derived only from those four: no wall clock, no uninitialised reads, no
address-derived hashing, no iteration over a container whose order depends on allocation. If
that holds, the decisions are a function of the inputs and the claim survives; if it does not,
every recorded result becomes unreproducible and the whole identity apparatus from #282 is
decoration. This is **testable**, and §7.1 is how.

Two identity consequences follow, both in the shape of #282's "four inputs were really six":

- `RunIdentity` gains the **orchestrator image digest**. The decisions now come from that image,
  so two runs of the same model with different orchestrators are not the same run.
- It also gains the **declared tensor-data digest** and its verification state (§4) — declared
  by the producer, verified lazily or not at all, and labelled as such.

### 6.5 Deadlock-freedom is now the orchestrator's problem, and it is a measured one

This is the substantive engineering risk in the whole plan, and it is not hypothetical.

The L-T1 executor is deadlock-free because of one invariant: **new slots are acquired in program
order** — an op may take slots only when no earlier unfired op still needs any. An op that needs
*no* new slot, because every tile it touches is already resident, fires freely, since it cannot
contribute to hold-and-wait.

That invariant was not chosen for elegance. The weaker rule — "do not promote a ready op past an
earlier ready one" — **wedges at every finite capacity below the unbounded run's greedy peak**:
measured at 29 tiles for the derived matmul, so it fails at 25 even though the true live set is
21. Feeds are ready immediately and take every slot; the computes that would retire those tiles
are not ready yet. Classic hold-and-wait, and no ordering *among ready ops* breaks it.

**A runtime allocator is exactly the thing that can violate this.** If the orchestrator places
tiles in whatever order looks locally good, it reproduces the weaker rule and inherits its
wedge. Options:

| approach | deadlock-free | freedom | verdict |
|---|---|---|---|
| **acquire in program order** | yes, the existing proof applies verbatim | constrained, but the escape hatch is large: reuse-driven firing needs no new slot and stays free | **DECIDED: increment 2** — cheapest sound rule |
| **reserve-then-launch**: reserve an operator's whole slot set before any `PLACE` for it, reservations granted in a total order | yes, hold-and-wait is broken by atomic acquisition | out-of-order *placement* allowed once reservation is ordered | **DECIDED: from increment 3** |
| static partition per in-flight operator | yes | poor: capacity is wasted on operators not running | rejected |
| detect and recover | — | — | rejected: recovery needs rollback, and there is none |

**And the simulator must refuse rather than wedge, at the ABI.** §6.4's refusal is what turns a
potential deadlock into a diagnosable event: a reservation that cannot be satisfied is reported,
the orchestrator chooses again, and a test can assert that the machine never hangs. A hang in a
simulator is the worst failure mode available, because it produces no evidence at all.

## 7. Renode

Renode gives a RISC-V ISS, a platform description (`.repl`), an MMIO peripheral model, virtual
time, GDB and a scriptable console. We supply the KPU peripheral. Note that this repo has
**neither a RISC-V nor a .NET surface today**, so increment 4 is where the new-technology risk
concentrates (§9).

### 7.1 One orchestrator, two targets — and the descriptor stream is a *recording*

The first draft offered a static `descriptor_stream` as a peer orchestration kind. **Runtime
placement decisions retire that idea**: a static list of descriptors is not a program that makes
decisions, it is a record of one particular set of them. Keeping it as a peer would have been the
two-sources-of-truth problem the draft flagged about itself.

What replaces it is better:

| artifact | what it is | who runs it | why |
|---|---|---|---|
| **the orchestrator** | ONE source, compiled twice: host C++ for CI, cross-compiled RV for Renode | reference interpreter, and the ISS | one source of truth for the decisions, and a **differential test** between targets in the shape #285 established between levels |
| `riscv_elf` in the loadable | the cross-compiled image | the Renode ISS | the deployable artifact |
| **recorded descriptor trace** | the descriptors a run actually issued | nothing — it is evidence | a **regression fixture** and the **determinism check** of §6.4 |

The recorded trace earns its place for a reason the static stream never had: it is how
determinism is asserted. Run the same loadable twice and the traces must be byte-identical; run
the host build and the RV build and they must agree. That is the `Cursor::executes()` distinction
from #282 increment 5 in a new place — executing and replaying are different things, and the
artifact that records a run is not the program that produced it.

One consequence to accept: the orchestrator source must compile **freestanding** for the RV
target. That constrains it to a subset — no exceptions, no RTTI, an arena rather than a general
heap — and that constraint applies to the host build too, or the two targets stop being the same
program. It also tilts §10 Q2: an allocator written twice in C++ argues for RV64GC and ordinary
tooling over RV32IMAC bare-metal.

### 7.2 The bridge: do not reimplement the KPU in C#

| option | verdict |
|---|---|
| reimplement the KPU as a C# Renode peripheral | **rejected.** It would be a fifth execution engine, in a fifth language, while the values, the credit model and all four levels live in C++. This repo's whole recent history is removing parallel paths ("must connect, not add a fourth path"); adding one here would undo it. |
| Renode peripheral → C++ `VirtualPlatform` over a narrow IPC boundary | **recommended.** Shared memory for the DRAM region, a socket for descriptors and completions. The C# side stays a thin MMIO shim with no model in it. |
| Renode's Verilator co-simulation channel | not used directly, but **imitated**: Renode already documents a socket protocol for Verilated peripherals, and following its shape costs less than inventing one and inherits its time-synchronisation thinking. |

### 7.3 Time reconciliation, and the one place it could silently lie

A `LAUNCH` completion carries a cycle count; Renode advances virtual time at a declared clock
ratio. That works for L-T1, L-T2 and L-CA, which model time.

**L-B models no time at all.** Letting a timing-free level advance a virtual clock would invent a
number and then print it next to real ones. So at L-B a KPU call costs a **declared constant**
(possibly zero) and the provenance records *"timing not modelled (L-B); orchestration time
only"*. This is the same discipline as `RunOutcome::has_timing` and `unmodelled_fields`: a
report must not quietly become a measurement. Getting this wrong would produce end-to-end model
latencies that look authoritative and mean nothing.

## 8. Versioning: reuse #265's mechanisms, add the capability axis

The loadable is an ABI, so it gets the full stamp, and #265 already built and *tested* every
mechanism it needs:

- magic + container version; a **major** bump means an old reader refuses;
- `min_consumer` **derived from the records present**, not hand-maintained — the rule increment 2
  established and increment 4 applied;
- `producer` = the build that wrote it, so R4's `bad_producers` has something to match;
- refusal **causes**, so a caller can tell "not for me" from "broken";
- a **golden corpus executed in CI**, because a version policy that is not tested is one that
  will be wrong — and `kernels/bin/*.kpubin` is the in-repo proof;
- `*.kpuld -text` in `.gitattributes` from the first commit. The L0 corpus cost #299 several CI
  rounds to learn that a checked-out CRLF conversion breaks byte comparison. A binary format
  makes it worse, not better.
- **new here:** the R6 **capability profile** — required compute-tile kinds, dtypes, and
  minimum resource counts. Checked against the `DeploymentSpec` at load, refused on mismatch.

**Version gates are checked before the orchestration image runs.** A RISC-V ELF from a producer
this build does not understand must be refused, not executed — the one place where "load then
find out" is unrecoverable.

## 9. Increments

Ordered so that **semantics land before toolchain**. A RISC-V image that sequences wrongly is far
harder to debug than a C++ reference that sequences rightly, and the reference is what the image
is then tested against.

1. **The container.** FlatBuffers schema, writer, verifying reader with refusal causes, and a
   golden corpus. No RISC-V, no Renode. Producer: a tool that takes derived L0 programs plus a
   tensor file and emits a `.kpuld`. **Done when:** byte-stable round-trip; an external tensor
   too large to inline is referenced and read through the DMA; malformed, too-new and
   capability-mismatched fixtures are each refused with the right cause.
2. **Orchestration semantics, with a deciding orchestrator.** The §6.2 descriptor vocabulary,
   the §6.4 status surface, and the orchestrator itself — written once, built here for the host
   and run against `VirtualPlatform` (#282). Allocation follows **program-order acquisition**
   (§6.5), the cheapest rule for which the existing deadlock proof applies verbatim.
   **Done when:** a two-operator model (GEMM → bias+activation epilogue) runs from a loadable
   and agrees **bit-exactly** with the in-process path at L-B and L-T1; **statefulness is
   proved** — the second operator consumes a tile the first left resident and no second `PLACE`
   is issued for it; the **recorded trace is byte-identical across two runs** (§6.4
   determinism); and a capacity too small to satisfy a reservation produces a **refusal with a
   diagnosis, never a hang** (§6.5).
3. **The call ABI as MMIO, and reserve-then-launch.** Descriptor ring, doorbell, completion
   ring, status surface. The same orchestrator drives the same model through MMIO instead of
   direct calls, and allocation moves from program-order acquisition to **reserve-then-launch**
   (§6.5), which is what buys out-of-order placement. **Done when:** identical values and
   identical residency behaviour through MMIO; a test asserts **no descriptor and no status read
   carries payload** (§3, §6.4); and the freedom is demonstrated — a placement order the
   program-order rule would have forbidden completes, at a capacity where it is sound.
4. **RISC-V under Renode.** The *same orchestrator source* cross-compiled, reading its own
   FlatBuffers tables in place, run on the ISS, with the KPU peripheral bridging to the C++
   platform. **Done when:** the model produces the same values as increment 2's host build
   **and the recorded descriptor traces are identical** — a differential test between two
   compilations of one program, not a smoke test — and the L-B time policy of §7.3 is visible
   in the provenance. **This is the expensive increment**: new toolchain, new language surface,
   new IPC. Note it is now *less* risky than in the first draft, because the semantics were
   settled in increment 2 by the same source rather than by a different artifact.
5. **Heterogeneous compute tiles.** Fixed-ISA (VIO, FFT) and programmable DFP tiles; the
   capability check of §6.3. **Done when:** a loadable requiring an FFT tile is refused on a
   deployment without one, and a programmable tile computes from a `CONFIGURE`d domain-flow
   program as data is pushed in.
6. **Scale.** ONNX in (#229 [A]) → a loadable with external weights, mapped not loaded.
   **Done when:** a model whose tensors exceed host RAM runs, and peak host memory tracks the
   working set rather than the model.

Increments 1–3 are the ones that make the encapsulation real; 4 is the one that makes it a
virtual platform; 5–6 are what make it interesting.

## 10. Open questions — decisions I should not make alone

1. **Container format.** FlatBuffers is the recommendation (§5). ONNX-with-external-data is the
   alternative worth arguing for if interoperability with other runtimes matters more than a
   small consumer.
2. **Orchestrator ISA and environment.** RV32IMAC bare-metal is smallest and most TFLM-like;
   RV64GC makes ordinary C++ and 64-bit addressing easy. §4's 64-bit **descriptor** fields are
   required either way — this is about the core, not the ABI. **Q4's answer tilts this**: a
   runtime allocator is real, non-trivial code that must be maintained and cross-compiled, so
   ordinary C++ tooling is worth more than a small core.
3. **Renode bridge.** IPC to the C++ platform (recommended), versus a C# reimplementation
   (rejected here), versus something closer to Renode's Verilator channel.
4. ~~How much decides at runtime.~~ **Answered (2026-09-25): the orchestration program makes
   real placement decisions at runtime.** The consequences are folded in: §6.4 (a status surface
   of metadata, never payload; an ABI that can refuse; determinism as a requirement), §6.5
   (deadlock-freedom becomes the orchestrator's problem, against a *measured* invariant), and
   §7.1 (the static descriptor stream is retired as an orchestration kind and becomes a recorded
   trace; one orchestrator source is compiled for two targets). **§6.5's allocation rule is also
   settled**: program-order acquisition in increment 2, reserve-then-launch from increment 3.
5. **Name and extension.** `.kpuld` ("loadable", after NVDLA's term, which this repo already
   cites as its reference architecture).
6. **Does this become ADR 0003?** It moves a boundary #229 named and amends a recommendation in
   the versioning plan. Both were recorded decisions, so amending them by plan alone is thinner
   than they deserve.

## 11. Deferred, with the reason rather than the label

- **Multi-device orchestration.** #282's naming map spans devices; its *execution* refuses more
  than one, deliberately, because `device_view()` selects device 0 and running it while ignoring
  the rest would be a deployment described and a machine run that are not the same machine.
  The loadable should carry a device axis it does not yet use, for the same reason the naming map
  did: retrofitting one is a format change.
- **L-T2 and L-CA under Renode.** Reachable, but the time reconciliation of §7.3 gets harder as
  the level gets finer. Start at L-B and L-T1, where the answer is known.
- **Preemption and multi-tenancy.** Nothing in the call ABI forbids them later; nothing here
  needs them.
- **The backdoor (#284) stays the only unphysical path.** §3 is what keeps this plan from
  quietly adding a second one.

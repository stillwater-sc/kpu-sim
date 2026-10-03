# #305 Increment 3: the call ABI as MMIO, and reserve-then-launch

**Date:** 2026-10-03
**Status:** Implemented (2026-10-03); Q1 and Q2 taken at their recommended defaults
**Issue:** #305, increment 3. Parent plan: `docs/plans/program-encapsulation-and-orchestration.md`
(§6.2 vocabulary, §6.4 status surface, §6.5 allocation, §7.1 one orchestrator / two targets)
**Builds on:** increment 2 (PR #308): `include/sw/kpu/orchestration/{descriptor,status,orchestrator}.hpp`,
`src/program/orchestrator.cpp`, `TileExecutionRequest::{initially_resident, retained_by_caller,
foreign_held_slots}`

**Done when** (from the parent plan):

1. identical values **and** identical residency behaviour through MMIO as through direct calls;
2. a test asserts **no descriptor and no status read carries payload**;
3. the freedom is demonstrated: a placement order the program-order rule would have forbidden
   **completes**, at a capacity where it is sound.

---

## 1. Problem Statement

Increment 2 produced a deciding orchestrator, but it does its deciding from the **inside of the
machine**. Four facts about `orchestrate()` in `src/program/orchestrator.cpp` show it:

| what increment 2 does | why that cannot survive increment 4 |
|---|---|
| takes `TensorStore&` and calls `load_into` / `store_from` itself | that is the DMA's job. The orchestrator is *handed* the data plane and promises in a comment not to read it. Under Renode the orchestrator is a RISC-V guest, so a comment is not enough. Only an address map can make it true. |
| calls `platform.run(handle, …, seeded, retained, foreign)` directly | there is no ABI between the decider and the machine, only a C++ call with seven arguments. The RV build cannot make that call. |
| keeps the credit ledger (`resident`, `resident_refs`) as its **own** bookkeeping, and builds a `StatusView` from it | the machine never sees the orchestrator's ledger. If the two disagree, nothing detects it. In increment 2 the executor enforces capacity and the orchestrator re-derives it, so there are two spellings of one fact. |
| parses each operator's L0 text (`serialize::from_string`) to find which tiles it reads | a freestanding RV64 guest should not carry an L0 text parser just to learn which tiles an operator reads |

Increment 2's allocation rule also limits what it can decide. **Program-order acquisition**
completes operator *k* before acquiring anything for *k+1*. That is why the existing deadlock
proof applies verbatim, and it is also why the orchestrator has no freedom at all: it cannot
place for *k+1* while *k* is still unfired. Increment 3 buys that freedom deliberately.

There is also a **soundness gap in the capacity check** that increment 3 must close, not
inherit. Increment 2 refuses when `resident + |to_place| > capacity`. That is a necessary
condition, not a sufficient one. The executor's in-run need is bounded by
`peak_live_tiles(prog)` (`characterize/characterization.hpp:64`), which includes output tiles
and the in-run live set. A run can therefore pass the orchestrator's check and then refuse
*inside* the executor. Today that inner refusal surfaces as an exception, reported as
`RefusedUnsupported`. It is not a hang, but it is the wrong cause, and it happens at the wrong
layer. A reservation must be a **sufficient** bound, or "a granted reservation completes"
cannot be stated.

## 2. Architecture

```
┌──────────────────────────────────────────┐
│  Orchestrator (the DECIDER)              │   one source; host build here, RV64GC in inc. 4
│                                          │
│  • AllocationPolicy                      │   sees: status registers, completions,
│    ProgramOrder | ReserveThenLaunch      │         operator manifests      (METADATA)
│  • decides RESERVE / PLACE / RELEASE /   │   never: TensorStore, VirtualPlatform,
│    LAUNCH, in descriptor-id order        │          tensor DRAM            (PAYLOAD)
└──────────────────┬───────────────────────┘
                   │  KpuPort  (submit · poll_completion · read_status · manifest)
          ┌────────┴──────────────┐
          ▼                       ▼
┌───────────────────┐   ┌──────────────────────────────────────────────────────┐
│   DirectPort      │   │   MmioPort                                           │
│                   │   │                                                      │
│ function calls,   │   │  encode ──► descriptor ring ─┐    ┌── completion ring │
│ structs in/out    │   │  (64 B each, ControlMemory)  │    │   (64 B each)     │
│                   │   │  write DRING_TAIL = DOORBELL │    │   read CRING_HEAD │
│ (inc. 2 shape,    │   │  read  STATUS_* registers    │    │   write CRING_TAIL│
│  the reference)   │   │                              ▼    │                   │
│                   │   │            ┌──────────────────────┴──────────┐        │
│                   │   │            │  Bus: address-range map + log   │        │
│                   │   │            │  [CTRL mem] [MMIO regs]         │        │
│                   │   │            │  [TENSOR DRAM] ── access FAULTS │        │
│                   │   │            └──────────────┬──────────────────┘        │
│                   │   │                           ▼                           │
│                   │   │                  KpuMmioDevice (decode ↔ structs)     │
└─────────┬─────────┘   └───────────────────────────┬──────────────────────────┘
          │                                         │
          └──────────────────┬──────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  KpuDevice (the MACHINE side of the ABI)                                     │
│                                                                              │
│  ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────────────┐  │
│  │ CreditLedger       │ │ ReservationTable   │ │ OperatorManifest[]       │  │
│  │ held / reserved /  │ │ granted in OPERATOR│ │ tensor tiles read/written│  │
│  │ free L3 slots      │ │ ORDER, all-or-none │ │ sufficient reservation   │  │
│  └─────────┬──────────┘ └─────────┬──────────┘ └──────────────────────────┘  │
│            └────────── LAUNCH ────┘                                          │
│                          │ seeded / retained / foreign  (operand keys,       │
│                          ▼                               derived HERE)       │
│  ┌────────────────────────────────┐    ┌──────────────────────────────┐      │
│  │ VirtualPlatform::run  (L-B,    │◄──►│ TensorStore (the data plane) │      │
│  │ L-T1 executor, credits inside) │    │ load_into / store_from       │      │
│  └────────────────────────────────┘    └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘

        CREDITS (L3 slots) are the DEVICE's: granted by RESERVE, consumed by PLACE,
        returned by RELEASE / at last reader. The orchestrator only ever READS them.
```

Both ports reach the **same** `KpuDevice`. That is what makes "identical through MMIO" a
differential test and not an assertion: one orchestrator and one machine, with two
transports between them.

## 3. Design

### 3.1 The split: the orchestrator loses the machine

```cpp
// WRONG — increment 2's shape: the decider holds the data plane and the platform
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,              // payload in reach
                                const OrchestratorOptions& opt);
//   ... tensors.load_into(prog, operand, tensor);                 // the DMA's job
//   ... platform.run(handle, level, snap, placement, nullptr, seeded, retained, foreign);
```

```cpp
// CORRECT — the decider sees a port and nothing else
class KpuPort {
public:
    virtual ~KpuPort() = default;
    virtual void submit(const Descriptor& d) = 0;              // enqueue; may ring the doorbell
    virtual bool poll_completion(Completion& out) = 0;         // false = none pending
    virtual StatusSnapshot read_status() const = 0;            // registers, never contents
    virtual std::uint32_t operator_count() const = 0;
    virtual OperatorManifest manifest(std::uint32_t op) const = 0;   // metadata, see §3.4
};

OrchestrationResult run_orchestrator(KpuPort& port, const OrchestratorOptions& opt);

// The machine side. Owns everything the orchestrator must not touch.
class KpuDevice {
public:
    KpuDevice(const loadable::Loadable& l, program::platform::VirtualPlatform& platform,
              TensorStore& tensors, ExecutionLevel level, DeviceOptions dopt = {});
    Completion execute(const Descriptor& d);                   // the one entry point
    StatusSnapshot status() const;
    const OperatorManifest& manifest(std::uint32_t op) const;
    const DeviceStats& stats() const;                          // per-launch RunOutcome etc.
};
```

`orchestrate(l, platform, tensors, opt)` **stays**, with the same signature. It becomes a
wrapper that builds a `KpuDevice` and a `DirectPort` and calls `run_orchestrator`. The
increment 2 tests keep compiling unchanged, and they are the regression bar for step 1 of §4.

### 3.2 Descriptors become numbers

The MMIO encoding cannot carry `std::string`. Rather than give the two ports different
descriptor types, the **one** `Descriptor` type moves to indices, and its string rendering is
reconstructed from the manifest, so the trace text keeps its readable form.

```cpp
// WRONG — names on the wire: unbounded size, and a string is a place payload could hide
struct Descriptor { TileRef tile; /* tensor name as std::string */ std::string target; ... };
```

```cpp
// CORRECT — every field is an index, a count, or an id; nothing could hold an element
struct TileId { std::uint32_t tensor; std::uint32_t ti; std::uint32_t tj; };  // tensor = index
                                                                              // into l.tensors
enum class DescriptorKind : std::uint8_t {
    Place, Release, Configure, Launch, Fence,
    Reserve,     // NEW: reserve `slots` L3 credits for operator `op`, all or none (§3.3)
};

struct Descriptor {
    std::uint64_t id = 0;
    DescriptorKind kind = DescriptorKind::Place;
    TileId tile{};                 // PLACE / RELEASE
    Hop leg = Hop::DmaDramToL3;    // PLACE: ONE leg, still
    PackedResource resource{};     // PLACE: #282 naming map, packed (device, kind, index path)
    std::uint32_t op = 0;          // RESERVE / PLACE / LAUNCH / CONFIGURE: operator index
    std::uint32_t slots = 0;       // RESERVE
    std::uint32_t flags = 0;       // RELEASE: kReleaseAtLastRead (§3.5)
    Dim compute_tile = 0;
    std::uint64_t wait_for = 0;    // FENCE
    // There is deliberately NO payload field, and there never will be.
};
```

`PLACE` gains `op`. A placement is made **under a reservation**, and the device has to know
which one (§3.3). `TileRef` (with names) stays as the orchestrator's internal vocabulary and
the trace's rendering. `TileId` is what crosses the ABI.

**Refusals become structured.** A free-text `diagnosis` cannot cross a fixed-size completion
record. Both ports would also need to produce byte-identical text, which is easiest to
guarantee if neither of them writes it:

```cpp
enum class RefusalCause : std::uint8_t {
    None, InsufficientCredit, ReservationOutOfOrder, PlaceWithoutReservation,
    ReservationExceeded, Unsupported,
};
struct Completion {
    std::uint64_t descriptor_id; CompletionStatus status;
    Cycle cycles; bool timed;
    RefusalCause cause;
    std::uint64_t needed, available, capacity, blocking_op;   // the arithmetic, as numbers
    std::vector<TileId> released;                             // inline up to 2; see §3.6
    std::string diagnosis() const;                            // RENDERED from the above
};
```

The increment 2 refusal test checks that the diagnosis names the operator and mentions
"slot". That keeps passing, because the rendering uses the manifest's operator name.

### 3.3 Reserve-then-launch, and the rule that keeps it deadlock-free

A reservation is an **atomic, all-or-none** claim on L3 slots for one operator's run. The
device grants it, enforces it, and refuses (never blocks) when it cannot be met.

**The total order is operator order.** This is the one rule that replaces program-order
acquisition:

> **R1.** A `RESERVE` for operator *j* is granted only if every operator *i < j* that has not
> completed already holds a granted reservation. Otherwise it is refused with
> `ReservationOutOfOrder` and `blocking_op = i`.
>
> **R2.** A `RESERVE` is granted only if `slots ≤ free`, where `free = capacity − held −
> Σ outstanding reservations`. Otherwise it is refused with `InsufficientCredit`. It is never
> queued.
>
> **R3.** A `PLACE` must name an operator holding a granted reservation with unconsumed slots.
> Otherwise it is refused (`PlaceWithoutReservation` / `ReservationExceeded`).
>
> **R4.** A `LAUNCH` of operator *j* requires *j*'s reservation and every *i < j* complete.
> When the launch completes, the reservation's unused remainder returns to `free`.

**Placement under a reservation is unordered.** Tiles for *k+1* may be placed before tiles
for *k*, and before `LAUNCH k`. Program-order acquisition forbids that, and it is the freedom
this increment demonstrates.

**Deadlock-freedom.** This is the proof the parent plan says must be written down rather than
assumed:

1. By R1, at any instant the **earliest uncompleted operator** *e* either holds a granted
   reservation, or no reservation for any later operator exists.
2. If *e* holds one: its reservation covers a **sufficient** bound for its run (§3.4). Its
   PLACEs succeed under R3 because they draw on its own reservation and nobody else's. Its
   LAUNCH is admissible under R4, the executor completes at that capacity (the executor's own
   program-order proof, applied inside the run), and the run returns credits. *e* advances.
3. If *e* holds none: every reservation in existence belongs to an operator > *e*. That is
   impossible by R1, unless none exist. In that case `free = capacity − held`, and either
   *e*'s RESERVE fits (case 2), or `held + bound(e) > capacity`, which is a **refusal with a
   diagnosis**: the machine is too small for *e*, given what the orchestrator chose to hold.
   It is never a hang.
4. Nothing ever waits on an unbounded condition: R2 and R3 refuse rather than block. So the
   only outcomes are progress or a refusal, by induction on *e*.

Hold-and-wait is broken because no operator holds slots that an *earlier* operator still
needs. R1 makes later reservations strictly subordinate to earlier ones. Atomic grant (R2)
removes partial holds.

```cpp
// WRONG — greedy prefetch with no reservation order. This is the weaker rule's wedge again:
// k+1's tiles take the slots, k cannot place, and k+1 cannot launch before k.
for (auto& t : manifest(k + 1).reads) port.submit(place(t));   // takes the slots
for (auto& t : manifest(k).reads)     port.submit(place(t));   // no credit left -> stuck

// CORRECT — reserve k, then k+1 if it fits; place in any order under granted reservations
submit(reserve(k, bound(k)));                       // R1: k is earliest, granted or refused
if (submit(reserve(k + 1, bound(k + 1))).granted)   // R2: refused = decision point, not a stall
    for (auto& t : to_place(k + 1)) submit(place(k + 1, t));   // out of program order: allowed
for (auto& t : to_place(k)) submit(place(k, t));
submit(launch(k));
```

The WRONG shape is not just a code smell to avoid; it is kept **as a test**. `DeviceOptions::
enforce_reservations = false` is an ablation used only by the test in §5, and it must
**refuse with a diagnosis naming the later operator holding the credits, never hang**. The
diagnosis must be accurate even when the rule it exists to enforce is switched off.

### 3.4 The operator manifest: metadata the orchestrator may read

The device derives a manifest per operator at load time. It parses the L0 program, so the
orchestrator does not have to:

```cpp
struct OperatorManifest {
    std::uint32_t index;
    std::vector<TileId> reads;          // distinct TENSOR tiles read, first-appearance order
    std::vector<TileId> writes;
    std::uint32_t peak_live_tiles;      // characterize::peak_live_tiles(prog)
    // Sufficient reservation, given what the caller holds that this op reads:
    //   bound = peak_live_tiles + |retained beyond this run|
    // seeded tiles are already inside peak_live_tiles (they are live tiles of this program),
    // so they are not added again; RETAINED tiles extend lifetimes past their last reader, so
    // they can raise the in-run live set by at most their count.
};
```

**The bound is conservative by design.** It replaces increment 2's `|to_place|`, which is not
sufficient (§1). The cost is that some capacities increment 2 accepted will now refuse **at
reservation**, where before they could pass the check and refuse later inside the executor.
That is the point. Every capacity that changes in an increment 2 test is re-derived and
justified in the PR. None is silently loosened.

> **Open question for review (Q1).** Should a tighter bound come from the executor itself? It
> would be an exact minimum-feasible-L3 per operator, computed by a capacity search at load
> time. It is still a pure function of the program, so determinism holds. **Recommendation:
> not in this increment.** `peak_live_tiles` already has a proof behind it. A search is an
> optimisation with its own correctness argument, and can follow once a test shows the
> conservatism costs something real.

> **Open question for review (Q2).** The manifest is derived **device-side** (no format change).
> The alternative is a compiler-written table inside the `.kpuld`. **Recommendation:
> device-side now.** The RV guest reads it through MMIO in increment 4 just as easily, and
> moving it into the schema later is additive. Revisit if load-time cost shows up at #305
> increment 6 scale.

### 3.5 Retention and release at the ABI

Increment 2 computes `retained` before `LAUNCH` and issues `RELEASE` *after* it, "recording
a decision made before". Through an ABI, that ordering is wrong: the device needs the
decision **at launch time**, to know which credits to return at last reader. So:

```cpp
// WRONG — the decision travels after the event it governs
submit(launch(k));  ...  submit(release(t));        // device already freed or kept t

// CORRECT — RELEASE before LAUNCH, flagged: "this launch is t's last reader"
submit(release(t, kReleaseAtLastRead));             // device: do NOT retain t in launch k
submit(launch(k));
// t's RELEASE completion is posted AFTER launch k's, listing t in `released`, because that
// is when the credit actually came back. Completion order != issue order, deliberately.
```

The device's retained set for a launch is: every tile *held* (placed or seeded, not
released) that this operator reads, minus the ones flagged `kReleaseAtLastRead`. `foreign` =
held tiles this operator does not read, **plus** other operators' unconsumed reservation
slots. A prefetched *k+1* tile therefore occupies a slot during *k*'s run, which is the
honest arithmetic.

**L-T1 honesty about prefetch.** A tile `PLACE`d for *k+1* before `LAUNCH k` is **not
seeded** into *k+1*'s run. Its DMA leg is charged to the first run that reads it. At L-T1 a
`PLACE` still has no latency of its own (parent §6.2), and there is no standalone leg to
charge it to. So at this level, prefetch buys **allocation freedom, not makespan**, and the
DMA transfer count is unchanged. Saying otherwise would invent an overlap the executor never
modelled. #283 (L-T2) is where a `PLACE` becomes a timed transaction and prefetch can show a
speed-up.

### 3.6 The MMIO layout

All addresses are **64-bit** regardless of XLEN (parent §4). Register offsets are from
`KPU_MMIO_BASE`. Descriptors and completions are 64 bytes each (as implemented; §8.2), little-endian, fixed
layout, and `static_assert`ed:

| offset | register | access | meaning |
|---|---|---|---|
| 0x000 | `ID` / `ABI_VERSION` | RO | magic + ABI major.minor |
| 0x010 | `CAP_L3_TILES`, `CAP_COMPUTE_TILES`, `CAP_INVENTORY` | RO | 0 = unbounded, as `StatusView` |
| 0x020 | `INV_INDEX` / `INV_ENTRY` | WO / RO | naming-map inventory, one packed resource per index |
| 0x040 | `DRING_BASE` (64), `DRING_SIZE` | RW | descriptor ring in ControlMemory |
| 0x050 | `DRING_TAIL` | WO | **the doorbell**: writing the tail submits |
| 0x058 | `DRING_HEAD` | RO | device's consume index |
| 0x060 | `CRING_BASE` (64), `CRING_SIZE` | RW | completion ring |
| 0x070 | `CRING_HEAD` | RO | device's produce index |
| 0x078 | `CRING_TAIL` | WO | orchestrator acknowledges |
| 0x080 | `ST_CREDITS_FREE`, `ST_RESERVED`, `ST_HELD` | RO | §6.4 credits, now the **device's** numbers |
| 0x090 | `RES_QUERY` / `RES_RESULT` | WO / RO | write a packed `TileId`, read resident(1)/not(0) |
| 0x0A0 | `MAN_OP` / `MAN_*` | WO / RO | operator manifest, one field per register (counts, bound, tile list by index) |
| 0x0C0 | `IRQ_STATUS` / `IRQ_ACK` | RO / W1C | completion-pending notifier |

`Completion::released` takes at most 1 inline `TileId` per record (§8.2). A RELEASE releases
exactly one tile, so in practice that is always enough. A completion with more sets a `more`
flag and continues in the next record. That keeps the record fixed-size, and the case is
tested rather than assumed.

**Timing of the host build.** The device services the ring **synchronously on the doorbell
write**, and completions are visible before the write returns. That is deterministic and
needs no clock. Increment 4 replaces the trigger with Renode's virtual time. The ring
protocol does not change, which is why it is specified now rather than then.

```cpp
// WRONG — a status register that reflects contents, however indirectly
reg ST_LAST_TILE_CHECKSUM;     // a checksum of a buffer IS payload: it varies with the data
reg ST_TILE_NONZERO;           // likewise: one bit of payload is payload

// CORRECT — every status value is a function of placement state alone
reg ST_CREDITS_FREE;           // counts
reg RES_RESULT;                // resident or not; never what is in it
```

### 3.7 Making "no payload" a property of the run, not just the type

Increment 2's tripwire is a structured binding over `Descriptor` and `Completion`. It
survives, extended to the wire structs. But a type check cannot see a register. Increment 3
adds two **dynamic** assertions on the `Bus`:

1. **Isolation.** The tensor DRAM range is mapped on the bus as **fault**. Any orchestrator
   access to it throws `BusFault` and the run fails. A negative test performs one deliberate
   read and asserts the fault: the property is an absence, so it needs a test that tries.
2. **Non-interference.** Run the same loadable twice, with **different tensor values** of the
   same shapes, and compare the bus access logs (every address, width, direction and value
   the orchestrator read or wrote). They must be **byte-identical**. If any value the
   orchestrator observed depended on tensor contents, the logs would differ. This is the
   general form of "no status read carries payload": it covers registers that do not exist
   yet.

## 4. Implementation Steps

Each step lands green on its own. Steps 1–2 change no behaviour, and the increment 2
suite plus a captured trace digest are the bar for that.

1. **Split decider from machine; DirectPort only.**
   - New `include/sw/kpu/orchestration/kpu_device.hpp`, `src/program/kpu_device.cpp`. Move the
     data plane, the platform calls, the seeded/retained/foreign derivation and the operand
     binding out of `orchestrator.cpp`.
   - New `include/sw/kpu/orchestration/port.hpp` (`KpuPort`, `DirectPort`).
   - `orchestrator.hpp/.cpp`: `run_orchestrator(KpuPort&, opt)`. Keep the
     `orchestrate(l, platform, tensors, opt)` wrapper.
   - Before starting, capture the increment 2 `DescriptorTrace::digest()` for
     `two_gemms_sharing_weights` and the three-GEMM chain as fixtures. After this step they must
     match exactly.
2. **Indices on the wire, structured refusals.**
   - `descriptor.hpp`: `TileId`, `op`/`slots`/`flags` fields, `RefusalCause`, numeric refusal
     fields, `Completion::diagnosis()` rendered.
   - The trace rendering resolves indices through the manifest, so `canonical_bytes()` is
     unchanged for increment 2 runs. The step 1 digests still match.
   - Update the structured-binding tripwire in `tests/program/test_orchestrator.cpp` (it is
     meant to break here, and the PR justifies every new field as non-payload).
3. **Device-owned ledger and reservations (R1–R4).**
   - `CreditLedger`, `ReservationTable`, `OperatorManifest` in `kpu_device.*`. `StatusView`
     (`status.hpp`) is built from the device's ledger, not the orchestrator's.
   - `AllocationPolicy::ProgramOrder` (default) now issues `RESERVE(k, bound(k))` immediately
     before k's PLACEs. Expect capacity changes where `|to_place|` under-counted (§3.4).
     Re-derive each one and record it in the PR.
   - `DeviceOptions::enforce_reservations` (default true; false only for the ablation test).
4. **`RELEASE` before `LAUNCH` (§3.5)**, with `kReleaseAtLastRead` and out-of-order completion
   posting. Trace bytes change here, deliberately: regenerate the step 1 fixtures and justify
   the diff (the RELEASE descriptors move ahead of their LAUNCH).
5. **`AllocationPolicy::ReserveThenLaunch`**: reserve *k*, and *k+1* if R2 allows; place *k+1*
   before *k* when granted; fall back without prefetch on refusal.
6. **The ABI and MMIO transport.**
   - New `include/sw/kpu/orchestration/abi.hpp`: register offsets, `WireDescriptor` (64 B),
     `WireCompletion` (64 B, §8.2), `encode`/`decode`, `static_assert`s on size and offsets.
   - New `include/sw/kpu/orchestration/mmio.hpp`, `src/program/mmio.cpp`: `ControlMemory`, `Bus`
     (range map, fault ranges, access log), `KpuMmioDevice`, `MmioPort`.
7. **Tests** (§5): new `tests/program/test_orchestration_mmio.cpp` and
   `tests/program/test_reservation.cpp`, registered in `tests/program/CMakeLists.txt` with labels
   `program;orchestration;v0.9`.
8. **Freestanding guard for the decider.** Compile `src/program/orchestrator.cpp` a second time
   as an object library with `-fno-exceptions -fno-rtti` on GCC/Clang (skipped on MSVC). Nothing
   links it. It exists so that the subset increment 4 needs cannot drift (parent §7.1). The
   decider reports refusals as results already. This makes it a build error to start throwing.
9. **Docs.** Update the parent plan's increment 3 entry with what was found, `CHANGELOG.md`, and
   the #305 checkboxes for increments 1–3. Add a section to ADR 0003 for the ABI (register map +
   R1–R4), since the ABI is now real rather than proposed.

## 5. Verification

```bash
cmake --preset release && cmake --build --preset release
cd build && ctest -L orchestration --output-on-failure      # increment 2 + 3 suites
cd build && ctest -L program --output-on-failure            # nothing upstream regressed
cd build && ctest --output-on-failure                       # full suite (187/187 at inc. 2)
```

Expected outcomes, one test per line, each verified by reverting what it guards:

| test | asserts |
|---|---|
| **DoD 1** `mmio and direct agree` | same loadable at L-B and L-T1 through `DirectPort` and `MmioPort`: tensor outputs **bit-identical**, `dma_transfers()` equal, per-launch `peak_l3_residency` equal, `DescriptorTrace::canonical_bytes()` **byte-identical** |
| **DoD 1** `mmio agrees with in-process` | increment 2's in-process comparison repeated through MMIO (bit-exact vs the direct `run_at` path) |
| **DoD 2** `tensor DRAM faults` | a deliberate orchestrator-side read of the tensor range raises `BusFault`; revert the fault mapping → test fails |
| **DoD 2** `non-interference` | two runs, different tensor values, same shapes → bus access logs byte-identical; plant a `ST_TILE_NONZERO`-style register → test fails |
| **DoD 2** wire tripwires | `static_assert(sizeof(WireDescriptor)==64)`, `(WireCompletion)==32`, structured bindings over both |
| **DoD 3** `forbidden order completes` | `ReserveThenLaunch` at capacity ≥ bound(k)+bound(k+1)+held: trace shows a `PLACE` for *k+1* **before** `LAUNCH k`; values bit-exact with `ProgramOrder`; DMA count equal (§3.5) |
| **DoD 3** `the wedge is refused, not hung` | ablation `enforce_reservations=false` with greedy prefetch at a capacity between bound(k) and the sum: refusal naming *k+1* as `blocking_op`; the test has a hard iteration cap, so a hang fails it |
| **DoD 3** `refused reservation is a decision point` | same capacity, `ReserveThenLaunch`: `RESERVE k+1` refused, orchestrator continues without prefetch, run completes |
| R1 | `RESERVE` for *k+1* before *k* → `ReservationOutOfOrder`, `blocking_op = k` |
| sufficiency | sweep capacity from 1 to unbounded on the three-GEMM chain: **every granted reservation completes**, i.e. no executor-level refusal ever follows a grant. This is the §3.3 proof checked empirically |
| determinism | increment 2's two-run digest test, through both ports |
| completion overflow | a completion releasing > 2 tiles spans records with `more`, and decodes to the same `Completion` |

Clang must stay clean on changed TUs (`release-werror`). Both compilers must pass. The
`-fno-exceptions` object library must build.

## 6. Key Invariants

1. **No descriptor, completion, register or ring entry carries payload.** Every value the
   orchestrator can observe is a function of placement state, and the non-interference test
   is the definition.
2. **The orchestrator cannot reach tensor DRAM.** On the MMIO path the bus faults. On the
   direct path the type has no route there (`TensorStore` lives only in `KpuDevice`).
3. **Credits are the device's.** The orchestrator reads `ST_*`. It never computes residency the
   device does not also hold. One ledger, one spelling.
4. **Reservations are granted in operator order, atomically, and refused rather than queued**
   (R1, R2). Every `PLACE` is covered by a granted reservation (R3).
5. **A granted reservation completes.** The bound is sufficient: no executor-level refusal
   follows a grant.
6. **The machine never hangs.** Every descriptor completes, with `Done` or a refusal whose cause
   and arithmetic are numbers and whose rendering names the operator.
7. **One orchestrator, one machine, two transports.** `DirectPort` and `MmioPort` produce
   byte-identical descriptor traces. Any divergence is a transport bug by definition.
8. **Determinism.** The trace is a pure function of (loadable, deployment, level, policy). No
   container whose iteration order depends on allocation, and no address-derived values.
9. **Hops do not collapse; RELEASE, never evict.** Inherited from increment 2, unchanged.
10. **L-T1 does not time a PLACE.** Prefetch changes allocation, not makespan or DMA count,
    until #283.

## 7. Risks

- **The conservative bound refuses capacities increment 2 accepted.** This is intended (§1), but
  it will move numbers in existing tests. Mitigation: every change is re-derived in the PR, and
  Q1 is the path to tightening.
- **Trace churn in step 4.** RELEASE moving ahead of LAUNCH changes canonical bytes. It is
  sequenced as its own step so the diff is reviewable in isolation.
- **Scope creep into increment 4.** No IPC, no Renode, no cross-compiler here. The freestanding
  guard (step 8) is the only increment 4 concession, because it is cheap now and expensive to
  retrofit.

## 8. Implementation notes: what the code does differently, and why

Every DoD clause landed. Five places where the code departs from §3–§4 are recorded here, so
that a reader comparing the plan with the code sees each departure as a decision, not drift.

1. **`Descriptor` keeps names in-process; only the wire uses indices.** §3.2 proposed moving
   the one `Descriptor` type to `TileId` indices. Instead, `abi::NameTable` maps names to
   indices at the encoding boundary (`abi.hpp`), and both ends build it from metadata they
   already hold. The payload argument is unchanged, because the wire record is what crosses
   the ABI, and it holds no string. The trace stays readable, and increment 2's tests that
   compare tile names did not have to change. An unknown name cannot be encoded at all, and a
   test asserts that.
2. **Completion records are 64 bytes, not 32.** The cause, four 32-bit refusal numbers, a
   diagnosis locator and one inline released tile do not fit in 32. The diagnosis text is
   written by the device into a DIAG area in control memory, and the record carries its
   offset and length. The register offsets in `abi.hpp` are authoritative. Every register is
   64 bits wide, so they do not match §3.6's table exactly.
3. **R3's capacity check applies only to a prefetch.** Counting every `PLACE` against the
   reservation would refuse the ordinary case. An operator reads more distinct tiles (8 for
   this matmul) than its live set (6), and at L-T1 those tiles flow through the reserved slots
   *inside* its own run. Only tiles placed for an operator that is *not* next to launch occupy
   slots during someone else's run, and those are the ones that must fit.
4. **Two rules the plan did not anticipate.** (a) A prefetch never places a tile of a tensor
   the current operator writes, since that is a read-after-write hazard. At L-T1 it would not
   change any value, which is exactly why it must be excluded by a rule rather than by a value
   comparison. (b) Before each operator, the orchestrator reads `ST_HELD`/`ST_COMPLETED` and
   refuses if they disagree with its own decisions. This keeps one ledger with one spelling,
   and it also puts status reads on every run's bus log. Without them the non-interference test
   could not see a leaking register. A mutation planted in `ST_RESERVED` confirmed it now does.
5. **Step 8 (the `-fno-exceptions` guard) is deferred to increment 4.** `descriptor.hpp`
   includes the executor and naming-map headers, whose inline functions `throw`. GCC and clang
   reject those under `-fno-exceptions` before any orchestrator code is reached. Making the
   decider freestanding first needs `descriptor.hpp` split from those headers. That is
   increment 4's first task, not a flag to add here.

**Measured.** The reservation bound is sufficient, and on these fixtures it is tight: with the
bound reduced by one, the capacity sweep catches LAUNCH refusals inside the executor. The
minimum L3 for the three-GEMM chain is 6 slots cold and 10 warm. Increment 2 reported 8 / 12,
but its numbers came from its own `|tiles to place|` check, not from the machine. Prefetch at
L-T1 leaves the DMA count unchanged (§3.5).

**Mutation checks** (each reverted after confirming the test fails): tensor DRAM unmapped
(the isolation test, after tightening it to require the fault to name the tensor; it first
passed, because an unmapped address faults too); a tensor-dependent status register
(non-interference); reservation bound minus one (the sufficiency sweep).

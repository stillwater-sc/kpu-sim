# The virtual platform (#282)

**Status:** design note, for review before implementation
**Issue:** #282, which **blocks** #283 (L-T2), #284 (the backdoor) and increment 5 of #285
**Authorities:** ADR 0002 §3.2 (API sketch), §3.3 (resource vocabulary), §3.5 (deployment as
data), §4 (driver architecture)

## 1. Why this is on the critical path

Three issues are waiting on the same missing thing: **an object that owns a deployment.**
L-T2 needs somewhere for its resources to live, the backdoor needs every resource to be
addressable, and `kpu-run --deploy` needs something to deploy *to*. All three would otherwise
each invent their own.

Today the driver builds a `DeviceDescriptor` from flags and hands it to `run_at`, which
switches on the level. That works, and it is why increments 1–4 of #285 could land before
this. What it cannot do is hold **state**, which is what makes a run reproducible, and what
the ADR's central claim depends on:

> a run is a pure function of `(program, initial_state, deployment, level)`

That claim is false today in a specific, checkable way: `run_at` takes three of the four.

## 2. The problem to avoid: a fifth description of a device

The repo already describes a device in four places:

| Description | Who uses it | Carries |
|---|---|---|
| `driver::DeviceSpec` | CLI flags | topology, counts, bytes/cycle, MACs/cycle, L3 tiles |
| `characterize::DeviceDescriptor` | what the executors schedule on | the same, plus analytical coefficients |
| `Placement` | which compute tiles an op runs on | pinning |
| ADR §3.5 `DeploymentSpec` (JSON) | nothing yet | **all of the above, plus L3 banks, L2 banks per tile, L1 vectors, DMA burst** |

Adding `DeploymentSpec` as a peer of the other three is the obvious mistake, and it is the
one this repo has made before under a different name — four disconnected execution engines,
each with its own notion of the machine.

**Recommendation: `DeploymentSpec` becomes *the* description, and the others become views of
it.**

- `deployment.device_view()` projects a `DeviceDescriptor` for the levels that need only
  counts and bandwidths (L-B, L-T1).
- `driver::DeviceSpec` stops building a `DeviceDescriptor` and starts building a
  `DeploymentSpec`. The CLI flags do not change.
- `Placement` stays a separate argument, because it is a property of a *run*, not of the
  machine — the same deployment runs a pinned and an unpinned placement.

The spec is a superset, so the projection loses fields, and **a level must say which ones it
ignored rather than ignoring them quietly.** A spec declaring `"l3": {"banks": 8}` run at
L-T1 must report *"L3 banks declared (8) but not modelled at this level"* in the provenance.
This is the same discipline as `not_implemented_reason`: the tool already refuses to let a
clean report be mistaken for full coverage, and a deployment field that silently evaporates
is exactly that mistake one layer down.

## 3. What `initial_state` can honestly mean today

This is the part most likely to be built wrong, because the ADR describes the end state and
the end state does not exist yet.

**The only mutable state a run touches today is the `TileProgram`'s operand values.** L-T1
tracks tile *identity*, credits and residency — never payload — and L-B computes straight
into the operands. There is no L3 tile with contents, no L2 bank, no L1 vector: §3.3's
resource vocabulary is #283's work.

So a `StateSnapshot` today is operand values and nothing else. That is genuinely complete for
the state that exists, and it is genuinely *incomplete* against §3.5. Both must be true in the
code, or the next person reads a snapshot digest and believes something false.

**Therefore the snapshot carries a coverage tag, and the digest covers the tag.** A v1
snapshot says "operands only"; an L-T2 snapshot will say "operands + resource residency".
Without the tag in the digest, two runs whose snapshots cover *different state classes* would
produce the same digest whenever their operands matched — so a cache would serve an
operands-only result for a run that also staged L3 contents. The tag makes that collision
impossible instead of unlikely.

## 4. Run identity: a digest for labelling, bytes for the claim

```cpp
struct RunIdentity {
    std::string program_digest;      // over write_l0 bytes -- byte-stable, already asserted
    std::string snapshot_digest;     // over the coverage tag + the state it covers
    std::string deployment_digest;   // over the canonical serialized spec
    ExecutionLevel level;
};
```

The program digest works because increment 3 of #265 made the L0 bytes stable and *tested*
that they are — which is why that check was worth its strictness. The deployment digest needs
the same property, so **spec round-trip byte-stability is an increment-1 test**, not an
afterthought.

**The four-tuple is really a six-tuple, and the ADR does not say so.** `run()` also takes a
`Placement` and an optional L1 stream annotation, and both change what happens: the placement
decides which compute tile an op lands on, the annotation decides per-op timing. An identity
over four of six inputs calls two different runs the same run, which review caught. Both are
recorded — the placement as its whole assignment rather than its label, since two different
pinned placements over the same compute tiles share a label; the annotation as its space-time
map's **name**, because a `StreamProgram` is a pure function of `(program, map)` and both are
already in the identity (the reasoning #265 increment 4 settled on for the file format).

This is worth flagging upward: §3.5's "pure function of `(program, initial_state, deployment,
level)`" is the shape of the claim, not its arity. `Placement` is deliberately not part of the
deployment (§2 above), so it is a fifth input by construction.

**The digest is for provenance and cache lookup. It is not the identity claim.** The issue's
definition of done says *"two runs with identical `(program, initial_state, deployment,
level)` produce identical results — asserted, not assumed"*, and asserting that through a
64-bit hash would be asserting the absence of a collision. The test compares the **canonical
bytes** of all four inputs and then compares the results. The digest appears in the
provenance beside them, where a collision is a cosmetic annoyance rather than a wrong answer.

## 5. The naming map: identity now, state binding with L-T2

```cpp
struct ResourceName {                 // "dev0/l3[3]/bank[2]+64"
    Dim device; ResourceKind kind; Dim instance; std::uint64_t offset;
};
```
with `ResourceKind ∈ {Dram, L3Tile, L3Bank, L2Bank, L1Vector, ComputeTile, RegisterFile}`.

The map is **derived from the deployment**, so it names exactly what the deployment declares
and nothing else — a name for an L2 bank that the spec does not declare must not resolve.
Multi-device from the start, as the issue requires; a single-device deployment is a map with
one device in it, which costs one index and removes a migration later.

**Two questions, and only one is answerable now.** *"Does this resource exist in this
deployment?"* needs the spec, which exists. *"What is in it?"* needs resources that hold
state, which is #283. So the map resolves **identity** in this issue and gains **state
binding** in #283. That split is not a hedge: #284 (the backdoor) and #286 (the spatial event
record) both need to *name* and *enumerate* resources before anything can read them, and
holding the map back until state exists would block both for no gain.

## 6. "No engine constructed by hand" needs a boundary

The issue says *no engine is constructed by hand in any test or demo*. Taken literally that
would forbid `tests/timing/test_credit_pool.cpp` from constructing a `CreditPool`, which is
what a unit test of a credit pool is.

The intent is about **deployments**, not about objects. The boundary this note proposes:

- **Goes through the platform:** anything that assembles a *set* of engines and calls it a
  machine — `tests/program/`, `examples/`, `tools/`, and the characterization harness.
- **Keeps constructing directly:** a unit test of the single process under test. A test that
  drove `CreditPool` through a whole platform would be testing the platform.

Stated here so the unfinished half is a documented boundary rather than a quiet omission.

## 7. What the platform takes: a handle, not a program type

ADR §3.2 sketches `load_program(const CspProgram&)`. `CspProgram` is #281 and does not exist;
the program that exists is the L0 `TileProgram`. The platform therefore takes a `TileProgram`
now and returns a `ProgramHandle`, and **the handle is the point**: when #281 lands, the
program type behind the handle changes without touching a caller. Sketching the future type
in the signature today would only produce a name with nothing behind it.

For the same reason `IInterpreter` (§4) is **not** introduced here. `run_at` is already the
level→interpreter seam and it is tested; turning it into a virtual interface before the L-T2
implementation exists would be designing an interface from one example. #283 is where a second
real implementation appears, and that is when the shape of the interface is knowable.

## 8. Increments

1. **`DeploymentSpec`, JSON, and one description of a device** — **done.** `from_json` /
   `to_json` with a **byte-stable canonical form**, `device_view()` projecting a
   `DeviceDescriptor`, `DeviceSpec` building a spec instead of a descriptor, and
   `unmodelled_fields()` naming every declared field the chosen level does not model. JSON
   lives in `src/program/deployment_json.cpp` behind a declaration-only header, following
   `tools/dfg/common/dfg_json.hpp`: the `program/` tree is header-only and std-only, and
   pulling `nlohmann/json.hpp` into it would hand that dependency to every consumer of a
   `TileProgram`. The one new library is linked PUBLIC into `kpu_simulator`, so no test or
   tool changed its link list.

   **The round-trip claim is precise rather than convenient.** *Canonical* bytes round-trip
   byte-exactly; a non-canonical spec normalizes on the first pass and is stable after it.
   Claiming the stronger thing would be false the moment someone writes `64` where a double
   belongs — and the ADR's own example does exactly that. Both halves are tested, and the
   byte check is verified to bite: tampering with one field, or converting the fixture to
   CRLF, each fails exactly one assertion. The CRLF case is why
   `tests/program/deploy/*.json` is `-text` in `.gitattributes` before CI could find it the
   hard way, as it did on #299.

   **The ADR's flat spelling loads.** Someone will paste §3.5's example, and a format that
   refused the spelling its own authority document shows would make the documentation a
   trap. Its `"burst"` key is read as `burst_bytes` for the same reason; giving both is
   refused, because two spellings of one field silently resolved is a machine that differs
   from the file whenever they disagree. **An unknown key is refused** and the refusal lists
   the keys that would work: a silently ignored `"compute_tile"` means the machine is not
   the one the file describes while the run reports success.

   **`l3.tiles` and `l3.capacity_tiles` are kept apart**, because `DeviceDescriptor::l3_tiles`
   is the *capacity* despite its name. Wiring the module count into it would bound the credit
   model by the wrong number and produce plausible timing for a machine nobody described.

   A **refactor guard** pins every field of `make_device()` across all three topologies.
   Every timing number in the repo depends on them, so a field shifted by the indirection
   would move calibration and makespans everywhere while every other test still passed.

   **Review found a hole in the validation, and the interesting half was not the one
   reported.** `!(x > 0.0)` accepts `+inf`, and `std::stod` parses `"inf"`, so
   `--dma-bytes-per-cycle inf` passed — an infinite bandwidth being a makespan of 0 or a NaN
   reported as a result. But nlohmann writes a non-finite double as JSON `null`, so a spec
   `validate()` **accepted** could serialize to bytes `from_json()` **refused**: the
   round-trip invariant the digest depends on was violated by values the validator itself
   admitted, and a cache key would have been taken over bytes nobody can load. A validator
   that admits values the format cannot represent is not a validator. Every double is
   finite-checked now, `analytical.pj_per_mac` / `pj_per_byte` are validated at all (finite
   and non-negative — zero energy is a legitimate modelling choice, a negative one is not),
   the flag reports it under its own name, and the property is asserted directly rather than
   implied: **anything `validate()` accepts can be written and read back.**
2. **`VirtualPlatform` with state** — **done.** `load_program`, `snapshot`, `restore`,
   `run(handle, level, const StateSnapshot&)` restoring **first**, the coverage-tagged digest
   of §3, and a `RunIdentity` naming all four inputs.

   **Tile LU is what makes the reproducibility test mean anything.** It factors `A` in place,
   so running it on its own output gives a different answer — which is what distinguishes
   "the restore worked" from "the program happens to be idempotent". A matmul that zeroes and
   re-accumulates `C` would pass either way, so on its own it proves nothing. The test asserts
   both halves: same snapshot twice gives the same answer, *and* the previous output gives a
   different one.

   **The restore is platform-wide, and that has a usage consequence worth writing down.** A
   run resets *every* loaded program, not only the one it executes — it has to, or the
   snapshot's digest would claim state the run did not restore and the identity would be a
   promise the platform does not keep. So a caller comparing several runs must capture each
   result as it is produced; reading `platform.program(h)` after a loop reads the input the
   last restore put back. The differential test got exactly this wrong first, and its own
   assertion caught it — which is the cheapest place to learn it.

   **The program digest covers structure, the snapshot covers values.** Folding values into
   the program digest would make two of the four inputs cover the same bytes, and "identical
   inputs" would stop meaning four independent things.

   **The coverage guard is at compile time.** A `StateSnapshot` is never deserialized — it
   only ever comes from this platform in this process — so an unknown coverage cannot arrive
   at runtime, and a check for one would be unreachable code pretending to be a safeguard.
   `restore()` switches exhaustively instead, so #283 breaks the build at the site that has to
   learn to restore the new coverage.

   A `ProgramHandle` is strongly typed and not default-usable: a bare `std::size_t` would let
   an uninitialised value index a program, and the failure would be a run of the **wrong
   program** reporting success.
3. **The naming map** — **done.** `ResourceName`, `format`/`parse_resource_name`,
   `ResourceMap{exists, index_of, enumerate, why_not, require}`, identity only.

   **A name carries a PATH, not an instance number.** L2 banks and L1 vectors are per
   *compute tile* and L3 banks are per L3 module, so an address is `dev/cf[2]/l2[3]`, not a
   flattened `(kind, instance)` pair. Flattening would have to fold two indices into one and
   lose the structure — the same class of error as conflating `l3.tiles` with
   `l3.capacity_tiles`, where the wrong number looks entirely plausible. The issue's own
   definition of done forces this: an L2 bank cannot be addressed without its compute tile.

   **Devices are addressed by name, not by index**, since a positional address would move
   silently when a deployment is reordered. That made the device name part of the grammar, so
   `validate()` now refuses a name containing `/[]+` — a device nothing can address is a
   device the backdoor cannot reach, and learning that at the first backdoor write is far
   worse than learning it at deployment.

   **The map's domain is exactly what the deployment declares.** An undeclared `l3.banks`
   means this machine's bank structure is unspecified, so `dev0/l3[0]/bank[0]` names nothing
   and does not resolve — and declaring one level does not imply the next: L3 modules can
   exist while banks remain unaddressable. `why_not()` distinguishes **undeclared** from
   **out of range**, because those are different problems with different fixes and collapsing
   them sends the reader to the wrong one.

   Two resources are declared by *inference* rather than by a field, and the inference is
   stated where it is made: a device's **DRAM**, because `validate()` requires at least one
   DMA engine and a DMA moves DRAM↔L3, so engines with no DRAM side would have nothing to
   read; and a compute tile's **register file**, because the fabric cannot hold an operand
   without one. Neither has a count to declare.

   **The offset is carried but never bounded**, and that is said rather than implied: a spec
   declares no sizes — no bytes per L3 tile, no L2 bank width — so nothing here can check an
   offset. Sizes are an additive field (R8) that #283 needs anyway to give a resource
   contents. The offset is also **not part of identity**: two writes at different offsets are
   two writes to the same resource, and counting them as two stations would be wrong for
   #286.
4. **`kpu-run --deploy spec.json`**, and the characterization harness becomes a consumer of
   the platform rather than a parallel path. The flags stay; `--deploy` replaces them, and
   giving both is refused for the same reason `--program` and `--algo` are.
5. **`step_begin` / `step` on the platform**, reusing the existing `Stepper`, so #286 and
   L-T2 have one stepping seam rather than two.

Increments 1–2 are what unblock #283. Increment 3 is what unblocks #284 and #286.

## 9. Definition of done, and what is deferred

From the issue, with the honest status of each:

- [x] a deployment spec round-trips — **increment 1**, as canonical byte-exactness plus
      idempotent normalization of a non-canonical spec (§8)
- [x] `run()` restores the passed snapshot first, and its digest is in the cache key and the
      provenance — **increment 2**
- [x] two runs with identical inputs produce identical results, asserted — **increment 2**,
      by comparing bytes rather than digests (§4)
- [x] a run that reads state a previous run left behind is impossible by construction —
      **increment 2**
- [ ] no test or demo constructs an engine directly; the characterization harness goes
      through the platform — **increment 4**, within the boundary of §6
- [x] the naming map resolves an L3 tile, an L2 bank, an L1 vector and a compute-tile
      register file on a two-device deployment — **increment 3**, as *identity*; resolving to
      **state** is #283, because the state does not exist yet (§5)

**Deferred, with the reason rather than the label:**

- **The backdoor (#284).** Out of scope here, but §3's snapshot is the surface it writes
  through, so the coverage tag exists partly for it: a backdoor-staged tensor must be
  distinguishable in the provenance from one that was moved.
- **Resource-state snapshots and state resolution of names (#283).** Both need resources that
  hold state. Building either now would mean inventing the state model in the wrong issue.
- **`IInterpreter` as a virtual interface (§4).** One real implementation is not enough to
  design an interface from (§7).

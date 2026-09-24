# Serializing the L0 program (#265)

**Status:** design note, for review before implementation
**Issue:** #265, which unblocks increment 4 of #285 (`kpu-run --program foo.l0`)
**Authorities:** ADR 0001 D1 (L0 is the portable program), `docs/plans/dfg-kpu-versioning.md`
(R1–R9)

## 1. Why this is on the critical path

`kpu-run` can only execute programs it can *derive* — matmul and LU, from `--size`/`--tile`.
Increment 4 of the driver is "a program from a file", and there is no file: nothing
serializes a `TileProgram`. Until there is, "load a program and execute it" is not literally
true, and every program the simulator runs is one the simulator wrote.

## 2. A decision that should not be made silently

`docs/plans/dfg-kpu-versioning.md` §4 recommends **"one binary format, one source format —
do not add a third path"**. Taken literally, an L0 program file is a third path.

It predates ADR 0001. That ADR made the **L0 `TileProgram` the portable program** and
demoted `.kpubin`/DMProgram to *driver-JIT output for one device*. So the artifact the
versioning plan calls "the ABI" is no longer the portable one, and the layer that needs a
durable format is L0.

**Recommendation: serialize L0, and say plainly in the versioning plan that its §4
recommendation is superseded by ADR 0001 D1 for the portable layer.** The alternative —
treating `.kpubin` as the portable format — was considered and rejected by the ADR, because
it bakes in a device and a placement. Reviving it here would reverse an accepted decision by
implication, which is worse than amending a plan on purpose.

The "no third path" instinct still applies to *quantity*: this note adds **one** format, and
`.dfg` stays compiler-internal.

## 3. What the format must carry

Everything that makes a run reproducible, and nothing that makes it device-specific:

| Carried | Not carried |
|---|---|
| operand registry: name, logical shape, tile shape (so ragged trailing tiles are recoverable) | any `DeviceDescriptor` |
| op list: kind, declared tile inputs/outputs, `alpha`, port kind + port, pivot slot, label | any `Placement` |
| optionally the L1 `StreamProgram` annotations (ADR §7.4, optional at the transactional level) | any timing, cycle count or calibration |
| operand **values**, optionally | any execution result |

Values are optional and that is a real decision: a program with values is a *test case*, a
program without them is a *kernel*. Both are wanted, and a reader must be able to tell which
it has rather than inferring it from zeros.

## 4. Versioning — the subset of R1–R9 that applies

Applied, with the reason each one is not ceremony:

- **R1/R3 — three axes, semver.** Format version, op-set version, producer. The op set is
  the `TileOpKind` surface; it versions separately from the container because adding an op
  is not the same kind of change as adding a field.
- **R4 — `min_consumer` gate.** The file declares the minimum reader it needs, and an older
  reader **refuses cleanly**. This is the requirement with teeth: `kernels/bin/*.kpubin`
  renumbered opcodes with no version bump and those files now abort with `std::get: wrong
  index for variant`. A crash is not a diagnostic.
- **R5 — add-only, freeze on release.** Field and op numbering never reused. Enforced by the
  golden corpus, not by intention.
- **R8 — unknown-element policy, stated per element.** Unknown **op**: refuse, because
  executing a program with an op you do not understand computes the wrong answer silently.
  Unknown **optional attribute**: accept and ignore, so a minor producer bump stays readable.
- **R9 — golden corpus in CI.** Files checked in and executed, so the format cannot rot the
  way `.kpubin` did.

Deliberately **not** applied: R2 (op domains), R6 (capability profiles) and R7 (type-system
pinning) belong to `.dfg`'s tensor-type grammar and a vendor-op ecosystem. L0 has one
producer, a closed op enum and no type strings. Adopting them here would be cargo-culting a
requirement without its problem.

## 5. Text or binary

**Text, line-oriented, one record per line** — the same shape as `disassemble()`.

Binary is the wrong default for this artifact. L0 files are small (an op list, not weights),
and the value of being greppable, diffable and reviewable in a PR outweighs parse speed for
a format whose consumer then runs a simulation lasting orders of magnitude longer. A golden
corpus you can read is a corpus whose rot is visible in a diff.

Values are the exception: a large operand belongs in a side file or an explicitly encoded
block, not in decimal text. That choice is deferred until a program with real weights exists
to measure, and the format reserves a record for it rather than guessing now.

## 6. Increments

1. **Round-trip without values.** Write and read the registry and op list; assert the
   reloaded program is structurally identical and **executes bit-identically** at L-B and
   L-T1. Version preamble with `min_consumer` and a clean refusal on an unsupported version.
2. **Values, optionally**, with the reader able to say whether it got a kernel or a test case.
3. **Golden corpus in CI**: matmul and tile LU, checked in, loaded and executed.
4. **Stream annotations** alongside (ADR §7.4).
5. **`kpu-run --program`** — increment 4 of #285, which is the point of all of this.

## 7. Definition of done for the issue

- [ ] `TileProgram` → file → `TileProgram` round-trips, and the reloaded program executes
      **bit-identically** to the in-memory one on matmul and tile LU
- [ ] An unsupported version, a corrupt preamble and an unknown op each produce a readable
      diagnostic — never a variant crash, never a silent mis-parse
- [ ] The golden corpus loads and executes in CI
- [ ] A derived GEMM, written to a file, loads and runs through `run_at` at L-T1 with the
      same values as the in-memory program

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
- **R4 — `min_consumer` gate, computed from the file's CONTENT.** The file declares the
  minimum reader it needs, and an older reader **refuses cleanly**. The demand depends on
  what is in the file, not on who wrote it: a **test case** requires 1.1.0 because a 1.0.0
  reader does not know `VALUES_ROW` and would skip every value record, accept the file, and
  execute with zero-initialised inputs — a confident wrong answer rather than a refusal. A
  **kernel** stays readable by 1.0.0, because nothing was added to it, and a blanket bump
  would needlessly orphan files that are still perfectly readable.

  **The bump rule for future changes:** a new record carrying *semantics* must raise
  `min_consumer` for files that use it; a new optional *attribute* need not, because
  ignoring it is harmless by construction (R8). A new container record is also **not** a new
  operator — the op-set axis stays where it is, which is why the axes are separate. This is the requirement with teeth: `kernels/bin/*.kpubin`
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

**Measured, before building increment 2 on an assumption** (2026-09-24):

| encoding | finite values | non-finite |
|---|---|---|
| `std::hexfloat` | **does not round-trip through iostreams** — inexact even with the manipulator set on input, because libstdc++ `operator>>` does not parse hex floats | fails to parse |
| decimal at `max_digits10` | **exact for every finite value tried**: `1.0000001f`, `-0.0f`, denormal min, `FLT_MIN`, `FLT_MAX`, π, `1e-7` | fails to parse |

So the encoding is **decimal at `max_digits10` through a classic-locale stream** — which is
also the readable choice, and the same rule already applied to `alpha`. Hexfloat looked like
the obvious answer for exactness and is simply wrong here.

**Non-finite values need explicit tokens.** `inf`, `-inf` and `nan` fail to parse via
`operator>>` in both encodings, so the writer emits them as literal tokens and the reader
recognises them rather than relying on the stream. They are not hypothetical: a masked
attention value is `-inf`, and this repo does softmax and attention work.

## 6. Increments

1. **Round-trip without values.** — **done.** `serialize/l0_format.hpp` writes and reads the
   operand registry and op list; the reloaded program is asserted structurally identical
   **and to execute bit-identically at both L-B and L-T1**, with matching makespan and (for
   LU) matching swap count and permutation. Structural equality alone would not be enough —
   a silently dropped field can leave two programs that look alike and compute differently.

   Op fields are **keyed** (`kind=`, `in=`, `out=`) rather than positional, which is what
   makes the add-only rule cheap: a new optional attribute is a new key that old readers
   ignore, and no existing field moves.

   Refusals carry a `FormatError::Cause`, so a caller can tell "not for me" from "broken"
   without parsing a message: `NotAnL0File`, `MalformedPreamble`, `UnsupportedVersion`,
   `UnknownOp`, `MalformedRecord`, `Truncated`. A missing `MIN_CONSUMER` is refused rather
   than assumed, and a file ending before `END` is refused because a partial program would
   execute a partial answer.

   `PRODUCER` carries the **build** version, not the format version — reusing the format
   version there would make the field useless for the thing R4's `bad_producers` list needs
   it for.
2. **Values, optionally** — **done.** `WriteOptions{include_values}` (or `to_test_case()`)
   emits `VALUES inline` plus one **`VALUES_ROW` per operand row**, and `read_l0` fills a
   `LoadInfo{has_values}` so the caller never infers "kernel or test case" from zeros — an
   all-zero operand is a legitimate kernel input, and guessing would make the two
   indistinguishable.

   One record per *row* is what keeps the text justification honest: a one-element change
   moves **one line** in a diff, which is asserted. A whole operand per line would make
   every change look like a rewrite.

   Values are decimal at `max_digits10` through a classic-locale stream, with `inf`,
   `-inf` and `nan` as explicit tokens (§5). Verified bit-exact — via `memcmp`, so `-0.0f`
   is distinguished from `0.0f` — across `1.0000001f`, denormal min, `FLT_MIN`, `FLT_MAX`,
   `lowest()`, π, `1e-7`, both infinities and NaN.

   **Strict about completeness**, deliberately: a missing row, a wrong value count, a
   duplicated row, a row outside the operand, a row for an undeclared operand, and a file
   whose preamble says `VALUES none` while carrying `VALUES_ROW` records are all refused. A
   test case that silently lost some of its inputs is worse than one that will not load,
   because it would run and produce an answer nobody could tell was wrong.
3. **Golden corpus in CI** — **done.** `tests/program/corpus/` holds matmul 48³ and tile LU
   64, each as a pair: `<case>.l0` carrying the **inputs** and `<case>.result.l0` carrying
   the **expected outputs**. Two files rather than one because LU factors `A` **in place** —
   a single post-execution snapshot would have overwritten the input it was meant to
   preserve.

   `test_l0_corpus` loads each input, executes it at every implemented level, and compares
   **every operand** against the expected file. No `fill()` call appears in the test: a
   corpus entry has to be self-contained or it is not evidence.

   **The recorded answers are compared within a tolerance, not bit-exactly**, and the reason
   is specific: Release builds with `-march=native -mtune=native`, so instruction selection
   follows the host CPU and two machines differ in the last bits. The first version compared
   bit-exactly and passed locally — which was luck, not evidence, because `fill_matmul`
   produces exact quarter-integers and matmul's arithmetic stays exactly representable. LU
   divides, and CI failed on LU alone, identically at both levels, which is what
   distinguishes a machine difference from a model disagreement. Bit-exactness is asserted
   where it is genuinely promised: **same machine**, corpus file versus fresh derivation.

   Three further checks, each answering a different question:

   - **byte-stable re-serialization**. Deliberately strict, and it will fail on any format
     change — that is the point, and verified to bite: tampering with one field makes two
     assertions fail.
   - **a hand-written refusal fixture** declaring `MIN_CONSUMER 9.0.0`, which must fail to
     load. Never regenerated, precisely so no tool can quietly bring it in line.
   - **derivation equivalence**, kept separate from the execution check because they can
     diverge: a derivation change leaves the corpus executing correctly while silently
     making it stale, and conflating the two would hide which moved.

   Regeneration is a documented command (`kpu-run --emit-l0 / --emit-l0-result`) rather than
   a script, and the corpus README asks the question that matters before you run it: **does
   this change need a version bump?** Regenerating without answering that turns a corpus
   into a rubber stamp — the files still load, because the code that reads them also wrote
   them.

   **The bytes are the evidence, so nothing may transform them.** `*.l0` is marked `-text`
   in `.gitattributes` and `--emit-l0` writes in binary mode. Both are needed for the same
   reason: a byte-stability check cannot survive an encoding that depends on the platform
   that checked the file out or wrote it. This was not theoretical — CI went red on all three
   builds while the corpus was green locally, because a Windows checkout had rewritten every
   LF as CRLF. Reproduced locally by converting a corpus file to CRLF, which fails exactly
   the three cases CI failed.

   **The corpus earned its place immediately**: the first draft of the refusal fixture put a
   comment before the magic line, and the test caught it — reporting `NotAnL0File` where the
   fixture was meant to exercise `UnsupportedVersion`. The magic must be the first line,
   with nothing before it, because a file has to be identifiable by its opening bytes.
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

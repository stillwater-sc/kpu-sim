# The L0 golden corpus

Checked-in `.l0` files that CI **loads and executes**, so the format cannot rot quietly.

This exists because `kernels/bin/*.kpubin` did rot: opcodes were renumbered with no version
bump, and those files now abort with `std::get: wrong index for variant`. A version policy
that is not tested is a version policy that will be wrong.

## What is here

| File | Role |
|---|---|
| `<case>.l0` | the program with its **inputs** inline — a test case |
| `<case>.result.l0` | the same program **after execution** — the expected outputs |
| `matmul_32x32x32_t16_kernel.l0` | a **kernel**: structure only, `VALUES none`, `MIN_CONSUMER 1.0.0` |
| `needs_a_newer_reader.l0` | **hand-written**: a supported container version with `MIN_CONSUMER 9.0.0`, so only the min_consumer gate can refuse it |
| `needs_a_newer_container.l0` | **hand-written**: `KPUL0 9.0.0`, so only the container-major check can refuse it |

Two refusal fixtures rather than one, because a single file cannot test both gates. An
earlier draft declared `KPUL0 9.0.0` *and* `MIN_CONSUMER 9.0.0` — and the reader rejects the
container major **before** reading `MIN_CONSUMER`, so the fixture passed with
`UnsupportedVersion` even if the min_consumer gate were broken. A test that passes for the
wrong reason guards nothing.

Two files per case rather than one, because LU factors `A` **in place**: a single
post-execution snapshot would have overwritten the input it was supposed to preserve.

**The kernel entry is here because `VALUES none` is a mode the format supports.** Without a
checked-in file written that way, the kernel path existed only in a round-trip test that
never touched a file, and a reader change could have broken it in CI's blind spot. Its
`MIN_CONSUMER` is `1.0.0` and that is the property it guards: a structure-only file needs
nothing beyond the original reader, so demanding more would lock out a consumer for no
reason. Its **container** line is deliberately unasserted — a file that still loads on a
newer reader is the whole point of the version policy.

It carries no expected-output half, because a program with no inputs has no answer to
record. `test_l0_corpus` runs it the way `kpu-run --program ... --fill-inputs` does: inputs
are **synthesized** first, and every level must then agree. Executing it as loaded would run
on zeros, where every level agrees about nothing — which is why the driver refuses that
combination outright.

## How CI uses them

`test_l0_corpus` makes **two separate claims**, and keeping them apart matters because they
fail for different reasons:

| Claim | Scope | Strictness |
|---|---|---|
| the recorded answer is still the answer | **cross-machine** | within tolerance (`atol 1e-5`, `rtol 1e-4`) |
| executing a corpus file equals executing a fresh derivation | **same machine** | **bit-identical** |
| re-serializing a corpus file reproduces it | any | **byte-identical** |

**Why every fixture value must be exactly representable.** `driver::fill` produces only
values that need no rounding: matmul's `A` in quarters and `B` in **eighths**, LU in eighths.
The per-operand granularity is incidental — the property that matters is that none of them
round. That is a requirement, not a coincidence: Release builds with `-march=native`, so
host-specific code
generation is free to compute `a - b * c` with one rounding instead of two, and a value that
needs rounding can therefore differ between machines. If the *inputs* differ, no checked-in
file can be reproduced anywhere else.

The LU fill used `* 0.1f`, which left **3129** of its off-diagonal values inexact, and CI
failed on the LU corpus **input** — not its output. With a power of two the product is exact,
so nothing rounds and there is nothing for code generation to vary. Verified: the LU input
file now hashes identically under `-O0`, `-O3 -march=native -ffp-contract=fast`,
`-O3 -march=x86-64 -ffp-contract=off` and `-O2 -ffp-contract=fast`.

**Why the recorded answers are still not compared bit-exactly.** Exact *inputs* do not make
exact *outputs*: LU divides, so its intermediates need rounding no matter how the inputs are
chosen, and those are the values host-specific code generation can vary. This project builds Release with
`-march=native -mtune=native`, so instruction selection — FMA contraction, vectorisation
width, reduction order — follows the *host CPU*. Two machines compute different last bits for
the same program, and no checked-in file can promise otherwise.

The first version of this corpus did compare bit-exactly and passed locally, which was luck
rather than evidence: `fill_matmul` produces exact quarter-integers, so matmul's arithmetic
stays exactly representable whatever the compiler emits. Tile LU divides, its intermediates
are not representable, and **CI failed on LU alone** — on both levels identically, which is
what distinguishes a machine difference from a model disagreement.

The tolerance still has teeth: perturbing one recorded element by 50% fails the check by four
orders of magnitude. It catches an op that starts computing something else, which is what a
golden output is for; it does not pretend to catch a differing last bit.

The byte-stability check *is* exact and will fail on any format change — which is the point.
It is not a nuisance to route around: the checked-in files are the evidence that old files
still load.

## The bytes are the evidence

`*.l0` is marked `-text` in `.gitattributes`, so git never converts line endings on these
files, and `--emit-l0` writes them in **binary** mode. Both are required for the same reason:
a byte-stability check cannot survive an encoding that depends on the platform that checked
the file out or wrote it.

This was not theoretical. CI went red on three builds while the corpus was green locally,
because a Windows checkout had rewritten every LF as CRLF and the re-serialization comparison
saw different bytes.

## Regenerating, which is a decision and not a chore

```
kpu-run --algo matmul --size 48 --tile 16 \
        --emit-l0 tests/program/corpus/matmul_48x48x48_t16.l0 \
        --emit-l0-result tests/program/corpus/matmul_48x48x48_t16.result.l0
```

Before regenerating, answer this: **does the change need a version bump?**

- a new record carrying *semantics*, or any change to an existing record's meaning → raise
  `min_consumer` for files that use it, so older readers refuse rather than mis-read
- a new *optional attribute* → no bump; ignoring it is harmless by construction (R8)
- a change to what an op *computes* → that is an **op-set** bump, not a container bump

Regenerating without answering that is how a corpus becomes a rubber stamp: the files still
load, because they were written by the same code that reads them.

The magic is the **first line**, with nothing before it — not even a comment — because a
file has to be identifiable by its opening bytes. Comments are allowed on every line after
it. The first draft of `needs_a_newer_reader.l0` put a comment first, and the corpus test
caught it on the day it was written: the file was refused as `NotAnL0File` when the point of
the fixture was to exercise `UnsupportedVersion`.

`needs_a_newer_reader.l0` is never regenerated. It is hand-written on purpose, so no tool
can quietly bring it in line with the current version.

The kernel entry has no `--emit-l0` form, because `kpu-run` writes test cases: every emitted
file carries its inputs, which is what makes a corpus entry self-contained. The kernel file
is `serialize::to_string(derive_matmul_tile_program(32, 32, 32, 16, 16, 16))` — the default
`WriteOptions`, no values — and `test_l0_corpus` asserts exactly that equality, so a
derivation change or a format change fails there and forces the version question rather than
being absorbed silently.

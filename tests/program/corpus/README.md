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
| `needs_a_newer_reader.l0` | **hand-written**: a supported container version with `MIN_CONSUMER 9.0.0`, so only the min_consumer gate can refuse it |
| `needs_a_newer_container.l0` | **hand-written**: `KPUL0 9.0.0`, so only the container-major check can refuse it |

Two refusal fixtures rather than one, because a single file cannot test both gates. An
earlier draft declared `KPUL0 9.0.0` *and* `MIN_CONSUMER 9.0.0` — and the reader rejects the
container major **before** reading `MIN_CONSUMER`, so the fixture passed with
`UnsupportedVersion` even if the min_consumer gate were broken. A test that passes for the
wrong reason guards nothing.

Two files per case rather than one, because LU factors `A` **in place**: a single
post-execution snapshot would have overwritten the input it was supposed to preserve.

## How CI uses them

`test_l0_corpus` makes **two separate claims**, and keeping them apart matters because they
fail for different reasons:

| Claim | Scope | Strictness |
|---|---|---|
| the recorded answer is still the answer | **cross-machine** | within tolerance (`atol 1e-5`, `rtol 1e-4`) |
| executing a corpus file equals executing a fresh derivation | **same machine** | **bit-identical** |
| re-serializing a corpus file reproduces it | any | **byte-identical** |

**Why the recorded answers are not compared bit-exactly.** This project builds Release with
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

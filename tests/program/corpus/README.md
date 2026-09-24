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
| `needs_a_newer_reader.l0` | **hand-written**, declares `MIN_CONSUMER 9.0.0`, and must **fail** to load |

Two files per case rather than one, because LU factors `A` **in place**: a single
post-execution snapshot would have overwritten the input it was supposed to preserve.

## How CI uses them

`test_l0_corpus` loads each `<case>.l0`, executes it at every implemented level, and
compares **every operand** against `<case>.result.l0`. It also re-serializes both files and
requires the output to be **byte-identical** to what is checked in.

That byte-stability check is deliberate and will fail on any format change — which is the
point. It is not a nuisance to route around: the checked-in files are the evidence that old
files still work.

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

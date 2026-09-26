# The `.kpuld` golden corpus

Checked-in loadables that CI **loads**, so the container cannot rot quietly (#305).

| File | Role |
|---|---|
| `matmul_32_external.kpuld` | one operator, three tensors, **data external** — compared **byte for byte** |
| `needs_a_newer_reader.kpuld` | **hand-built**: a supported container version with `min_consumer 9.0.0`, so only the min_consumer gate can refuse it |

## Why a corpus at all

`kernels/bin/*.kpubin` **rotted**: opcodes were renumbered with no version bump, and those
files now abort with `std::get: wrong index for variant`. A version policy that is not tested
is a version policy that will be wrong, so a checked-in file is the evidence that an existing
loadable still works.

A binary container makes this *more* important than it was for the L0 text, not less: the
generated FlatBuffers accessors do not bounds-check, so a file this reader mis-parses is
undefined behaviour rather than a wrong answer.

## What each file proves

`matmul_32_external.kpuld` is compared **byte for byte** against what this build writes. That
is deliberately strict and will fail on a change to the container, to the embedded L0 text, or
to the project version that stamps `producer_version`. Before regenerating, answer the question
that matters: **did the format change?** If only the project version moved, regeneration is
routine. If the format moved, it needs a version decision first — a regenerated fixture always
passes, because the code that writes it is the code that reads it.

`needs_a_newer_reader.kpuld` is **hand-built and never regenerated**, because the normal writer
*cannot* produce it: `write()` always stamps the current version and derives `min_consumer` from
the records present. That impossibility is what makes the fixture trustworthy — no tool can
quietly bring it into line. It declares a container version this reader **does** support, so the
only thing that can refuse it is the `min_consumer` gate itself; the test also checks that the
*message* names the demand rather than the container, because #265's corpus once had a fixture
that passed for the wrong reason by tripping the earlier gate.

## The bytes are the evidence

`*.kpuld` is `-text -diff` in `.gitattributes`, from the format's first commit. Without `-text`,
a Windows checkout rewrites bytes that happen to look like line endings and the byte comparison
fails there and nowhere else — which cost #299 several CI rounds on the L0 corpus, where the
files were at least human-readable. `-diff` keeps git from trying to render a binary diff.

## Regenerating

`matmul_32_external.kpuld` is `write()` applied to the loadable that
`tests/program/test_loadable.cpp` builds in `canonical_fixture()` — the test asserts that
equality, so the two cannot drift. There is deliberately **no tool** that regenerates the
refusal fixture.

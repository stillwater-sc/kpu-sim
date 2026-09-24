# Deployment spec fixtures

Checked-in `.json` deployment specs that CI **loads** (#282 increment 1).

| File | Role |
|---|---|
| `canonical_single.json` | the **canonical** form of the default single-device deployment; compared **byte for byte** |
| `adr_checkerboard16.json` | ADR 0002 §3.5's example **verbatim**, in its flat spelling |

## Why two, and why they are checked differently

`canonical_single.json` is the format's own output, so it is compared byte for byte. That
check is deliberately strict and will fail on any change to the canonical form — which is
the point: a deployment digest is part of a run's identity (design note §4), so the bytes
have to be a stable function of the spec. It also pins the **default machine**, so a drifting
default is caught here rather than as a moved makespan somewhere else.

`adr_checkerboard16.json` is the ADR's example, unmodified. It is **not** byte-compared,
because it cannot be: `"engines": 4` is an integer literal where `bytes_per_cycle` is a
double, and no single writer reproduces both spellings. What is asserted instead is that it
**loads**, that it means the machine the ADR describes, and that normalizing it is
**idempotent** — rewritten once, stable thereafter.

Keeping it verbatim is the point of having it: someone will paste §3.5's example, and a
format that refused the spelling its own authority document shows would make the
documentation a trap. Its flat form (no `"devices"` array) and its `"burst"` key are both
accepted for that reason and canonicalized on the way out.

## The bytes are the evidence

`tests/program/deploy/*.json` is marked `-text` in `.gitattributes`, so git never converts
line endings here. Without it a Windows checkout rewrites every LF as CRLF and the
byte-comparison fails there and nowhere else — the failure that cost `#299` several CI
rounds on the L0 corpus. Verified by converting the fixture locally and watching exactly
that assertion break.

## Regenerating

`canonical_single.json` is `to_json(DeploymentSpec{})`. Before regenerating, answer the same
question the L0 corpus asks: **is the change to the canonical form intentional?** A
regenerated fixture always passes, because the code that writes it is the code that reads it.

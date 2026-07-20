# Versioning the `.dfg` source IR and the `.kpubin` binary program

**Status:** requirements analysis (companion to `model-ingestion-compilation-epic.md`,
epic #229). Resolves the plan's §9 Q1.

**Motivation:** ONNX, StableHLO, and TOSA/MLIR all carry explicit version numbers and
compatibility policies. domain_flow's `.dfg` today has **none** — no format version,
and the `DomainFlowOperator` enum is an *implicit, unversioned* opset. The `.kpubin`
binary program (the compiler↔simulator ABI, epic decision D4/D5) has a partial header
(magic + version) but lacks the full stamp. This note extracts the versioning
requirements from prior art.

---

## 1. How the major formats version themselves

### ONNX — three orthogonal axes
ONNX versions three independent things
([ONNX Versioning](https://onnx.ai/onnx/repo-docs/Versioning.html),
[IR spec](https://onnx.ai/onnx/repo-docs/IR.html)):
- **IR / format version** (`ir_version`) — the container structure; a monotonic
  integer, bumped atomically.
- **Operator-set version** (`opset_import`) — a list of `(domain, version)` pairs.
  `""` = the standard `ai.onnx` set; other domains = vendor sets, each versioned
  independently. Any op add/remove/**semantic change** MUST bump the domain version;
  each op carries a `since_version`.
- **Model version** (`model_version`) — the trained model's own version. Plus
  `producer_name`/`producer_version`.

**Takeaway:** *format structure* and *operator semantics* version separately; ops
live in versioned **domains**.

### StableHLO — semver + explicit windows + a separate serialization dialect (VHLO)
[StableHLO Compatibility](https://openxla.org/stablehlo/compatibility),
[VHLO](https://openxla.org/stablehlo/vhlo):
- `MAJOR.MINOR.PATCH`; explicit compatibility windows (≈2 years forward; backward
  across 1 major release).
- Serialization goes through **VHLO**, an **add-only** dialect: every op/type/attr is
  individually versioned with a `[min, max]` range; once released, semantics are
  **frozen**; changes require a *new* version. Serialization targets a specific
  version; deserialization upgrades/downgrades. Both directions are **CI-tested every
  PR**.

**Takeaway:** *serialization stability is a first-class, separately-versioned,
freeze-on-release concern — real only if enforced by a golden-artifact suite.*

### TOSA — version number plus capability dimensions
[TOSA dialect](https://mlir.llvm.org/docs/Dialects/TOSA/),
[RFC v1.0](https://discourse.llvm.org/t/rfc-tosa-dialect-increment-to-v1-0/83708):
- Versioned (→ v1.0; minor versions backward-compatible), but layered with
  **profiles** (base/main inference, training), **levels**, **extensions**, and
  **datatype combinations**. A module can be *syntactically valid but invalid for a
  given profile/level*. The dialect is a **superset** of the spec, with a validation
  pass checking conformance.

**Takeaway:** *a version number is not enough — an artifact declares required
capabilities, a consumer advertises supported ones, mismatch is a clean refusal.*

### Supporting patterns
- **TF GraphDef** `VersionDef { producer, min_consumer, bad_consumers[] }` — the
  cleanest **forward-compat gate**: an artifact states the minimum consumer version
  (old readers refuse too-new files cleanly) + a **blocklist of known-bad producers**.
- **PyTorch `torch.export`** versions its serialization schema as a `(major, minor)`
  tuple with defined bump rules; **TFLite** carries a flatbuffer schema `version`.

---

## 2. Requirements for `.dfg`

| # | Requirement | Sourced from |
|---|---|---|
| **R1** | **Three orthogonal version axes**: (a) `.dfg` format/container version, (b) operator-set version(s), (c) producer name+version. | ONNX |
| **R2** | **Operators namespaced by domain** — every op is `(domain, name, version)`; a core `stillwater.dfa` opset versions independently of vendor/experimental ops. | ONNX |
| **R3** | **Semver with documented bump rules** per axis: MAJOR = breaking, MINOR = additive/back-compat, PATCH = fixes. | StableHLO, torch.export |
| **R4** | **`min_consumer` gate**: the file declares the minimum reader version; an older loader **refuses cleanly** rather than mis-parsing. Plus a `bad_producers` blocklist. | TF GraphDef, StableHLO |
| **R5** | **Add-only / freeze-on-release** for the serialized op/attr/type surface, backed by a versioned registry with per-element `[min,max]` ranges. | StableHLO VHLO |
| **R6** | **Capability/profile metadata beyond the version**: datatype set + (for KPU) target-config assumptions; artifact declares *required* profiles, consumer advertises *supported*. | TOSA |
| **R7** | **Pin the type-system version** — `.dfg` embeds `tensor<…>` strings; that grammar is a versioned surface and must be declared, not assumed. | MLIR/TOSA, ONNX |
| **R8** | **Defined unknown-op/attr policy**: strict-fail on unknown *ops* in a required opset; additive-tolerant on optional *attrs*. | ONNX, StableHLO |
| **R9** | **Explicit compatibility windows**, documented and **CI-tested** with a golden-`.dfg` corpus. | StableHLO |

### Illustrative `.dfg` versioned preamble (the current format has none)

The example below is **illustrative** — it shows the *fields* the requirements imply,
not the final grammar. The **complete, normative grammar** (exact field order,
repetition, semver comparison + bump rules, legacy-file policy, and the precise
refusal behavior for malformed / unsupported-version / capability-mismatch /
unknown-op cases) is a **Phase 0 deliverable** of the epic (issue #230), written once
and shared by producer (domain_flow) and consumer (kpu-sim's loader).

```text
DFG 1.0.0                     # R1a/R3 format version (MAJOR.MINOR.PATCH semver)
MIN_CONSUMER 1.0.0            # R4 minimum reader format version; older readers refuse
PRODUCER domain_flow 0.4.2    # R1c producer name + semver
BAD_PRODUCERS domain_flow 0.4.0  # R4 known-bad producer blocklist (repeatable)
OPSET   stillwater.dfa 3.0.0  # R1b/R2/R3 (domain, opset semver) — repeatable
TYPESYS mlir-tensor 1.0.0     # R7 type-string grammar semver
REQUIRES_PROFILE fp32         # R6 profiles this artifact NEEDS (repeatable)
                              #    (consumer separately advertises SUPPORTS_PROFILE;
                              #     REQUIRES ⊄ SUPPORTS ⇒ clean refusal)
... existing DIRECTED/NODES/EDGES/ADJACENCY body ...
```

Semver comparison (R3): a consumer accepts an artifact iff its reader format version
≥ `MIN_CONSUMER`, the producer is not in its `BAD_PRODUCERS` set, each required
`OPSET`/`TYPESYS` MAJOR matches a supported one (MINOR/PATCH additive-tolerant, R8),
and `REQUIRES_PROFILE ⊆ SUPPORTS_PROFILE`; otherwise it refuses with a specific
diagnostic. Unknown *ops* in a required opset are hard errors; unknown optional
*attrs* are ignored (R8).

---

## 3. The `.kpubin` binary program needs its own, stricter versioning

`.dfg` is the compiler-front-facing *source* IR. The **`.kpubin` binary program (the
`DMProgram` ISA stream) is the hardware ABI** — the D5 contract between the
domain_flow compiler and the simulated KPU (epic decision D4) — and a binary ABI is
exactly where versioning matters most.

**Prior art already in-repo:** `ProgramSerializer` (`src/software/isa/program_serializer.cpp`)
already writes a header with `DMPROGRAM_MAGIC` + `DMPROGRAM_VERSION` and magic-checks
on read ("format v2"). That is the **seed** — Phase 0 extends it to the full stamp:
- **(a) ISA / format version** — already present (`DMPROGRAM_VERSION`); add a
  `MIN_CONSUMER` gate (R4) so an old loader refuses a too-new binary cleanly.
- **(b) KPU functional-spec profile it targets** — fabric config + datatype support.
  The R6 capability dimension is load-bearing: an int8 program on an fp32-only fabric
  must be **rejected, not mis-run**. (Not represented in the current header.)
- **(c) producer compiler version** (+ `bad_producers`, R4). (Not present today.)

Treat it like a StableHLO portable artifact: freeze-on-release + `min_consumer` gate +
a **golden-binary conformance suite** — the epic's D5 / Phase 0 deliverable (#230).
kpu-sim owns this spec (it is the hardware); the compiler targets it. **Do not add a
second binary format** — extend `.kpubin`/`ProgramSerializer`; the higher-level `dfx`
`.kpu` object's fate (intermediate vs dropped) is a separate Phase 0 decision (epic
D4).

---

## 4. Recommendation

- **Version `.dfg` in place** by adding the preamble (§2) — do **not** fork a parallel
  schema; extend domain_flow's existing text format. `.dfg` is **compiler-internal**
  (the source IR that ONNX imports into and passes rewrite); it is versioned for the
  *compiler front-end*, **not** as a kpu-sim runtime input. kpu-sim's runtime
  execution contract is the `.kpubin` binary (epic D4/D5), not `.dfg`.
- **Version `.kpubin` from day one** (Phase 0) by extending the existing
  `ProgramSerializer` header (§3) — it is the ABI; retrofitting a fuller version onto a
  binary contract later is far more expensive.
- **One binary format, one source format** — do not add a third path; the `dfx` `.kpu`
  object's status (intermediate vs dropped) is a Phase 0 decision (epic D4).
- **Enforce with golden corpora** in CI for both — a version policy that isn't tested
  rots (StableHLO's discipline).

# Versioning the `.dfg` source IR and the `.kpu` binary program

**Status:** requirements analysis (companion to `model-ingestion-compilation-epic.md`,
epic #229). Resolves the plan's §9 Q1.

**Motivation:** ONNX, StableHLO, and TOSA/MLIR all carry explicit version numbers and
compatibility policies. domain_flow's `.dfg` today has **none** — no format version,
and the `DomainFlowOperator` enum is an *implicit, unversioned* opset. The `.kpu`
binary program (the compiler↔simulator ABI, epic decision D5) has the same gap and a
higher stakes. This note extracts the versioning requirements from prior art.

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

### Proposed `.dfg` versioned preamble (the current format has none)

```
DFG 1.0                       # R1a format version (MAJOR.MINOR)
PRODUCER domain_flow 0.4.2    # R1c producer + version
OPSET   stillwater.dfa 3      # R1b/R2 (domain, opset_version) — repeatable
MIN_CONSUMER 1.0              # R4 minimum reader; older readers refuse
PROFILE fp32                  # R6 datatype/capability profile(s)
TYPESYS mlir-tensor 1         # R7 type-string grammar version
... existing DIRECTED/NODES/EDGES/ADJACENCY body ...
```

---

## 3. The `.kpu` binary program needs its own, stricter versioning

`.dfg` is the compiler-front-facing *source* IR. The **`.kpu` binary program is the
hardware ABI** — the D5 contract between the domain_flow compiler and the simulated
KPU — and a binary ABI is exactly where versioning matters most. It needs its **own**
version stamp:
- **(a) ISA / format version** of the binary program.
- **(b) KPU functional-spec profile it targets** — fabric config + datatype support.
  The R6 capability dimension is load-bearing here: an int8 program on an fp32-only
  fabric must be **rejected, not mis-run**.
- **(c) producer compiler version.**

Treat it like a StableHLO portable artifact: freeze-on-release + `min_consumer` gate +
a **golden-binary conformance suite** — which is already the epic's D5 / Phase 0
deliverable. kpu-sim owns this spec (it is the hardware); the compiler targets it.

---

## 4. Recommendation

- **Version `.dfg` in place** by adding the preamble (§2) — do **not** fork a parallel
  schema; extend the existing text format so kpu-sim's current reader evolves rather
  than a second path appearing.
- **Version `.kpu` from day one** (Phase 0) — it is the ABI; retrofitting a version
  onto an unversioned binary contract later is far more expensive.
- **Enforce with golden corpora** in CI for both — a version policy that isn't tested
  rots (StableHLO's discipline).

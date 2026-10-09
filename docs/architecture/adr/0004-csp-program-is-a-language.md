# ADR 0004 — The CSP program is a written language, not only a derived IR

| | |
|---|---|
| **Status** | **Accepted** (2026-10-09) |
| **Date** | 2026-10-09 |
| **Amends** | ADR 0002 §6, answer 1 ("never hand-authored, so it needs no source language or parser"); ADR 0001 D1 ("the portable program is the L0 `TileProgram`"), by §4 |
| **Preserves** | ADR 0002 §2 (CSP is the program layer that every level interprets); ADR 0001 D5 (values answer to the L0 reference, which L0 instances carry) |
| **Context docs** | `docs/plans/csp-program-tile-sequencing.md` (the CSP program as the only source of tile sequencing), `docs/plans/csp-language.md` (this language) |

---

## 1. Context

ADR 0002 made the CSP program the layer every level interprets: processes, channels, and the
ordered tile sequences they move. It decided that the program is *derived* from the Domain
Flow Program and never written, so "an in-memory IR with a disassembler is sufficient".

That decision was implemented in `include/sw/kpu/program/csp/` (#341). The `CspProgram` IR is
lowered from L0, printed by a disassembler, and executed at L-B and L-CA (#342). It has one
producer, the L0 lowering, and no form anyone can write.

The sequencing *is* the program a human or an AI assistant needs to express. It covers:
- what comes into L3, and how long it stays;
- the order in which tiles meet the compute fabric;
- what is fused onto a tile as it moves.

Those decisions determine reuse, and so energy (`csp-program-tile-sequencing.md` §1). A
compiler is one author of them, and not the only one. Without a language, every sequencing
experiment means editing a C++ deriver. Two forms are writable today, and neither works at this
level:
- **L0 (`.l0`)** says what to compute, as a flat list of tile ops with no residency.
- **KPUASM (`.kpuasm`)** says how one engine moves bytes: physical L3 tiles, offsets, L2 banks,
  and synchronization by barriers. It runs on the older instruction-set executors, which are
  not connected to the four levels.

## 2. Decision

1. **The CSP program has a written form,** a structured language (`docs/plans/csp-language.md`).
   Humans, AI assistants and compilers all write it. The DFP lowering and the L0 deriver are
   producers of it, not its only source.
2. **It is structured, not a flat action list.** It has loops over the tile index space,
   explicit residency (`resident` / `release`), tile-function calls standing in for the inner
   loops, and contexts attached to tile moves. The flat per-process action list is its
   disassembly and its IR, not what anyone writes.
3. **Processes are implicit at level 1.** On the flat machine (one L3, one compute fabric), each
   statement implies the actions of the dma, bm, str and cf processes. **Placement is explicit
   at level 2:** a distributed program names its partition map, its owners (L3 tile, compute
   tile) and its communication (broadcasts on the NoC), as an MPI program names ranks and
   collectives.
4. **Every level executes the written program,** through one validator and one lowering to the
   `CspProgram` IR: L-B and L-T1 interpret it, and L-CA through `CspDriver`. The validator owns
   the checks the lowering and the behavioral interpreter already make: capacity, operands
   present, nothing read after release.
5. **KPUASM stays, for now,** as the engine level below this language. A distributed CSP
   program could one day emit it for a specific device. It is neither extended nor retired
   until that reuse is evaluated.

## 3. Consequences

- **The ADR 0002 §6.1 premise changes, its intent does not.** The CSP program is still the one
  thing every level executes. What changes is that it can be written as well as derived.
- **The disassembler and the printer converge.** Printing the IR as the language makes a derived
  program something you can read and edit. The flat action listing remains for debugging.
- **New drivers are written, not coded:** matmul, LU, and a linear operator (matmul, bias and
  activation) that expresses fusion. The linear operator requires the language to say *where* an
  epilogue runs: in the compute fabric, on a streamer's drain, or on a BlockMover's ingress or
  egress. That is the **tile context** of `csp-language.md` §3.4.
- **The language needs versioning,** as `.l0` has (format, opset, reader versions).

## 4. Amendment (2026-10-09): L0 is a trace format; the CSP program is the portable program

The architect observed that L0 cannot express a large operator. Its tile sequence is fully
unrolled: a 1M x 1M matmul with 32 x 32 tiles is about 3 x 10^13 tile ops, so no L0 file of it
can exist. **L0 is therefore an instance, or trace, format, not a program.**

The same applies to the flat `CspProgram` action list of step 1. The `.csp` language keeps its
loops, but the step-1 compiler unrolled them, the driver enqueued every action up front, and
the validator walked every unrolled statement. The language was scalable; its representation
was not.

Decided:

1. **The portable program is the CSP program (`.csp`).** L0's role narrows to what it is good at:
   - the golden corpus;
   - a carrier of inline values for the value oracle;
   - a debugging artifact.

   This amends ADR 0001 D1.
2. **The structured program is what executes.**
   - Interpreters pull actions lazily from the program's loop nests, in memory bounded by
     L3 capacity and loop depth, not by the problem size.
   - The flat action list is a trace, emitted on request for small cases, debugging and the
     record. It is never required to run.
3. **Validation is symbolic.** Residency, capacity and index bounds are checked over the loop
   structure: affine indices, per-iteration residency balance, and one representative
   iteration. Nothing is unrolled.
4. **Derived programs should arrive with their loops.** The L0 → `.csp` emitter is a bridge for
   today's corpus, not the path forward; DFP → `.csp` is.


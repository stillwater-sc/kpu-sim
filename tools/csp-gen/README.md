# kpu-csp-gen: a generator of test and experiment programs

`kpu-csp-gen` writes CSP programs (`.csp`) for three operators, matmul, the linear operator and
LU, at a chosen size, tiling and L3 capacity. `kpu-run` executes them.

**It is not the source of the KPU's operator library.** It is a generator of test and experiment
programs for the simulator. It stands in for the compiler's lowering until that exists. Programs
at the scale of MTL5 or PyTorch come from lowering Domain Flow Programs, not from this tool.

## Where programs are meant to come from

ADR 0002 ([`docs/architecture/adr/0002-csp-interpretations-and-virtual-platform.md`](../../docs/architecture/adr/0002-csp-interpretations-and-virtual-platform.md))
says that a CSP program is **derived** from a Domain Flow Program (DFP) and never hand-written:
"Nothing below the DFP is a separate program to maintain."

Its §3.1 places that derivation on the compiler side (domain_flow), not in the platform:

```cpp
CspProgram lower_to_csp(const DomainFlowProgram&, const LoweringOptions&);
```

So the answer to thousands of operators is a lowering pass over one DFP per operator, not one
hand-written schedule per operator. That lowering does not exist yet: it is #281, "derive
explicit CSP processes and channels from the Domain Flow Program", on the roadmap for v0.11.0.

## What `kpu-csp-gen` solves

The tool came out of [`docs/plans/kpu-run-csp-programs.md`](../../docs/plans/kpu-run-csp-programs.md)
(October 2026), for three reasons.

1. **It takes program generation out of the simulator.** Before it, `kpu-run` derived its own
   programs (`--algo`) or ran unrolled L0 traces, and its L-T1 *inferred* residency rather than
   executing the program's. The plan made `kpu-run` an executor of a `.csp` program only, and
   moved generation into a separate tool, whose role is described as "Not its role: simulating
   anything".
2. **It supplies valid programs while no lowering exists.** Every level, the golden corpus
   (`tests/program/corpus/`), the viewers and the bubble report need real CSP programs. The
   generator writes each operator's canonical tiled schedule at any size, tiling and L3. It
   validates what it writes (`lang::validate`), and writes ordinary text, which can be printed,
   edited and fed back in.
3. **Its options probe the architecture, not operator coverage.** Each option switches on a
   machine feature under test:

   | Option | Exercises |
   |---|---|
   | `--dataflow os\|ws\|is` | where partials live: the fabric, or L2 slots merged by the Combiner |
   | `--orient`, `--stage-a` | panel order, and L2 staging as stated reuse |
   | `--init` | the accumulator's initial value, streamed in from the array's edge |
   | `--dtype`, `--requant`, `--requant-channels` | number systems, and requantization on the Combiner |
   | `--place`, `--act` | where an epilogue runs: the fabric, the Combiner, the BlockMover's egress unit, or unfused |

   The aim is to compare data paths on one operator, which is the architecture work. The plan's
   Q4 says the same: one canonical schedule per operator, and a later schedule search "belongs to
   the generator, not the simulator".

## Scope

The tool has grown past its original brief. The plan settled on one canonical schedule per
operator with two options. Since then it has gained `--init`, `--dtype`, `--requant`,
`--requant-channels`, `--stage-a` and three dataflows, so it is becoming a small compiler for
three operators. That serves the architecture experiments well, but it is not the road to MTL5 or
PyTorch: each new operator added here would be hand-written schedule code that ADR 0002 says
should not exist.

**Recommended** (not yet decided):
- Keep `kpu-csp-gen` as the generator of test and experiment programs. Add options when an
  architecture question needs one; do not add operators for coverage.
- Put operator-coverage work into `lower_to_csp` (#281).
- A useful first milestone for the lowering: lower matmul's DFP and match `kpu-csp-gen`'s output.
  That cross-checks the lowering, and after it this tool is a test fixture rather than a source
  of programs.

## Usage

```bash
kpu-csp-gen --algo matmul --size 64 --tile 16 --l3 32 --dataflow ws -o ws.csp
kpu-run --program ws.csp --deploy tests/program/deploy/kpu_s1_combiner.json --level all --bubbles
```

- `--algo`, `--size` and `--tile` are required, and so is one of `--l3` or `--target <spec.json>`
  (a deployment, which also gives the vector units an epilogue's placement is checked against).
- `kpu-csp-gen --help` lists every option and the combinations each accepts.
- [`docs/ONBOARDING.md`](../../docs/ONBOARDING.md) step 5 shows what to look for in the programs
  it writes.

## The code

| Piece | Where |
|---|---|
| The generators, a library | `include/sw/kpu/program/csp/gen/generate.hpp` |
| The executable, a thin wrapper | `tools/csp-gen/csp_gen.cpp` |
| Its CLI tests | `tools/csp-gen/CMakeLists.txt` |

The build target is `csp-gen`; the binary it writes is `kpu-csp-gen`. In a Visual Studio
solution it is under **Tools → Driver**.

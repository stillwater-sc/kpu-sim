# Program-Stack Examples

These examples show how kpu-sim runs a program. A CSP program (`.csp`) states the data movement:
what is resident in L3, the order tiles meet the fabric, and what is staged in L2. Every
fidelity level executes that one program:

| Level | Decomposes a transaction into | Role |
|---|---|---|
| L-B behavioral | a whole block move, atomic | computes the values: **the oracle** |
| L-T1 block-sequential | one tile move, under the program's L3 credits | timing, fast |
| L-T2 resource-transactional | read/write per resource | not implemented yet (#283) |
| L-CA cycle-accurate | protocol events, per cycle | **the timing reference** |

Every level must compute L-B's values. A level that times a program without computing it is
unvalidated, not cheaper.

## `program_run_levels`

```bash
build/examples/program/program_run_levels                    # a corpus matmul on kpu_s1
build/examples/program/program_run_levels \
    --program examples/program/matmul_48_stage_a.csp --placement
build/examples/program/program_run_levels --program <file.csp> --deploy <spec.json>
```

**Why.** This is the smallest complete use of the program API, in about 200 lines of
`run_levels.cpp`. It does what `kpu-run` does, without the tool's options.

**What.** It reads a `.csp` program and a deployment spec (the machine L-CA builds), loads the
program on the `VirtualPlatform`, runs L-B as the oracle, then runs every level. For each level it
prints the makespan and checks every operand, per tile, against L-B (`check_operands` in
`sw/kpu/program/tile_check.hpp`: `|a − o| ≤ atol + rtol·|o|` for fp32, exact for integers).

With `--placement` it also shows the **L3 placement** and the **L2 staging**:
- The walker gives every residency an L3 address (module, slot, bank) and every staging an L2
  slot.
- L-B executes those addresses, and L-T1 and L-CA time them. No level allocates a slot of its
  own (ADR 0004 §5).
- The example checks that L-T1 and L-CA hold each tile in exactly the walker's slot.
- `matmul_48_stage_a.csp` stages A's row panel in L2 (`stage A[i, :] in l2`), so it exercises
  both. Generate others with `kpu-csp-gen --stage-a`.

**Output.** One line per level, then the placement:

```text
block-sequential        makespan 774       values within bound of L-B (max abs 0)
resource-transactional  skipped: the L-T2 resource-transactional interpreter is not implemented (#283)
cycle-accurate          makespan 1673      values within bound of L-B (max abs 0)

placement (tile@slot, in the walker's order)
    L3  45 residencies:  A#0#0@0  A#0#1@1  A#0#2@2  B#0#0@3  ...
    L2  9 stagings:  A#0#0@0  A#0#1@1  A#0#2@2  A#1#0@5  ...
    L-T1 L3: executes the walker's slots
    ...
```

The L-T1 and L-CA makespans differ: L-T1 is uncalibrated, and L-CA is the timing reference.

**Checks.** The program exits 1 if any level's values fall outside the bound of L-B's, or if a
level holds a tile in a slot the walker didn't assign. CTest runs it twice:
`example_program_run_levels` (the corpus matmul) and `example_program_placement` (the staged
program with `--placement`).

## Where the bubbles and records come from

`kpu-run` runs the same program and writes what a run leaves behind:

```bash
P=examples/program/matmul_48_stage_a.csp; D=tests/program/deploy/kpu_s1.json
build/tools/kpu-run/kpu-run --program $P --deploy $D --bubbles     # the reuse and bubble report, L-T1 and L-CA
build/tools/kpu-run/kpu-run --program $P --deploy $D --tflow lt1.tflow   # L-T1's tile-flow record
build/tools/kpu-run/kpu-run --program $P --deploy $D --dflow mm.dflow    # L-B's domain-flow record
```

The bubble report says why each compute lane sat idle, for example "BlockMover lanes busy" or "a
result's way out ahead of it". It also reports the reuse (feeds per load), and the largest bubbles
with the chain of actions each one waited on. `docs/ONBOARDING.md` §4 walks through reading the
records with the viewers.

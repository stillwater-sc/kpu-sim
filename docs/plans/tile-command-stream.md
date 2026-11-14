# One Program on Both Sides: the Tile Command Stream (`kpu-tile-cmds` Version 1, #397)

**Date:** 2026-11-14
**Status:** Version 1 built: kpu-sim lowers a CSP program to the RTL's tile commands, and a
reference executor runs them. The RTL generator's runtime executing the file is
branes-ai/embodied-ai-soc-generator#49.
**Issue:** #397. Builds on the shared record (`shared-tile-flow-record.md`, 6.1 for the RTL's
acc → l3 store and word-addressed L3) and the placement (`l3-placement-at-lb.md`, #399).

## 1. Decision

The RTL's programs are Python command generators; kpu-sim's are `.csp`. The issue gave three
directions. **Option 1: kpu-sim lowers a `.csp` program to the RTL's tile commands**, so kpu-sim
stays the one source of programs. The issue recommended it. It was taken on 2026-11-14 while the
user was away, as that recommendation; it can be revisited.

## 2. What the RTL executes

From kpugen (`src/kpugen/tiles/systolic/`):

- **One flat command record**, the same fields on the RTL port, the CSRs and the runtime's
  `Command` dataclass (`runtime/driver.py`): `type, opcode, id, dram_addr, local_addr, len, rs1,
  rd, k_dim, stride, rs2, row_bytes, activation, sync, chain`.
  - `type`: 0 LOAD, 1 STORE, 2 EXEC.
  - DMA opcodes: `MEMCPY` 0, `TRANSPOSE` 4 (a corner turn).
  - EXEC opcodes: `CONFIG` 0x00 (`rs1` = dataflow, 0 is output-stationary), `COMPUTE` 0x11.
  - `dram_addr` is a byte address in the tile's L3 memory cell.
  - `local_addr` bit 31 selects the accumulator; otherwise it is a scratchpad row.
- **`sync`**: `src` in bits 3:0, `dst` in bits 7:4, then flags `WAIT` 1<<8, `RELEASE` 1<<9,
  `ACQUIRE` 1<<10, `SIGNAL` 1<<11 (`controller/sync.py`). ACQUIRE starts only when `dst` is free,
  and takes it. SIGNAL marks `dst` ready on completion. WAIT starts only when `src` is ready.
  RELEASE frees `src` on completion. L2 pair slots and accumulator slots are separate pools.
- **`chain`**, output-stationary only: bit 0 continues the previous reduction, bit 1 continues
  into the next.

## 3. The lowering (version 1)

Version 1 lowers a matmul on one output-stationary tile, int8 × int8 → int32, every tile the
array's `dim × dim`. The target is the RTL's defaults: `dim` (from the program's tiles),
`buffers` 2 (L2 pair slots), `acc_slots` 2.

| CSP | Commands |
|---|---|
| (start) | `EXEC CONFIG`, `rs1` 0 (output-stationary) |
| `LOAD` | none: the host fills L3 before the tile runs (`l3_init`) |
| `MOVE`, `FEED` | none: the call's LOADs move and feed its pair |
| `CALL gemm` | `LOAD A` MEMCPY into pair slot *p* (ACQUIRE *p*); `LOAD B` TRANSPOSE into B slot `buffers + p` (SIGNAL *p*); `COMPUTE` (WAIT, RELEASE *p*; ACQUIRE the accumulator slot on the reduction's first step, SIGNAL it on its last; chain bits) |
| `DRAIN` + `WRITEBACK` | one `STORE` from the accumulator slot into L3 (WAIT, RELEASE the slot): the RTL's store reads the accumulator straight to L3 |
| `STORE` | none: the host reads the results after (`results`) |
| `RELEASE` | none |

- The pair slot *p* cycles call by call; an output's accumulator slot cycles reduction by
  reduction.
- L3 is the RTL's flat memory. A residency lives at its walker slot × a pitch of `dim·dim·4`
  bytes, so the stream's L3 addresses are the program's placement (#399).
- **Refused, by name:**
  - another element type, or a tile not the array's size;
  - an L3 slot holding two residencies: the host fills L3 once, so a program must give every
    residency its own slot;
  - L2 staging, a tile context, an accumulator init, a loaded result (accumulate onto C);
  - inherit and retain, any call other than gemm with alpha 1.

The 4×4×4 matmul lowers to the five commands kpugen's own `isa/matmul.py` issues for it. Two
LOADs, the COMPUTE with sync `0xF00`, and the STORE with sync `0x300`. `test_tile_commands`
checks that command for command.

## 4. The file

```json
{
 "format": "kpu-tile-cmds", "version": 1, "producer": "kpu-sim 0.9.0-...", "program": "mm8",
 "tile": {"dim": 4, "buffers": 2, "acc_slots": 2, "dataflow": "output_stationary",
          "input": "int8", "accumulator": "int32"},
 "l3": {"bytes": 768, "pitch": 64},
 "l3_init": [{"tile": "A[0,0]", "addr": 0, "row_bytes": 4, "element": "int8", "bytes": "<hex>"}, ...],
 "results": [{"tile": "C[0,0]", "addr": 512, "row_bytes": 16, "element": "int32", "bytes": "<hex>"}, ...],
 "fields": ["type", "opcode", "id", "dram_addr", "local_addr", "len", "rs1", "rd", "k_dim",
            "stride", "rs2", "row_bytes", "activation", "sync", "chain"],
 "annotations": ["tile", "action"],
 "commands": [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, null, null], ...]
}
```

- Each command is its `fields` values, in the RTL's order, then its `annotations`:
  - `tile`, which the RTL's `Command.tile` carries, not sent to the hardware;
  - `action`, the CSP action it implements, which joins it to the program and to its `.dflow`
    steps.
- `bytes` is row-major and little-endian, each element in its type's width, as hex. It is `null`
  when no values were given.
- `results[].bytes` are L-B's values: what the run must leave in L3.
- `ids` count from 0, modulo 256, as the RTL's issuer numbers them.
- Changing `fields` or a value's encoding needs a new version.

`kpu-run --program p.csp --level behavioral --tile-commands out.json` writes it.

## 5. The reference executor

`tools/trace/tile_cmds_run.py` runs a stream the way the tile's controller does, functionally:
- L3 bytes, scratchpad rows and accumulator rows;
- LOAD's corner turn, COMPUTE's output-stationary reduction (int32 wrap), STORE into L3;
- the slot protocol on every command.

It reports a broken credit, an address outside L3 or a result that is not L-B's. It is
kpu-sim's reading of the RTL's semantics, not the RTL. `test_tile_cmds_run.py` breaks each rule
and requires it to fail.

## 6. Not in version 1

- **The RTL running it.** That is the generator's runtime, generator#49. "Done when" (both
  sides' `.dflow` results agree) needs it.
- **A host DMA stream**, so a program can reuse L3 slots.
- **Weight- and input-stationary, accumulate onto C (an accumulator LOAD), epilogues** (STORE's
  `activation`), **several tiles** (`DistributedGemm`'s ranks).

## 7. Verification

```bash
build/tests/program/test_tile_commands
build/tools/kpu-run/kpu-run --program tests/program/tilecmd/matmul_int8_8_t4.csp --level behavioral \
    --tile-commands mm8.json
python3 tools/trace/tile_cmds_run.py mm8.json          # exit 0: protocol held, results are L-B's
python3 tools/trace/test_tile_cmds_run.py mm8.json
(cd build && ctest -L tilecmd)
```

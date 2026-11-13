# Typed Value Path: Number Systems from Universal (#394)

**Status:** done: steps 1 (the type in `.csp`, the registry, L0, L-B and L-T1, the generator), 2 (L-CA) and 3 (`.dflow`)
**Issue:** #394 (v0.9.0), the generator side's branes-ai/embodied-ai-soc-generator#51
**Related:** #384 (end-to-end tests, which this unblocks), #406 step 4 (bit-equal bias and
accumulate), `docs/plans/shared-tile-flow-record.md` §7 (the record's element types)

## 1. The problem

The KPU RTL computes int8 × int8 → int32. kpu-sim's value path is fp32 only:
- `TensorOperand::values` is `std::vector<float>`;
- L-B, L-T1 and L-CA move and compute floats;
- `tile_check` refuses any other format.

So kpu-sim can't be the RTL's value reference. More broadly, it can't be used to experiment with
the KPU's arithmetic, which is half the reason for a hardware simulator.

## 2. The decision: number systems come from Universal

kpu-sim already depends on stillwater-sc/universal: it is fetched at v3.91 in
`cmake/Dependencies.cmake` and used by `include/sw/kpu/quantization/`. Universal provides the
arbitrary user-defined arithmetic:
- `integer<N>`, which wraps modulo 2^N as the RTL's accumulator does (checked: max + 1 = −2³¹);
- `cfloat<…>`: fp32, fp16, fp8 E4M3/E5M2, and the rest;
- `bfloat16`, `posit<N,es>`, `fixpnt`, `lns`, `takum`.

The value path carries an operand's **number system**, named as in **universal_dtypes**. That
package registers the same Universal types as NumPy dtypes, so the Python side gets a NumPy
reference that uses the same names: tests, `dflow_check.py`, and the RTL's tools.

- **A program declares each operand's type:** `tensor A[16,16] int8 tile 4x4 in;`. A
  declaration without a type is `fp32`, so every existing program, trace and golden file is
  unchanged.
- **A compiled registry,** modelled on universal_dtypes', maps a name to its Universal type. The
  registry gives:
  - the bit width;
  - `encode(double) -> word`, rounding as Universal converts;
  - `decode(word) -> double`, for checks, printing and synthesis.
- **The fabric's MAC is typed.** A gemm reads A and B in their type and accumulates in the
  accumulator's type, which is the result operand's type (`C int32`). The tile kernels are
  instantiated per supported (input, accumulator) pair and dispatched at run time:
  - `(fp32, fp32)`: today's arithmetic, bit for bit;
  - `(int8, int32)`: this issue.

  Adding a pair is one table entry: for example `(bfloat16, fp32)`, `(fp8e4m3, fp32)` or
  `(posit16, posit16)`. A pair that isn't registered is refused by name.
- **One path for every level.** L-B, the L0 replay (`TileProgramReference`), L-T1 and L-CA call
  the same kernels. Their values therefore agree bit for bit, for every type, and the existing
  "same computation, same bits" rule holds.
- **Checks by type:** integers are compared bit for bit; floating types use the proximity bound of
  ADR 0001 D5, per format.

## 3. Storage: typed 32-bit words

Each element keeps its 4-byte slot. A non-fp32 element holds its Universal bit pattern,
zero-extended, read and written through typed accessors (`word`, `set_word`). fp32 words are the
floats themselves.

So code that only moves tiles (L3, L2, L1, the executor's payloads, copies) is unchanged. Only
code that interprets values decodes by type: the kernels, checks, synthesis, `.l0`, `.dflow`
and printing. That is the interpretation layer `tile_check.hpp` already describes ("each
tile's stored bits are read ... in the format the program intends").

**The limit:** formats up to 32 bits. Wider ones (posit64, the cascades, a quire accumulator)
need a wider word. That is a later step, the typed element buffer, kept out of #394.

## 4. The pieces

| Piece | Change |
|---|---|
| `.csp` | an optional type after the shape: `tensor A[M,K] int8 tile …`; the validator checks a gemm's (input, accumulator) pair is registered |
| `TensorOperand` | `std::string type = "fp32"`, `word`/`set_word`, `value(r,c)` (decoded) |
| `numeric/` registry | names → Universal types, encode/decode, the MAC table |
| L0 | `OPERAND … type=int8`; values written as decoded integers or exact floats. Container format 1.3.0 and `MIN_CONSUMER 1.3.0`, only when an operand isn't fp32: a 1.2.0 reader would ignore the attribute and read the words as floats. The op set is unchanged |
| L-B, the replay, L-T1 | the typed kernels; the fabric accumulator held in the accumulator's type |
| L-CA | the executor's matmul uses the same MAC; payloads carry words |
| `tile_check` | decodes by type; integers exact |
| `csp_inputs` | synthesizes in range: int8 in [−128, 127] |
| `.dflow` v6 | the C++ writer and reader carry int8 and int32 in the byte payload (the format has them) |
| `kpu-csp-gen` | `--dtype int8` writes an int8 matmul with an int32 C |

## 5. Steps

1. The type in `.csp`, `TensorOperand`, the registry (fp32, int8, int32), L0 and L-B, plus
   `kpu-csp-gen --dtype int8`. The tests:
   - values bit for bit against an int32 reference computed from the inputs, including a wrap
     case;
   - the trace replays bit for bit;
   - `.l0` round-trips;
   - fp32 programs unchanged;
   - an unregistered pair is refused.

   L-T1 carries the typed values too, since it takes them from L-B. L-CA refuses a typed program
   until step 2, and the `.dflow` writer refuses one until step 3.

   **Done:** `tests/program/test_typed_values.cpp`.
   - The registry and the wrapping MAC.
   - The syntax and its refusals.
   - Every init form in both loop orders, plus a C near INT32_MAX that wraps on every element,
     all bit for bit.
   - The trace's replay and the `.l0` round trip; an out-of-range value is refused.
   - L-T1 agrees, and the check reads an integer exactly.
2. L-CA carries the values, through the same MAC: the executor's payloads carry words, and its
   matmul calls the registered MAC. The test is L-CA bit-equal to L-B on int8.

   **Done:**
   - `MatMulComputeSpec::mac` holds the registered MAC. The CSP driver sets it on a typed gemm.
   - The executor's typed matmul streams the init, the accumulator and the k-slices as before. It
     computes on words, and every copy is a `memcpy`.
   - The driver writes payloads and reads results back as words.
   - Tests:
     - L-CA equals L-B on every operand, for every init form in both loop orders;
     - C seeded with words shaped as signalling NaNs survives the run.
     - fp32 keeps the executor's own arithmetic, unchanged.
3. The C++ `.dflow` writer and reader carry int8 and int32; `dflow_check.py` accepts the record.
   That meets #394's done-when.

   **Done:**
   - Each operand's manifest `element` is its number system.
   - The payload holds each element at its width, little-endian: int8 is one byte, int32 and fp32
     are four. The values cap counts those bytes.
   - The reader accepts any element that both the format and the registry name (fp32, int8, int32),
     and decodes it back into words.
   - fp32 bundles are unchanged byte for byte.
   - Tests:
     - the C++ round trip, word for word, with `elements.bin` exactly the payload's bytes;
     - every final equal to L-B's;
     - the reader refuses fp8 (format-only) and non-dense domains;
     - a ctest pair: `kpu-run --dflow` on `tests/program/csp/matmul_int8_16_t4_bias_row.csp`, then
       `dflow_check.py`, which passes. Its decoded A, B and bias row reproduce C in Python,
       wrapped to int32.

Later, outside #394:
- more pairs (bfloat16, fp8, posit);
- a Universal upgrade from v3.91;
- the typed element buffer for formats wider than 32 bits;
- **traffic sized by number system.** The timed levels size an element as the device's
  `element_bytes` (L-T1) or 4 bytes (L-CA's driver). An int8 tile therefore moves 4× the bytes the
  RTL moves. The values are right; the timing of an int8 program is an fp32 program's. Sizing
  per operand from `NumberSystem::bits` belongs with the RTL timing cross-check.

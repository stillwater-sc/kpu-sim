# E4 In-Flight Layout-Transform Pattern (T1)

**Issue:** #109, sub-issue T1 of #73

**Baseline:** `f790e9c7ccede501cd0bf4e318d2919949905085` (`origin/main`, 2026-07-15)

**Status:** implementation-ready design; implementation is T2-T5

## 1. Decision summary

E4 will be a bounded BlockMover capability, not an arbitrary permutation engine. A
single versioned `LayoutTransformDescriptor` carries common shape, element, source,
destination, and layout-key fields. The operation kind is a closed enum with four
E4 core semantics:

1. physical rank-2 transpose;
2. metadata-only compact reshape performed while the bytes move;
3. physical attention-head split and merge;
4. physical non-overlapping 2-D patchify with deterministic zero padding.

The descriptor does not contain a programmable axis permutation, index bytecode,
or arbitrary source/destination strides. Compact row-major byte strides are derived
from the shapes and element size and are validated. The common BlockMover machinery
may share admission, scratch, timing, atomic completion, tracing, and serialization,
but each named operation has its own validator and independent index map.

Every E4 operation is an L3-to-L2 movement into separate destination storage. Even
metadata-only reshape is not a zero-copy alias. A physical transform materializes
its complete output in private reorder storage, then makes the destination payload
and destination TagCAM key visible atomically. Source storage remains resident until
that completion point.

## 2. Grounded current-state inventory

The paths requested as `docs/kpu-execution-model.md` and
`docs/SIMULATION_FIDELITY_FRAMEWORK.md` no longer exist at this baseline. Their
current, reviewed locations are `docs/01-architecture/kpu-execution-model.md` and
`docs/02-simulation/fidelity-framework.md`. The former defines the current
credit/TagCAM dataflow and the latter requires semantic agreement between behavioral
and timing levels.

The labels in this table describe payload behavior, not names, comments, or intent.

| Capability | Classification | Exact current path and symbol | Observed behavior |
|---|---|---|---|
| Explicit BlockMover transpose in `BehavioralProgramExecutor` | **functionally implemented** | `src/software/isa/behavioral_program_executor.cpp:388` `dispatch_bm_move`; transpose branch at `:404` | Copies each `element_size` byte group from `(r,c)` in contiguous `[height,width]` to `(c,r)` in `[width,height]`. It is synchronous and has no destination-shape descriptor. |
| Temporal byte-holding BlockMover transpose | **functionally implemented** | `src/models/temporal/datamovement/block_mover.cpp:217` completion transform; transpose loop at `:240` | Materializes a complete transposed byte vector and commits it at modeled completion. Existing tests cover square tiles, not the complete E4 contract. |
| CSP/schedule transpose | **declaration/annotation-only** | `include/sw/kpu/timing/schedule/schedule_generator_interface.hpp:90` `ScheduleOperation::transpose`; `include/sw/kpu/timing/block_mover_process.hpp:95` `schedule_move`; pending flag erased at `:472`; `include/sw/kpu/timing/concurrent_timing_executor.hpp:1337` `apply_payload_event` | The boolean reaches BlockMover scheduling, but no timing or payload path reads it. `apply_payload_event` copies L3 payload to L2 unchanged. There is no transformed destination identity. |
| GEMM B transpose | **declaration/annotation-only** | `include/sw/kpu/timing/schedule/matmul_schedule_generator.hpp:263`, `:313`, `:381`, and `:428` emit `move(..., true)`; `include/sw/kpu/timing/concurrent_timing_executor.hpp:1230` functional matmul | All strategies annotate B moves as transposed. The functional consumer still reads canonical row-major `[K,N]`, so present results are correct precisely because the boolean is ignored. This is not genuine end-to-end B transpose. |
| `BM_RESHAPE_TILE` in `BehavioralProgramExecutor` | **absent** | opcode dispatch `src/software/isa/behavioral_program_executor.cpp:100`; default path through `:261` | There is no reshape case. The instruction silently falls through without moving payload. |
| `BLOCK_RESHAPE` in the temporal BlockMover | **declaration/annotation-only** | `src/software/isa/program_executor.cpp:219` maps the opcode; `src/models/temporal/datamovement/block_mover.cpp:230` reshape case | It performs the identity byte copy. No destination shape is carried, checked, or attached to the result. |
| `BM_RESHAPE_TILE` in concurrent and transactional ISA timing lanes | **timing-only** | `src/software/isa/concurrent_executor.cpp:217` and `:503`; `src/software/isa/transactional_program_executor.cpp:303` and `:665` | It is grouped with ordinary move/transpose for latency and completion. No payload reshape occurs. |
| ISA transform description | **declaration/annotation-only** | `include/sw/kpu/isa/data_movement_isa.hpp:50` `DMOpcode`; `:174` `Transform`; `:214` `BlockMoverOperands`; `:367` `AutoBlockMoverOperands` | Explicit operands carry only one `[height,width]`, `element_size`, and a small transform enum. AUTO operands also declare `Transform`, but cannot describe either shape. Neither form can encode destination rank/shape, head count, patch geometry, or distinct layout identity. |
| Assembler transform support | **declaration/annotation-only** | `src/software/isa/assembler.cpp:620` `parse_bm_move_tile`, `:645` transpose, `:658` reshape | It parses names and legacy operands but does not validate opcode/transform agreement, products, shapes, overflow, or aliases. AUTO movement at `:1095` is forced to identity. |
| Program serialization | **declaration/annotation-only** | `include/sw/kpu/isa/program_serializer.hpp:21` version 3; `src/software/isa/program_serializer.cpp:157` and `:284` binary operand paths; `:574` and `:726` JSON paths | The legacy enum round-trips by raw value. It has no layout descriptor or semantic validation. AUTO operands are not serialized. `tests/isa/test_serialization.cpp:54` exercises identity movement only. |
| AUTO BlockMover transform | **declaration/annotation-only** | `src/software/isa/assembler.cpp:1095` `parse_bm_move_tile_auto`; `src/software/isa/behavioral_program_executor.cpp:906` `dispatch_bm_move_auto` | The assembler forces `Transform::IDENTITY`; behavioral AUTO movement copies bytes unchanged and never reads `ops.transform`; serialization omits the operand variant. |
| Standalone behavioral BlockMover transpose | **functionally implemented** | `src/models/behavioral/datamovement/block_mover.cpp:42` transform dispatch; `:168` `copy_block_transpose` | Performs physical byte transpose and supports explicit source/destination row strides. It is a separate model from the CSP value path. |
| Harness transpose | **functionally implemented** | `include/sw/kpu/harness/harness_config.hpp:98` harness `Transform`; `src/harness/block_mover_harness.cpp:358` request lowering | The harness lowers only `Transform::TRANSPOSE` to the standalone behavioral BlockMover's transpose boolean. |
| Harness swizzle and FP16 pack/unpack names | **declaration/annotation-only** | `include/sw/kpu/harness/harness_config.hpp:99` `Transform`; `src/harness/block_mover_harness.cpp:361` lowering and `:370` statistics | `SWIZZLE_4x4`, `SWIZZLE_8x8`, `PACK_FP16`, and `UNPACK_FP16` are counted as transforms but physically execute the identity-copy branch. The FP16 names describe intended numeric conversion, not E4 spatial pack/unpack. |
| CSP destination payload creation | **functionally implemented** | `include/sw/kpu/timing/concurrent_timing_executor.hpp:1312` `copy_payload` and `:1337` `apply_payload_event` | Identity movement only: a BlockMover-complete event copies the source `TilePayload` unchanged under the same rank-2 `TileID`. |
| CSP BlockMover admission/completion | **functionally implemented** | `include/sw/kpu/timing/block_mover_process.hpp:407` source match and issue; `:451` L2 credit admission; `:239` completion | Identity timing only: requires source TagCAM arrival and an L2 credit, admits one transfer per mover, inserts the L2 tag at completion, then invalidates one L3 reference and returns its credit. There is no reorder-storage accounting. |
| CSP transform latency | **absent** | `include/sw/kpu/timing/block_mover_process.hpp:228` transfer duration | Duration is startup plus bytes/bandwidth only. Transform throughput and scratch pressure are not modeled. |
| CSP buffer-slot byte-fit validation | **absent** | `include/sw/kpu/timing/concurrent_timing_executor.hpp:96` `l3_buffer_size`; `:105` `l2_bank_size`; `include/sw/kpu/timing/block_mover_process.hpp:451` admission | Configured slot sizes are not consulted by movement admission. One descriptor consumes one credit regardless of bytes, and `size_bytes` is not checked against `height*width*element_size`. |
| Distinct source/destination layout identity | **absent** | `include/sw/kpu/timing/tag_cam.hpp:90` `insert`, `:149` `match`, and `:334` map key; `include/sw/kpu/timing/process_interface.hpp:119` `TimingEvent`, `:263` `InFlightTransfer` | TagCAM and events use only `TileID`. A transposed or head-split form can collapse onto the canonical form; duplicate handling ignores transform kind. |
| Rank-general payload | **absent** | `include/sw/kpu/timing/tile_descriptor.hpp:116` `TileDescriptor`; `:171` `TilePayload` | Both are rank-2-oriented; payload values are `float`. Element widths and destination rank/shape are not carried through CSP functional execution. |
| Partitioned L3/L2 credits | **functionally implemented** | `include/sw/kpu/timing/credit_pool.hpp:39` `PartitionedCreditPool`; `:221` `CreditPool`; `include/sw/kpu/timing/concurrent_timing_executor.hpp:646` construction | Equal A/B/C partitioning and conservation checks exist. BlockMover admission uses the tile's matrix partition. |
| Accounted L1 credits | **absent** | `include/sw/kpu/timing/streamer_process.hpp:56` `Config::l1_depth`; `include/sw/kpu/timing/concurrent_timing_executor.hpp:514` credit-pool state; `include/sw/kpu/timing/streamer_process.hpp:242` feed completion | L3 and L2 have `CreditPool` instances. L1 has a configured depth but no acquired/released, partitioned `CreditPool`. L2 is held until the final feed. |
| Conv2D im2col and filter flattening | **host-side transformation** | `include/sw/kpu/timing/schedule/conv2d_im2col.hpp:130` `im2col_nchw`; `:179` `filter_to_bw_nchw`; `:202` `conv2d_reference_nchw` | Host C++ builds `[M,K]` and `[K,Cout]` vectors before scheduling. `tests/timing/test_functional_conv2d.cpp:96` and `examples/schedule/conv2d_simulator.cpp:96` seed those materialized values into the executor. |
| Conv2D gather declaration | **declaration/annotation-only** | `include/sw/kpu/dsl/schedule.hpp:131` `LOAD_GATHER`/`Im2ColParams`; `src/schedules/conv2d_schedule.cpp:83` `load_gather`; `src/dsl/schedule_compiler.cpp:116` lowering | The compiler lowers gather to an ordinary DMA load and retains only a label. `src/software/isa/behavioral_program_executor.cpp:251` explicitly treats gather/scatter as non-functional annotations. |
| In-flight Conv2D patchify/im2col | **absent** | `include/sw/kpu/timing/schedule/conv2d_schedule_generator.hpp:271` generated dataflow; `docs/plans/e6_conv2d_pattern.md:88` current-state discussion | The generator schedules ordinary load/move/feed over already flattened A. General overlapping im2col remains an E6/DMA-gather concern, not an implemented movement transform. |
| E4-complete transform verification | **absent** | `tests/block_mover/test_block_mover_basic.cpp:116` and `:310`; `tests/timing/test_schedule_generators.cpp:53`; `tests/isa/test_serialization.cpp:54` | Existing tests cover square temporal transpose, storage of the schedule boolean, and identity serialization. There is no CSP value, rectangular/partial, malformed, round-trip, randomized, low-credit transform, or mutation coverage. |

`tests/coverage/pattern_coverage.json:308` therefore correctly records E4 design and
functional/regression closure as missing at this baseline. Some prose in older plans
describes intended gather or transpose capability; those declarations do not override
the traced payload behavior above.

The four current-state verdicts are therefore direct:

- B-tile transpose is physically functional in the explicit behavioral ISA executor,
  temporal BlockMover, and standalone behavioral BlockMover/harness paths, but not
  end-to-end in the CSP executor used by schedule integration.
- `BM_RESHAPE_TILE` is a silent no-op in `BehavioralProgramExecutor`, an identity byte
  movement without shape metadata in the temporal lane, and timing-only in the other
  ISA timing lanes. It is not a functional reshape anywhere.
- CSP timing represents transform intent with one boolean, plus an unused
  `BlockMoverProcess::Config::supports_transpose` declaration at
  `include/sw/kpu/timing/block_mover_process.hpp:49`; neither changes transform latency
  or payload.
- Conv2D patchification/im2col is performed by host preparation before CSP scheduling;
  modeled movement sees an already flattened matrix.

## 3. Semantic vocabulary and scope boundary

These terms are not interchangeable:

- **Physical transpose** changes byte positions. For compact rank-2 source `[R,C]`,
  destination `[C,R]` stores destination element `[c,r]` from source `[r,c]`.
- **Metadata-only reshape** changes rank and extents but preserves the compact linear
  element sequence exactly. In E4 it still copies bytes L3 to L2 and creates a distinct
  destination layout key; "metadata-only" describes the transform, not an alias.
- **Byte-permuting reshape** changes both shape and linear element order. That is not
  `RESHAPE` in this design. It must select a named physical operation with a defined
  map or be rejected.
- **Pack** takes a source logical shape `X`, source byte strides `ss`, and an active
  index set `A` in lexicographic logical order. Compact destination element `q` receives
  the source element at the `q`th index in `A`. **Unpack** is the inverse map into a
  declared destination shape/strides and fills every destination element outside `A`
  with a declared deterministic value. These operations change storage span while
  preserving active element bytes. General pack/unpack is excluded from E4 v1 and
  belongs with E16 spatial layout work.
- The existing harness enum names `PACK_FP16`/`UNPACK_FP16` mean intended FP32/FP16
  numeric conversion, which changes element width and storage bytes. Those declarations
  are neither functional today nor the spatial pack/unpack definition above; numeric
  conversion is outside E4.
- **Split attention heads** maps `[B,S,D]` to `[B,H,S,Dh]` where `D=H*Dh` and changes
  physical order. **Merge attention heads** is its exact inverse.
- **Patchify** maps one compact channel-first image tile `[C,H,W]` into non-overlapping
  flattened patches `[Gh*Gw,C*Ph*Pw]`, with zeros only at declared image edges.
- **Unpatchify** accepts `[Gh*Gw,C*Ph*Pw]` plus exact destination `[C,H,W]`, requires
  `Gh=ceil(H/Ph)` and `Gw=ceil(W/Pw)`, and maps
  `dst[c,gh*Ph+kh,gw*Pw+kw] = src[gh*Gw+gw,(c*Ph+kh)*Pw+kw]` only when the destination
  coordinates are in range; padded patch elements are discarded. It is a bijection on
  non-padding positions for non-overlapping patches. E4 v1 includes forward patchify
  only; unpatchify is deferred to E16.

The following are explicitly deferred:

- E16: general pack/unpack, unpatchify, depth/space rearrangement, upsample,
  interpolation, transposed-convolution scattering, and other spatial layout maps;
- E6 plus the E1 DMA path: overlapping/dilated Conv2D im2col gather from global image
  memory, because it may duplicate source elements and needs address-generation rather
  than a one-source/one-destination permutation;
- a general axis permutation, programmable gather/scatter, conversion between element
  types, fused arithmetic, QKV projection, and fused `[B,S,3D]` slicing.

## 4. Common descriptor and opcode contract

### 4.1 Closed descriptor

T2 shall introduce the following normalized in-memory concept. Names may follow local
C++ style, but fields and validation are normative.

```text
LayoutTransformDescriptor {
  version = 1
  kind: IDENTITY | TRANSPOSE_2D | RESHAPE | HEAD_SPLIT | HEAD_MERGE | PATCHIFY_2D
  src_key: PayloadKey { TileID tile; uint32 layout_id; }
  dst_key: PayloadKey { TileID tile; uint32 layout_id; }
  element_size_bytes: uint8
  src_rank: uint8
  src_shape[4]: uint32       // unused tail entries are zero
  dst_rank: uint8
  dst_shape[4]: uint32       // unused tail entries are zero
  params: closed tagged union {
    heads { uint32 head_count; uint32 head_dim; }
    patch { uint32 patch_h; uint32 patch_w; }
  }
}
```

`layout_id=0` means the canonical layout at a level. In E4 v1, `IDENTITY` requires
`dst_key == src_key`. Every other kind requires `dst_key.tile == src_key.tile` and
`dst_key.layout_id != src_key.layout_id`: a transform changes the representation of
one logical tile but cannot change its matrix or tile coordinates. A patch-cell schedule
therefore assigns a source tile coordinate to each compact cell before E4 runs. TagCAM
equality and hashing shall include both `TileID` and `layout_id`. The destination
TagCAM entry shall retain the descriptor kind and a collision-safe,
equality-comparable descriptor fingerprint
for trace/debug validation; correctness may not depend on a hash match alone.

Source/destination buffer IDs and byte offsets remain movement operands surrounding
the descriptor. Bounds are checked with overflow-safe arithmetic. Shapes, not free-form
strides, are serialized. For rank `r`, element size `e`, and compact row-major shape
`X[0..r-1]`, derived byte strides are:

```text
stride[r-1] = e
stride[i]   = stride[i+1] * X[i+1], for i = r-2 ... 0
offset(i0,...,ir-1) = sum(ij * stride[j])
bytes(X,e) = e * product(X[j])
```

All multiplication/addition uses checked `uint64` arithmetic before narrowing to
container sizes or offsets. E4 v1 accepts `element_size_bytes` in `{1,2,4,8}` and
treats each element as opaque bytes; it performs no byte-order or numeric conversion.

The opcode remains explicit and must agree with `kind`:

| Opcode | Descriptor kind |
|---|---|
| `BM_MOVE_TILE` | `IDENTITY` |
| `BM_TRANSPOSE_TILE` | `TRANSPOSE_2D` |
| `BM_RESHAPE_TILE` | `RESHAPE` |
| `BM_HEAD_SPLIT_TILE` (new) | `HEAD_SPLIT` |
| `BM_HEAD_MERGE_TILE` (new) | `HEAD_MERGE` |
| `BM_PATCHIFY_TILE` (new) | `PATCHIFY_2D` |

There is no `BM_ARBITRARY_PERMUTE`. AUTO movement remains identity-only in v1; an AUTO
instruction requesting a transform is malformed until an envelope-aware AUTO contract
is designed.

### 4.2 Serialization compatibility

New opcode values are append-only after the highest v3 opcode. T2 shall not insert the
three new opcodes into the middle of `DMOpcode` and renumber raw v3 `uint8_t` values;
either assign every existing value explicitly or append without changing any existing
number. Binary v4 writes the common descriptor fields in the declaration order above,
including all four shape entries, followed by the fixed two-word kind parameter payload
(zeros for kinds without parameters). JSON v4 uses a `layout_transform` object with
keys `version`, `kind`, `src_layout_id`, `dst_layout_id`, `element_size_bytes`,
`src_rank`, `src_shape`, `dst_rank`, `dst_shape`, and a kind-specific `params` object.
The surrounding operand continues to serialize source/destination component IDs,
offsets, `TileID`, and buffer selection. Both readers run the same checked validator
after decoding.

The v3 normalization policy is exact:

| v3 opcode | v3 `Transform` | v4 result |
|---|---|---|
| `BM_MOVE_TILE` or `BM_WRITEBACK_TILE` | `IDENTITY` | Normalize to compact rank-2 identity using legacy `height,width,element_size`; source and destination layout ID 0. |
| `BM_TRANSPOSE_TILE` | `TRANSPOSE` | Reject with `legacy transpose lacks downstream layout identity`. v3 FEED/compute operands cannot name the transformed layout, so normalizing only the producer would be unsafe. |
| `BM_RESHAPE_TILE` | `RESHAPE` | Reject with `legacy reshape lacks destination shape`; no shape may be invented. |
| any move/writeback opcode | `SHUFFLE` | Reject as unsupported; current execution had no defined physical map. |
| any other opcode/transform pairing, including `BM_MOVE_TILE+TRANSPOSE` or `BM_TRANSPOSE_TILE+IDENTITY` | any | Reject as an opcode/transform mismatch. |

No well-formed v3 field encoding exists for AUTO operands; the current writer records
the operand tag but omits the operand fields. The v4 writer rejects a non-identity AUTO
transform rather than dropping it. A v4 reader rejects unknown enum values, ranks,
params, and trailing/non-zero unused shape or parameter fields.

### 4.3 General rules shared by every operation

- Ranks and every used extent are positive; unused shape entries are zero.
- Source and destination memory ranges must be disjoint. In-place and partially
  overlapping transforms are rejected, including reshape.
- Source and destination compact storage must fit their configured buffer-slot byte
  capacities. E4 v1 uses one source tile credit and one destination tile credit per
  descriptor. A larger logical operation is tiled by T3.
- Slot bytes beyond the descriptor's destination storage are set to zero before atomic
  commit and are not part of the logical shape.
- There is no partial destination visibility. A destination payload is absent until
  all required source bytes have arrived, all mapped destination bytes and padding
  have been produced in private storage, and timing completion has been reached.
- A BlockMover request targeting an already resident or reserved `dst_key` is rejected.
  Reuse is represented by explicit downstream references to one completed destination,
  not by issuing duplicate transforms.
- A descriptor is validated before queue insertion or resource acquisition. A malformed
  descriptor produces a deterministic executor error naming the failed condition and
  changes no payload, TagCAM entry, trace completion count, or credit count.
- A runtime integrity failure after admission atomically releases the destination and
  reorder reservations, leaves the source reference resident, emits one error event,
  and never inserts the destination key. It may not degrade to identity or no-op.

## 5. Normative operation maps

Let `copy_e(dst,src)` copy exactly `e=element_size_bytes` opaque bytes.

### 5.1 `TRANSPOSE_2D`

- Source rank/shape: rank 2, `[R,C]`.
- Destination rank/shape: rank 2, exactly `[C,R]`.
- Source strides: `[C*e,e]`; destination strides: `[R*e,e]`.
- Map for `0 <= r < R`, `0 <= c < C`:
  `dst[c,r] = src[r,c]`.
- Bytes: `Bsrc=Bdst=R*C*e`; no logical padding is synthesized.
- Partial tiles: any positive `R,C` are valid, including non-square edge tiles.
- Round trip: applying transpose to `[C,R]` returns the original bytes and `[R,C]`.
- Completion: every one of the `R*C` elements is present exactly once in private
  destination storage and the common timing/resource completion conditions hold.

### 5.2 `RESHAPE`

- Source rank/shape: rank 1 through 4, `S`.
- Destination rank/shape: rank 1 through 4, `D`.
- Validation: `product(S)=product(D)`.
- Strides: the compact derived strides for each shape.
- Map: for linear element index `q` in `[0,product(S))`, destination linear element
  `q` receives source linear element `q`; equivalently the entire byte sequence is
  copied unchanged.
- Bytes: `Bsrc=Bdst=product(S)*e`; no logical padding is synthesized.
- Partial tiles: valid whenever their actual positive extents satisfy equal products;
  nominal untiled extents are irrelevant.
- Round trip: reshape from `S` to `D`, then `D` to `S`, is byte-identical.
- Completion: all bytes have moved into distinct destination storage, destination shape
  metadata is attached to `dst_key`, and common completion conditions hold.

Any unequal product, non-compact stride request, or requested element reordering is a
malformed reshape. It does not silently become identity movement.

### 5.3 `HEAD_SPLIT` and `HEAD_MERGE`

For split:

- Source rank/shape: rank 3, `[B,S,D]`.
- Destination rank/shape: rank 4, exactly `[B,H,S,Dh]`.
- Parameters: positive `H,Dh` and exact `D=H*Dh`.
- Source strides: `[S*D*e,D*e,e]`.
- Destination strides: `[H*S*Dh*e,S*Dh*e,Dh*e,e]`.
- Map: `dst[b,h,s,d] = src[b,s,h*Dh+d]`.

For merge, source/destination shapes and the map reverse:

```text
src [B,H,S,Dh] -> dst [B,S,H*Dh]
dst[b,s,h*Dh+d] = src[b,h,s,d]
```

For both, `Bsrc=Bdst=B*S*H*Dh*e`, with no logical padding. Partial batch and sequence
tiles are valid. A partial embedding/head dimension is not valid: a descriptor must
contain all `H*Dh` elements for each included token. The scheduler must retile or defer
until that complete span is resident. Split followed by merge with the same `H,Dh` is
an exact round trip. Completion requires every element to have been mapped exactly
once plus the common completion conditions.

### 5.4 `PATCHIFY_2D`

One descriptor operates on one compact batch item. A batch is a schedule-level loop.

- Source rank/shape: rank 3 channel-first `[C,H,W]`.
- Parameters: positive patch height `Ph` and width `Pw`.
- Derived grid: `Gh=ceil(H/Ph)`, `Gw=ceil(W/Pw)`.
- Destination rank/shape: rank 2, exactly
  `[Gh*Gw,C*Ph*Pw]`.
- Source strides: `[H*W*e,W*e,e]`.
- Destination strides: `[C*Ph*Pw*e,e]`.
- Patch row `q=gh*Gw+gw`; flattened feature
  `d=(c*Ph+kh)*Pw+kw`.
- Map:

```text
y = gh*Ph + kh
x = gw*Pw + kw
dst[q,d] = src[c,y,x]  if y < H and x < W
dst[q,d] = all-zero e-byte element otherwise
```

Patchify is non-overlapping: each source element appears in exactly one non-padding
destination position. It may synthesize edge padding only. Define:

```text
Nsrc = C*H*W
Ndst = Gh*Gw*C*Ph*Pw
Npad = Ndst-Nsrc
Bsrc = Nsrc*e
Bdst = Ndst*e = Bsrc + Npad*e
```

A large image may be tiled into independent patch-cell groups. Every group must contain
compact source bytes already resident in L3; extracting a non-contiguous global image
window is a DMA/gather scheduling responsibility, not hidden inside E4. For an edge
cell, the local source shape may be `[C,valid_h,valid_w]` and the destination is
`[1,C*Ph*Pw]`; the same formula applies with the local origin at zero. Completion
requires every source element and every declared zero-padding element to be present in
private destination storage plus the common completion conditions.

## 6. Dataflow, ownership, and exact completion

The BlockMover owns transform admission, physical transformation, private reorder
storage, timing, atomic destination publication, source release, and transform trace
events. The schedule generator owns tiling, source/destination keys, descriptors,
partition selection, and proof that the requested burst fits the declared envelope.
The CSP executor payload store owns the actual L3/L2 byte vectors; it applies the
BlockMover's completed private output rather than independently recomputing a different
map.

The state transition for one descriptor is:

1. **Validate:** reject malformed fields, range overlap, overflow, slot overflow,
   conflicting destination key, or unsupported kind before resource changes.
2. **Wait for source:** `src_key` must match an arrived L3 TagCAM entry and the complete
   source payload bytes must be present. A queued operation cannot read source bytes
   merely because it was scheduled.
3. **Atomic admission:** claim one arrived, unclaimed source TagCAM reference without
   removing its resident credit; reserve all required destination L2 credits, one
   transform slot, and the complete reorder-storage allocation. If any item is
   unavailable, claim/reserve none and remain queued. The source L3 entry remains held.
4. **Transform:** read only arrived source bytes and build private output. No destination
   TagCAM entry or L2 payload is visible during this phase. A v1 mover has one active
   transform; multiple movers share global reorder/credit accounting.
5. **Complete atomically at cycle `c_done`:** only when the modeled duration has elapsed,
   the private output contains every destination byte including deterministic padding,
   and the destination reservation still exists: commit the output to L2, insert
   `dst_key`, convert the reserved L2 credit(s) to visible resident credit(s), invalidate
   exactly one source reference, return its L3 credit if that was the last reference,
   release the source-reference claim, reorder storage, and transform slot, and emit
   exactly one completion event. These changes are one observable transition.

Source credit therefore cannot return before the transform has finished. It can return
at `c_done`, before any downstream consumer has consumed the L2 destination. The L2
credit remains held until the final downstream feed/reference releases it. The reorder
allocation is private and cannot serve as destination buffering.

The timing model for transform `t` is:

```text
beta_r = source bytes readable per cycle
beta_w = destination bytes writable per cycle
rho_k = destination elements transformable per cycle for kind k
T_t = L_start + max(ceil(Bsrc_t/beta_r),
                    ceil(Bdst_t/beta_w),
                    ceil(Ndst_t/rho_k))
c_done = c_start + T_t
```

The timing model assumes source read, destination write, and named transform are fully
pipelined after one shared `L_start`; that overlap is why their costs use `max` rather
than a sum. `c_start` is the atomic-admission cycle, no destination is visible during
that cycle, and completion is the single transition at the first cycle satisfying
`current_cycle >= c_done`. `L_start`, both bandwidths, and each closed-kind throughput
are explicit configuration. For identity/reshape, `rho_k` may be configured not to
limit bandwidth, preserving the existing startup-plus-transfer model. Patch padding
still consumes destination write and transform throughput. A future sequential engine
must declare a different summed timing model rather than silently reusing this formula.

Every pipeline consumer uses `PayloadKey`, not bare `TileID`: MOVE names both keys;
FEED, resident dependencies, compute dependencies, functional payload maps, and
matmul A/B operand lists name the exact layout they require. Loading produces the
declared canonical source key, and a transform completion produces only `dst_key`.
This propagation is mandatory; changing only TagCAM storage would still allow a
consumer to accept the wrong bytes.

### TagCAM and duplicate policy

TagCAMs shall key entries by `PayloadKey`, not bare `TileID`. Pending destinations use
the same key space, so two operations cannot race to publish one key. A descriptor
fingerprint mismatch for an expected key is an error. BlockMover transform requests do
not perform implicit duplicate coalescing. If several consumers need one transformed
payload, the schedule establishes multiple consumer references to the one completed
entry. This prevents transform annotations from being lost through current bare-tile
duplicate handling.

For each source key `k`, TagCAM source references also obey
`ref_count(k)=unclaimed(k)+claimed(k)`. Admission moves one reference from unclaimed to
claimed without changing `ref_count` or its one resident credit. Successful completion
removes one claimed reference and decrements `ref_count`; error rollback moves the
claim back to unclaimed. The resident L3 credit returns only when the resulting
`ref_count` is zero.

### Backpressure and deadlock

- Destination-credit, reorder-storage, and transform-slot acquisition is all-or-none;
  partial acquisition cannot form a wait cycle.
- The source is held while waiting for execution, but a queued operation does not hold
  destination or reorder resources. Once admitted it holds all resources needed to
  reach completion and never waits for destination consumption.
- Publication does not require an L1 credit. Downstream feed may wait for L1, while the
  completed tile safely occupies its accounted L2 credit.
- Schedules reserve the output/drain headroom specified below. If the constructive
  bound is zero, the generator refuses the schedule rather than emitting a burst that
  can consume every partition credit.
- A single full-output scratch allocation is required in v1. Double buffering is
  achieved by two transform slots and two scratch allocations, not by exposing half
  an output. Streaming fragments are a future optimization only if they preserve the
  same atomic publication and resource equations.

## 7. Resource envelope

### 7.1 Definitions and conservation

For level `l` in `{3,2,1}` and credit partition `p`, define:

```text
C[l,p] = configured credit capacity
a[l,p] = currently available credits
v[l,p] = credits held by visible resident payloads
q[l,p] = credits reserved by in-flight work not yet visible
C[l,p] = a[l,p] + v[l,p] + q[l,p]
```

This equation is checked after every issue, completion, feed, drain, refusal, reset,
and error path. In shared-pool mode there is one partition. In partitioned mode the
equation holds independently for the A, B, and C partitions selected by the payload's
`TileID::matrix`, and summing all partitions equals the physical pool capacity. T2
must add accounted L1 credits; the current executor only accounts L3 and L2.

For v1, source partition `ps` and destination partition `pd` are derived from
`src_key.tile.matrix` and `dst_key.tile.matrix`. The identity rule in section 4 requires
the same `TileID`, so `ps=pd`; both symbols remain in the equations to make the two
lifetimes explicit. There are no free-form partition fields in the descriptor.

Let `U3`, `U2`, and `U1` be payload bytes represented by one credit at each level. For
descriptor `t`:

```text
s_t = ceil(Bsrc_t/U3)       source-tile credits
d_t = ceil(Bdst_t/U2)       destination-tile credits
Bfeed_t = Bdst_t            E4 v1 feeds the complete transformed tile
l_t = ceil(Bfeed_t/U1)      maximum simultaneous downstream L1 credits
r_t = align_up(Bdst_t, W_r) private reorder bytes
```

`W_r` is the reorder-storage allocation quantum. E4 v1 requires `s_t=d_t=1`; otherwise
validation refuses the descriptor and T3 tiles it. `l_t` is a schedule-envelope term,
not a BlockMover admission reservation. The transform completes without waiting for L1.

For global reorder storage and transform engines:

```text
C_R = a_R + sum(r_t for admitted t)
F   = a_F + n_in_flight
n_in_flight <= number_of_block_movers
```

`C_R` is bytes, `F` is transform slots, and both conservation equations include error
rollback. A transform slot may not be overcommitted merely because another operation
is waiting on timing completion.

### 7.2 Atomic admission and concurrency

Let required free headroom after admission be `h[l,p]`, and define saturated free
capacity `free[l,p]=max(0,a[l,p]-h[l,p])`. Let `R_ready[p]` count arrived,
unclaimed source references in partition `p`, and let `M_idle` be the number of
currently idle BlockMovers capable of this kind. Descriptor `t` may start iff:

```text
source_arrived(src_key)
source_reference_unclaimed(src_key)
a[3,ps] already accounts for the resident source; v[3,ps] >= s_t
a[2,pd] >= h[2,pd] + d_t
a_R >= r_t
a_F >= 1
M_idle >= 1
dst_key is neither resident nor reserved
```

The source credit is not newly acquired by the BlockMover; it was acquired by the
upstream load and remains in `v[3,ps]`. Admission marks one source reference claimed,
moves `d_t` from `a[2,pd]` to `q[2,pd]`, subtracts `r_t` from `a_R`, and subtracts one
from `a_F` atomically. Error rollback unclaims that reference without invalidating it.

For identical descriptors under current availability, the exact simultaneous-start
bound is:

```text
f_max = min(M_idle,
            a_F,
            floor(a_R/r_t),
            floor(free[2,pd]/d_t),
            R_ready[ps])
```

For mixed descriptors, admit the longest queue prefix whose sums satisfy:

```text
every src_key in the prefix is arrived and has an unclaimed reference
count(t by ps) <= R_ready[ps]  for every ps
sum(d_t by pd) <= free[2,pd]  for every pd
sum(r_t)       <= a_R
count(t)       <= min(a_F, M_idle)
```

There is no greedy partial reservation: if the next descriptor breaks any inequality,
it waits with no new resources held.

### 7.3 Constructively safe burst bound

The schedule generator must account for the source working set, transformed L2 working
set, downstream L1 window, reorder storage, and transform slots. For a homogeneous
burst with per-transform requirements `(s,d,l,r)`, source/destination headroom
`h3,h2,h1`, `free3=max(0,a[3,ps]-h3)`, `free2=max(0,a[2,pd]-h2)`,
`free1=max(0,a[1,pd]-h1)`, and `F_avail=min(a_F,M_idle)`, the conservative bound is:

```text
b_safe = min(floor(free3/s),
             floor(free2/d),
             floor(free1/l),
             F_avail,
             floor(a_R/r))
```

The L3 term applies to distinct sources not already resident for the burst. If they
are already resident, replace it with the number of arrived, unclaimed source
references, checked by exact key. For mixed sizes, let `P` be an ordered candidate
prefix; `New(P,p)` be the set of distinct source keys in partition `p` that require a
new L3 load; `uses(P,k)` be the number of transform requests in `P` that claim source
key `k`; and `refs(k)` be the source-reference count declared by its upstream load.
The exact conservative schedule bound is the largest prefix `P` satisfying all of:

```text
for every source partition ps:
  sum(s_k for k in New(P,ps)) <= free3[ps]
for every source key k:
  uses(P,k) <= refs(k)
for every destination partition pd:
  sum(d_t for t in P with destination pd) <= free2[pd]
  sum(l_t for t in P with destination pd) <= free1[pd]
sum(r_t for t in P) <= a_R
|P| <= a_F
|P| <= M_idle
```

At runtime, every request in the admitted sub-prefix must additionally have its exact
source key arrived and an unclaimed reference, as specified in section 7.2. A failed
inequality terminates the prefix before that descriptor; it does not partially reserve
it.

If requested burst `b_req` exceeds the bound, emit `min(b_req,b_safe)` and serialize
the reduced value in schedule metadata and trace output. If `b_safe=0`, refuse with a
diagnostic naming every limiting term. Do not apply the current unconditional
`max(1,...)` behavior from
`include/sw/kpu/timing/schedule/schedule_generator_interface.hpp:34`; a mathematical
zero is a refusal, not permission to oversubscribe. The existing
`per_matrix_burst_share(C3,C2)=max(1,min(floor(C3/4),floor(C2/4)))` is insufficient
because it omits L1, reorder bytes, transform slots, per-partition headroom, and
source/destination byte asymmetry.

## 8. Worked examples

### 8.1 GEMM B-tile transpose

Take a `float32` B tile in canonical K-major order:

```text
src shape [K,N] = [2,3]
src strides = [12,4] bytes
src = [[b00,b01,b02],
       [b10,b11,b12]]

dst shape [N,K] = [3,2]
dst strides = [8,4] bytes
dst = [[b00,b10],
       [b01,b11],
       [b02,b12]]
Bsrc=Bdst=r=24 bytes; s=d=1 when U3,U2 >= 24
```

The downstream compute contract must consume the distinct B-transposed layout and read
`dst[n,k]`; it must not continue indexing the bytes as `[k,n]`. T3 must update both the
matmul schedule and functional consumer together. In particular, blindly activating
the existing `move(..., true)` annotations would break current GEMM and Conv2D values,
because current payloads such as `B_w` are already seeded `[K,N]`/`[K,Cout]` and the
consumer assumes that order.

### 8.2 QKV attention-head split and merge

For one tensor (Q, K, or V), use `B=1`, `S=2`, `D=8`, `H=2`, `Dh=4`, and `e=2`:

```text
src [1,2,8], strides [32,16,2]
dst [1,2,2,4], strides [32,16,8,2]
dst[0,h,s,d] = src[0,s,4*h+d]
Bsrc=Bdst=r=32 bytes; s=d=1 when U3,U2 >= 32
```

The destination order groups all sequence positions for head 0, followed by all
sequence positions for head 1. `HEAD_MERGE` with the same `H=2,Dh=4` reconstructs the
original 32-byte sequence exactly. A tile containing only six of the eight embedding
elements for a token is refused; the scheduler must assemble a complete D span.

### 8.3 ViT patchification with non-aligned edges

For one channel-first image `src [C,H,W]=[3,5,7]`, `Ph=Pw=4`, and `e=2`:

```text
Gh=ceil(5/4)=2
Gw=ceil(7/4)=2
dst shape = [4, 3*4*4] = [4,48]
Bsrc = 3*5*7*2 = 210 bytes
Bdst = 4*48*2 = 384 bytes
padding = (192-105) elements = 174 zero bytes
```

The four patch rows have these active/padding counts:

| Patch `(gh,gw)` | Valid source shape | Active elements | Zero elements | Source bytes | Destination/reorder bytes |
|---|---:|---:|---:|---:|---:|
| `(0,0)` | `[3,4,4]` | 48 | 0 | 96 | 96 |
| `(0,1)` | `[3,4,3]` | 36 | 12 | 72 | 96 |
| `(1,0)` | `[3,1,4]` | 12 | 36 | 24 | 96 |
| `(1,1)` | `[3,1,3]` | 9 | 39 | 18 | 96 |

Across the operation, the 105 source elements appear exactly once and the remaining
87 destination elements are zeros. No edge source byte is read outside `[3,5,7]`.
With sufficiently large slots this can be one descriptor; a smaller envelope emits
four one-patch descriptors with destination shape `[1,48]`.

### 8.4 Deliberately constrained envelope

Use 96-byte slots and one credit in each relevant partition:

```text
U3=U2=U1=96 bytes
C[3,ps]=C[2,pd]=C[1,pd]=1, all initially available
h3=h2=h1=0
C_R=96 bytes
F=1, number_of_block_movers=1
```

The full-image descriptor above is refused (`s=ceil(210/96)=3`,
`d=ceil(384/96)=4`, violating v1's `s=d=1`). Each patch-cell descriptor has
`s=d=l=1` and `r=96`, so:

```text
b_safe=min(1,1,1,1,1)=1
```

A requested burst of two is reduced to one. The schedule performs load, transform,
and downstream feed for each cell before admitting the next; double buffering is
refused because both the credit and reorder terms are exhausted. Source credit returns
at transform completion, while the sole destination credit returns only after the
patch is fed. This schedule makes progress without inventing a second buffer credit.
The four compact patch-cell source payloads in this constrained example are assumed to
have been produced by prior contiguous materialization or the E6/E1 DMA-gather path;
E4 does not crop non-contiguous NCHW rows out of the global image.

## 9. Required invariants

These are executable assertions and test properties, not aspirational prose.

1. **Byte provenance/conservation:** for transpose, reshape, split, and merge,
   `Bsrc=Bdst` and every source byte group appears exactly once in the destination. For
   patchify, `Bdst=Bsrc+Bpad`, every source element appears exactly once, and every
   additional byte is part of an explicitly declared all-zero padding element.
2. **Bijection:** transpose, reshape, split, and merge are bijections over active
   element indices. Patchify is injective over source indices; its only non-source
   indices are declared padding.
3. **Transpose round trip:** `transpose(transpose(X))` restores shape and bytes for
   square, rectangular, singleton, and partial tiles.
4. **Reshape round trip:** compact reshape `S -> D -> S` restores identical shape
   metadata and bytes.
5. **Head round trip:** `merge(split(X,H,Dh),H,Dh)` restores `X` exactly.
6. **Atomic visibility:** no destination payload or destination TagCAM key exists before
   the exact completion transition.
7. **Arrival before read:** no source byte is read before the source TagCAM entry's
   arrival cycle and complete payload availability.
8. **No credit creation or loss:** all L3/L2/L1 partition, reorder-byte, and transform-
   slot conservation equations hold after success, stall, refusal, reset, and error.
9. **Single completion:** each admitted transform emits exactly one terminal completion
   or error event, never both and never neither.
10. **Layout identity:** canonical and transformed payloads with the same `TileID` remain
    distinct by `layout_id`; a consumer cannot match the wrong layout.

## 10. Selected implementation and rejected alternatives

### Selected: in-flight materialization in BlockMover reorder storage

This places the transformation at the architectural movement boundary already
responsible for L3 source arrival, L2 destination credit, transfer timing, TagCAM
publication, and source release. A full private output makes atomic visibility and
failure rollback explicit. It also gives behavioral, temporal, and CSP executors one
normative mapping contract.

### Rejected: host/DRAM materialization

Precomputing transposed B, head-major QKV, or image patches before simulation would
produce values but would not model the intended BlockMover resource pressure,
backpressure, latency, or credit lifetime. That is the current Conv2D shortcut and is
not E4 closure.

### Rejected: annotation-only reshape or transpose

Carrying a flag while copying identity bytes allows schedules and values to disagree.
It is the present CSP failure mode. A transform either satisfies its exact map or fails
validation.

### Rejected: direct streaming publication

Writing visible destination fragments as bytes arrive complicates consumer ordering,
rollback, TagCAM meaning, and scratch equations. A future implementation may stream
internally into reserved destination storage, but it must retain full private ownership
and atomic publication; it is observationally identical to the selected model.

### Rejected: arbitrary permutation descriptor

A free axis list, arbitrary strides, or index program would exceed current rank-2
payload, TagCAM, serializer, timing, and verification architecture. The closed named
maps cover the E4 milestone needs and can be validated independently. New maps require
their own bounded descriptor extension and oracle.

### Rejected: zero-copy reshape alias

The current pattern is specifically movement from L3 to L2, with separate credits and
completion. Aliasing would require lifetime/ownership semantics absent from the CSP
model and would make source release ambiguous. A future view operation is a separate
contract.

## 11. T2-T5 implementation plan

### T2 — ISA and executor capability closure

1. Add `PayloadKey`, `LayoutTransformKind`, fixed parameter structs, and the normalized
   descriptor in `include/sw/kpu/isa/data_movement_isa.hpp` and
   `src/software/isa/data_movement_isa.cpp`; add the three explicit opcodes. Make opcode/
   kind mismatch and AUTO transforms validation errors.
2. Update `src/software/isa/assembler.cpp` (and its public header if signatures change)
   with explicit source/destination ranks and shapes, head parameters, and patch
   parameters. Use one shared checked validator; remove silent defaults.
3. Bump `DMPROGRAM_VERSION` from 3 to 4 in
   `include/sw/kpu/isa/program_serializer.hpp`. Update binary and JSON paths in
   `src/software/isa/program_serializer.cpp`, including every operand-variant switch and
   static exhaustiveness check. Decode only v3 identity move/writeback into normalized
   v4 descriptors; reject v3 transpose because consumers cannot name its layout, and
   reject legacy reshape rather than inventing a destination shape. Serialize identity
   AUTO operands and reject non-identity AUTO at save time.
4. Make `src/software/isa/behavioral_program_executor.cpp`,
   `src/models/temporal/datamovement/block_mover.cpp`,
   `src/software/isa/program_executor.cpp`, `src/software/isa/concurrent_executor.cpp`,
   and `src/software/isa/transactional_program_executor.cpp` use one normative semantic
   contract. Remove reshape no-op/identity fallbacks and opcode-vs-field ambiguity.
5. Extend `include/sw/kpu/timing/tile_descriptor.hpp` from rank-2-only payload metadata;
   carry opaque element bytes or a byte-exact typed representation through the CSP
   payload store.
6. Extend `include/sw/kpu/timing/process_interface.hpp` events and transfers with source
   and destination keys, descriptor kind, logical/source/destination bytes, padding,
   scratch bytes, and terminal status. Update `include/sw/kpu/timing/tag_cam.hpp` to key
   by `PayloadKey`.
7. Implement all-or-none destination/reorder/slot admission, transform timing, atomic
   completion, and rollback in `include/sw/kpu/timing/block_mover_process.hpp` and
   payload commit in `include/sw/kpu/timing/concurrent_timing_executor.hpp`. Add global
   reorder-byte and transform-slot accounting, slot byte-fit checks, and partitioned L1
   credits wired through `include/sw/kpu/timing/streamer_process.hpp`.
8. Replace bare `TileID` with `PayloadKey` throughout FEED queues, Streamer matching,
   schedule and resident dependencies, functional payload maps, pending-compute state,
   and `ConcurrentTimingExecutor::MatMulComputeSpec::{a_tiles,b_tiles,resident_tiles}`.
   A compatibility helper may default legacy callers to `layout_id=0`, but transformed
   consumers must name `dst_key` explicitly.

### T3 — envelope-aware schedule generation

1. Replace `ScheduleOperation::transpose` in
   `include/sw/kpu/timing/schedule/schedule_generator_interface.hpp` with the normalized
   descriptor; thread it through `schedule_executor.hpp` and the concurrent executor.
   Change FEED and COMPUTE factories/dependency vectors in the same interface to carry
   the exact `PayloadKey` expected downstream.
2. Add `include/sw/kpu/timing/schedule/layout_transform_schedule_generator.hpp` with
   named builders for transpose, reshape, head split/merge, and patchify. Configuration
   includes L3/L2/L1 slot bytes and partition credits, reorder bytes, transform slots,
   BlockMover count, and per-partition headroom.
3. Implement the exact homogeneous and mixed-prefix bounds from section 7. Record
   requested/selected burst and every limiting term in schedule metadata. A zero bound
   is a refusal with an actionable diagnostic.
4. Update `include/sw/kpu/timing/schedule/schedule_validator.hpp` to independently
   recompute shape validity, byte fit, per-partition working sets, and the constructive
   bound rather than trusting generator metadata.
5. Update `include/sw/kpu/timing/schedule/matmul_schedule_generator.hpp` and the
   functional matmul/streamer consumer together so B-transpose bytes and indexing agree.
   Audit `include/sw/kpu/timing/schedule/conv2d_schedule_generator.hpp`: its preflattened
   `[K,Cout]` payload must either use identity movement or a matching physical layout and
   consumer. Do not retain a boolean whose semantics depend on the operator.
6. Add head-layout and patch-cell strategies. Patch scheduling may split an image only
   at compact source-tile boundaries already produced by a legal contiguous load or an
   explicit E6/E1 gather. It must not imply that E4 itself crops arbitrary global NCHW
   rows or implements overlapping Conv2D gather.

### T4 — functional integration and independent oracle

1. Add a test-only independent reference module, for example
   `tests/timing/layout_transform_reference.hpp`, implemented as straightforward nested
   loops that do not call production mapping helpers, descriptor offset helpers, or
   Conv2D `detail::padded_coord`.
2. Add `tests/timing/test_functional_layout_transform.cpp` and a schedule example. Check
   byte-exact transpose (including non-square), reshape, head split/merge, aligned and
   non-aligned patchify, and complete CSP pipeline values.
3. Extend `tests/isa/test_serialization.cpp` and assembler tests with v4 binary/JSON
   round trips, v3 compatibility cases, every malformed descriptor class, opcode/kind
   mismatch, overflow, unsupported element size, aliasing, and truncated data.
4. Add deterministic randomized property tests across ranks, extents, element widths,
   head counts, patch sizes, partial tiles, and padding. Check every invariant in section
   9 against the independent reference, including round trips.
5. Retain direct Conv2D/reference tests and add an integration check that makes any
   change to the existing B layout explicit; enabling E4 must not silently transpose an
   already materialized `[K,Cout]` operand.

### T5 — regression, trace characterization, and mutation sensitivity

1. Add `tests/timing/test_layout_transform_regression.cpp` covering default and
   deliberately constrained envelopes, every single limiting term, reduced bursts,
   zero-bound refusal, multi-mover contention, partition starvation attempts, and
   transform-error rollback.
2. Extend BlockMover, concurrent-executor, schedule-generator, and component-integration
   tests with one-credit L3/L2/L1 cases, reorder capacity at `r-1/r/r+1`, one/two slots,
   destination backpressure, delayed source arrival, and downstream stalls. Assert
   progress and exact credit conservation at every cycle boundary.
3. Trace source/destination `PayloadKey`, kind, shapes, element size, source/destination/
   padding/reorder bytes, partition, requested/selected burst, start/completion cycles,
   and terminal status. Tests must prove there is no destination-visible event before
   completion and exactly one source-release and destination-publication transition.
4. Add deterministic mutation checks (a small CTest driver or script) for at least:
   transpose row/column offset swap, head-axis swap, missing patch-edge zero fill,
   early TagCAM insertion, omitted source-credit release, omitted reorder release, and
   acceptance of unequal-product reshape. Each mutant must be killed by a named test;
   do not rely only on line coverage.
5. Update `tests/coverage/pattern_coverage.json` only after the default and constrained
   functional suites pass. Record characterization in the #73 epic and close T2-T5 in
   dependency order.

## 12. Milestone impact

- **M4 attention:** head split/merge becomes a modeled, byte-correct movement rather
  than host reshaping. Distinct layout keys prevent Q/K/V consumers from accepting the
  wrong physical order. This closes the E4 side of M4; projections, attention math, and
  other dependencies remain separate.
- **M5 ViT:** non-overlapping patchification, including non-aligned image edges, becomes
  an in-flight CSP operation with real destination expansion, zero-padding cost,
  reorder pressure, and constrained-envelope behavior. Global non-contiguous image
  gather still depends on the appropriate DMA schedule.
- **M6 spatial operations:** the descriptor/resource pattern provides the bounded
  foundation for later pack/unpack and spatial maps without pretending E4 is a general
  scatter engine. Overlapping Conv2D im2col remains E6/E1; broader rearrangements are
  E16.
- **M8 JEPA:** ViT patch tokens and head-major attention tensors can stay inside the
  modeled pipeline, making encoder/predictor value checks and low-credit runs reflect
  actual layout traffic instead of unaccounted host passes.

## 13. T1 acceptance checklist

- Current transpose, reshape, timing, serialization, TagCAM, credit, and Conv2D payload
  paths are classified by observed behavior.
- Selected maps have exact ranks, shapes, strides, index functions, byte/padding rules,
  partial-tile policy, alias policy, malformed behavior, and completion conditions.
- Dataflow ownership, source arrival, destination admission, scratch, backpressure,
  source release, and atomic visibility are specified.
- L3/L2/L1 partition, reorder, slot, source/destination, concurrency, refusal, and
  reduced-burst equations are explicit, with four worked examples.
- Invariants and independent verification/mutation expectations are executable.
- No source, tests, roadmap, coverage data, or other design document is changed by T1.

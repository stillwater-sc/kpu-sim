// ============================================================================
// include/sw/kpu/orchestration/abi.hpp
// The call ABI on the wire (#305 increment 3): register map, descriptor and completion
// records, and the encoding between them and the in-process types.
//
// EVERY FIELD IS AN INDEX, A COUNT, AN ID OR A FLAG. Names cross as indices into tables both
// ends already hold -- the loadable's tensor and operator tables, the deployment's devices --
// so a record has no variable-length field, and nowhere a tensor element could ride along.
// That is §3's rule stated as a layout a reviewer can read off this file, and the size
// assertions below make widening it a compile error.
//
// All addresses are 64-bit regardless of the guest's XLEN (parent plan §4). Registers are
// 64 bits wide, accessed as aligned 8-byte words. Records are little-endian, packed by hand
// rather than by memcpy of a struct, so the layout does not depend on the host compiler.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/loadable/loadable.hpp>
#include <sw/kpu/orchestration/descriptor.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::orchestration::abi {

inline constexpr std::uint64_t kMagic = 0x4B50554C44000000ull;   // "KPULD"
inline constexpr std::uint64_t kAbiMajor = 1, kAbiMinor = 0;
inline constexpr std::uint32_t kNone = 0xFFFFFFFFu;

inline constexpr std::size_t kDescriptorBytes = 64;
inline constexpr std::size_t kCompletionBytes = 64;

using DescriptorRecord = std::array<std::uint8_t, kDescriptorBytes>;
using CompletionRecord = std::array<std::uint8_t, kCompletionBytes>;

// THE LAYOUT, CHECKED. The last field of each record (see the table below) ends exactly at the
// record size, and records move as aligned 64-bit words. The encoders write through byte
// offsets, so a field placed past the end would be out-of-bounds writes rather than a compile
// error -- these are what make widening a record a compile error instead.
inline constexpr std::size_t kDescriptorLastField = 56, kDescriptorLastWidth = 8;   // wait_for
inline constexpr std::size_t kCompletionLastField = 60, kCompletionLastWidth = 4;   // zero
static_assert(kDescriptorLastField + kDescriptorLastWidth == kDescriptorBytes,
              "descriptor layout must end at the record size");
static_assert(kCompletionLastField + kCompletionLastWidth == kCompletionBytes,
              "completion layout must end at the record size");
static_assert(kDescriptorBytes % 8 == 0 && kCompletionBytes % 8 == 0,
              "records are moved as aligned 64-bit words");

// ----------------------------------------------------------------------------
// Register map, offsets from the KPU MMIO base
// ----------------------------------------------------------------------------
namespace reg {
inline constexpr std::uint64_t ID              = 0x000;   // RO  kMagic
inline constexpr std::uint64_t ABI_VERSION     = 0x008;   // RO  major << 16 | minor
inline constexpr std::uint64_t CAP_L3_TILES    = 0x010;   // RO  0 = unbounded
inline constexpr std::uint64_t CAP_COMPUTE     = 0x018;   // RO  compute tiles
inline constexpr std::uint64_t CAP_INVENTORY   = 0x020;   // RO  resources in the naming map
inline constexpr std::uint64_t INV_INDEX       = 0x028;   // WO  select an inventory entry
inline constexpr std::uint64_t INV_ENTRY       = 0x030;   // RO  packed resource (pack_resource)
inline constexpr std::uint64_t OP_COUNT        = 0x038;   // RO  operators in the loadable
inline constexpr std::uint64_t DRING_BASE      = 0x040;   // RW  descriptor ring address
inline constexpr std::uint64_t DRING_SIZE      = 0x048;   // RW  entries
inline constexpr std::uint64_t DRING_TAIL      = 0x050;   // WO  THE DOORBELL
inline constexpr std::uint64_t DRING_HEAD      = 0x058;   // RO  device consume index
inline constexpr std::uint64_t CRING_BASE      = 0x060;   // RW  completion ring address
inline constexpr std::uint64_t CRING_SIZE      = 0x068;   // RW  entries
inline constexpr std::uint64_t CRING_HEAD      = 0x070;   // RO  device produce index
inline constexpr std::uint64_t CRING_TAIL      = 0x078;   // WO  orchestrator acknowledge
inline constexpr std::uint64_t ST_CREDITS_FREE = 0x080;   // RO  §6.4 credits: the DEVICE's
inline constexpr std::uint64_t ST_RESERVED     = 0x088;   // RO
inline constexpr std::uint64_t ST_HELD         = 0x090;   // RO
inline constexpr std::uint64_t ST_COMPLETED    = 0x098;   // RO  operators completed
inline constexpr std::uint64_t RES_Q_TENSOR    = 0x0A0;   // WO  residency query: tile id
inline constexpr std::uint64_t RES_Q_TI        = 0x0A8;   // WO
inline constexpr std::uint64_t RES_Q_TJ        = 0x0B0;   // WO
inline constexpr std::uint64_t RES_RESULT      = 0x0B8;   // RO  1 resident, 0 not. NEVER contents
inline constexpr std::uint64_t MAN_OP          = 0x0C0;   // WO  select an operator manifest
inline constexpr std::uint64_t MAN_VALID       = 0x0C8;   // RO
inline constexpr std::uint64_t MAN_PEAK_LIVE   = 0x0D0;   // RO
inline constexpr std::uint64_t MAN_N_READS     = 0x0D8;   // RO
inline constexpr std::uint64_t MAN_READ_IDX    = 0x0E0;   // WO  select a read tile
inline constexpr std::uint64_t MAN_READ_TENSOR = 0x0E8;   // RO
inline constexpr std::uint64_t MAN_READ_TI     = 0x0F0;   // RO
inline constexpr std::uint64_t MAN_READ_TJ     = 0x0F8;   // RO
inline constexpr std::uint64_t MAN_ERR_OFF     = 0x100;   // RO  invalid manifest: text in DIAG
inline constexpr std::uint64_t MAN_ERR_LEN     = 0x108;   // RO
inline constexpr std::uint64_t DIAG_BASE       = 0x110;   // RW  device-written diagnosis text
inline constexpr std::uint64_t DIAG_SIZE       = 0x118;   // RW
inline constexpr std::uint64_t IRQ_STATUS      = 0x120;   // RO  bit 0: completions pending
inline constexpr std::uint64_t IRQ_ACK         = 0x128;   // W1C
inline constexpr std::uint64_t kWindow         = 0x1000;  // bytes the device decodes
} // namespace reg

// ----------------------------------------------------------------------------
// Name tables: what lets a name cross as an index
// ----------------------------------------------------------------------------
// Both ends build this from what they already hold -- the loadable's metadata and the
// deployment -- so an index means the same thing on both sides by construction. An unknown
// name is a programming error on the encoding side and throws; an out-of-range index from the
// wire throws too, rather than decoding to something plausible.
class NameTable {
public:
    NameTable(const loadable::Loadable& l, const program::platform::DeploymentSpec& spec);

    std::uint32_t tensor_index(const std::string& name) const;     // kNone for ""
    std::string tensor_name(std::uint32_t i) const;                // "" for kNone
    std::uint32_t operator_index(const std::string& name) const;
    std::string operator_name(std::uint32_t i) const;
    std::uint32_t program_index(const std::string& name) const;    // domain-flow programs
    std::string program_name(std::uint32_t i) const;
    std::uint32_t device_index(const std::string& name) const;
    std::string device_name(std::uint32_t i) const;

private:
    std::vector<std::string> tensors_, operators_, programs_, devices_;
};

// ----------------------------------------------------------------------------
// Records
// ----------------------------------------------------------------------------
//   descriptor (64 B)                       completion (64 B)
//    0 u64 id                                0 u64 descriptor_id
//    8 u8  kind   9 u8 leg                   8 u64 cycles
//   10 u8  res_device  11 u8 res_kind       16 u8 status 17 u8 cause 18 u8 timed 19 u8 more
//   12 u32 flags                            20 u32 blocking_op
//   16 u32 tile.tensor 20 ti 24 tj          24 u32 needed 28 available 32 capacity
//   28 u32 target (operator / program)      36 u32 diag_off 40 u32 diag_len
//   32 u32 res_path[0] 36 res_path[1]       44 u32 n_released (0 or 1, inline)
//   40 u64 res_offset                       48 u32 tile.tensor 52 ti 56 tj
//   48 u32 compute_tile 52 u32 slots        60 u32 zero
//   56 u64 wait_for
//
// A completion releasing more than one tile sets `more` and continues in the next record,
// which repeats the id and carries the next tile. A RELEASE returns exactly one credit, so in
// practice one record is enough; the continuation keeps the record fixed-size without
// assuming that.
DescriptorRecord encode(const Descriptor& d, const NameTable& names);
Descriptor decode(const DescriptorRecord& r, const NameTable& names);

// `diag_off`/`diag_len` locate the diagnosis text, which the device writes into its DIAG area
// rather than into the record.
std::vector<CompletionRecord> encode(const Completion& c, const NameTable& names,
                                     std::uint32_t diag_off, std::uint32_t diag_len);

// Decode the FIRST record of a completion; continuation records are appended with
// `decode_continuation`. The diagnosis text is filled by the caller from the DIAG area.
Completion decode(const CompletionRecord& r, const NameTable& names, bool& more,
                  std::uint32_t& diag_off, std::uint32_t& diag_len);
void decode_continuation(const CompletionRecord& r, const NameTable& names, Completion& into,
                         bool& more);

// A naming-map resource as one 64-bit word: device(8) | kind(8) | path0(24) | path1(24).
std::uint64_t pack_resource(const program::platform::ResourceName& n, const NameTable& names);
program::platform::ResourceName unpack_resource(std::uint64_t w, const NameTable& names);

} // namespace sw::kpu::orchestration::abi

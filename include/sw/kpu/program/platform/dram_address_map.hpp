// ============================================================================
// include/sw/kpu/program/platform/dram_address_map.hpp
// Where a DRAM address lands: controller, channel, rank, bank group, bank, row and the burst
// within the page (docs/plans/dram-bank-model.md §3.2, step 1).
//
// ONE DECODE, DERIVED FROM THE SPEC. Before this, every consumer decoded addresses its own way:
// the CSP-tier MemoryControllerProcess hard-codes `bank = (addr >> col_bits) & 0xF` with no
// channel or controller bits, and the tile-flow binding guessed a top of memory because the
// spec declared none. Two decodes of one address disagree eventually, and then a bank
// conflict the static model predicts is one the controller never sees. So the controller, the
// static conflict model, the record and the viewer all take their coordinates from here.
//
// The map is a bijection on [0, capacity_bytes): decode(encode(c, o)) == (c, o) and
// encode(decode(a)) == a. XOR folds keep it one, because a field that is folded from is never
// folded into (dram_problem() refuses the other case).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>

namespace sw::kpu::program::platform {

struct DramMapError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Coordinates of one burst. `bank` is the bank WITHIN its group; flat_bank() numbers the
// banks of a rank 0 .. bank_groups * banks_per_group - 1, which is what a controller queues on.
struct DramCoord {
    Dim mc = 0, channel = 0, rank = 0, bank_group = 0, bank = 0;
    std::uint64_t row = 0;
    std::uint64_t col = 0;      // the burst within the page

    bool operator==(const DramCoord&) const = default;
};

class DramAddressMap {
public:
    // The map of device `d`; throws DramMapError when it declares no DRAM or an inconsistent
    // one (the message is dram_problem()'s).
    static DramAddressMap of(const DeviceSpecification& d);

    DramCoord decode(std::uint64_t addr) const;
    // The address of byte `offset` (< burst_bytes) of the burst at `c`.
    std::uint64_t encode(const DramCoord& c, std::uint64_t offset = 0) const;

    std::uint64_t capacity() const { return capacity_; }
    Dim burst_bytes() const { return Dim{1} << offset_bits_; }
    Dim controllers() const { return count(Field::Mc); }
    Dim channels() const { return count(Field::Ch); }
    Dim ranks() const { return count(Field::Rk); }
    Dim bank_groups() const { return count(Field::Bg); }
    Dim banks_per_group() const { return count(Field::Ba); }
    std::uint64_t rows() const { return std::uint64_t{1} << width_[Field::Ro]; }
    std::uint64_t bursts_per_page() const { return std::uint64_t{1} << width_[Field::Co]; }

    Dim flat_bank(const DramCoord& c) const { return c.bank_group * banks_per_group() + c.bank; }

    // "co[6,11) ch[11,12) ... xor ba^=ro[0,2)": the map as bit ranges, for a report.
    std::string describe() const;

private:
    enum Field : unsigned { Co, Ch, Rk, Bg, Ba, Mc, Ro, NFields };
    struct Fold { Field into, from; unsigned bits, from_lsb; };

    Dim count(Field f) const { return Dim{1} << width_[f]; }
    static Field field_of(const std::string& name);

    std::uint64_t capacity_ = 0;
    unsigned offset_bits_ = 0;
    std::array<unsigned, NFields> lsb_{};
    std::array<unsigned, NFields> width_{};
    std::vector<Fold> folds_;
};

} // namespace sw::kpu::program::platform

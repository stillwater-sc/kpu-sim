// ============================================================================
// src/program/dram_address_map.cpp
// The DRAM address map (docs/plans/dram-bank-model.md §3.2). See the header.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/platform/dram_address_map.hpp>

#include <algorithm>

namespace sw::kpu::program::platform {

namespace {
const char* kNames[] = {"co", "ch", "rk", "bg", "ba", "mc", "ro"};

std::uint64_t mask(unsigned bits) {
    return bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
}
} // namespace

DramAddressMap::Field DramAddressMap::field_of(const std::string& name) {
    for (unsigned f = 0; f < NFields; ++f)
        if (name == kNames[f]) return static_cast<Field>(f);
    throw DramMapError("dram map: unknown field '" + name + "'");
}

DramAddressMap DramAddressMap::of(const DeviceSpecification& d) {
    if (!d.memory.dram)
        throw DramMapError("dram map: device \"" + d.name + "\" declares no memory.dram");
    if (std::string why = dram_problem(d); !why.empty())
        throw DramMapError("dram map: device \"" + d.name + "\": " + why);
    const auto& m = *d.memory.dram;
    const auto bits = dram_field_bits(d);

    DramAddressMap a;
    a.capacity_ = m.capacity_bytes;
    a.offset_bits_ = log2_exact(m.burst_bytes);
    unsigned at = a.offset_bits_;
    for (const std::string& name : split_dram_map(m.map)) {
        const Field f = field_of(name);
        a.lsb_[f] = at;
        a.width_[f] = static_cast<unsigned>(bits.at(name));
        at += a.width_[f];
    }
    for (const auto& f : m.xor_folds)
        a.folds_.push_back({field_of(f.into), field_of(f.from), f.bits, f.from_lsb});
    return a;
}

DramCoord DramAddressMap::decode(std::uint64_t addr) const {
    if (addr >= capacity_)
        throw DramMapError("dram map: address " + std::to_string(addr) +
                           " is past the top of memory (" + std::to_string(capacity_) + ")");
    std::array<std::uint64_t, NFields> v{};
    for (unsigned f = 0; f < NFields; ++f) v[f] = (addr >> lsb_[f]) & mask(width_[f]);
    // `from` fields are never folded into, so reading them here sees the raw address bits.
    for (const Fold& x : folds_) v[x.into] ^= (v[x.from] >> x.from_lsb) & mask(x.bits);
    DramCoord c;
    c.col = v[Co];
    c.channel = static_cast<Dim>(v[Ch]);
    c.rank = static_cast<Dim>(v[Rk]);
    c.bank_group = static_cast<Dim>(v[Bg]);
    c.bank = static_cast<Dim>(v[Ba]);
    c.mc = static_cast<Dim>(v[Mc]);
    c.row = v[Ro];
    return c;
}

std::uint64_t DramAddressMap::encode(const DramCoord& c, std::uint64_t offset) const {
    std::array<std::uint64_t, NFields> v{c.col, c.channel, c.rank, c.bank_group,
                                         c.bank, c.mc,     c.row};
    for (unsigned f = 0; f < NFields; ++f)
        if (v[f] > mask(width_[f]))
            throw DramMapError(std::string("dram map: coordinate ") + kNames[f] + " = " +
                               std::to_string(v[f]) + " does not fit its " +
                               std::to_string(width_[f]) + " bits");
    if (offset >= burst_bytes())
        throw DramMapError("dram map: offset " + std::to_string(offset) +
                           " is not within one burst (" + std::to_string(burst_bytes()) + ")");
    // XOR is its own inverse, and the `from` values are the decoded ones unchanged.
    for (const Fold& x : folds_) v[x.into] ^= (v[x.from] >> x.from_lsb) & mask(x.bits);
    std::uint64_t addr = offset;
    for (unsigned f = 0; f < NFields; ++f) addr |= v[f] << lsb_[f];
    return addr;
}

std::string DramAddressMap::describe() const {
    std::string out = "off[0," + std::to_string(offset_bits_) + ")";
    std::array<unsigned, NFields> order{};
    for (unsigned f = 0; f < NFields; ++f) order[f] = f;
    std::sort(order.begin(), order.end(), [&](unsigned x, unsigned y) { return lsb_[x] < lsb_[y]; });
    for (unsigned f : order) {
        if (width_[f] == 0) continue;
        out += std::string(" ") + kNames[f] + "[" + std::to_string(lsb_[f]) + "," +
               std::to_string(lsb_[f] + width_[f]) + ")";
    }
    for (const Fold& x : folds_)
        out += std::string(" xor ") + kNames[x.into] + "[0," + std::to_string(x.bits) + ")^=" +
               kNames[x.from] + "[" + std::to_string(x.from_lsb) + "," +
               std::to_string(x.from_lsb + x.bits) + ")";
    return out;
}

} // namespace sw::kpu::program::platform

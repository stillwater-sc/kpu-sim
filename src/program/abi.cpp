// ============================================================================
// src/program/abi.cpp
// The call ABI's wire encoding (#305 increment 3). Layout tables live in abi.hpp.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/orchestration/abi.hpp>

#include <stdexcept>

namespace sw::kpu::orchestration::abi {

namespace {

// Little-endian by hand, so the layout is the documented one on every host.
void put8(std::uint8_t* p, std::uint8_t v) { p[0] = v; }
void put32(std::uint8_t* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void put64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
std::uint8_t get8(const std::uint8_t* p) { return p[0]; }
std::uint32_t get32(const std::uint8_t* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    return v;
}
std::uint64_t get64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

std::uint32_t index_in(const std::vector<std::string>& table, const std::string& name,
                       const char* what) {
    if (name.empty()) return kNone;
    for (std::size_t i = 0; i < table.size(); ++i)
        if (table[i] == name) return static_cast<std::uint32_t>(i);
    throw std::invalid_argument(std::string("abi: no ") + what + " named \"" + name + "\"");
}

std::string name_in(const std::vector<std::string>& table, std::uint32_t i, const char* what) {
    if (i == kNone) return {};
    if (i >= table.size())
        throw std::invalid_argument(std::string("abi: ") + what + " index " +
                                    std::to_string(i) + " is out of range");
    return table[i];
}

// Which table a descriptor's `target` indexes depends on its kind: CONFIGURE names a
// domain-flow program, everything else an operator.
bool targets_program(DescriptorKind k) { return k == DescriptorKind::Configure; }

void put_tile(std::uint8_t* p, const TileRef& t, const NameTable& names) {
    put32(p + 0, names.tensor_index(t.tensor));
    put32(p + 4, t.ti);
    put32(p + 8, t.tj);
}
TileRef get_tile(const std::uint8_t* p, const NameTable& names) {
    return TileRef{names.tensor_name(get32(p + 0)), get32(p + 4), get32(p + 8)};
}

} // namespace

NameTable::NameTable(const loadable::Loadable& l, const program::platform::DeploymentSpec& spec) {
    for (const auto& t : l.tensors) tensors_.push_back(t.name);
    for (const auto& o : l.operators) operators_.push_back(o.name);
    for (const auto& p : l.domain_flow_programs) programs_.push_back(p.name);
    for (program::Dim i = 0; i < spec.device_count(); ++i) devices_.push_back(spec.device(i).name);
}

std::uint32_t NameTable::tensor_index(const std::string& n) const { return index_in(tensors_, n, "tensor"); }
std::string NameTable::tensor_name(std::uint32_t i) const { return name_in(tensors_, i, "tensor"); }
std::uint32_t NameTable::operator_index(const std::string& n) const { return index_in(operators_, n, "operator"); }
std::string NameTable::operator_name(std::uint32_t i) const { return name_in(operators_, i, "operator"); }
std::uint32_t NameTable::program_index(const std::string& n) const { return index_in(programs_, n, "domain-flow program"); }
std::string NameTable::program_name(std::uint32_t i) const { return name_in(programs_, i, "domain-flow program"); }
std::uint32_t NameTable::device_index(const std::string& n) const { return index_in(devices_, n, "device"); }
std::string NameTable::device_name(std::uint32_t i) const { return name_in(devices_, i, "device"); }

// ---- descriptors ------------------------------------------------------------
DescriptorRecord encode(const Descriptor& d, const NameTable& names) {
    DescriptorRecord r{};
    std::uint8_t* p = r.data();
    put64(p + 0, d.id);
    put8(p + 8, static_cast<std::uint8_t>(d.kind));
    put8(p + 9, static_cast<std::uint8_t>(d.leg));
    const std::uint32_t dev = names.device_index(d.resource.device);
    put8(p + 10, dev == kNone ? 0xFF : static_cast<std::uint8_t>(dev));
    put8(p + 11, static_cast<std::uint8_t>(d.resource.kind));
    put32(p + 12, d.flags);
    put_tile(p + 16, d.tile, names);
    put32(p + 28, targets_program(d.kind) ? names.program_index(d.target)
                                          : names.operator_index(d.target));
    if (d.resource.path.size() > 2)
        throw std::invalid_argument("abi: a resource path has at most two components");
    put32(p + 32, d.resource.path.size() > 0 ? d.resource.path[0] : kNone);
    put32(p + 36, d.resource.path.size() > 1 ? d.resource.path[1] : kNone);
    put64(p + 40, d.resource.offset);
    put32(p + 48, d.compute_tile);
    put32(p + 52, d.slots);
    put64(p + 56, d.wait_for);
    return r;
}

Descriptor decode(const DescriptorRecord& r, const NameTable& names) {
    const std::uint8_t* p = r.data();
    Descriptor d;
    d.id = get64(p + 0);
    d.kind = static_cast<DescriptorKind>(get8(p + 8));
    d.leg = static_cast<program::Hop>(get8(p + 9));
    const std::uint8_t dev = get8(p + 10);
    d.resource.device = dev == 0xFF ? std::string{} : names.device_name(dev);
    d.resource.kind = static_cast<program::platform::ResourceKind>(get8(p + 11));
    d.flags = get32(p + 12);
    d.tile = get_tile(p + 16, names);
    const std::uint32_t target = get32(p + 28);
    d.target = targets_program(d.kind) ? names.program_name(target) : names.operator_name(target);
    for (std::size_t off : {std::size_t{32}, std::size_t{36}}) {
        const std::uint32_t c = get32(p + off);
        if (c != kNone) d.resource.path.push_back(c);
    }
    d.resource.offset = get64(p + 40);
    d.compute_tile = get32(p + 48);
    d.slots = get32(p + 52);
    d.wait_for = get64(p + 56);
    return d;
}

// ---- completions ------------------------------------------------------------
std::vector<CompletionRecord> encode(const Completion& c, const NameTable& names,
                                     std::uint32_t diag_off, std::uint32_t diag_len) {
    std::vector<CompletionRecord> out(c.released.size() > 1 ? c.released.size() : 1);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::uint8_t* p = out[i].data();
        put64(p + 0, c.descriptor_id);
        put8(p + 19, i + 1 < out.size() ? 1 : 0);                 // more
        if (i == 0) {
            put64(p + 8, c.cycles);
            put8(p + 16, static_cast<std::uint8_t>(c.status));
            put8(p + 17, static_cast<std::uint8_t>(c.cause));
            put8(p + 18, c.timed ? 1 : 0);
            put32(p + 20, c.blocking_op);
            put32(p + 24, c.needed);
            put32(p + 28, c.available);
            put32(p + 32, c.capacity);
            put32(p + 36, diag_off);
            put32(p + 40, diag_len);
        }
        put32(p + 44, i < c.released.size() ? 1 : 0);
        if (i < c.released.size()) put_tile(p + 48, c.released[i], names);
    }
    return out;
}

Completion decode(const CompletionRecord& r, const NameTable& names, bool& more,
                  std::uint32_t& diag_off, std::uint32_t& diag_len) {
    const std::uint8_t* p = r.data();
    Completion c;
    c.descriptor_id = get64(p + 0);
    c.cycles = get64(p + 8);
    c.status = static_cast<CompletionStatus>(get8(p + 16));
    c.cause = static_cast<RefusalCause>(get8(p + 17));
    c.timed = get8(p + 18) != 0;
    more = get8(p + 19) != 0;
    c.blocking_op = get32(p + 20);
    c.needed = get32(p + 24);
    c.available = get32(p + 28);
    c.capacity = get32(p + 32);
    diag_off = get32(p + 36);
    diag_len = get32(p + 40);
    if (get32(p + 44) != 0) c.released.push_back(get_tile(p + 48, names));
    return c;
}

void decode_continuation(const CompletionRecord& r, const NameTable& names, Completion& into,
                         bool& more) {
    const std::uint8_t* p = r.data();
    if (get64(p + 0) != into.descriptor_id)
        throw std::invalid_argument("abi: a continuation record names descriptor " +
                                    std::to_string(get64(p + 0)) + ", expected " +
                                    std::to_string(into.descriptor_id));
    more = get8(p + 19) != 0;
    if (get32(p + 44) != 0) into.released.push_back(get_tile(p + 48, names));
}

std::uint64_t pack_resource(const program::platform::ResourceName& n, const NameTable& names) {
    const std::uint64_t dev = names.device_index(n.device) & 0xFF;
    const std::uint64_t kind = static_cast<std::uint64_t>(n.kind) & 0xFF;
    const std::uint64_t p0 = n.path.size() > 0 ? (n.path[0] & 0xFFFFFF) : 0xFFFFFF;
    const std::uint64_t p1 = n.path.size() > 1 ? (n.path[1] & 0xFFFFFF) : 0xFFFFFF;
    return (dev << 56) | (kind << 48) | (p0 << 24) | p1;
}

program::platform::ResourceName unpack_resource(std::uint64_t w, const NameTable& names) {
    program::platform::ResourceName n;
    n.device = names.device_name(static_cast<std::uint32_t>(w >> 56));
    n.kind = static_cast<program::platform::ResourceKind>((w >> 48) & 0xFF);
    const std::uint64_t p0 = (w >> 24) & 0xFFFFFF, p1 = w & 0xFFFFFF;
    if (p0 != 0xFFFFFF) n.path.push_back(static_cast<program::Dim>(p0));
    if (p1 != 0xFFFFFF) n.path.push_back(static_cast<program::Dim>(p1));
    return n;
}

} // namespace sw::kpu::orchestration::abi

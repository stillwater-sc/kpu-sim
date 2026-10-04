// ============================================================================
// include/sw/kpu/program/platform/resource_map.hpp
// The global naming map (#282 increment 3): every resource a deployment declares,
// addressable by name, across devices (ADR 0002 §3.4, §6.4).
//
// WHY A MAP AND WHY NOW. Two issues need to NAME and ENUMERATE resources before
// anything can read one: the backdoor (#284) writes to "any addressable resource", and
// the spatial event record (#286) has to say WHERE an event happened in order to show
// concurrency. Neither needs the resource to hold state yet, so the map answers the
// question it can answer:
//
//     "Does this resource exist in this deployment?"     -- here, from the spec
//     "What is in it?"                                   -- #283, when resources
//                                                           hold state
//
// THE DOMAIN IS EXACTLY WHAT THE DEPLOYMENT DECLARES, which is the whole discipline of
// increment 1 carried through. If a spec does not declare `l3.banks`, then this machine's
// bank structure is unspecified and `dev0/l3[0]/bank[0]` NAMES NOTHING -- so it does not
// resolve, and the diagnostic says which field is missing rather than inventing a count.
// Resolving it against a default would hand the backdoor an address for a resource nobody
// described.
//
// A NAME CARRIES A PATH, NOT AN INSTANCE NUMBER. L2 banks and L1 vectors are per COMPUTE
// TILE and L3 banks are per L3 module, so an address needs the whole path: a flat
// (kind, instance) pair would have to flatten `cf[2]/l2[3]` into one number and lose the
// structure -- the same class of error as conflating `l3.tiles` with `l3.capacity_tiles`,
// where the wrong number looks plausible.
//
// DEVICES ARE ADDRESSED BY NAME, not by index. A spec names its devices, and a positional
// address would silently move when a deployment is reordered.
//
// WHAT THIS DELIBERATELY CANNOT CHECK: the OFFSET. A DeploymentSpec declares no sizes --
// no bytes per L3 tile, no L2 bank width -- so an offset is carried and formatted but
// never bounded. Saying so beats implying a check that does not happen; the sizes are an
// additive spec field (R8) and #283 needs them anyway to give a resource contents.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::platform {

// ----------------------------------------------------------------------------
// The kinds of resource an address can name
// ----------------------------------------------------------------------------
enum class ResourceKind {
    Dram,           // dev/dram              -- the far side of the DMA
    L3Tile,         // dev/l3[t]
    L3Bank,         // dev/l3[t]/bank[b]
    ComputeTile,    // dev/cf[c]
    L2Bank,         // dev/cf[c]/l2[b]       -- L2 is PER COMPUTE TILE
    L1Vector,       // dev/cf[c]/l1[v]       -- so is L1
    RegisterFile,   // dev/cf[c]/regs
    // ---- the physical shape (#286 step 1). Appended, so existing kinds keep their values,
    //      and named only when the spec declares what they need (array_layout.hpp).
    BlockMover,     // dev/l3[t]/bm[e]       -- e = the edge it sits on: 0 N, 1 E, 2 S, 3 W
    NocRouter,      // dev/l3[t]/noc         -- the L3 tile's hub on the folded torus
    NocPort,        // dev/noc/port[k]       -- a fold-end port, where traffic enters a loop
    MemoryController, // dev/mc[m]
    DmaEngine,      // dev/mc[m]/dma[d]      -- a DMA engine belongs to one controller
    CpuHart,        // dev/cpu/hart[h]       -- the attached RV64 orchestrator (ADR 0003)
    CpuSram,        // dev/cpu/sram
    DescriptorRing, // dev/cpu/dring
    CompletionRing, // dev/cpu/cring
};

inline const char* to_string(ResourceKind k) {
    switch (k) {
        case ResourceKind::Dram:         return "dram";
        case ResourceKind::L3Tile:       return "l3";
        case ResourceKind::L3Bank:       return "l3bank";
        case ResourceKind::ComputeTile:  return "cf";
        case ResourceKind::L2Bank:       return "l2";
        case ResourceKind::L1Vector:     return "l1";
        case ResourceKind::RegisterFile: return "regs";
        case ResourceKind::BlockMover:   return "bm";
        case ResourceKind::NocRouter:    return "noc";
        case ResourceKind::NocPort:      return "port";
        case ResourceKind::MemoryController: return "mc";
        case ResourceKind::DmaEngine:    return "dma";
        case ResourceKind::CpuHart:      return "hart";
        case ResourceKind::CpuSram:      return "sram";
        case ResourceKind::DescriptorRing: return "dring";
        case ResourceKind::CompletionRing: return "cring";
    }
    return "?";
}

// How many path components a kind takes. Stated once, so the parser, the formatter and
// the resolver cannot disagree about the shape of an address.
inline std::size_t path_arity(ResourceKind k) {
    switch (k) {
        case ResourceKind::Dram:         return 0;
        case ResourceKind::L3Tile:       return 1;   // {tile}
        case ResourceKind::L3Bank:       return 2;   // {tile, bank}
        case ResourceKind::ComputeTile:  return 1;   // {cf}
        case ResourceKind::L2Bank:       return 2;   // {cf, bank}
        case ResourceKind::L1Vector:     return 2;   // {cf, vector}
        case ResourceKind::RegisterFile: return 1;   // {cf}
        case ResourceKind::BlockMover:   return 2;   // {tile, edge}
        case ResourceKind::NocRouter:    return 1;   // {tile}
        case ResourceKind::NocPort:      return 1;   // {port}
        case ResourceKind::MemoryController: return 1;   // {mc}
        case ResourceKind::DmaEngine:    return 2;   // {mc, engine}
        case ResourceKind::CpuHart:      return 1;   // {hart}
        case ResourceKind::CpuSram:      return 0;
        case ResourceKind::DescriptorRing: return 0;
        case ResourceKind::CompletionRing: return 0;
    }
    return 0;
}

inline const std::vector<ResourceKind>& all_resource_kinds() {
    static const std::vector<ResourceKind> k = {
        ResourceKind::Dram,   ResourceKind::L3Tile, ResourceKind::L3Bank,
        ResourceKind::ComputeTile, ResourceKind::L2Bank, ResourceKind::L1Vector,
        ResourceKind::RegisterFile, ResourceKind::BlockMover, ResourceKind::NocRouter,
        ResourceKind::NocPort, ResourceKind::MemoryController, ResourceKind::DmaEngine,
        ResourceKind::CpuHart, ResourceKind::CpuSram, ResourceKind::DescriptorRing,
        ResourceKind::CompletionRing};
    return k;
}

// ----------------------------------------------------------------------------
// An address
// ----------------------------------------------------------------------------
struct ResourceName {
    std::string device;              // the device's NAME, as the spec spells it
    ResourceKind kind = ResourceKind::Dram;
    std::vector<Dim> path;           // length == path_arity(kind)
    std::uint64_t offset = 0;        // bytes into the resource; NEVER bounded here

    bool operator==(const ResourceName& o) const {
        return device == o.device && kind == o.kind && path == o.path && offset == o.offset;
    }
};

class NameError : public std::runtime_error {
public:
    explicit NameError(const std::string& what) : std::runtime_error(what) {}
};

// ---- formatting -------------------------------------------------------------
inline std::string format(const ResourceName& n) {
    if (n.path.size() != path_arity(n.kind))
        throw NameError(std::string("resource name: ") + to_string(n.kind) + " takes " +
                        std::to_string(path_arity(n.kind)) + " path component(s), got " +
                        std::to_string(n.path.size()));
    std::string out = n.device;
    switch (n.kind) {
        case ResourceKind::Dram:
            out += "/dram";
            break;
        case ResourceKind::L3Tile:
            out += "/l3[" + std::to_string(n.path[0]) + "]";
            break;
        case ResourceKind::L3Bank:
            out += "/l3[" + std::to_string(n.path[0]) + "]/bank[" +
                   std::to_string(n.path[1]) + "]";
            break;
        case ResourceKind::ComputeTile:
            out += "/cf[" + std::to_string(n.path[0]) + "]";
            break;
        case ResourceKind::L2Bank:
            out += "/cf[" + std::to_string(n.path[0]) + "]/l2[" +
                   std::to_string(n.path[1]) + "]";
            break;
        case ResourceKind::L1Vector:
            out += "/cf[" + std::to_string(n.path[0]) + "]/l1[" +
                   std::to_string(n.path[1]) + "]";
            break;
        case ResourceKind::RegisterFile:
            out += "/cf[" + std::to_string(n.path[0]) + "]/regs";
            break;
        case ResourceKind::BlockMover:
            out += "/l3[" + std::to_string(n.path[0]) + "]/bm[" + std::to_string(n.path[1]) + "]";
            break;
        case ResourceKind::NocRouter:
            out += "/l3[" + std::to_string(n.path[0]) + "]/noc";
            break;
        case ResourceKind::NocPort:
            out += "/noc/port[" + std::to_string(n.path[0]) + "]";
            break;
        case ResourceKind::MemoryController:
            out += "/mc[" + std::to_string(n.path[0]) + "]";
            break;
        case ResourceKind::DmaEngine:
            out += "/mc[" + std::to_string(n.path[0]) + "]/dma[" + std::to_string(n.path[1]) + "]";
            break;
        case ResourceKind::CpuHart:
            out += "/cpu/hart[" + std::to_string(n.path[0]) + "]";
            break;
        case ResourceKind::CpuSram:        out += "/cpu/sram";  break;
        case ResourceKind::DescriptorRing: out += "/cpu/dring"; break;
        case ResourceKind::CompletionRing: out += "/cpu/cring"; break;
    }
    // A zero offset is omitted, so the common address has one spelling. Without that,
    // "dev0/l3[0]" and "dev0/l3[0]+0" would be the same resource under two names, and any
    // map keyed on the text would hold it twice.
    if (n.offset != 0) out += "+" + std::to_string(n.offset);
    return out;
}

// ---- parsing ----------------------------------------------------------------
namespace detail {

// "name[<digits>]" -> the index, or nothing when the shape is wrong. Checked rather than
// stoul'd: this file is a parser for text that may come from a command line, and an
// unchecked stoul turns "-1" into an enormous index instead of an error. That mistake has
// already been made twice in this repo.
inline std::optional<Dim> bracket_index(const std::string& token, const std::string& head) {
    if (token.size() < head.size() + 3) return std::nullopt;         // head + "[x]"
    if (token.compare(0, head.size(), head) != 0) return std::nullopt;
    if (token[head.size()] != '[' || token.back() != ']') return std::nullopt;
    const std::string digits = token.substr(head.size() + 1, token.size() - head.size() - 2);
    if (digits.empty()) return std::nullopt;
    unsigned long long v = 0;
    for (char c : digits) {
        if (c < '0' || c > '9') return std::nullopt;
        v = v * 10 + static_cast<unsigned long long>(c - '0');
        if (v > 0xFFFFFFFFull) return std::nullopt;
    }
    return static_cast<Dim>(v);
}

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

} // namespace detail

// Parse an address. Throws NameError with the text in the message -- a name that does not
// parse is a usage error, and the caller should not have to guess which part was wrong.
inline ResourceName parse_resource_name(const std::string& text) {
    const std::string where = "resource name '" + text + "': ";
    std::string body = text;
    ResourceName n;

    // The offset comes last and is optional.
    const std::size_t plus = body.rfind('+');
    if (plus != std::string::npos) {
        const std::string digits = body.substr(plus + 1);
        if (digits.empty()) throw NameError(where + "'+' with no offset");
        std::uint64_t v = 0;
        for (char c : digits) {
            if (c < '0' || c > '9') throw NameError(where + "offset is not a number");
            const std::uint64_t d = static_cast<std::uint64_t>(c - '0');
            // CHECKED BEFORE THE MULTIPLY, not after. `next < v` looks like an overflow test
            // and is not one: for v = 3689348814741910323, v*10 wraps to a value GREATER
            // than v, so "+36893488147419103230" parsed clean and produced a wrong offset --
            // the precise failure this parser exists to prevent, in the code that claims to
            // prevent it.
            if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 10)
                throw NameError(where + "offset is out of range");
            v = v * 10 + d;
        }
        n.offset = v;
        body = body.substr(0, plus);
    }

    const std::vector<std::string> part = detail::split(body, '/');
    if (part.size() < 2) throw NameError(where + "expected <device>/<resource>");
    n.device = part[0];
    if (n.device.empty()) throw NameError(where + "empty device name");

    if (part.size() == 2) {
        if (part[1] == "dram") { n.kind = ResourceKind::Dram; return n; }
        if (const auto i = detail::bracket_index(part[1], "l3")) {
            n.kind = ResourceKind::L3Tile;
            n.path = {*i};
            return n;
        }
        if (const auto i = detail::bracket_index(part[1], "cf")) {
            n.kind = ResourceKind::ComputeTile;
            n.path = {*i};
            return n;
        }
        if (const auto i = detail::bracket_index(part[1], "mc")) {
            n.kind = ResourceKind::MemoryController;
            n.path = {*i};
            return n;
        }
        throw NameError(where + "'" + part[1] +
                        "' is not a resource; expected dram, l3[i], cf[i] or mc[i]");
    }
    if (part.size() == 3) {
        if (const auto t = detail::bracket_index(part[1], "l3")) {
            if (const auto b = detail::bracket_index(part[2], "bank")) {
                n.kind = ResourceKind::L3Bank;
                n.path = {*t, *b};
                return n;
            }
            if (const auto e = detail::bracket_index(part[2], "bm")) {
                n.kind = ResourceKind::BlockMover;
                n.path = {*t, *e};
                return n;
            }
            if (part[2] == "noc") {
                n.kind = ResourceKind::NocRouter;
                n.path = {*t};
                return n;
            }
            throw NameError(where + "under l3[i], expected bank[j], bm[e] or noc");
        }
        if (const auto m = detail::bracket_index(part[1], "mc")) {
            if (const auto e = detail::bracket_index(part[2], "dma")) {
                n.kind = ResourceKind::DmaEngine;
                n.path = {*m, *e};
                return n;
            }
            throw NameError(where + "under mc[i], expected dma[j]");
        }
        if (part[1] == "noc") {
            if (const auto k = detail::bracket_index(part[2], "port")) {
                n.kind = ResourceKind::NocPort;
                n.path = {*k};
                return n;
            }
            throw NameError(where + "under noc, expected port[k]");
        }
        if (part[1] == "cpu") {
            if (const auto h = detail::bracket_index(part[2], "hart")) {
                n.kind = ResourceKind::CpuHart;
                n.path = {*h};
                return n;
            }
            if (part[2] == "sram")  { n.kind = ResourceKind::CpuSram; return n; }
            if (part[2] == "dring") { n.kind = ResourceKind::DescriptorRing; return n; }
            if (part[2] == "cring") { n.kind = ResourceKind::CompletionRing; return n; }
            throw NameError(where + "under cpu, expected hart[h], sram, dring or cring");
        }
        if (const auto c = detail::bracket_index(part[1], "cf")) {
            if (part[2] == "regs") {
                n.kind = ResourceKind::RegisterFile;
                n.path = {*c};
                return n;
            }
            if (const auto b = detail::bracket_index(part[2], "l2")) {
                n.kind = ResourceKind::L2Bank;
                n.path = {*c, *b};
                return n;
            }
            if (const auto v = detail::bracket_index(part[2], "l1")) {
                n.kind = ResourceKind::L1Vector;
                n.path = {*c, *v};
                return n;
            }
            throw NameError(where + "under cf[i], expected l2[j], l1[j] or regs");
        }
        throw NameError(where + "'" + part[1] + "' has no sub-resources");
    }
    throw NameError(where + "too many path components");
}

// ----------------------------------------------------------------------------
// The map
// ----------------------------------------------------------------------------
class ResourceMap {
public:
    explicit ResourceMap(const DeploymentSpec& spec) : spec_(spec) {
        const std::string bad = spec_.validate();
        if (!bad.empty()) throw NameError("deployment: " + bad);
        build();
    }

    // Identity, not state. True when the deployment DECLARES this resource.
    bool exists(const ResourceName& n) const { return index_of(n).has_value(); }

    // A dense id, stable for one map, so an event record or a backdoor table can key on a
    // small integer instead of re-formatting a string per event (#286).
    std::optional<std::size_t> index_of(const ResourceName& n) const {
        for (std::size_t i = 0; i < names_.size(); ++i)
            if (names_[i].device == n.device && names_[i].kind == n.kind &&
                names_[i].path == n.path)
                return i;                      // the OFFSET is not part of identity
        return std::nullopt;
    }

    // Every resource the deployment declares, in a stable order: device, then kind, then
    // path. This is what #286 enumerates to lay out its stations.
    const std::vector<ResourceName>& enumerate() const { return names_; }

    std::size_t size() const { return names_.size(); }

    // Why a name does not resolve, phrased for someone who wrote the name. Empty when it
    // does resolve. An UNDECLARED FIELD is reported as such rather than as "out of range":
    // "l3.banks is not declared" and "there are only 8 banks" are different problems with
    // different fixes, and collapsing them sends the reader to the wrong one.
    std::string why_not(const ResourceName& n) const {
        if (exists(n)) return {};
        const std::size_t dev = spec_.index_of(n.device);
        if (dev == spec_.devices.size()) {
            std::string known;
            for (const DeviceSpecification& d : spec_.devices)
                known += (known.empty() ? "" : ", ") + d.name;
            return "no device named '" + n.device + "' in this deployment (has " + known + ")";
        }
        const DeviceSpecification& d = spec_.device(static_cast<Dim>(dev));
        if (n.path.size() != path_arity(n.kind))
            return std::string(to_string(n.kind)) + " takes " +
                   std::to_string(path_arity(n.kind)) + " path component(s)";
        switch (n.kind) {
            case ResourceKind::Dram:
                // Unreachable in practice: build() declares a DRAM for every device, and
                // both the device and the arity were checked above. Answered anyway,
                // because require() turns an empty diagnosis into a throw with no reason.
                return "dram: this device is declared but its DRAM is not in the map";
            case ResourceKind::L3Tile:
            case ResourceKind::L3Bank:
                if (!d.l3.tiles)
                    return "l3.tiles is not declared in this deployment, so no L3 module "
                           "can be addressed";
                if (n.path[0] >= *d.l3.tiles)
                    return "l3[" + std::to_string(n.path[0]) + "]: this device declares " +
                           std::to_string(*d.l3.tiles) + " L3 module(s)";
                if (n.kind == ResourceKind::L3Bank) {
                    if (!d.l3.banks)
                        return "l3.banks is not declared in this deployment, so no bank "
                               "can be addressed";
                    if (n.path[1] >= *d.l3.banks)
                        return "bank[" + std::to_string(n.path[1]) + "]: each L3 module "
                               "declares " + std::to_string(*d.l3.banks) + " bank(s)";
                }
                return {};
            case ResourceKind::ComputeTile:
            case ResourceKind::RegisterFile:
            case ResourceKind::L2Bank:
            case ResourceKind::L1Vector:
                if (n.path[0] >= d.compute_tiles)
                    return "cf[" + std::to_string(n.path[0]) + "]: this device has " +
                           std::to_string(d.compute_tiles) + " compute tile(s)";
                if (n.kind == ResourceKind::L2Bank) {
                    if (!d.l2.banks_per_tile)
                        return "l2.banks_per_tile is not declared in this deployment, so "
                               "no L2 bank can be addressed";
                    if (n.path[1] >= *d.l2.banks_per_tile)
                        return "l2[" + std::to_string(n.path[1]) + "]: each compute tile "
                               "declares " + std::to_string(*d.l2.banks_per_tile) +
                               " L2 bank(s)";
                }
                if (n.kind == ResourceKind::L1Vector) {
                    if (!d.l1.vectors)
                        return "l1.vectors is not declared in this deployment, so no L1 "
                               "vector can be addressed";
                    if (n.path[1] >= *d.l1.vectors)
                        return "l1[" + std::to_string(n.path[1]) + "]: each compute tile "
                               "declares " + std::to_string(*d.l1.vectors) + " L1 vector(s)";
                }
                return {};
            case ResourceKind::BlockMover:
            case ResourceKind::NocRouter:
            case ResourceKind::NocPort: {
                // Named from the LAYOUT, so the reason a name is missing is usually that the
                // device has none -- and that reason is the one worth reporting.
                std::string lw;
                const auto L = ArrayLayout::of(d, &lw);
                if (!L)
                    return std::string(to_string(n.kind)) +
                           ": this device has no array layout, because " + lw;
                if (n.kind == ResourceKind::NocPort) {
                    if (!L->has_noc()) return "port: " + L->noc_reason();
                    if (n.path[0] >= L->ports().size())
                        return "port[" + std::to_string(n.path[0]) + "]: the torus has " +
                               std::to_string(L->ports().size()) + " fold-end port(s)";
                    return {};
                }
                if (n.path[0] >= L->l3_count())
                    return "l3[" + std::to_string(n.path[0]) + "]: this device declares " +
                           std::to_string(L->l3_count()) + " L3 module(s)";
                if (n.kind == ResourceKind::NocRouter)
                    return L->has_noc() ? std::string{} : "noc: " + L->noc_reason();
                if (n.path[1] > 3)
                    return "bm[" + std::to_string(n.path[1]) +
                           "]: a BlockMover is named by its edge, 0 N, 1 E, 2 S, 3 W";
                if (!L->abutting_cf(n.path[0], static_cast<Edge>(n.path[1])))
                    return "l3[" + std::to_string(n.path[0]) + "] has no compute tile on its " +
                           to_string(static_cast<Edge>(n.path[1])) +
                           " edge, so it has no BlockMover there";
                return {};
            }
            case ResourceKind::MemoryController:
            case ResourceKind::DmaEngine: {
                if (!d.memory.controllers)
                    return "memory.controllers is not declared in this deployment, so no "
                           "memory controller or DMA engine can be addressed";
                if (n.path[0] >= *d.memory.controllers)
                    return "mc[" + std::to_string(n.path[0]) + "]: this device declares " +
                           std::to_string(*d.memory.controllers) + " memory controller(s)";
                const Dim per = d.dma.engines / *d.memory.controllers;
                if (n.kind == ResourceKind::DmaEngine && n.path[1] >= per)
                    return "dma[" + std::to_string(n.path[1]) + "]: each memory controller has " +
                           std::to_string(per) + " DMA engine(s)";
                return {};
            }
            case ResourceKind::CpuHart:
            case ResourceKind::CpuSram:
            case ResourceKind::DescriptorRing:
            case ResourceKind::CompletionRing:
                if (!d.cpu.harts)
                    return "cpu.harts is not declared in this deployment, so the attached CPU "
                           "cannot be addressed";
                if (n.kind == ResourceKind::CpuHart && n.path[0] >= *d.cpu.harts)
                    return "hart[" + std::to_string(n.path[0]) + "]: this device declares " +
                           std::to_string(*d.cpu.harts) + " hart(s)";
                return {};
        }
        return "unknown resource kind";
    }

    // Convenience: parse and resolve in one step, throwing with why_not()'s diagnosis.
    ResourceName require(const std::string& text) const {
        const ResourceName n = parse_resource_name(text);
        if (exists(n)) return n;
        const std::string bad = why_not(n);
        throw NameError("resource name '" + text + "': " +
                        (bad.empty() ? "this deployment does not declare it" : bad));
    }

private:
    void build() {
        for (const DeviceSpecification& d : spec_.devices) {
            // DRAM exists because the data path starts there: every device has DMA engines
            // (validate() requires at least one), and a DMA moves DRAM <-> L3, so a device
            // without a DRAM side would have engines with nothing to read. Its EXTENT is
            // unknown -- the spec declares no size -- which is why an offset is never
            // bounded (see the header).
            names_.push_back(ResourceName{d.name, ResourceKind::Dram, {}, 0});

            if (d.l3.tiles) {
                for (Dim t = 0; t < *d.l3.tiles; ++t) {
                    names_.push_back(ResourceName{d.name, ResourceKind::L3Tile, {t}, 0});
                    if (d.l3.banks)
                        for (Dim b = 0; b < *d.l3.banks; ++b)
                            names_.push_back(
                                ResourceName{d.name, ResourceKind::L3Bank, {t, b}, 0});
                }
            }
            for (Dim c = 0; c < d.compute_tiles; ++c) {
                names_.push_back(ResourceName{d.name, ResourceKind::ComputeTile, {c}, 0});
                // A compute tile has a register file for the same reason it has a DRAM on
                // the far side of its DMA: the fabric cannot hold an operand without one.
                // The spec declares no count because there is one per tile.
                names_.push_back(ResourceName{d.name, ResourceKind::RegisterFile, {c}, 0});
                if (d.l2.banks_per_tile)
                    for (Dim b = 0; b < *d.l2.banks_per_tile; ++b)
                        names_.push_back(ResourceName{d.name, ResourceKind::L2Bank, {c, b}, 0});
                if (d.l1.vectors)
                    for (Dim v = 0; v < *d.l1.vectors; ++v)
                        names_.push_back(ResourceName{d.name, ResourceKind::L1Vector, {c, v}, 0});
            }

            // ---- the physical shape, AFTER every pre-existing kind, so a resource that had
            //      a dense index before this was added keeps it.
            if (const auto L = ArrayLayout::of(d)) {
                for (const BlockMoverSite& m : L->block_movers())
                    names_.push_back(ResourceName{d.name, ResourceKind::BlockMover,
                                                  {m.l3, static_cast<Dim>(m.edge)}, 0});
                if (L->has_noc()) {
                    for (Dim t = 0; t < L->l3_count(); ++t)
                        names_.push_back(ResourceName{d.name, ResourceKind::NocRouter, {t}, 0});
                    for (const NocPort& p : L->ports())
                        names_.push_back(ResourceName{d.name, ResourceKind::NocPort, {p.index}, 0});
                }
            }
            if (d.memory.controllers) {
                const Dim per = d.dma.engines / *d.memory.controllers;
                for (Dim m = 0; m < *d.memory.controllers; ++m) {
                    names_.push_back(ResourceName{d.name, ResourceKind::MemoryController, {m}, 0});
                    for (Dim e = 0; e < per; ++e)
                        names_.push_back(ResourceName{d.name, ResourceKind::DmaEngine, {m, e}, 0});
                }
            }
            if (d.cpu.harts) {
                for (Dim h = 0; h < *d.cpu.harts; ++h)
                    names_.push_back(ResourceName{d.name, ResourceKind::CpuHart, {h}, 0});
                names_.push_back(ResourceName{d.name, ResourceKind::CpuSram, {}, 0});
                names_.push_back(ResourceName{d.name, ResourceKind::DescriptorRing, {}, 0});
                names_.push_back(ResourceName{d.name, ResourceKind::CompletionRing, {}, 0});
            }
        }
    }

    DeploymentSpec spec_;
    std::vector<ResourceName> names_;
};

} // namespace sw::kpu::program::platform

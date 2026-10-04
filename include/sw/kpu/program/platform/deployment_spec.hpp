// ============================================================================
// include/sw/kpu/program/platform/deployment_spec.hpp
// A deployment is DATA, not code (ADR 0002 §3.5).
//
// THIS IS THE DEVICE DESCRIPTION, not a fifth one. The repo already described a
// machine in four places — `driver::DeviceSpec` (CLI flags), `DeviceDescriptor`
// (what the executors schedule on), `Placement` (which compute tiles), and the ADR's
// JSON sketch (nothing yet). Adding a peer to those would repeat, one layer up, the
// mistake this program exists to undo: four disconnected engines, each with its own
// notion of the machine.
//
// So `DeploymentSpec` is the description and the others become VIEWS of it:
//
//   spec.device_view(i)  ->  DeviceDescriptor   for the levels that need only counts
//                                               and bandwidths (L-B, L-T1)
//   driver::DeviceSpec   ->  DeploymentSpec     the CLI builds a spec, not a descriptor
//   Placement                                   stays separate: it is a property of a
//                                               RUN, not of the machine
//
// The spec is a strict SUPERSET of what any level models today — L3 banks, L2 banks
// per compute tile, L1 vectors and DMA burst belong to the §3.3 resource vocabulary,
// which is L-T2's work (#283). The projection therefore loses fields, and a level
// must SAY which ones it ignored rather than ignoring them quietly (see
// `unmodelled_fields` in driver/execution_level.hpp). That is the same discipline as
// `not_implemented_reason`: this tool already refuses to let a clean report be
// mistaken for full coverage, and a deployment field that silently evaporates is
// exactly that mistake one layer down.
//
// WHY THE UNMODELLED FIELDS ARE `std::optional`. Absent means "not declared", and a
// default value is not a declaration. Without that distinction every run would
// report four unmodelled fields forever, the report would become noise, and noise is
// how a real one gets missed. Optionality is how the spec says "I meant this".
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/platform/digest.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::platform {

using characterize::DeviceDescriptor;
using characterize::Topology;

// ----------------------------------------------------------------------------
// One device
// ----------------------------------------------------------------------------
struct DeviceSpecification {
    // A name, because the naming map (#282 increment 3) addresses resources as
    // (device, kind, instance, offset) and a positional index is not a name.
    std::string name = "dev0";
    std::string topology = "single";            // single | news | checkerboard
    Dim compute_tiles = 1;
    double macs_per_cycle = 256.0;              // ONE compute tile's throughput
    Dim element_bytes = 4;

    // DMA: DRAM <-> L3.
    struct Dma {
        Dim engines = 1;
        double bytes_per_cycle = 64.0;          // PER ENGINE (§6.3: per lane, always)
        std::optional<Dim> burst_bytes;         // §3.3 resource vocabulary — L-T2
    } dma;

    // L3. `tiles` and `capacity_tiles` are DIFFERENT THINGS and conflating them would
    // be a silently wrong machine: `tiles` is how many L3 modules exist (what the
    // naming map enumerates), `capacity_tiles` is how many tile-sized buffers the
    // whole L3 holds (what the credit model bounds). `DeviceDescriptor::l3_tiles` is
    // the CAPACITY, despite its name.
    struct L3 {
        std::optional<Dim> tiles;               // how many L3 modules — naming map
        std::optional<Dim> banks;               // §3.3 — L-T2
        Dim capacity_tiles = 0;                 // 0 = unbounded; modelled at L-T1
    } l3;

    struct L2 { std::optional<Dim> banks_per_tile; } l2;   // §3.3 — L-T2
    struct L1 { std::optional<Dim> vectors; } l1;          // §3.3 — L-T2

    // Movers, per CSP process. Lanes belong to the process and are shared between
    // directions; there is no collapsed hop and no aggregate movement pool.
    struct Movers {
        Dim block_movers = 1;
        double bm_bytes_per_cycle = 128.0;      // per mover, L3 <-> L2
        Dim streamers = 1;
        double str_bytes_per_cycle = 256.0;     // per streamer, L2 <-> L1
        Dim noc_links = 0;                      // 0 = no L3 <-> L3 path
        double noc_bytes_per_cycle = 128.0;     // per link
    } movers;

    // The PHYSICAL SHAPE (#286 step 1; array_layout.hpp). All optional, all additive (R8):
    // absent means "not declared", and a deployment without them is exactly as valid as
    // before. They are what lets the naming map name a BlockMover by the L3 edge it sits on
    // and a DMA engine by the memory controller it hangs off.
    //
    //   array.rows/cols     the checkerboard's grid; absent = derived (squarest even grid)
    //   memory.controllers  how many memory controllers; dma.engines are split evenly
    //   cpu.harts           the attached RV64 orchestrator's harts (ADR 0003)
    struct Array { std::optional<Dim> rows, cols; } array;
    //
    //   memory.dram         the DRAM behind the controllers and how an address lands in it
    //                       (docs/plans/dram-bank-model.md §3.1). Absent = no geometry
    //                       declared, and the controller model keeps its built-in mapping.
    struct Memory {
        std::optional<Dim> controllers;
        // One XOR fold of the address map: the low `bits` bits of field `into` are XORed
        // with bits [from_lsb, from_lsb + bits) of field `from`. `from` is never itself
        // folded into, so decode and encode stay inverses (dram_problem() enforces it).
        struct XorFold {
            std::string into, from;
            Dim bits = 0;
            Dim from_lsb = 0;
        };
        // Every count is a power of two, because each one is a bit field of the address.
        // The map lists the fields low -> high ABOVE the burst offset:
        //   co  burst within the page (page_bytes / burst_bytes)    ch  channel per controller
        //   rk  rank                  bg  bank group                ba  bank within its group
        //   mc  memory controller (memory.controllers, else 1)      ro  row
        // `ro` takes whatever capacity_bytes leaves, so the map covers the capacity exactly.
        struct Dram {
            std::string technology = "lpddr5x";
            Dim channels = 2;                   // per controller: x16 channels, so 2 = 32-bit
            Dim channel_width_bits = 16;
            Dim ranks = 1;
            Dim bank_groups = 4;
            Dim banks_per_group = 4;
            Dim page_bytes = 2048;              // row-buffer size, per channel
            Dim burst_bytes = 64;
            Dim data_rate_mtps = 8533;
            std::uint64_t capacity_bytes = 0;   // top of memory; required, a power of two
            std::string map = "co:ch:rk:bg:ba:mc:ro";
            std::vector<XorFold> xor_folds;
        };
        std::optional<Dram> dram;
    } memory;
    struct Cpu { std::optional<Dim> harts; } cpu;

    // Analytical-harness coefficients. The executors do not use these; the
    // first-order model is a different tier with a different job.
    struct Analytical {
        double bytes_per_cycle = 64.0;
        double pj_per_mac = 1.0;
        double pj_per_byte = 20.0;
    } analytical;
};

// ----------------------------------------------------------------------------
// The deployment
// ----------------------------------------------------------------------------
// MULTI-DEVICE FROM THE START. The backdoor (#284) requires every resource to be
// addressable and the naming map spans devices (§3.4, §6.4), so a single-device
// deployment is a list with one device in it — one index now, instead of a format
// migration later.
struct DeploymentSpec {
    std::vector<DeviceSpecification> devices{DeviceSpecification{}};

    Dim device_count() const { return static_cast<Dim>(devices.size()); }

    const DeviceSpecification& device(Dim i) const { return devices.at(i); }
    DeviceSpecification& device(Dim i) { return devices.at(i); }

    // Index of a named device, or devices.size() if there is none.
    std::size_t index_of(const std::string& name) const {
        for (std::size_t i = 0; i < devices.size(); ++i)
            if (devices[i].name == name) return i;
        return devices.size();
    }

    // Empty when valid; otherwise the first problem, phrased for a user who wrote a
    // spec file. Checked rather than trusted: a spec with zero compute tiles or a
    // zero bandwidth would divide by zero deep inside an executor, where the message
    // names an executor internal instead of the field that was wrong.
    std::string validate() const;

    // The projection onto what L-B and L-T1 schedule. Loses every field marked "L-T2"
    // above, which is why a caller reports them (see unmodelled_fields).
    DeviceDescriptor device_view(Dim i = 0) const;

    std::string label(Dim i = 0) const {
        return devices.empty() ? std::string("<no devices>")
                               : device_view(i).label() +
                                     (devices.size() > 1
                                          ? "/x" + std::to_string(devices.size())
                                          : std::string());
    }
};

inline bool known_topology_name(const std::string& t) {
    return t == "single" || t == "news" || t == "checkerboard";
}

// A device name is also an ADDRESS: the naming map spells a resource
// "<device>/cf[2]/l2[3]+64" (resource_map.hpp), so these four characters are the grammar
// and a name containing one would be unparseable. Enforced in validate(), because a device
// nothing can address is a device the backdoor (#284) cannot reach -- and finding that out
// at the first backdoor write is far worse than finding it at deployment.
inline constexpr const char* kAddressReservedChars = "/[]+";

// Every double in a spec has to survive the JSON round trip, and a non-finite one does not:
// it serializes as `null`. So "finite" is a representability requirement, not a taste.
inline bool finite_positive(double v) { return std::isfinite(v) && v > 0.0; }
inline bool finite_non_negative(double v) { return std::isfinite(v) && v >= 0.0; }

inline constexpr const char* kFinitePos = " must be finite and positive";
inline constexpr const char* kFiniteNonNeg = " must be finite and non-negative";

// ----------------------------------------------------------------------------
// DRAM geometry (docs/plans/dram-bank-model.md §3.1)
// ----------------------------------------------------------------------------
// The address-map field names, in no particular order; a map lists each exactly once.
inline const std::vector<std::string>& dram_map_fields() {
    static const std::vector<std::string> f = {"co", "ch", "rk", "bg", "ba", "mc", "ro"};
    return f;
}

inline bool known_dram_technology(const std::string& t) {
    return t == "lpddr5" || t == "lpddr5x" || t == "ddr5" || t == "hbm2" || t == "hbm3" ||
           t == "gddr6" || t == "gddr7";
}

inline bool power_of_two(std::uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

inline unsigned log2_exact(std::uint64_t v) {
    unsigned n = 0;
    while (v > 1) { v >>= 1; ++n; }
    return n;
}

// The map split into its fields, or empty when a field is unknown, repeated or missing.
inline std::vector<std::string> split_dram_map(const std::string& map) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at <= map.size()) {
        const std::size_t colon = std::min(map.find(':', at), map.size());
        out.push_back(map.substr(at, colon - at));
        at = colon + 1;
    }
    const auto& known = dram_map_fields();
    if (out.size() != known.size()) return {};
    for (const std::string& f : known)
        if (std::count(out.begin(), out.end(), f) != 1) return {};
    return out;
}

// Bit width of each map field, keyed by name, for a device whose dram is declared and whose
// counts are powers of two. `ro` may come out negative when the capacity is too small for the
// rest of the geometry, which dram_problem() refuses.
inline std::map<std::string, int> dram_field_bits(const DeviceSpecification& d) {
    const auto& m = *d.memory.dram;
    std::map<std::string, int> w;
    w["co"] = static_cast<int>(log2_exact(m.page_bytes / m.burst_bytes));
    w["ch"] = static_cast<int>(log2_exact(m.channels));
    w["rk"] = static_cast<int>(log2_exact(m.ranks));
    w["bg"] = static_cast<int>(log2_exact(m.bank_groups));
    w["ba"] = static_cast<int>(log2_exact(m.banks_per_group));
    w["mc"] = static_cast<int>(log2_exact(d.memory.controllers.value_or(1)));
    int below = static_cast<int>(log2_exact(m.burst_bytes));
    for (const auto& [k, v] : w) below += v;
    w["ro"] = static_cast<int>(log2_exact(m.capacity_bytes)) - below;
    return w;
}

// Empty when the device declares no DRAM or a consistent one; otherwise the first problem.
inline std::string dram_problem(const DeviceSpecification& d) {
    if (!d.memory.dram) return {};
    const auto& m = *d.memory.dram;
    const std::string w = "memory.dram";
    if (!known_dram_technology(m.technology))
        return w + ".technology '" + m.technology +
               "' is not one of lpddr5 | lpddr5x | ddr5 | hbm2 | hbm3 | gddr6 | gddr7";
    if (m.channel_width_bits == 0) return w + ".channel_width_bits must be non-zero";
    if (m.data_rate_mtps == 0) return w + ".data_rate_mtps must be non-zero";
    const std::pair<const char*, std::uint64_t> counts[] = {
        {"channels", m.channels},       {"ranks", m.ranks},
        {"bank_groups", m.bank_groups}, {"banks_per_group", m.banks_per_group},
        {"page_bytes", m.page_bytes},   {"burst_bytes", m.burst_bytes},
        {"capacity_bytes", m.capacity_bytes}};
    for (const auto& [name, v] : counts)
        if (!power_of_two(v))
            return w + "." + name + " (" + std::to_string(v) +
                   ") must be a non-zero power of two, because it is a bit field of the address";
    if (d.memory.controllers && !power_of_two(*d.memory.controllers))
        return "memory.controllers (" + std::to_string(*d.memory.controllers) +
               ") must be a power of two when memory.dram is declared, because the map "
               "interleaves controllers by address bits";
    if (m.page_bytes < m.burst_bytes)
        return w + ".page_bytes (" + std::to_string(m.page_bytes) +
               ") is smaller than a burst (" + std::to_string(m.burst_bytes) + ")";
    if (split_dram_map(m.map).empty())
        return w + ".map '" + m.map + "' must list each of co, ch, rk, bg, ba, mc, ro exactly "
               "once, separated by ':'";
    const auto bits = dram_field_bits(d);
    if (bits.at("ro") < 0)
        return w + ".capacity_bytes (" + std::to_string(m.capacity_bytes) +
               ") is smaller than one row across every controller, channel, rank and bank";
    if (d.dma.burst_bytes && *d.dma.burst_bytes % m.burst_bytes != 0)
        return "dma.burst_bytes (" + std::to_string(*d.dma.burst_bytes) +
               ") must be a whole number of DRAM bursts (" + std::to_string(m.burst_bytes) + ")";
    for (std::size_t i = 0; i < m.xor_folds.size(); ++i) {
        const auto& f = m.xor_folds[i];
        const std::string at = w + ".xor_folds[" + std::to_string(i) + "]";
        if (!bits.count(f.into) || !bits.count(f.from))
            return at + " names an unknown field (into '" + f.into + "', from '" + f.from + "')";
        if (f.into == f.from) return at + " folds a field into itself";
        if (f.bits == 0) return at + ".bits must be non-zero";
        // Compared in 64 bits: casting a Dim to int turns 2^32 - 1 into -1, which passes any
        // width, and from_lsb + bits in 32 bits can wrap below a width it exceeds.
        if (std::uint64_t{f.bits} > static_cast<std::uint64_t>(bits.at(f.into)))
            return at + " folds " + std::to_string(f.bits) + " bits into '" + f.into +
                   "', which has " + std::to_string(bits.at(f.into));
        if (std::uint64_t{f.from_lsb} + f.bits > static_cast<std::uint64_t>(bits.at(f.from)))
            return at + " reads bits [" + std::to_string(f.from_lsb) + ", " +
                   std::to_string(std::uint64_t{f.from_lsb} + f.bits) + ") of '" + f.from + "', which has " +
                   std::to_string(bits.at(f.from));
        // A field that is both folded into and folded from would make decode depend on the
        // order of the folds, and encode would no longer invert it.
        for (const auto& g : m.xor_folds)
            if (g.into == f.from)
                return at + " reads '" + f.from + "', which another fold writes; a field is "
                       "folded into or from, never both";
    }
    return {};
}

inline std::string DeploymentSpec::validate() const {
    if (devices.empty()) return "a deployment needs at least one device";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const DeviceSpecification& d = devices[i];
        const std::string where = "device " + std::to_string(i) + " (\"" + d.name + "\")";
        if (d.name.empty()) return where + ": a device needs a name";
        // Names are how the backdoor and the naming map address a resource, so two
        // devices sharing one name would make an address ambiguous rather than merely
        // confusing.
        for (std::size_t j = 0; j < i; ++j)
            if (devices[j].name == d.name)
                return where + ": duplicate device name; a name has to identify one device";
        if (d.name.find_first_of(kAddressReservedChars) != std::string::npos)
            return where + ": a device name may not contain any of " +
                   kAddressReservedChars + ", because those characters are the resource "
                   "address grammar";
        if (!known_topology_name(d.topology))
            return where + ": unknown topology '" + d.topology +
                   "' (single | news | checkerboard)";
        if (d.compute_tiles == 0) return where + ": compute_tiles must be non-zero";
        if (d.element_bytes == 0) return where + ": element_bytes must be non-zero";
        // FINITE, not merely positive. `!(x > 0.0)` lets +inf through, and an infinite
        // bandwidth is not a fast machine -- it is a makespan of 0 or a NaN, reported as
        // a result. It is reachable: std::stod parses "inf", so `--dma-bytes-per-cycle inf`
        // reached this check and passed it.
        //
        // It also broke the round-trip the digest depends on, which is the worse half:
        // nlohmann writes a non-finite double as JSON `null`, and null is not a number, so
        // a spec this function ACCEPTED could serialize to bytes it then REFUSED to read
        // back. A validator that admits values the format cannot represent is not a
        // validator.
        if (!finite_positive(d.macs_per_cycle)) return where + ": macs_per_cycle" + kFinitePos;
        if (d.dma.engines == 0) return where + ": dma.engines must be non-zero";
        if (!finite_positive(d.dma.bytes_per_cycle))
            return where + ": dma.bytes_per_cycle" + kFinitePos;
        if (d.movers.block_movers == 0) return where + ": movers.block_movers must be non-zero";
        if (!finite_positive(d.movers.bm_bytes_per_cycle))
            return where + ": movers.bm_bytes_per_cycle" + kFinitePos;
        if (d.movers.streamers == 0) return where + ": movers.streamers must be non-zero";
        if (!finite_positive(d.movers.str_bytes_per_cycle))
            return where + ": movers.str_bytes_per_cycle" + kFinitePos;
        if (!finite_positive(d.movers.noc_bytes_per_cycle))
            return where + ": movers.noc_bytes_per_cycle" + kFinitePos;
        if (!finite_positive(d.analytical.bytes_per_cycle))
            return where + ": analytical.bytes_per_cycle" + kFinitePos;
        // The ENERGY coefficients were not validated at all. Zero is a legitimate modelling
        // choice -- "ignore compute energy" -- so they are finite and non-negative rather
        // than positive.
        if (!finite_non_negative(d.analytical.pj_per_mac))
            return where + ": analytical.pj_per_mac" + kFiniteNonNeg;
        if (!finite_non_negative(d.analytical.pj_per_byte))
            return where + ": analytical.pj_per_byte" + kFiniteNonNeg;
        // Zero is legal for the optional counts only in the sense that declaring zero
        // of a resource is a statement; a zero BANK count is not, since a memory with
        // no banks cannot hold anything.
        if (d.l3.banks && *d.l3.banks == 0) return where + ": l3.banks declared as zero";
        if (d.l3.tiles && *d.l3.tiles == 0) return where + ": l3.tiles declared as zero";
        if (d.l2.banks_per_tile && *d.l2.banks_per_tile == 0)
            return where + ": l2.banks_per_tile declared as zero";
        if (d.l1.vectors && *d.l1.vectors == 0) return where + ": l1.vectors declared as zero";
        if (d.dma.burst_bytes && *d.dma.burst_bytes == 0)
            return where + ": dma.burst_bytes declared as zero";
        // The physical shape. Validated only when DECLARED: an absent shape is derived where
        // it can be (array_layout.hpp), and a spec that never mentions one stays valid.
        if (d.array.rows.has_value() != d.array.cols.has_value())
            return where + ": array.rows and array.cols are declared together or not at all";
        if (d.array.rows) {
            if (*d.array.rows == 0 || *d.array.cols == 0)
                return where + ": array.rows and array.cols must be non-zero";
            if (d.topology != "checkerboard")
                return where + ": an array shape applies to the checkerboard topology, not '" +
                       d.topology + "'";
            if (*d.array.rows % 2 != 0 || *d.array.cols % 2 != 0)
                return where + ": array.rows and array.cols must be even, because the folded "
                       "torus pairs rows and columns into loops";
            if (static_cast<std::uint64_t>(*d.array.rows) * *d.array.cols !=
                2ull * d.compute_tiles)
                return where + ": an alternating checkerboard of " +
                       std::to_string(*d.array.rows) + "x" + std::to_string(*d.array.cols) +
                       " holds " + std::to_string(*d.array.rows * *d.array.cols / 2) +
                       " compute tiles, but compute_tiles is " +
                       std::to_string(d.compute_tiles);
            if (d.l3.tiles && *d.l3.tiles != d.compute_tiles)
                return where + ": an alternating checkerboard has as many L3 tiles as compute "
                       "tiles, so a declared array needs l3.tiles = compute_tiles";
        }
        if (d.memory.controllers) {
            if (*d.memory.controllers == 0)
                return where + ": memory.controllers declared as zero";
            if (d.dma.engines % *d.memory.controllers != 0)
                return where + ": dma.engines (" + std::to_string(d.dma.engines) +
                       ") must divide evenly across memory.controllers (" +
                       std::to_string(*d.memory.controllers) +
                       "), because each DMA engine belongs to one memory controller";
        }
        if (d.cpu.harts && *d.cpu.harts == 0) return where + ": cpu.harts declared as zero";
        if (std::string why = dram_problem(d); !why.empty()) return where + ": " + why;
    }
    return {};
}

inline DeviceDescriptor DeploymentSpec::device_view(Dim i) const {
    const DeviceSpecification& s = device(i);
    DeviceDescriptor d = DeviceDescriptor::single();
    if (s.topology == "news") d = DeviceDescriptor::news();
    else if (s.topology == "checkerboard") d = DeviceDescriptor::checkerboard(s.compute_tiles);

    d.compute_tiles = s.compute_tiles;
    // Aggregate lanes, for the analytical harness only — the presets above already set
    // this, but compute_tiles may have moved since.
    if (s.topology == "single") d.move_lanes = 1;
    else if (s.topology == "news") d.move_lanes = 4;
    else d.move_lanes = s.compute_tiles;

    d.fabric_macs_per_cycle = s.macs_per_cycle;
    d.element_bytes = static_cast<double>(s.element_bytes);
    d.l3_tiles = s.l3.capacity_tiles;            // CAPACITY, not the module count

    d.dma_engines = s.dma.engines;
    d.dma_bytes_per_cycle = s.dma.bytes_per_cycle;
    d.block_movers = s.movers.block_movers;
    d.bm_bytes_per_cycle = s.movers.bm_bytes_per_cycle;
    d.streamers = s.movers.streamers;
    d.str_bytes_per_cycle = s.movers.str_bytes_per_cycle;
    d.noc_links = s.movers.noc_links;
    d.noc_bytes_per_cycle = s.movers.noc_bytes_per_cycle;

    d.bytes_per_cycle = s.analytical.bytes_per_cycle;
    d.pj_per_mac = s.analytical.pj_per_mac;
    d.pj_per_byte = s.analytical.pj_per_byte;
    return d;
}

// ----------------------------------------------------------------------------
// The fields a level may have to admit it ignored
// ----------------------------------------------------------------------------
// Named as data so the report and the "does this level model it" table cannot drift
// apart — a table keyed on a string typed twice is a table that disagrees with itself.
enum class SpecField { L3Tiles, L3Banks, L2BanksPerTile, L1Vectors, DmaBurst, L3Capacity, Dram };

inline const char* to_string(SpecField f) {
    switch (f) {
        case SpecField::L3Tiles:        return "l3.tiles";
        case SpecField::L3Banks:        return "l3.banks";
        case SpecField::L2BanksPerTile: return "l2.banks_per_tile";
        case SpecField::L1Vectors:      return "l1.vectors";
        case SpecField::DmaBurst:       return "dma.burst_bytes";
        case SpecField::L3Capacity:     return "l3.capacity_tiles";
        case SpecField::Dram:           return "memory.dram";
    }
    return "?";
}

// Was the field DECLARED? For the optional ones that is "is it set"; `l3.capacity_tiles`
// is not optional because 0 already means "unbounded", so declaring it means a non-zero
// bound.
inline bool declared(const DeviceSpecification& s, SpecField f) {
    switch (f) {
        case SpecField::L3Tiles:        return s.l3.tiles.has_value();
        case SpecField::L3Banks:        return s.l3.banks.has_value();
        case SpecField::L2BanksPerTile: return s.l2.banks_per_tile.has_value();
        case SpecField::L1Vectors:      return s.l1.vectors.has_value();
        case SpecField::DmaBurst:       return s.dma.burst_bytes.has_value();
        case SpecField::L3Capacity:     return s.l3.capacity_tiles != 0;
        case SpecField::Dram:           return s.memory.dram.has_value();
    }
    return false;
}

inline std::string declared_value(const DeviceSpecification& s, SpecField f) {
    switch (f) {
        case SpecField::L3Tiles:        return std::to_string(s.l3.tiles.value_or(0));
        case SpecField::L3Banks:        return std::to_string(s.l3.banks.value_or(0));
        case SpecField::L2BanksPerTile: return std::to_string(s.l2.banks_per_tile.value_or(0));
        case SpecField::L1Vectors:      return std::to_string(s.l1.vectors.value_or(0));
        case SpecField::DmaBurst:       return std::to_string(s.dma.burst_bytes.value_or(0));
        case SpecField::L3Capacity:     return std::to_string(s.l3.capacity_tiles);
        case SpecField::Dram: {
            if (!s.memory.dram) return "0";
            const auto& m = *s.memory.dram;
            return m.technology + ", " + std::to_string(s.memory.controllers.value_or(1)) +
                   " mc x " + std::to_string(m.channels) + " ch x " +
                   std::to_string(m.bank_groups * m.banks_per_group) + " banks, " +
                   std::to_string(m.capacity_bytes) + " B, map " + m.map;
        }
    }
    return "?";
}

inline const std::vector<SpecField>& all_spec_fields() {
    static const std::vector<SpecField> f = {
        SpecField::L3Tiles, SpecField::L3Banks, SpecField::L2BanksPerTile,
        SpecField::L1Vectors, SpecField::DmaBurst, SpecField::L3Capacity, SpecField::Dram};
    return f;
}

} // namespace sw::kpu::program::platform

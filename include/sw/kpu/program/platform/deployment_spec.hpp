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
        if (!known_topology_name(d.topology))
            return where + ": unknown topology '" + d.topology +
                   "' (single | news | checkerboard)";
        if (d.compute_tiles == 0) return where + ": compute_tiles must be non-zero";
        if (d.element_bytes == 0) return where + ": element_bytes must be non-zero";
        if (!(d.macs_per_cycle > 0.0)) return where + ": macs_per_cycle must be positive";
        if (d.dma.engines == 0) return where + ": dma.engines must be non-zero";
        if (!(d.dma.bytes_per_cycle > 0.0))
            return where + ": dma.bytes_per_cycle must be positive";
        if (d.movers.block_movers == 0) return where + ": movers.block_movers must be non-zero";
        if (!(d.movers.bm_bytes_per_cycle > 0.0))
            return where + ": movers.bm_bytes_per_cycle must be positive";
        if (d.movers.streamers == 0) return where + ": movers.streamers must be non-zero";
        if (!(d.movers.str_bytes_per_cycle > 0.0))
            return where + ": movers.str_bytes_per_cycle must be positive";
        if (!(d.movers.noc_bytes_per_cycle > 0.0))
            return where + ": movers.noc_bytes_per_cycle must be positive";
        if (!(d.analytical.bytes_per_cycle > 0.0))
            return where + ": analytical.bytes_per_cycle must be positive";
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
enum class SpecField { L3Tiles, L3Banks, L2BanksPerTile, L1Vectors, DmaBurst, L3Capacity };

inline const char* to_string(SpecField f) {
    switch (f) {
        case SpecField::L3Tiles:        return "l3.tiles";
        case SpecField::L3Banks:        return "l3.banks";
        case SpecField::L2BanksPerTile: return "l2.banks_per_tile";
        case SpecField::L1Vectors:      return "l1.vectors";
        case SpecField::DmaBurst:       return "dma.burst_bytes";
        case SpecField::L3Capacity:     return "l3.capacity_tiles";
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
    }
    return "?";
}

inline const std::vector<SpecField>& all_spec_fields() {
    static const std::vector<SpecField> f = {
        SpecField::L3Tiles, SpecField::L3Banks, SpecField::L2BanksPerTile,
        SpecField::L1Vectors, SpecField::DmaBurst, SpecField::L3Capacity};
    return f;
}

} // namespace sw::kpu::program::platform

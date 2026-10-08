// ============================================================================
// src/program/deployment_json.cpp
// The JSON edge of a DeploymentSpec. See the header for the format contract.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <set>
#include <sstream>

namespace sw::kpu::program::platform {

namespace {

// ORDERED, not the default. nlohmann's `json` sorts keys, which is deterministic but puts
// "analytical" before "name" -- and a spec file is meant to be read and diffed, which is the
// same reason the L0 format is text at all. Insertion order lets the canonical output read
// like the ADR's example, and the writer fixes the order, so the bytes stay deterministic.
using json = nlohmann::ordered_json;

// Every key this reader understands, per object. An unknown key is refused against
// these sets, so the vocabulary is stated ONCE rather than implied by a chain of
// if-present checks — which is what lets the refusal list the keys that would work.
const std::set<std::string>& device_keys() {
    static const std::set<std::string> k = {"name",          "topology", "compute_tiles",
                                            "macs_per_cycle", "element_bytes",
                                            "dma",           "l3",       "l2",
                                            "l1",            "movers",   "analytical",
                                            "array",         "memory",   "cpu",
                                            "noc"};
    return k;
}
const std::set<std::string>& dma_keys() {
    // "burst" is the ADR's spelling; "burst_bytes" is canonical (§ header).
    static const std::set<std::string> k = {"engines", "bytes_per_cycle", "burst",
                                            "burst_bytes", "window"};
    return k;
}
const std::set<std::string>& l3_keys() {
    static const std::set<std::string> k = {"tiles", "banks", "capacity_tiles"};
    return k;
}
const std::set<std::string>& l2_keys() {
    static const std::set<std::string> k = {"banks_per_tile"};
    return k;
}
const std::set<std::string>& l1_keys() {
    static const std::set<std::string> k = {"vectors"};
    return k;
}
const std::set<std::string>& array_keys() {
    static const std::set<std::string> k = {"rows", "cols"};
    return k;
}
const std::set<std::string>& memory_keys() {
    static const std::set<std::string> k = {"controllers", "dram"};
    return k;
}
const std::set<std::string>& dram_keys() {
    static const std::set<std::string> k = {
        "technology",  "channels",   "channel_width_bits", "ranks",
        "bank_groups", "banks_per_group", "page_bytes",    "burst_bytes",
        "data_rate_mtps", "capacity_bytes", "map",         "xor_folds"};
    return k;
}
const std::set<std::string>& xor_fold_keys() {
    static const std::set<std::string> k = {"into", "from", "bits", "from_lsb"};
    return k;
}
const std::set<std::string>& noc_keys() {
    static const std::set<std::string> k = {"hub_buffer_blocks", "port"};
    return k;
}
const std::set<std::string>& noc_port_keys() {
    static const std::set<std::string> k = {"input_queue_blocks", "output_queue_blocks",
                                            "arbitration"};
    return k;
}
const std::set<std::string>& cpu_keys() {
    static const std::set<std::string> k = {"harts"};
    return k;
}
const std::set<std::string>& mover_keys() {
    static const std::set<std::string> k = {"block_movers",   "bm_bytes_per_cycle",
                                            "streamers",      "str_bytes_per_cycle",
                                            "noc_links",      "noc_bytes_per_cycle"};
    return k;
}
const std::set<std::string>& analytical_keys() {
    static const std::set<std::string> k = {"bytes_per_cycle", "pj_per_mac", "pj_per_byte"};
    return k;
}

std::string joined(const std::set<std::string>& keys) {
    std::string out;
    for (const std::string& k : keys) out += (out.empty() ? "" : ", ") + k;
    return out;
}

void reject_unknown(const json& obj, const std::set<std::string>& allowed,
                    const std::string& where) {
    if (!obj.is_object())
        throw SpecError("deployment: " + where + " must be a JSON object");
    for (auto it = obj.begin(); it != obj.end(); ++it)
        if (allowed.count(it.key()) == 0)
            throw SpecError("deployment: " + where + ": unknown key '" + it.key() +
                            "'; understood keys are " + joined(allowed));
}

// Numbers are read through these rather than json::get<>, so a wrong TYPE is a spec
// error with a field name rather than an nlohmann type exception, and so a negative
// count cannot wrap into an enormous Dim.
Dim read_dim(const json& obj, const char* key, Dim fallback, const std::string& where) {
    if (!obj.contains(key)) return fallback;
    const json& v = obj.at(key);
    if (!v.is_number_integer() && !v.is_number_unsigned())
        throw SpecError("deployment: " + where + "." + key + " must be an integer");
    const long long raw = v.get<long long>();
    if (raw < 0)
        throw SpecError("deployment: " + where + "." + key + " must not be negative (got " +
                        std::to_string(raw) + ")");
    if (raw > 0xFFFFFFFFll)
        throw SpecError("deployment: " + where + "." + key + " is out of range");
    return static_cast<Dim>(raw);
}

double read_double(const json& obj, const char* key, double fallback,
                   const std::string& where) {
    if (!obj.contains(key)) return fallback;
    const json& v = obj.at(key);
    if (!v.is_number())
        throw SpecError("deployment: " + where + "." + key + " must be a number");
    return v.get<double>();
}

std::string read_string(const json& obj, const char* key, const std::string& fallback,
                        const std::string& where) {
    if (!obj.contains(key)) return fallback;
    const json& v = obj.at(key);
    if (!v.is_string())
        throw SpecError("deployment: " + where + "." + key + " must be a string");
    return v.get<std::string>();
}

// A 64-bit count. Capacities pass 4 GiB, so read_dim's 32-bit range is not enough; the
// representability limit is 2^53, past which a JSON number is no longer an exact integer for
// every reader (the same bound the tile-flow record keeps).
std::uint64_t read_u64(const json& obj, const char* key, std::uint64_t fallback,
                       const std::string& where) {
    if (!obj.contains(key)) return fallback;
    const json& v = obj.at(key);
    if (v.is_number_unsigned()) {
        const std::uint64_t raw = v.get<std::uint64_t>();
        if (raw > (std::uint64_t{1} << 53))
            throw SpecError("deployment: " + where + "." + key + " is out of range");
        return raw;
    }
    if (v.is_number_integer())
        throw SpecError("deployment: " + where + "." + key + " must not be negative (got " +
                        std::to_string(v.get<long long>()) + ")");
    throw SpecError("deployment: " + where + "." + key + " must be an integer");
}

std::optional<Dim> read_optional_dim(const json& obj, const char* key,
                                     const std::string& where) {
    if (!obj.contains(key)) return std::nullopt;   // ABSENT means "not declared"
    return read_dim(obj, key, 0, where);
}

DeviceSpecification read_device(const json& obj, const std::string& where) {
    reject_unknown(obj, device_keys(), where);
    DeviceSpecification d;
    d.name = read_string(obj, "name", d.name, where);
    d.topology = read_string(obj, "topology", d.topology, where);
    d.compute_tiles = read_dim(obj, "compute_tiles", d.compute_tiles, where);
    d.macs_per_cycle = read_double(obj, "macs_per_cycle", d.macs_per_cycle, where);
    d.element_bytes = read_dim(obj, "element_bytes", d.element_bytes, where);

    if (obj.contains("dma")) {
        const json& s = obj.at("dma");
        const std::string w = where + ".dma";
        reject_unknown(s, dma_keys(), w);
        if (s.contains("burst") && s.contains("burst_bytes"))
            throw SpecError("deployment: " + w +
                            ": 'burst' and 'burst_bytes' are the same field spelled two "
                            "ways; give one");
        d.dma.engines = read_dim(s, "engines", d.dma.engines, w);
        d.dma.bytes_per_cycle = read_double(s, "bytes_per_cycle", d.dma.bytes_per_cycle, w);
        d.dma.burst_bytes = s.contains("burst") ? read_optional_dim(s, "burst", w)
                                                : read_optional_dim(s, "burst_bytes", w);
        d.dma.window = read_optional_dim(s, "window", w);
    }
    if (obj.contains("l3")) {
        const json& s = obj.at("l3");
        const std::string w = where + ".l3";
        reject_unknown(s, l3_keys(), w);
        d.l3.tiles = read_optional_dim(s, "tiles", w);
        d.l3.banks = read_optional_dim(s, "banks", w);
        d.l3.capacity_tiles = read_dim(s, "capacity_tiles", d.l3.capacity_tiles, w);
    }
    if (obj.contains("l2")) {
        const json& s = obj.at("l2");
        const std::string w = where + ".l2";
        reject_unknown(s, l2_keys(), w);
        d.l2.banks_per_tile = read_optional_dim(s, "banks_per_tile", w);
    }
    if (obj.contains("l1")) {
        const json& s = obj.at("l1");
        const std::string w = where + ".l1";
        reject_unknown(s, l1_keys(), w);
        d.l1.vectors = read_optional_dim(s, "vectors", w);
    }
    if (obj.contains("movers")) {
        const json& s = obj.at("movers");
        const std::string w = where + ".movers";
        reject_unknown(s, mover_keys(), w);
        d.movers.block_movers = read_dim(s, "block_movers", d.movers.block_movers, w);
        d.movers.bm_bytes_per_cycle =
            read_double(s, "bm_bytes_per_cycle", d.movers.bm_bytes_per_cycle, w);
        d.movers.streamers = read_dim(s, "streamers", d.movers.streamers, w);
        d.movers.str_bytes_per_cycle =
            read_double(s, "str_bytes_per_cycle", d.movers.str_bytes_per_cycle, w);
        d.movers.noc_links = read_dim(s, "noc_links", d.movers.noc_links, w);
        d.movers.noc_bytes_per_cycle =
            read_double(s, "noc_bytes_per_cycle", d.movers.noc_bytes_per_cycle, w);
    }
    if (obj.contains("array")) {
        const json& s = obj.at("array");
        const std::string w = where + ".array";
        reject_unknown(s, array_keys(), w);
        d.array.rows = read_optional_dim(s, "rows", w);
        d.array.cols = read_optional_dim(s, "cols", w);
    }
    if (obj.contains("memory")) {
        const json& s = obj.at("memory");
        const std::string w = where + ".memory";
        reject_unknown(s, memory_keys(), w);
        d.memory.controllers = read_optional_dim(s, "controllers", w);
        if (s.contains("dram")) {
            const json& r = s.at("dram");
            const std::string wd = w + ".dram";
            reject_unknown(r, dram_keys(), wd);
            DeviceSpecification::Memory::Dram m;
            m.technology = read_string(r, "technology", m.technology, wd);
            m.channels = read_dim(r, "channels", m.channels, wd);
            m.channel_width_bits = read_dim(r, "channel_width_bits", m.channel_width_bits, wd);
            m.ranks = read_dim(r, "ranks", m.ranks, wd);
            m.bank_groups = read_dim(r, "bank_groups", m.bank_groups, wd);
            m.banks_per_group = read_dim(r, "banks_per_group", m.banks_per_group, wd);
            m.page_bytes = read_dim(r, "page_bytes", m.page_bytes, wd);
            m.burst_bytes = read_dim(r, "burst_bytes", m.burst_bytes, wd);
            m.data_rate_mtps = read_dim(r, "data_rate_mtps", m.data_rate_mtps, wd);
            // Required: without it there is no top of memory and no row field.
            if (!r.contains("capacity_bytes"))
                throw SpecError("deployment: " + wd + ".capacity_bytes is required");
            m.capacity_bytes = read_u64(r, "capacity_bytes", 0, wd);
            m.map = read_string(r, "map", m.map, wd);
            if (r.contains("xor_folds")) {
                const json& folds = r.at("xor_folds");
                if (!folds.is_array())
                    throw SpecError("deployment: " + wd + ".xor_folds must be an array");
                for (std::size_t i = 0; i < folds.size(); ++i) {
                    const std::string wf = wd + ".xor_folds[" + std::to_string(i) + "]";
                    reject_unknown(folds[i], xor_fold_keys(), wf);
                    DeviceSpecification::Memory::XorFold f;
                    if (!folds[i].contains("into") || !folds[i].contains("from") ||
                        !folds[i].contains("bits"))
                        throw SpecError("deployment: " + wf + " needs into, from and bits");
                    f.into = read_string(folds[i], "into", "", wf);
                    f.from = read_string(folds[i], "from", "", wf);
                    f.bits = read_dim(folds[i], "bits", 0, wf);
                    f.from_lsb = read_dim(folds[i], "from_lsb", 0, wf);
                    m.xor_folds.push_back(f);
                }
            }
            d.memory.dram = m;
        }
    }
    if (obj.contains("noc")) {
        const json& s = obj.at("noc");
        const std::string w = where + ".noc";
        reject_unknown(s, noc_keys(), w);
        DeviceSpecification::Noc n;
        n.hub_buffer_blocks = read_dim(s, "hub_buffer_blocks", n.hub_buffer_blocks, w);
        if (s.contains("port")) {
            const json& p = s.at("port");
            const std::string wp = w + ".port";
            reject_unknown(p, noc_port_keys(), wp);
            n.port.input_queue_blocks = read_dim(p, "input_queue_blocks", n.port.input_queue_blocks, wp);
            n.port.output_queue_blocks = read_dim(p, "output_queue_blocks", n.port.output_queue_blocks, wp);
            n.port.arbitration = read_string(p, "arbitration", n.port.arbitration, wp);
        }
        d.noc = n;
    }
    if (obj.contains("cpu")) {
        const json& s = obj.at("cpu");
        const std::string w = where + ".cpu";
        reject_unknown(s, cpu_keys(), w);
        d.cpu.harts = read_optional_dim(s, "harts", w);
    }
    if (obj.contains("analytical")) {
        const json& s = obj.at("analytical");
        const std::string w = where + ".analytical";
        reject_unknown(s, analytical_keys(), w);
        d.analytical.bytes_per_cycle =
            read_double(s, "bytes_per_cycle", d.analytical.bytes_per_cycle, w);
        d.analytical.pj_per_mac = read_double(s, "pj_per_mac", d.analytical.pj_per_mac, w);
        d.analytical.pj_per_byte = read_double(s, "pj_per_byte", d.analytical.pj_per_byte, w);
    }
    return d;
}

json write_device(const DeviceSpecification& d) {
    json o = json::object();
    o["name"] = d.name;
    o["topology"] = d.topology;
    o["compute_tiles"] = d.compute_tiles;
    o["macs_per_cycle"] = d.macs_per_cycle;
    o["element_bytes"] = d.element_bytes;

    json dma = json::object();
    dma["engines"] = d.dma.engines;
    dma["bytes_per_cycle"] = d.dma.bytes_per_cycle;
    // OMITTED when absent, which is the whole mechanism behind "declared": writing a
    // default here would turn every round trip into a declaration.
    if (d.dma.burst_bytes) dma["burst_bytes"] = *d.dma.burst_bytes;
    if (d.dma.window) dma["window"] = *d.dma.window;
    o["dma"] = dma;

    json l3 = json::object();
    if (d.l3.tiles) l3["tiles"] = *d.l3.tiles;
    if (d.l3.banks) l3["banks"] = *d.l3.banks;
    l3["capacity_tiles"] = d.l3.capacity_tiles;
    o["l3"] = l3;

    if (d.l2.banks_per_tile) o["l2"] = json{{"banks_per_tile", *d.l2.banks_per_tile}};
    if (d.l1.vectors) o["l1"] = json{{"vectors", *d.l1.vectors}};

    json movers = json::object();
    movers["block_movers"] = d.movers.block_movers;
    movers["bm_bytes_per_cycle"] = d.movers.bm_bytes_per_cycle;
    movers["streamers"] = d.movers.streamers;
    movers["str_bytes_per_cycle"] = d.movers.str_bytes_per_cycle;
    movers["noc_links"] = d.movers.noc_links;
    movers["noc_bytes_per_cycle"] = d.movers.noc_bytes_per_cycle;
    o["movers"] = movers;

    // The physical shape, each object written only when something in it is declared -- so a
    // spec that never mentions one keeps its canonical bytes, and every existing
    // deployment_digest with it.
    if (d.array.rows || d.array.cols) {
        json a = json::object();
        if (d.array.rows) a["rows"] = *d.array.rows;
        if (d.array.cols) a["cols"] = *d.array.cols;
        o["array"] = a;
    }
    if (d.memory.controllers || d.memory.dram) {
        json mem = json::object();
        if (d.memory.controllers) mem["controllers"] = *d.memory.controllers;
        if (d.memory.dram) {
            const auto& m = *d.memory.dram;
            json r = json::object();
            r["technology"] = m.technology;
            r["channels"] = m.channels;
            r["channel_width_bits"] = m.channel_width_bits;
            r["ranks"] = m.ranks;
            r["bank_groups"] = m.bank_groups;
            r["banks_per_group"] = m.banks_per_group;
            r["page_bytes"] = m.page_bytes;
            r["burst_bytes"] = m.burst_bytes;
            r["data_rate_mtps"] = m.data_rate_mtps;
            r["capacity_bytes"] = m.capacity_bytes;
            r["map"] = m.map;
            if (!m.xor_folds.empty()) {
                json folds = json::array();
                for (const auto& f : m.xor_folds)
                    folds.push_back(json{{"into", f.into}, {"from", f.from},
                                         {"bits", f.bits}, {"from_lsb", f.from_lsb}});
                r["xor_folds"] = folds;
            }
            mem["dram"] = r;
        }
        o["memory"] = mem;
    }
    if (d.cpu.harts) o["cpu"] = json{{"harts", *d.cpu.harts}};
    // Written in full when declared, so a round trip states the defaults it relied on.
    if (d.noc) {
        const auto& n = *d.noc;
        o["noc"] = json{{"hub_buffer_blocks", n.hub_buffer_blocks},
                        {"port", json{{"input_queue_blocks", n.port.input_queue_blocks},
                                      {"output_queue_blocks", n.port.output_queue_blocks},
                                      {"arbitration", n.port.arbitration}}}};
    }

    json an = json::object();
    an["bytes_per_cycle"] = d.analytical.bytes_per_cycle;
    an["pj_per_mac"] = d.analytical.pj_per_mac;
    an["pj_per_byte"] = d.analytical.pj_per_byte;
    o["analytical"] = an;
    return o;
}

} // namespace

std::string to_json(const DeploymentSpec& spec) {
    json root = json::object();
    json devices = json::array();
    for (const DeviceSpecification& d : spec.devices) devices.push_back(write_device(d));
    root["devices"] = devices;
    // THE CANONICAL BYTES DEPEND ON INSERTION ORDER, because `json` is ordered_json (see
    // above). Reordering an assignment in write_device() therefore changes every
    // deployment_digest -- it is a format change, not a cosmetic one, and the checked-in
    // fixture in tests/program/deploy is what makes that visible instead of silent.
    //
    // An earlier version of this comment claimed the opposite, left over from the sorted
    // default. Two spaces and a trailing newline make a spec file reviewable in a diff,
    // which is the same reason the L0 format is text.
    return root.dump(2) + "\n";
}

DeploymentSpec from_json(const std::string& text) {
    json root;
    try {
        root = json::parse(text);
    } catch (const nlohmann::json::parse_error& e) {
        throw SpecError(std::string("deployment: not valid JSON: ") + e.what());
    }
    if (!root.is_object())
        throw SpecError("deployment: the top level must be a JSON object");

    DeploymentSpec spec;
    spec.devices.clear();
    if (root.contains("devices")) {
        if (root.size() != 1)
            throw SpecError("deployment: with a \"devices\" array, no other top-level key "
                            "is understood; device fields belong inside it");
        const json& arr = root.at("devices");
        if (!arr.is_array()) throw SpecError("deployment: \"devices\" must be an array");
        if (arr.empty()) throw SpecError("deployment: \"devices\" is empty");
        for (std::size_t i = 0; i < arr.size(); ++i)
            spec.devices.push_back(
                read_device(arr[i], "devices[" + std::to_string(i) + "]"));
    } else {
        // The ADR's own spelling: a flat object is one device (see the header).
        spec.devices.push_back(read_device(root, "device"));
    }

    const std::string bad = spec.validate();
    if (!bad.empty()) throw SpecError("deployment: " + bad);
    return spec;
}

DeploymentSpec read_spec_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw SpecError("deployment: cannot read '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    if (!in && !in.eof())
        throw SpecError("deployment: failed while reading '" + path + "'");
    try {
        return from_json(ss.str());
    } catch (const SpecError& e) {
        // The path belongs in the message: a spec error with no file name is unhelpful
        // the moment a sweep reads more than one spec.
        throw SpecError(std::string(e.what()) + " (in '" + path + "')");
    }
}

std::string deployment_digest(const DeploymentSpec& spec) {
    return digest_of(to_json(spec));
}

} // namespace sw::kpu::program::platform

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
                                            "l1",            "movers",   "analytical"};
    return k;
}
const std::set<std::string>& dma_keys() {
    // "burst" is the ADR's spelling; "burst_bytes" is canonical (§ header).
    static const std::set<std::string> k = {"engines", "bytes_per_cycle", "burst",
                                            "burst_bytes"};
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
    // Ordered output: nlohmann's default json keeps keys sorted, so the canonical bytes do
    // not depend on insertion order above. Two spaces and a trailing newline make a spec
    // file reviewable in a diff, which is the same reason the L0 format is text.
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

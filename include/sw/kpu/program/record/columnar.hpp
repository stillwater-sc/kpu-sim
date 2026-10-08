// ============================================================================
// include/sw/kpu/program/record/columnar.hpp
// The bundle container both records share: a manifest names tables, and each table is one file
// of little-endian columns, each padded to 8 bytes, so a viewer can view a column as a typed
// array without copying. The .tflow record (tile_flow_record.hpp) and the .mflow record
// (memory_flow_record.hpp) are written and read through these.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/record/tile_flow_record.hpp>   // RecordError, Cycle

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::program::record::columnar {

using json = nlohmann::ordered_json;

static_assert(sizeof(double) == 8, "f64 columns");

// Times are stored as f64, which holds every integer up to 2^53 exactly; past it a time column
// would round silently, so writers refuse.
inline constexpr Cycle kMaxExactCycle = (Cycle{1} << 53);

inline bool little_endian() {
    const std::uint16_t x = 1;
    unsigned char b = 0;
    std::memcpy(&b, &x, 1);
    return b == 1;
}

// A columnar table under construction.
struct Table {
    std::string name, file;
    std::size_t rows = 0;
    struct Col { std::string name, dtype; std::string bytes; };
    std::vector<Col> cols;

    template <class T> void put(const std::string& col, const std::string& dtype, const std::vector<T>& v) {
        Col c{col, dtype, {}};
        c.bytes.resize(v.size() * sizeof(T));
        if (!v.empty()) std::memcpy(c.bytes.data(), v.data(), c.bytes.size());   // host is LE: checked by writers
        cols.push_back(std::move(c));
    }
};

// Write `t` to dir/t.file; its manifest entry.
inline json write_table(const Table& t, const std::string& dir) {
    std::string blob;
    json cols = json::array();
    for (const Table::Col& c : t.cols) {
        while (blob.size() % 8) blob.push_back('\0');
        cols.push_back(json{{"name", c.name}, {"dtype", c.dtype}, {"offset", blob.size()}});
        blob += c.bytes;
    }
    std::ofstream out(dir + "/" + t.file, std::ios::binary);
    out << blob;
    if (!out) throw RecordError("record: cannot write " + dir + "/" + t.file);
    return json{{"file", t.file}, {"rows", t.rows}, {"columns", cols}};
}

inline std::vector<double> times(const std::vector<Cycle>& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

inline std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw RecordError("record: cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Column `name` of a table, checked: its dtype, and that `count` values fit inside the file.
template <class T>
std::vector<T> read_col(const json& table, const std::string& blob, const std::string& name,
                        const char* dtype, std::size_t count) {
    for (const json& c : table.at("columns")) {
        if (c.at("name").get<std::string>() != name) continue;
        if (c.at("dtype").get<std::string>() != dtype)
            throw RecordError("record: column " + name + " is " + c.at("dtype").get<std::string>() +
                              ", expected " + dtype);
        const std::size_t off = c.at("offset").get<std::size_t>();
        // Checked without arithmetic that can overflow: a hostile offset or row count must
        // fail here, not wrap around and pass.
        if (off > blob.size() || count > (blob.size() - off) / sizeof(T))
            throw RecordError("record: column " + name + " runs past the end of its file");
        std::vector<T> v(count);
        if (count) std::memcpy(v.data(), blob.data() + off, count * sizeof(T));
        return v;
    }
    throw RecordError("record: no column " + name);
}

} // namespace sw::kpu::program::record::columnar

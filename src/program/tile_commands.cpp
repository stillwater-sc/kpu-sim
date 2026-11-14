// ============================================================================
// src/program/tile_commands.cpp
// A CSP program lowered to the KPU RTL's tile command stream (#397; see tile_commands.hpp and
// docs/plans/tile-command-stream.md).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/csp/tile_commands.hpp>

#include <sw/kpu/program/numeric.hpp>

#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sw::kpu::program::csp {

namespace {

using namespace tilecmd;

// The number systems version 1 lowers (numeric.cpp's registry names them so).
const std::string kInt8 = "int8", kInt32 = "int32";

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("tile commands (version 1): " + why);
}

std::uint32_t element_bytes(const std::string& type) {
    if (type == kInt8) return 1;
    if (type == kInt32) return 4;
    refuse("element type " + type + ": version 1 lowers int8 inputs and an int32 accumulator");
}

// A tile's elements as the RTL's L3 holds them: row-major, little-endian, each in its type's width.
std::vector<std::uint8_t> tile_bytes(const TensorOperand& t, const TileCoord& c) {
    const std::uint32_t w = element_bytes(t.type);
    std::vector<std::uint8_t> out;
    for (Dim r = t.row_begin(c.ti); r < t.row_end(c.ti); ++r)
        for (Dim col = t.col_begin(c.tj); col < t.col_end(c.tj); ++col) {
            const numeric::Word v = t.word(r, col);
            for (std::uint32_t b = 0; b < w; ++b) out.push_back(static_cast<std::uint8_t>(v >> (8 * b)));
        }
    return out;
}

}  // namespace

TileCommandStream lower_tile_commands(const CspProgram& p, const TileTarget& t, const TileProgram* inputs,
                                      const TileProgram* expected) {
    if (t.dim == 0 || t.buffers == 0 || t.acc_slots == 0 || t.buffers > 8 || t.acc_slots > 16)
        refuse("the tile target needs dim >= 1, 1..8 pair buffers and 1..16 accumulator slots (4-bit slot tags)");
    TileCommandStream s;
    s.program = p.name;
    s.target = t;
    s.l3_pitch = static_cast<std::uint64_t>(t.dim) * t.dim * 4;      // the largest tile: dim x dim int32

    // The operands: int8 inputs, an int32 accumulator, every tile the array's size.
    const TileProgram& src = p.source;
    for (const auto& name : src.operand_order()) {
        const auto& o = src.operand(name);
        if (o.tile_rows != t.dim || o.tile_cols != t.dim || o.rows % t.dim || o.cols % t.dim)
            refuse("operand " + name + " is " + std::to_string(o.rows) + "x" + std::to_string(o.cols) + " in " +
                   std::to_string(o.tile_rows) + "x" + std::to_string(o.tile_cols) + " tiles; every tile must be the " +
                   std::to_string(t.dim) + "x" + std::to_string(t.dim) + " array's");
        (void)element_bytes(o.type);
    }

    // L3: each residency at its slot's address. The host fills L3 once, before the tile runs, so a
    // slot holds one residency for the whole program.
    std::map<std::uint32_t, std::size_t> slot_owner;
    std::uint32_t top = 0;
    for (std::size_t r = 0; r < p.residencies.size(); ++r) {
        const Residency& res = p.residencies[r];
        if (res.inherited || res.retained)
            refuse("residency of " + res.tile.to_string() + " crosses operators (inherit/retain): not lowered");
        if (res.place.l3_tile != 0) refuse("residency of " + res.tile.to_string() + " is in L3 tile " +
                                          std::to_string(res.place.l3_tile) + "; version 1 runs one tile");
        if (auto [it, fresh] = slot_owner.emplace(res.place.slot, r); !fresh)
            refuse("L3 slot " + std::to_string(res.place.slot) + " holds " + p.residencies[it->second].tile.to_string() +
                   " and then " + res.tile.to_string() + "; version 1's host fills L3 once (give the program an l3 "
                   "that holds every residency)");
        top = std::max(top, res.place.slot + 1);
    }
    s.l3_bytes = top * s.l3_pitch;
    auto addr_of = [&](std::size_t r) { return static_cast<std::uint64_t>(p.residencies.at(r).place.slot) * s.l3_pitch; };
    auto l3_tile = [&](std::size_t r, const TileProgram* values) {
        const Residency& res = p.residencies[r];
        const auto& o = src.operand(res.tile.operand);
        L3Tile x;
        x.tile = res.tile;
        x.addr = addr_of(r);
        x.row_bytes = t.dim * element_bytes(o.type);
        x.element = o.type;
        if (values && values->has_operand(res.tile.operand)) x.bytes = tile_bytes(values->operand(res.tile.operand), res.tile);
        return x;
    };

    // Each reduction's first and last call: the accumulator slot's ACQUIRE and SIGNAL, and the
    // chain bits. A reduction is the calls on one output between its drains.
    const auto& A = p.actions;
    std::map<std::size_t, std::pair<bool, bool>> ends;             // call action -> (first, last)
    {
        std::map<std::string, std::size_t> open;                   // output -> its reduction's last call so far
        for (std::size_t i = 0; i < A.size(); ++i) {
            if (A[i].kind == Action::Kind::Call) {
                const TileOp& op = src.ops().at(A[i].l0_op);
                const std::string y = op.outputs.at(0).to_string();
                auto it = open.find(y);
                ends[i] = {it == open.end(), false};
                open[y] = i;
            } else if (A[i].kind == Action::Kind::Drain) {
                auto it = open.find(A[i].tile.to_string());
                if (it == open.end()) refuse("action " + std::to_string(i) + " drains " + A[i].tile.to_string() +
                                             ", which no call computed");
                ends[it->second].second = true;
                open.erase(it);
            }
        }
        if (!open.empty()) refuse("the accumulator " + open.begin()->first + " is never drained");
    }

    auto emit = [&](TileCommand c, std::size_t action, const TileCoord& tile) {
        c.id = static_cast<std::uint8_t>(s.commands.size() & 0xFF);
        c.action = action;
        c.tile = tile;
        s.commands.push_back(c);
    };
    {
        TileCommand cfg;                                           // EXEC CONFIG: output-stationary
        cfg.type = kExec;
        cfg.opcode = kConfig;
        cfg.rs1 = kDataflowOs;
        emit(cfg, kNone, TileCoord{});
    }

    std::map<std::string, std::size_t> held;                       // tile -> its open residency
    std::map<std::string, std::uint32_t> acc_of;                   // output -> its accumulator slot
    std::map<std::string, std::uint32_t> drained;                  // output drained, its STORE still to come
    std::set<std::uint32_t> acc_busy;                              // accumulator slots held until their STORE
    std::uint32_t pair = 0, next_acc = 0;
    const std::uint32_t dim = t.dim;
    for (std::size_t i = 0; i < A.size(); ++i) {
        const Action& a = A[i];
        if (!a.context.empty()) refuse("action " + std::to_string(i) + " carries a tile context; not lowered");
        if (a.l2 || a.kind == Action::Kind::L2Release) refuse("action " + std::to_string(i) + " stages a tile in L2; "
                                                              "the RTL's pair slots are the stream's own");
        if (a.seeds) refuse("action " + std::to_string(i) + " feeds an accumulator's init; not lowered");
        const std::string key = a.tile.to_string();
        switch (a.kind) {
            case Action::Kind::Load: {
                const auto& o = src.operand(a.tile.operand);
                if (o.type != kInt8) refuse("action " + std::to_string(i) + " loads " + key + " (" + o.type +
                                                     "): accumulating onto a loaded result is not lowered");
                held[key] = a.residency;
                s.l3_init.push_back(l3_tile(a.residency, inputs));
                break;
            }
            case Action::Kind::Release:
                held.erase(key);
                break;
            case Action::Kind::Move:
            case Action::Kind::Feed:
                break;                                             // the call's LOADs move and feed its pair
            case Action::Kind::Call: {
                const TileOp& op = src.ops().at(a.l0_op);
                if (op.kind != TileOpKind::MatMulAccum || op.alpha != 1.0f)
                    refuse("action " + std::to_string(i) + " is " + to_string(op.kind) + "; version 1 lowers gemm "
                           "(alpha 1)");
                const TileCoord &x = op.inputs.at(0), &w = op.inputs.at(1), &y = op.outputs.at(0);
                // The tile computes int8 x int8 -> int32: its STORE writes int32 rows.
                if (src.operand(x.operand).type != kInt8 || src.operand(w.operand).type != kInt8 ||
                    src.operand(y.operand).type != kInt32)
                    refuse("action " + std::to_string(i) + " computes " + src.operand(x.operand).type + " x " +
                           src.operand(w.operand).type + " -> " + src.operand(y.operand).type +
                           "; the tile computes int8 x int8 -> int32");
                for (const TileCoord* in : {&x, &w})
                    if (!held.count(in->to_string()))
                        refuse("action " + std::to_string(i) + " reads " + in->to_string() + ", which is not resident in L3");
                const auto [first, last] = ends.at(i);
                if (first) {
                    // A free accumulator slot, held until its STORE releases it. The tile has
                    // acc_slots of them: a program holding more accumulators at once is refused.
                    std::optional<std::uint32_t> slot;
                    for (std::uint32_t k = 0; k < t.acc_slots && !slot; ++k) {
                        const std::uint32_t cand = (next_acc + k) % t.acc_slots;
                        if (!acc_busy.count(cand)) slot = cand;
                    }
                    if (!slot)
                        refuse("action " + std::to_string(i) + " opens accumulator " + y.to_string() + " while " +
                               std::to_string(t.acc_slots) + " are held (the tile has " + std::to_string(t.acc_slots) +
                               " accumulator slots); store one first");
                    acc_of[y.to_string()] = *slot;
                    acc_busy.insert(*slot);
                    next_acc = *slot + 1;
                }
                const std::uint32_t acc = acc_of.at(y.to_string());
                const std::uint32_t pr = pair++ % t.buffers;
                TileCommand la;                                    // A into pair slot pr: take the pair
                la.type = kLoad;
                la.opcode = kMemcpy;
                la.dram_addr = addr_of(held.at(x.to_string()));
                la.local_addr = pr * dim;
                la.len = static_cast<std::uint16_t>(dim);
                la.stride = dim;
                la.row_bytes = static_cast<std::uint16_t>(dim);
                la.sync = sync(0, pr, kAcquire);
                emit(la, i, x);
                TileCommand lb = la;                               // B, corner-turned: the pair is ready
                lb.opcode = kTranspose;
                lb.dram_addr = addr_of(held.at(w.to_string()));
                lb.local_addr = (t.buffers + pr) * dim;
                lb.sync = sync(0, pr, kSignal);
                emit(lb, i, w);
                TileCommand c;
                c.type = kExec;
                c.opcode = kCompute;
                c.rs1 = pr * dim;
                c.rs2 = (t.buffers + pr) * dim;
                c.rd = kAccumulatorBit | (acc * dim);
                c.k_dim = static_cast<std::uint16_t>(dim);
                c.chain = static_cast<std::uint8_t>((first ? 0 : 1) | (last ? 0 : 2));
                c.sync = sync(pr, acc, static_cast<std::uint16_t>(kWait | kRelease | (first ? kAcquire : 0) |
                                                                  (last ? kSignal : 0)));
                emit(c, i, y);
                break;
            }
            case Action::Kind::Drain:
                drained[key] = acc_of.at(key);                     // the STORE below drains it
                acc_of.erase(key);
                break;
            case Action::Kind::Writeback: {
                auto d = drained.find(key);
                if (d == drained.end()) refuse("action " + std::to_string(i) + " writes " + key + " back without a drain");
                TileCommand st;                                    // the accumulator slot straight into L3
                st.type = kStore;
                st.opcode = kMemcpy;
                st.dram_addr = addr_of(a.residency);
                st.local_addr = kAccumulatorBit | (d->second * dim);
                st.len = static_cast<std::uint16_t>(dim);
                st.stride = dim * 4;
                st.row_bytes = static_cast<std::uint16_t>(dim * 4);
                st.sync = sync(d->second, 0, kWait | kRelease);
                emit(st, i, a.tile);
                acc_busy.erase(d->second);                         // the STORE releases the slot
                drained.erase(d);
                held[key] = a.residency;
                break;
            }
            case Action::Kind::Store:
                s.results.push_back(l3_tile(a.residency, expected));
                break;
            default:
                refuse(std::string("action ") + std::to_string(i) + " is " + to_string(a.kind) + "; not lowered");
        }
    }
    if (!drained.empty()) refuse(drained.begin()->first + " is drained and never written back");
    return s;
}

namespace {

std::string hex(const std::vector<std::uint8_t>& b) {
    static const char* d = "0123456789abcdef";
    std::string out;
    out.reserve(2 * b.size());
    for (auto x : b) {
        out += d[x >> 4];
        out += d[x & 15];
    }
    return out;
}

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
            out += buf;
            continue;
        }
        out += c;
    }
    return out + "\"";
}

void l3_list(std::ostringstream& o, const char* name, const std::vector<L3Tile>& v) {
    o << " " << quote(name) << ": [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        const L3Tile& x = v[i];
        o << (i ? ",\n" : "\n") << "  {\"tile\": " << quote(x.tile.to_string()) << ", \"addr\": " << x.addr
          << ", \"row_bytes\": " << x.row_bytes << ", \"element\": " << quote(x.element) << ", \"bytes\": "
          << (x.bytes.empty() ? std::string("null") : quote(hex(x.bytes))) << "}";
    }
    o << (v.empty() ? "]" : "\n ]");
}

}  // namespace

std::string to_json(const TileCommandStream& s, const std::string& producer) {
    std::ostringstream o;
    o << "{\n \"format\": \"kpu-tile-cmds\",\n \"version\": " << tilecmd::kVersion << ",\n \"producer\": " << quote(producer)
      << ",\n \"program\": " << quote(s.program) << ",\n \"tile\": {\"dim\": " << s.target.dim << ", \"buffers\": "
      << s.target.buffers << ", \"acc_slots\": " << s.target.acc_slots
      << ", \"dataflow\": \"output_stationary\", \"input\": \"int8\", \"accumulator\": \"int32\"},\n"
      << " \"l3\": {\"bytes\": " << s.l3_bytes << ", \"pitch\": " << s.l3_pitch << "},\n";
    l3_list(o, "l3_init", s.l3_init);
    o << ",\n";
    l3_list(o, "results", s.results);
    o << ",\n \"fields\": [\"type\", \"opcode\", \"id\", \"dram_addr\", \"local_addr\", \"len\", \"rs1\", \"rd\", "
         "\"k_dim\", \"stride\", \"rs2\", \"row_bytes\", \"activation\", \"sync\", \"chain\"],\n"
         " \"annotations\": [\"tile\", \"action\"],\n \"commands\": [";
    for (std::size_t i = 0; i < s.commands.size(); ++i) {
        const TileCommand& c = s.commands[i];
        o << (i ? ",\n" : "\n") << "  [" << unsigned(c.type) << ", " << unsigned(c.opcode) << ", " << unsigned(c.id) << ", "
          << c.dram_addr << ", " << c.local_addr << ", " << c.len << ", " << c.rs1 << ", " << c.rd << ", " << c.k_dim
          << ", " << c.stride << ", " << c.rs2 << ", " << c.row_bytes << ", " << unsigned(c.activation) << ", " << c.sync
          << ", " << unsigned(c.chain) << ", "
          << (c.tile.operand.empty() ? std::string("null") : quote(c.tile.to_string())) << ", "
          << (c.action == kNone ? std::string("null") : std::to_string(c.action)) << "]";
    }
    o << "\n ]\n}\n";
    return o.str();
}

}  // namespace sw::kpu::program::csp

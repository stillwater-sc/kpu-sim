#pragma once
// ============================================================================
// include/sw/kpu/program/csp/lang/distributed_text.hpp
// The level-2 program's text form, `dcsp 1` (#605; docs/plans/distributed-csp-programs.md §3.2):
// a reader and a printer, so a person can write a distributed program and read one a generator
// wrote. The printer writes what the reader reads, and reading it back gives the same program.
//
//   dcsp 1
//   // one vector from DRAM to node 3 of a 2x2 mesh: in at edge node 0, then east, then south
//   program vector_to_node3 machine mesh(2x2) {
//     tensor A[16,16] int8 tile 16x16 in;
//
//     channel ld0 : dma[0, 0] -> l3[0] slots 0;
//     channel e01 : bm.E[0]   -> l3[1] slots 0;
//     channel s13 : bm.S[1]   -> l3[3] slots 0;
//
//     process dma[0, 0] { load A[0,0] to l3[0]:0 reads 1 via ld0; }
//     process bm.E[0]   { push A[0,0] from l3[0]:0 to l3[1]:0 reads 1 via e01; }
//     process bm.S[1]   { push A[0,0] from l3[1]:0 to l3[3]:0 reads 0 via s13; }
//   }
//
// A program is flat: every process lists its actions (no loops yet). Generators write the large
// ones (#608-#616); this form is for reading them, and for writing the small ones a person traces
// by hand. The actions and what they mean are distributed.hpp's.
//
// Errors are SyntaxError (parse.hpp), "csp line N: ...". Structure is checked after parsing
// (validate_structure), and reported with the action's line.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>

#include <optional>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::lang {

inline constexpr const char* kDistributedVersion = "1";

class DistributedParser {
public:
    explicit DistributedParser(const std::string& src) : t_(lex(src)) {}

    DistributedProgram parse() {
        DistributedProgram p;
        expect_word("dcsp");
        const std::string v = take().text;
        if (v != kDistributedVersion)
            throw SyntaxError(line(), "this reader reads dcsp " + std::string(kDistributedVersion) + ", not " + v);
        expect_word("program");
        p.name = ident("a program name");
        expect_word("machine");
        expect_word("mesh");
        expect("(");
        p.rows = dim("the mesh's rows");
        expect("x");
        p.cols = dim("the mesh's columns");
        expect(")");
        expect("{");
        while (word("tensor") || word("vector")) p.operands.push_back(operand());
        while (word("channel")) channel(p);
        while (word("process")) process(p);
        expect("}");
        if (peek().kind != Token::Kind::End) throw SyntaxError(line(), "text after the program");
        if (const std::string bad = validate_structure(p); !bad.empty())
            throw SyntaxError(first_line_, "dcsp program " + p.name + ": " + bad);
        return p;
    }

private:
    std::vector<Token> t_;
    std::size_t at_ = 0;
    int first_line_ = 1;

    const Token& peek() const { return t_[at_]; }
    int line() const { return peek().line; }
    bool sym(const char* s) const { return peek().kind == Token::Kind::Sym && peek().text == s; }
    bool word(const char* s) const { return peek().kind == Token::Kind::Ident && peek().text == s; }
    Token take() { return t_[at_ < t_.size() - 1 ? at_++ : at_]; }
    void expect(const char* s) {
        if (!sym(s)) throw SyntaxError(line(), std::string("expected '") + s + "', found '" + peek().text + "'");
        take();
    }
    void expect_word(const char* s) {
        if (!word(s)) throw SyntaxError(line(), std::string("expected '") + s + "', found '" + peek().text + "'");
        take();
    }
    std::string ident(const char* what) {
        if (peek().kind != Token::Kind::Ident) throw SyntaxError(line(), std::string("expected ") + what + ", found '" + peek().text + "'");
        return take().text;
    }
    std::uint64_t integer(const char* what) {
        if (peek().kind != Token::Kind::Int) throw SyntaxError(line(), std::string("expected ") + what + ", found '" + peek().text + "'");
        const std::string s = take().text;
        if (s.size() > 9) throw SyntaxError(line(), std::string(what) + " " + s + " is out of range");
        return std::stoull(s);
    }
    Dim dim(const char* what) { return static_cast<Dim>(integer(what)); }

    // tensor A[16,16] int8 tile 16x16 in;   vector b[16] int32 tile 16 in;
    Operand2 operand() {
        Operand2 o;
        o.is_vector = take().text == "vector";
        o.name = ident("an operand name");
        expect("[");
        o.rows = dim("its rows");
        if (!o.is_vector) {
            expect(",");
            o.cols = dim("its columns");
        } else {
            o.cols = 1;
        }
        expect("]");
        if (!word("tile")) o.type = ident("a number system or 'tile'");
        expect_word("tile");
        o.tile_rows = dim("its tile's rows");
        if (!o.is_vector) {
            expect("x");
            o.tile_cols = dim("its tile's columns");
        } else {
            o.tile_cols = 1;
        }
        o.io = ident("in, out or inout");
        if (o.io != "in" && o.io != "out" && o.io != "inout") throw SyntaxError(line(), "'" + o.io + "': in, out or inout");
        expect(";");
        return o;
    }

    // dma[m, e]  |  bm.N[t] ... bm.c[t]  |  cf[t]
    ProcessId process_id() {
        const int ln = line();
        const std::string n = ident("a process: dma[m, e], bm.<N|E|S|W|a|b|x|c>[t] or cf[t]");
        const auto r = role_from(n);
        if (!r) throw SyntaxError(ln, "'" + n + "' is not a process: dma, bm.N, bm.E, bm.S, bm.W, bm.a, bm.b, bm.x, bm.c or cf");
        ProcessId id;
        id.role = *r;
        expect("[");
        if (*r == Role::Dma) {
            id.mc = dim("the memory controller");
            expect(",");
            id.engine = dim("the DMA engine");
        } else {
            id.node = dim("the node");
        }
        expect("]");
        return id;
    }

    // l3[t]:s  |  l2[t]:s
    SlotRef slot() {
        const int ln = line();
        const std::string n = ident("a slot, l3[t]:s or l2[t]:s");
        SlotRef s;
        if (n == "l3") s.pool = SlotRef::Pool::L3;
        else if (n == "l2") s.pool = SlotRef::Pool::L2;
        else throw SyntaxError(ln, "'" + n + "' is not a slot pool: l3 or l2");
        expect("[");
        s.node = dim("the node");
        expect("]");
        expect(":");
        s.slot = static_cast<std::uint32_t>(integer("the slot"));
        return s;
    }

    // A[i, j]  |  b[i]
    TileCoord tile(const DistributedProgram& p) {
        TileCoord c;
        c.operand = ident("a tile");
        expect("[");
        c.ti = dim("the tile's row");
        if (sym(",")) {
            take();
            c.tj = dim("the tile's column");
        }
        expect("]");
        (void)p;
        return c;
    }

    std::size_t via(const DistributedProgram& p) {
        expect_word("via");
        const int ln = line();
        const std::string n = ident("a channel");
        const std::size_t c = p.channel_index(n);
        if (c == kNoChannel) throw SyntaxError(ln, "channel '" + n + "' is not declared");
        return c;
    }

    std::uint32_t reads() {
        expect_word("reads");
        return static_cast<std::uint32_t>(integer("the reads the residency serves"));
    }

    // channel e01 : bm.E[0] -> l3[1] slots 0, 1;     channel st0 : bm.N[0] -> dma[0, 0] credits 4;
    void channel(DistributedProgram& p) {
        take();
        Channel2 c;
        c.name = ident("a channel name");
        expect(":");
        c.from = process_id();
        expect("->");
        const int ln = line();
        const std::string d = ident("l3[t], l2[t] or dma[m, e]");
        expect("[");
        if (d == "dma") {
            c.to = Channel2::To::Dma;
            c.mc = dim("the memory controller");
            expect(",");
            c.engine = dim("the DMA engine");
        } else if (d == "l3" || d == "l2") {
            c.to = d == "l3" ? Channel2::To::L3 : Channel2::To::L2;
            c.node = dim("the node");
        } else {
            throw SyntaxError(ln, "'" + d + "' is not a destination: l3[t], l2[t] or dma[m, e]");
        }
        expect("]");
        if (c.to == Channel2::To::Dma) {
            expect_word("credits");
            c.credits = static_cast<std::uint32_t>(integer("the store buffer's credits"));
        } else {
            expect_word("slots");
            c.slots.push_back(static_cast<std::uint32_t>(integer("a slot")));
            while (sym(",")) {
                take();
                c.slots.push_back(static_cast<std::uint32_t>(integer("a slot")));
            }
        }
        expect(";");
        if (p.channel_index(c.name) != kNoChannel) throw SyntaxError(ln, "channel " + c.name + " is declared twice");
        p.channels.push_back(c);
    }

    void process(DistributedProgram& p) {
        take();
        Process2 pr;
        pr.id = process_id();
        expect("{");
        while (!sym("}")) {
            Action2 a = action(p);
            pr.actions.push_back(p.actions.size());
            p.actions.push_back(a);
        }
        expect("}");
        p.processes.push_back(pr);
    }

    Action2 action(const DistributedProgram& p) {
        Action2 a;
        a.line = line();
        const std::string k = ident("an action: load, push, feed, drain, store or call");
        if (k == "load") {
            a.kind = Action2::Kind::Load;
            a.tile = tile(p);
            expect_word("to");
            a.to = slot();
            a.reads = reads();
            a.channel = via(p);
        } else if (k == "push" || k == "feed" || k == "drain") {
            a.kind = k == "push" ? Action2::Kind::Push : k == "feed" ? Action2::Kind::Feed : Action2::Kind::Drain;
            a.tile = tile(p);
            expect_word("from");
            a.from = slot();
            expect_word("to");
            std::optional<std::pair<Dim, Dim>> engine;
            if (a.kind == Action2::Kind::Push && word("dma")) {      // out of the edge port, to a Store
                take();
                expect("[");
                const Dim mc = dim("the memory controller");
                expect(",");
                engine = std::make_pair(mc, dim("the DMA engine"));
                expect("]");
            } else {
                a.to = slot();
                a.reads = reads();
            }
            a.channel = via(p);
            if (engine) {
                const Channel2& c = p.channels[a.channel];
                if (c.to != Channel2::To::Dma || c.mc != engine->first || c.engine != engine->second)
                    throw SyntaxError(a.line, "the push names dma[" + std::to_string(engine->first) + ", " +
                                              std::to_string(engine->second) + "], but channel " + c.name + " goes to " +
                                              c.destination());
            }
        } else if (k == "store") {
            a.kind = Action2::Kind::Store;
            a.tile = tile(p);
            a.channel = via(p);
        } else if (k == "call") {
            a.kind = Action2::Kind::Call;
            a.op = ident("a tile function");
            expect("(");
            a.args.push_back(slot());
            while (sym(",")) {
                take();
                a.args.push_back(slot());
            }
            expect(")");
            if (word("acc")) {
                take();
                a.accumulate = true;
                a.to = slot();
            } else {
                expect_word("to");
                a.tile = tile(p);
                expect_word("at");
                a.to = slot();
                a.reads = reads();
                a.channel = via(p);
            }
        } else {
            throw SyntaxError(a.line, "'" + k + "' is not an action: load, push, feed, drain, store or call");
        }
        expect(";");
        return a;
    }
};

inline DistributedProgram parse_distributed(const std::string& text) { return DistributedParser(text).parse(); }

// The canonical text: one declaration or action per line, channels before processes, each in the
// program's own order.
inline std::string print_distributed(const DistributedProgram& p) {
    std::ostringstream o;
    o << "dcsp " << kDistributedVersion << "\n";
    o << "program " << p.name << " machine mesh(" << p.rows << "x" << p.cols << ") {\n";
    for (const Operand2& d : p.operands) {
        if (d.is_vector)
            o << "  vector " << d.name << "[" << d.rows << "] " << d.type << " tile " << d.tile_rows << " " << d.io << ";\n";
        else
            o << "  tensor " << d.name << "[" << d.rows << "," << d.cols << "] " << d.type << " tile " << d.tile_rows << "x"
              << d.tile_cols << " " << d.io << ";\n";
    }
    if (!p.channels.empty()) o << "\n";
    for (const Channel2& c : p.channels) {
        o << "  channel " << c.name << " : " << c.from.to_string() << " -> " << c.destination();
        if (c.to == Channel2::To::Dma) {
            o << " credits " << c.credits;
        } else {
            o << " slots ";
            for (std::size_t i = 0; i < c.slots.size(); ++i) o << (i ? ", " : "") << c.slots[i];
        }
        o << ";\n";
    }
    const auto tile = [&](const TileCoord& t) {
        const Operand2* d = p.operand(t.operand);
        if (d && d->is_vector) return t.operand + "[" + std::to_string(t.ti) + "]";
        return t.operand + "[" + std::to_string(t.ti) + "," + std::to_string(t.tj) + "]";
    };
    const auto via = [&](std::size_t c) { return " via " + p.channels.at(c).name; };
    for (const Process2& pr : p.processes) {
        o << "\n  process " << pr.id.to_string() << " {\n";
        for (std::size_t ai : pr.actions) {
            const Action2& a = p.actions.at(ai);
            o << "    ";
            switch (a.kind) {
                case Action2::Kind::Load:
                    o << "load " << tile(a.tile) << " to " << a.to->to_string() << " reads " << a.reads << via(a.channel);
                    break;
                case Action2::Kind::Push:
                case Action2::Kind::Feed:
                case Action2::Kind::Drain:
                    o << (a.kind == Action2::Kind::Push ? "push " : a.kind == Action2::Kind::Feed ? "feed " : "drain ")
                      << tile(a.tile) << " from " << a.from->to_string() << " to ";
                    if (a.to) o << a.to->to_string() << " reads " << a.reads;
                    else o << p.channels.at(a.channel).destination();
                    o << via(a.channel);
                    break;
                case Action2::Kind::Store:
                    o << "store " << tile(a.tile) << via(a.channel);
                    break;
                case Action2::Kind::Call:
                    o << "call " << a.op << "(";
                    for (std::size_t i = 0; i < a.args.size(); ++i) o << (i ? ", " : "") << a.args[i].to_string();
                    o << ")";
                    if (a.accumulate) o << " acc " << a.to->to_string();
                    else o << " to " << tile(a.tile) << " at " << a.to->to_string() << " reads " << a.reads << via(a.channel);
                    break;
            }
            o << ";\n";
        }
        o << "  }\n";
    }
    o << "}\n";
    return o.str();
}

} // namespace sw::kpu::program::csp::lang

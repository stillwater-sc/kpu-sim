// ============================================================================
// tests/program/test_csp_distributed.cpp
// The level-2 CSP program and its text form, `dcsp 1` (#605; docs/plans/distributed-csp-programs.md
// §3.2): the samples read, print and read back to the same program; a program built in C++ prints
// as its text; and a program whose names or roles do not fit is refused, naming the action.
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>

#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using sw::kpu::program::csp::lang::parse_distributed;
using sw::kpu::program::csp::lang::print_distributed;
using sw::kpu::program::csp::lang::SyntaxError;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kDir = "tests/program/csp/distributed/";

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// The gemm sample with one line replaced, for the refusals.
std::string gemm_with(const std::string& from, const std::string& to) {
    std::string t = read_file(std::string(kDir) + "gemm_one_tile_node1.dcsp");
    const auto at = t.find(from);
    REQUIRE(at != std::string::npos);
    t.replace(at, from.size(), to);
    return t;
}

std::string refusal(const std::string& text) {
    try {
        parse_distributed(text);
    } catch (const SyntaxError& e) {
        return e.what();
    }
    return {};
}

} // namespace

TEST_CASE("the samples read, and print and read back to the same program", "[program][csp][distributed]") {
    for (const char* f : {"vector_to_node3.dcsp", "gemm_one_tile_node1.dcsp"}) {
        INFO(f);
        const DistributedProgram p = parse_distributed(read_file(std::string(kDir) + f));
        CHECK(validate_structure(p).empty());
        const std::string once = print_distributed(p);
        const DistributedProgram q = parse_distributed(once);
        CHECK(print_distributed(q) == once);
        CHECK(q.actions.size() == p.actions.size());
        CHECK(q.channels.size() == p.channels.size());
    }
}

TEST_CASE("the vector sample: one load, two one-hop pushes, each in its own process", "[program][csp][distributed]") {
    const DistributedProgram p = parse_distributed(read_file(std::string(kDir) + "vector_to_node3.dcsp"));
    CHECK(p.rows == 2);
    CHECK(p.cols == 2);
    REQUIRE(p.processes.size() == 3);
    CHECK(p.processes[0].id.to_string() == "dma[0, 0]");
    CHECK(p.processes[1].id.to_string() == "bm.E[0]");
    CHECK(p.processes[2].id.to_string() == "bm.S[1]");
    const Action2& last = p.actions.at(p.processes[2].actions.at(0));
    CHECK(last.kind == Action2::Kind::Push);
    CHECK(last.from->to_string() == "l3[1]:0");
    CHECK(last.to->to_string() == "l3[3]:0");
    CHECK(last.reads == 0);                                // held at node 3 to the end
    CHECK(last.opens());
    CHECK(p.channels[last.channel].name == "s13");
}

TEST_CASE("the gemm sample: a residency per write, an edge push to a store", "[program][csp][distributed]") {
    const DistributedProgram p = parse_distributed(read_file(std::string(kDir) + "gemm_one_tile_node1.dcsp"));
    std::size_t opens = 0;
    for (const Action2& a : p.actions) opens += a.opens();
    CHECK(opens == 6);                                     // two loads, two feeds, the call, the drain
    const Channel2& st = p.channels.at(p.channel_index("st"));
    CHECK(st.to == Channel2::To::Dma);
    CHECK(st.credit_count() == 2);
    CHECK(p.channels.at(p.channel_index("ld")).credit_count() == 2);
}

TEST_CASE("a program built in C++ prints as its text", "[program][csp][distributed]") {
    DistributedProgram p;
    p.name = "built";
    p.rows = 1;
    p.cols = 2;
    p.operands.push_back(Operand2{"x", true, 8, 1, 8, 1, "int8", "in"});
    p.channels.push_back(Channel2{"in0", ProcessId{Role::Dma, 0, 0, 1}, Channel2::To::L3, 0, 0, 0, {3}, 0});
    p.channels.push_back(Channel2{"e", ProcessId{Role::SendE, 0, 0, 0}, Channel2::To::L3, 1, 0, 0, {0}, 0});
    Action2 load;
    load.kind = Action2::Kind::Load;
    load.tile = TileCoord{"x", 0, 0};
    load.to = SlotRef{SlotRef::Pool::L3, 0, 3};
    load.reads = 1;
    load.channel = 0;
    Action2 push;
    push.kind = Action2::Kind::Push;
    push.tile = TileCoord{"x", 0, 0};
    push.from = SlotRef{SlotRef::Pool::L3, 0, 3};
    push.to = SlotRef{SlotRef::Pool::L3, 1, 0};
    push.channel = 1;
    p.actions = {load, push};
    p.processes = {Process2{ProcessId{Role::Dma, 0, 0, 1}, {0}}, Process2{ProcessId{Role::SendE, 0, 0, 0}, {1}}};
    REQUIRE(validate_structure(p).empty());
    CHECK(print_distributed(p) ==
          "dcsp 1\n"
          "program built machine mesh(1x2) {\n"
          "  vector x[8] int8 tile 8 in;\n"
          "\n"
          "  channel in0 : dma[0, 1] -> l3[0] slots 3;\n"
          "  channel e : bm.E[0] -> l3[1] slots 0;\n"
          "\n"
          "  process dma[0, 1] {\n"
          "    load x[0] to l3[0]:3 reads 1 via in0;\n"
          "  }\n"
          "\n"
          "  process bm.E[0] {\n"
          "    push x[0] from l3[0]:3 to l3[1]:0 reads 0 via e;\n"
          "  }\n"
          "}\n");
}

TEST_CASE("a program whose names, roles or slots do not fit is refused, naming the action",
          "[program][csp][distributed]") {
    // An action in a process of another role.
    CHECK_THAT(refusal(gemm_with("feed A[0,0] from l3[1]:0", "drain A[0,0] from l3[1]:0")),
               ContainsSubstring("a drain does not run in bm.a"));
    // A channel another process sends on.
    CHECK_THAT(refusal(gemm_with("to l2[1]:1 reads 1 via fb", "to l2[1]:1 reads 1 via fa")),
               ContainsSubstring("channel fa is bm.a[1]'s, not this process's"));
    // A slot its channel does not own.
    CHECK_THAT(refusal(gemm_with("load B[0,0] to l3[1]:1", "load B[0,0] to l3[1]:5")),
               ContainsSubstring("slot l3[1]:5 is not one of channel ld's"));
    // Two channels owning one slot.
    CHECK_THAT(refusal(gemm_with("channel fb : bm.b[1] -> l2[1] slots 1;", "channel fb : bm.b[1] -> l2[1] slots 0;")),
               ContainsSubstring("slot l2[1]:0 belongs to channel fa already"));
    // A feed into another node's compute tile.
    CHECK_THAT(refusal(gemm_with("channel fa : bm.a[1] -> l2[1] slots 0;", "channel fa : bm.a[1] -> l2[0] slots 0;")),
               ContainsSubstring("which goes to l2[0]"));
    // A mover reading another node's L3.
    CHECK_THAT(refusal(gemm_with("drain C[0,0] from l2[1]:2", "drain C[0,0] from l2[0]:2")),
               ContainsSubstring("reads l2[0]:2, not its own node's L2"));
    // A tile outside its operand.
    CHECK_THAT(refusal(gemm_with("load B[0,0]", "load B[1,0]")), ContainsSubstring("tile B[1,0] is outside B's tiles"));
    CHECK_THAT(refusal(gemm_with("load B[0,0]", "load Q[0,0]")), ContainsSubstring("tile Q[0,0] is not a declared operand's"));
    // A channel that is not declared, and one declared twice.
    CHECK_THAT(refusal(gemm_with("reads 1 via fb;", "reads 1 via nope;")), ContainsSubstring("channel 'nope' is not declared"));
    CHECK_THAT(refusal(gemm_with("channel fb :", "channel fa :")), ContainsSubstring("channel fa is declared twice"));
    // A store through a channel into another engine, and a push naming the wrong engine.
    CHECK_THAT(refusal(gemm_with("store C[0,0] via st;", "store C[0,0] via ld;")),
               ContainsSubstring("a store receives from a channel into its own engine"));
    CHECK_THAT(refusal(gemm_with("to dma[0, 0] via st;", "to dma[1, 0] via st;")),
               ContainsSubstring("the push names dma[1, 0], but channel st goes to dma[0, 0]"));
    // Off the mesh.
    CHECK_THAT(refusal(gemm_with("channel dc : bm.c[1] -> l3[1] slots 2;", "channel dc : bm.c[1] -> l3[9] slots 2;")),
               ContainsSubstring("l3[9] is off the mesh"));
    // A store buffer with no credits, and syntax.
    CHECK_THAT(refusal(gemm_with("credits 2", "credits 0")), ContainsSubstring("needs credits, its store buffer"));
    CHECK_THAT(refusal(gemm_with("call gemm", "kall gemm")), ContainsSubstring("'kall' is not an action"));
    CHECK_THAT(refusal(gemm_with("process bm.N[1]", "process bm.Q[1]")), ContainsSubstring("'bm.Q' is not a process"));
    CHECK_THAT(refusal("dcsp 2\nprogram x machine mesh(1x1) { }"), ContainsSubstring("reads dcsp 1, not 2"));
    // The refusal names the action's line.
    CHECK_THAT(refusal(gemm_with("load B[0,0] to l3[1]:1", "load B[0,0] to l3[1]:5")), ContainsSubstring("line 19"));
}

TEST_CASE("an accumulating call merges into a live result and names no channel", "[program][csp][distributed]") {
    const std::string acc = gemm_with("    call gemm(l2[1]:0, l2[1]:1) to C[0,0] at l2[1]:2 reads 1 via cc;\n",
                                      "    call gemm(l2[1]:0, l2[1]:1) to C[0,0] at l2[1]:2 reads 1 via cc;\n"
                                      "    call gemm(l2[1]:0, l2[1]:1) acc l2[1]:2;\n");
    const DistributedProgram p = parse_distributed(acc);
    const Action2& a = p.actions.at(p.processes.at(3).actions.at(1));
    CHECK(a.accumulate);
    CHECK_FALSE(a.opens());
    CHECK(print_distributed(parse_distributed(print_distributed(p))) == print_distributed(p));
    CHECK_THAT(refusal(gemm_with("    call gemm(l2[1]:0, l2[1]:1) to C[0,0] at l2[1]:2 reads 1 via cc;\n",
                                 "    call gemm(l2[1]:0, l2[1]:1) acc l2[0]:2;\n")),
               ContainsSubstring("a call accumulates into its own compute tile's L2"));
}

TEST_CASE("a program with CRLF line endings reads as the same program", "[program][csp][distributed]") {
    const std::string lf = read_file(std::string(kDir) + "gemm_one_tile_node1.dcsp");
    std::string crlf;
    for (char c : lf) {
        if (c == '\n') crlf += '\r';
        crlf += c;
    }
    CHECK(print_distributed(parse_distributed(crlf)) == print_distributed(parse_distributed(lf)));
}

// ---- regressions from the review of #631

TEST_CASE("an operand's I/O mode is in, out or inout", "[program][csp][distributed]") {
    DistributedProgram p = parse_distributed(read_file(std::string(kDir) + "vector_to_node3.dcsp"));
    for (const char* io : {"in", "out", "inout"}) {
        p.operands[0].io = io;
        CHECK(validate_structure(p).empty());
    }
    p.operands[0].io = "";                                 // would print `... tile 16x16 ;`, which no reader reads
    CHECK_THAT(validate_structure(p), ContainsSubstring("operand A is '', not in, out or inout"));
}

TEST_CASE("each action kind fills its own pool", "[program][csp][distributed]") {
    // A feed into an L3 slot its channel owns: the channel and node checks alone let it through.
    std::string t = gemm_with("channel fa : bm.a[1] -> l2[1] slots 0;", "channel fa : bm.a[1] -> l3[1] slots 3;");
    const std::string from = "feed A[0,0] from l3[1]:0 to l2[1]:0 reads 1 via fa;";
    t.replace(t.find(from), from.size(), "feed A[0,0] from l3[1]:0 to l3[1]:3 reads 1 via fa;");
    CHECK_THAT(refusal(t), ContainsSubstring("a feed fills an L2 slot, not l3[1]:3"));
    // A drain into an L2 slot.
    std::string d = gemm_with("channel dc : bm.c[1] -> l3[1] slots 2;", "channel dc : bm.c[1] -> l2[1] slots 7;");
    const std::string dfrom = "drain C[0,0] from l2[1]:2 to l3[1]:2 reads 1 via dc;";
    d.replace(d.find(dfrom), dfrom.size(), "drain C[0,0] from l2[1]:2 to l2[1]:7 reads 1 via dc;");
    CHECK_THAT(refusal(d), ContainsSubstring("a drain fills an L3 slot, not l2[1]:7"));
}

TEST_CASE("any 32-bit count prints and reads back; a larger one is refused", "[program][csp][distributed]") {
    DistributedProgram p = parse_distributed(read_file(std::string(kDir) + "vector_to_node3.dcsp"));
    p.actions[0].reads = 4000000000u;                      // ten digits, within uint32
    const DistributedProgram q = parse_distributed(print_distributed(p));
    CHECK(q.actions[0].reads == 4000000000u);
    CHECK_THAT(refusal(gemm_with("load A[0,0] to l3[1]:0 reads 1", "load A[0,0] to l3[1]:0 reads 4294967296")),
               ContainsSubstring("is out of range (at most 4294967295)"));
}

TEST_CASE("an operand near the 32-bit maximum still has its tiles", "[program][csp][distributed]") {
    std::string t = read_file(std::string(kDir) + "vector_to_node3.dcsp");
    const std::string from = "tensor A[16,16] int8 tile 16x16 in;";
    t.replace(t.find(from), from.size(), "tensor A[4294967295,16] int8 tile 16x16 in;");
    const DistributedProgram p = parse_distributed(t);     // A[0,0] is inside A's tiles
    CHECK(p.operands[0].tile_grid_rows() == 268435456u);   // ceil(4294967295 / 16)
    CHECK(p.operands[0].tile_grid_cols() == 1u);
}

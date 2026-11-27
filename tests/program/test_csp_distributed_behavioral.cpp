// ============================================================================
// tests/program/test_csp_distributed_behavioral.cpp
// L-B for level-2 programs (#606; docs/plans/distributed-csp-programs.md §3.3): the samples run with
// real values and reproduce the L0 kernels' result bit for bit (D6); credits bound every channel; the
// result does not depend on the order the processes are listed in; and a program that cannot finish
// is reported with its stuck processes (D5).
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/distributed_behavioral.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using sw::kpu::program::csp::lang::parse_distributed;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kDir = "tests/program/csp/distributed/";

std::string read_file(const std::string& name) {
    std::ifstream in(std::string(kDir) + name, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

std::string with(std::string t, const std::string& from, const std::string& to) {
    const auto at = t.find(from);
    REQUIRE(at != std::string::npos);
    t.replace(at, from.size(), to);
    return t;
}

// The L0 reference: the same kernels on the whole operands, C := sum over k of A[i,k] . B[k,j].
TileProgram gemm_reference(const TileProgram& inputs, Dim kt) {
    TileProgram ref = inputs;
    for (Dim k = 0; k < kt; ++k) {
        TileOp op;
        op.kind = TileOpKind::MatMulAccum;
        op.inputs = {TileCoord{"A", 0, k}, TileCoord{"B", k, 0}};
        op.outputs = {TileCoord{"C", 0, 0}};
        TileKernelState st;
        apply(ref, op, st);
    }
    return ref;
}

} // namespace

TEST_CASE("one vector, hop by hop: it lands at node 3 with DRAM's values, and every slot it left is free",
          "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(read_file("vector_to_node3.dcsp"));
    const TileProgram in = distributed_inputs(p);
    DistributedBehavioral lb;
    const auto sum = lb.run(p, in);
    CHECK(sum.actions == 3);
    const auto at3 = lb.slot(SlotRef{SlotRef::Pool::L3, 3, 0});
    REQUIRE(at3.has_value());
    CHECK(at3->first == "A[0,0]");
    CHECK(at3->second == BehavioralInterpreter::extract_tile(in.operand("A"), 0, 0));
    CHECK_FALSE(lb.slot(SlotRef{SlotRef::Pool::L3, 0, 0}).has_value());      // read once: freed
    CHECK_FALSE(lb.slot(SlotRef{SlotRef::Pool::L3, 1, 0}).has_value());
}

TEST_CASE("one gemm tile on node 1 reproduces the L0 kernels bit for bit (D6)", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(read_file("gemm_one_tile_node1.dcsp"));
    const TileProgram in = distributed_inputs(p);
    DistributedBehavioral lb;
    const auto sum = lb.run(p, in);
    CHECK(sum.calls == 1);
    CHECK(compare_values(lb.result(), gemm_reference(in, 1)).empty());
    // A wrong reference is reported at its first differing element.
    TileProgram off = gemm_reference(in, 1);
    off.operand("C").set_value(3, 5, off.operand("C").value(3, 5) + 1);
    CHECK_THAT(compare_values(lb.result(), off), ContainsSubstring("C(3, 5) is"));
}

TEST_CASE("a K-split gemm: credits make the loads wait, merges make the drain wait", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(read_file("gemm_ksplit_node0.dcsp"));
    const TileProgram in = distributed_inputs(p, 3);
    DistributedBehavioral lb;
    const auto sum = lb.run(p, in);
    CHECK(sum.calls == 2);
    CHECK(sum.peak_live <= 5);                                 // two L3 load slots, the drain's, L2's
    CHECK(compare_values(lb.result(), gemm_reference(in, 2)).empty());
}

TEST_CASE("the result does not depend on the order the processes are listed in", "[program][csp][distributed][behavioral]") {
    DistributedProgram p = parse_distributed(read_file("gemm_ksplit_node0.dcsp"));
    const TileProgram in = distributed_inputs(p, 7);
    const TileProgram want = gemm_reference(in, 2);
    std::sort(p.processes.begin(), p.processes.end(), [](const Process2& a, const Process2& b) { return b.id < a.id; });
    for (int i = 0; i < 6; ++i) {
        std::rotate(p.processes.begin(), p.processes.begin() + 1, p.processes.end());
        DistributedBehavioral lb;
        lb.run(p, in);
        INFO("rotation " << i);
        CHECK(compare_values(lb.result(), want).empty());
    }
}

TEST_CASE("a program that cannot finish is a deadlock, naming every stuck process (D5)", "[program][csp][distributed][behavioral]") {
    // A second push from a residency its one read already freed: the tile never arrives again.
    const std::string t = with(read_file("vector_to_node3.dcsp"),
                               "    push A[0,0] from l3[1]:0 to l3[3]:0 reads 0 via s13;     // held at node 3 to the end\n",
                               "    push A[0,0] from l3[1]:0 to l3[3]:0 reads 1 via s13;\n"
                               "    push A[0,0] from l3[1]:0 to l3[3]:0 reads 0 via s13;\n");
    DistributedBehavioral lb;
    const DistributedProgram p = parse_distributed(t);
    try {
        lb.run(p, distributed_inputs(p));
        FAIL("no deadlock reported");
    } catch (const DistributedDeadlock& e) {
        REQUIRE(e.stuck().size() == 1);
        CHECK_THAT(e.stuck()[0], ContainsSubstring("bm.S[1], line 22: its source l3[1]:0 holds nothing yet"));
        CHECK_THAT(std::string(e.what()), ContainsSubstring("deadlock (D5)"));
    }
    // A drain that never sees its merges: the accumulating call was left out.
    const DistributedProgram q = parse_distributed(with(read_file("gemm_ksplit_node0.dcsp"),
                                                        "    call gemm(l2[0]:0, l2[0]:1) acc l2[0]:2;\n", ""));
    try {
        lb.run(q, distributed_inputs(q));
        FAIL("no deadlock reported");
    } catch (const DistributedDeadlock& e) {
        bool drain = false;
        for (const std::string& s : e.stuck()) drain |= s.find("bm.c[0]") != std::string::npos &&
                                                        s.find("still expects 1 merge(s)") != std::string::npos;
        CHECK(drain);
    }
    // A store waiting for a tile its channel delivers second.
    const std::string two = with(with(read_file("gemm_one_tile_node1.dcsp"), "store C[0,0] via st;", "store C[0,0] via st;\n    store C[0,0] via st;"),
                                 "credits 2", "credits 2");
    const DistributedProgram r = parse_distributed(two);
    CHECK_THROWS_AS(lb.run(r, distributed_inputs(r)), DistributedDeadlock);
}

TEST_CASE("a program that ends with a residency still expecting reads is refused", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(with(read_file("vector_to_node3.dcsp"),
                                                        "load A[0,0] to l3[0]:0 reads 1", "load A[0,0] to l3[0]:0 reads 2"));
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(p, distributed_inputs(p)),
                      ContainsSubstring("ends with A[0,0] in l3[0]:0 still expecting 1 read(s), so its credit never returns"));
}

TEST_CASE("L-B refuses what it cannot run: an unknown tile function, an operand without values", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(with(read_file("gemm_one_tile_node1.dcsp"), "call gemm(", "call conv("));
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(p, distributed_inputs(p)), ContainsSubstring("tile function 'conv' is not one L-B runs at level 2 yet"));
    const DistributedProgram g = parse_distributed(read_file("gemm_one_tile_node1.dcsp"));
    TileProgram partial("x");
    CHECK_THROWS_WITH(lb.run(g, partial), ContainsSubstring("no values for operand A"));
}

TEST_CASE("merges print and read back, and only a result expects them", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(read_file("gemm_ksplit_node0.dcsp"));
    const std::string once = lang::print_distributed(p);
    CHECK_THAT(once, ContainsSubstring("to C[0,0] at l2[0]:2 merges 1 reads 1 via cc;"));
    CHECK(lang::print_distributed(parse_distributed(once)) == once);
    DistributedProgram bad = p;
    bad.actions[0].merges = 1;                                 // a load
    CHECK_THAT(validate_structure(bad), ContainsSubstring("only a call that opens a result expects merges"));
}

TEST_CASE("without merges, the drain would take the partial: the accumulation then has no result", "[program][csp][distributed][behavioral]") {
    const DistributedProgram p = parse_distributed(with(read_file("gemm_ksplit_node0.dcsp"), " merges 1 reads 1", " reads 1"));
    DistributedBehavioral lb;
    try {
        lb.run(p, distributed_inputs(p));
        FAIL("the partial was drained and nothing noticed");
    } catch (const DistributedDeadlock& e) {
        bool acc = false;
        for (const std::string& s : e.stuck()) acc |= s.find("cf[0]") != std::string::npos && s.find("is not live yet") != std::string::npos;
        CHECK(acc);
    }
}

// ---- regressions from the review of #633

namespace {

// Z := X . Y (+ Z when accumulating), on separate one-tile operands: the reference for the calls below.
void mm(TileProgram& t, const char* x, const char* y, const char* z) {
    TileOp op;
    op.kind = TileOpKind::MatMulAccum;
    op.inputs = {TileCoord{x, 0, 0}, TileCoord{y, 0, 0}};
    op.outputs = {TileCoord{z, 0, 0}};
    TileKernelState st;
    apply(t, op, st);
}

const char* kOverlap = R"(dcsp 1
program overlap machine mesh(1x1) {
  tensor A[4,4] fp32 tile 4x4 inout;
  tensor B[4,4] fp32 tile 4x4 in;
  tensor C[4,4] fp32 tile 4x4 out;
  channel ld : dma[0, 0] -> l3[0] slots 0, 1;
  channel fa : bm.a[0] -> l2[0] slots 0;
  channel fb : bm.b[0] -> l2[0] slots 1;
  channel cc : cf[0] -> l2[0] slots 2, 3;
  channel dc : bm.c[0] -> l3[0] slots 2, 3;
  channel st : bm.N[0] -> dma[0, 0] credits 2;
  process dma[0, 0] {
    load A[0,0] to l3[0]:0 reads 1 via ld;
    load B[0,0] to l3[0]:1 reads 1 via ld;
    store A[0,0] via st;
    store C[0,0] via st;
  }
  process bm.a[0] { feed A[0,0] from l3[0]:0 to l2[0]:0 reads 2 via fa; }
  process bm.b[0] { feed B[0,0] from l3[0]:1 to l2[0]:1 reads 1 via fb; }
  process cf[0] {
    call gemm(l2[0]:0, l2[0]:1) to A[0,0] at l2[0]:2 reads 2 via cc;
    call gemm(l2[0]:2, l2[0]:0) to C[0,0] at l2[0]:3 reads 1 via cc;
  }
  process bm.c[0] {
    drain A[0,0] from l2[0]:2 to l3[0]:2 reads 1 via dc;
    drain C[0,0] from l2[0]:3 to l3[0]:3 reads 1 via dc;
  }
  process bm.N[0] {
    push A[0,0] from l3[0]:2 to dma[0, 0] via st;
    push C[0,0] from l3[0]:3 to dma[0, 0] via st;
  }
}
)";

} // namespace

TEST_CASE("a call's inputs and result never share storage", "[program][csp][distributed][behavioral]") {
    // A' := A . B into A's own coordinate, then C := A' . A: an output naming its input's tile, and
    // two versions of A[0,0] as one call's two operands.
    const DistributedProgram p = parse_distributed(kOverlap);
    const TileProgram in = distributed_inputs(p, 2);
    DistributedBehavioral lb;
    lb.run(p, in);
    TileProgram ref("ref");
    for (const char* n : {"a", "b", "a1", "c"}) {
        TensorOperand o(n, 4, 4, 4, 4);
        if (std::string(n) == "a") o.values = in.operand("A").values;
        if (std::string(n) == "b") o.values = in.operand("B").values;
        ref.add_operand(o);
    }
    mm(ref, "a", "b", "a1");
    mm(ref, "a1", "a", "c");
    CHECK(lb.result().operand("A").values == ref.operand("a1").values);
    CHECK(lb.result().operand("C").values == ref.operand("c").values);
}

TEST_CASE("an accumulator that shares an input's coordinate keeps both values", "[program][csp][distributed][behavioral]") {
    // A' := B . B, then A' += A . B, where A' is A[0,0]'s coordinate and A[0,0] is an operand.
    std::string t = kOverlap;
    const auto cut = [&](const std::string& from, const std::string& to) { t = with(t, from, to); };
    cut("feed B[0,0] from l3[0]:1 to l2[0]:1 reads 1 via fb;", "feed B[0,0] from l3[0]:1 to l2[0]:1 reads 3 via fb;");
    cut("feed A[0,0] from l3[0]:0 to l2[0]:0 reads 2 via fa;", "feed A[0,0] from l3[0]:0 to l2[0]:0 reads 1 via fa;");
    cut("    call gemm(l2[0]:0, l2[0]:1) to A[0,0] at l2[0]:2 reads 2 via cc;\n"
        "    call gemm(l2[0]:2, l2[0]:0) to C[0,0] at l2[0]:3 reads 1 via cc;\n",
        "    call gemm(l2[0]:1, l2[0]:1) to A[0,0] at l2[0]:2 merges 1 reads 1 via cc;\n"
        "    call gemm(l2[0]:0, l2[0]:1) acc l2[0]:2;\n");
    cut("    drain C[0,0] from l2[0]:3 to l3[0]:3 reads 1 via dc;\n", "");
    cut("    push C[0,0] from l3[0]:3 to dma[0, 0] via st;\n", "");
    cut("    store C[0,0] via st;\n", "");
    const DistributedProgram p = parse_distributed(t);
    const TileProgram in = distributed_inputs(p, 4);
    DistributedBehavioral lb;
    lb.run(p, in);
    TileProgram ref("ref");
    for (const char* n : {"a", "b", "z"}) {
        TensorOperand o(n, 4, 4, 4, 4);
        if (std::string(n) == "a") o.values = in.operand("A").values;
        if (std::string(n) == "b") o.values = in.operand("B").values;
        ref.add_operand(o);
    }
    mm(ref, "b", "b", "z");
    mm(ref, "a", "b", "z");
    CHECK(lb.result().operand("A").values == ref.operand("z").values);

    // The same slot twice in one call is two reads: a residency with one is refused by name.
    const DistributedProgram short_of = parse_distributed(with(t, "reads 3 via fb;", "reads 1 via fb;"));
    CHECK_THROWS_WITH(lb.run(short_of, distributed_inputs(short_of)),
                      ContainsSubstring("the call reads l2[0]:1 2 times, but its residency (B[0,0]) has 1 read(s) left"));
}

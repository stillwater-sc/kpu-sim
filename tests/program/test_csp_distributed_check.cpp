// ============================================================================
// tests/program/test_csp_distributed_check.cpp
// The distributed checker, D0-D4 (#607; docs/plans/distributed-csp-programs.md §3.3): the samples
// pass on their meshes, and each rule refuses what it exists to refuse, by name.
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/distributed_check.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using sw::kpu::program::csp::lang::parse_distributed;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}
std::string sample(const char* f) { return read_file(std::string("tests/program/csp/distributed/") + f); }
platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}
std::string with(std::string t, const std::string& from, const std::string& to) {
    const auto at = t.find(from);
    REQUIRE(at != std::string::npos);
    t.replace(at, from.size(), to);
    return t;
}
// Every violation's text, one per line, prefixed with its rule.
std::string violations(const std::string& text, const char* deploy) {
    std::string out;
    for (const auto& v : check_distributed(parse_distributed(text), device(deploy))) out += v.rule + " " + v.message + "\n";
    return out;
}

} // namespace

TEST_CASE("the samples pass D0-D4 on their meshes", "[program][csp][distributed][check]") {
    CHECK(violations(sample("vector_to_node3.dcsp"), "kpu_dmm_2x2.json").empty());
    CHECK(violations(sample("gemm_one_tile_node1.dcsp"), "kpu_dmm_2x2.json").empty());
    CHECK(violations(sample("gemm_ksplit_node0.dcsp"), "kpu_dmm_1x1.json").empty());
}

TEST_CASE("D0: the program's mesh is the device's", "[program][csp][distributed][check]") {
    CHECK_THAT(violations(sample("vector_to_node3.dcsp"), "kpu_dmm_4x4.json"),
               ContainsSubstring("D0 the program is written for a 2x2 mesh; device \"dmm4x4\" is 4x4"));
    CHECK_THAT(violations(sample("vector_to_node3.dcsp"), "kpu_t4.json"), ContainsSubstring("is not a dmm_mesh"));
}

TEST_CASE("D1: every process exists, and a mover carries one flow", "[program][csp][distributed][check]") {
    // The 2x2 has two controllers with two engines each: dma[0, 2] is not there.
    const std::string engine = with(with(sample("vector_to_node3.dcsp"), "channel ld0 : dma[0, 0]", "channel ld0 : dma[0, 2]"),
                                    "process dma[0, 0]", "process dma[0, 2]");
    CHECK_THAT(violations(engine, "kpu_dmm_2x2.json"), ContainsSubstring("D1 dma[0, 2] does not exist on device \"dmm2x2\""));
    // One mover on two channels.
    // (built directly: a second channel from bm.S[1], with a push on it)
    DistributedProgram p = parse_distributed(sample("vector_to_node3.dcsp"));
    p.channels.push_back(Channel2{"s13b", p.channels[2].from, Channel2::To::L3, 3, 0, 0, {1}, 0});
    Action2 a = p.actions[2];
    a.from = SlotRef{SlotRef::Pool::L3, 1, 0};
    a.to = SlotRef{SlotRef::Pool::L3, 3, 1};
    a.channel = 3;
    a.line = 0;
    p.actions.push_back(a);
    p.processes[2].actions.push_back(p.actions.size() - 1);
    std::string got;
    for (const auto& v : check_distributed(p, device("kpu_dmm_2x2.json"))) got += v.rule + " " + v.message + "\n";
    CHECK_THAT(got, ContainsSubstring("D1 bm.S[1] sends on 2 channels (s13, s13b): one CSP carries one flow"));
}

TEST_CASE("D2: one hop, to the neighbour on the mover's side, or out of an edge port to a DMA engine",
          "[program][csp][distributed][check]") {
    // East from node 0 is node 1; node 3 is two hops away.
    std::string two_hops = sample("vector_to_node3.dcsp");
    two_hops = with(two_hops, "channel e01 : bm.E[0] -> l3[1] slots 0;", "channel e01 : bm.E[0] -> l3[3] slots 5;");
    two_hops = with(two_hops, "push A[0,0] from l3[0]:0 to l3[1]:0 reads 1 via e01;", "push A[0,0] from l3[0]:0 to l3[3]:5 reads 0 via e01;");
    two_hops = with(two_hops, "  channel s13 : bm.S[1] -> l3[3] slots 0;\n", "");
    two_hops = with(two_hops, "  process bm.S[1] {\n    push A[0,0] from l3[1]:0 to l3[3]:0 reads 0 via s13;     // held at node 3 to the end\n  }\n", "");
    CHECK_THAT(violations(two_hops, "kpu_dmm_2x2.json"),
               ContainsSubstring("D2 channel e01 (bm.E[0] -> l3[3]): node 0's E link goes one hop, to l3[1]"));
    // Node 1's north side is an edge port: it carries DMA flows only, not a push into an L3 tile.
    std::string out = sample("gemm_one_tile_node1.dcsp");
    out = with(out, "channel st : bm.N[1] -> dma[0, 0] credits 2;", "channel st : bm.N[1] -> l3[0] slots 7;");
    out = with(out, "push C[0,0] from l3[1]:2 to dma[0, 0] via st;", "push C[0,0] from l3[1]:2 to l3[0]:7 reads 0 via st;");
    out = with(out, "    store C[0,0] via st;\n", "");
    CHECK_THAT(violations(out, "kpu_dmm_2x2.json"),
               ContainsSubstring("node 1's N side is an edge port, onto the periphery, which carries DMA flows only"));
    // An interior node of the 4x4 has no edge port: no DMA engine reaches it directly.
    std::string far = with(sample("vector_to_node3.dcsp"), "machine mesh(2x2)", "machine mesh(4x4)");
    far = with(far, "channel ld0 : dma[0, 0] -> l3[0] slots 0;", "channel ld0 : dma[0, 0] -> l3[5] slots 0;");
    far = with(far, "load A[0,0] to l3[0]:0 reads 1 via ld0;", "load A[0,0] to l3[5]:0 reads 0 via ld0;");
    CHECK_THAT(violations(far, "kpu_dmm_4x4.json"),
               ContainsSubstring("node 5 has no edge port; a DMA engine reaches the mesh only at an edge node"));
}

TEST_CASE("D3: a channel nothing sends on is refused", "[program][csp][distributed][check]") {
    CHECK_THAT(violations(with(sample("vector_to_node3.dcsp"), "  channel s13 : bm.S[1] -> l3[3] slots 0;\n",
                               "  channel s13 : bm.S[1] -> l3[3] slots 0;\n  channel idle : bm.W[1] -> l3[0] slots 4;\n"),
                          "kpu_dmm_2x2.json"),
               ContainsSubstring("D3 channel idle is never used: its 1 credit(s) would be held by a flow that never sends"));
}

TEST_CASE("D4: slots fit, and each slot's reads and merges are the ones its residencies declare",
          "[program][csp][distributed][check]") {
    // A slot beyond the tile's L3: 64 slots over 4 tiles is 16 per tile.
    CHECK_THAT(violations(with(with(sample("vector_to_node3.dcsp"), "channel s13 : bm.S[1] -> l3[3] slots 0;",
                                    "channel s13 : bm.S[1] -> l3[3] slots 16;"),
                               "to l3[3]:0 reads 0", "to l3[3]:16 reads 0"),
                          "kpu_dmm_2x2.json"),
               ContainsSubstring("D4 channel s13: slot 16 of l3[3] is outside its 16 slot(s)"));
    // Declared reads that the program does not make (L-B would end with the credit outstanding).
    CHECK_THAT(violations(with(sample("vector_to_node3.dcsp"), "load A[0,0] to l3[0]:0 reads 1", "load A[0,0] to l3[0]:0 reads 2"),
                          "kpu_dmm_2x2.json"),
               ContainsSubstring("D4 slot l3[0]:0: its residencies declare 2 read(s), and the program makes 1"));
    // A call naming one slot twice makes two reads: a residency declaring one is caught before any run.
    CHECK_THAT(violations(with(sample("gemm_ksplit_node0.dcsp"), "call gemm(l2[0]:0, l2[0]:1) acc l2[0]:2;",
                               "call gemm(l2[0]:0, l2[0]:0) acc l2[0]:2;"),
                          "kpu_dmm_1x1.json"),
               ContainsSubstring("D4 slot l2[0]:0: its residencies declare 2 read(s), and the program makes 3"));
    // A result expecting a merge that never comes.
    CHECK_THAT(violations(with(sample("gemm_ksplit_node0.dcsp"), "    call gemm(l2[0]:0, l2[0]:1) acc l2[0]:2;\n", ""),
                          "kpu_dmm_1x1.json"),
               ContainsSubstring("D4 slot l2[0]:2: its results expect 1 merge(s), and the program makes 0"));
    // A held residency followed by another write.
    CHECK_THAT(violations(with(sample("gemm_ksplit_node0.dcsp"), "load A[0,0] to l3[0]:0 reads 1", "load A[0,0] to l3[0]:0 reads 0"),
                          "kpu_dmm_1x1.json"),
               ContainsSubstring("a residency held to the end (reads 0) is followed by another write"));
    // A tile read from a slot no write places it in.
    std::string wrong = sample("vector_to_node3.dcsp");
    wrong = with(wrong, "tensor A[16,16] int8 tile 16x16 in;", "tensor A[32,16] int8 tile 16x16 in;");
    wrong = with(wrong, "push A[0,0] from l3[1]:0 to l3[3]:0", "push A[1,0] from l3[1]:0 to l3[3]:0");
    CHECK_THAT(violations(wrong, "kpu_dmm_2x2.json"),
               ContainsSubstring("D4 slot l3[1]:0: A[1,0] is read from it, but no write places A[1,0] there"));
}

// ---- regressions from the review of #634

TEST_CASE("D3: a store receives from its channel, and does not make it used", "[program][csp][distributed][check]") {
    // The push out of the edge port removed: only the DMA engine's store names `st`.
    std::string t = sample("gemm_one_tile_node1.dcsp");
    t = with(t, "  process bm.N[1] {\n    push C[0,0] from l3[1]:2 to dma[0, 0] via st;\n  }\n", "");
    t = with(t, "drain C[0,0] from l2[1]:2 to l3[1]:2 reads 1 via dc;", "drain C[0,0] from l2[1]:2 to l3[1]:2 reads 0 via dc;");
    const std::string v = violations(t, "kpu_dmm_2x2.json");
    CHECK_THAT(v, ContainsSubstring("D3 channel st is never used"));
    // A push without its store, and a store without its push, are unmatched.
    const std::string extra = with(sample("gemm_one_tile_node1.dcsp"), "    store C[0,0] via st;\n", "    store C[0,0] via st;\n    store C[0,0] via st;\n");
    CHECK_THAT(violations(extra, "kpu_dmm_2x2.json"),
               ContainsSubstring("D3 channel st delivers 1 tile(s) to dma[0, 0], which stores 2"));
    CHECK(violations(sample("gemm_one_tile_node1.dcsp"), "kpu_dmm_2x2.json").empty());
}

TEST_CASE("D4: the residencies before a held one must have all their reads", "[program][csp][distributed][check]") {
    // Two loads into one slot: the first declares a read nobody makes, the second is held.
    std::string t = sample("vector_to_node3.dcsp");
    t = with(t, "tensor A[16,16] int8 tile 16x16 in;", "tensor A[32,16] int8 tile 16x16 in;");
    t = with(t, "  channel e01 : bm.E[0] -> l3[1] slots 0;\n", "");
    t = with(t, "  channel s13 : bm.S[1] -> l3[3] slots 0;\n", "");
    t = with(t, "    load A[0,0] to l3[0]:0 reads 1 via ld0;\n", "    load A[0,0] to l3[0]:0 reads 1 via ld0;\n    load A[1,0] to l3[0]:0 reads 0 via ld0;\n");
    t = with(t, "  process bm.E[0] {\n    push A[0,0] from l3[0]:0 to l3[1]:0 reads 1 via e01;\n  }\n\n", "");
    t = with(t, "  process bm.S[1] {\n    push A[0,0] from l3[1]:0 to l3[3]:0 reads 0 via s13;     // held at node 3 to the end\n  }\n", "");
    CHECK_THAT(violations(t, "kpu_dmm_2x2.json"),
               ContainsSubstring("D4 slot l3[0]:0: its residencies before the held one declare 1 read(s), and the program makes 0"));
}

TEST_CASE("D1: a DMA engine carries one flow too", "[program][csp][distributed][check]") {
    // dma[0, 0] loading into two edge nodes through two channels: one CSP, two flows.
    std::string t = sample("vector_to_node3.dcsp");
    t = with(t, "tensor A[16,16] int8 tile 16x16 in;", "tensor A[32,16] int8 tile 16x16 in;");
    t = with(t, "  channel ld0 : dma[0, 0] -> l3[0] slots 0;\n",
             "  channel ld0 : dma[0, 0] -> l3[0] slots 0;\n  channel ld2 : dma[0, 0] -> l3[2] slots 4;\n");
    t = with(t, "    load A[0,0] to l3[0]:0 reads 1 via ld0;\n",
             "    load A[0,0] to l3[0]:0 reads 1 via ld0;\n    load A[1,0] to l3[2]:4 reads 0 via ld2;\n");
    CHECK_THAT(violations(t, "kpu_dmm_2x2.json"),
               ContainsSubstring("D1 dma[0, 0] sends on 2 channels (ld0, ld2): one CSP carries one flow"));
    // One channel owning both slots is one flow.
    std::string one = sample("gemm_one_tile_node1.dcsp");
    CHECK(violations(one, "kpu_dmm_2x2.json").empty());
}

TEST_CASE("D4: the L3 tile's own capacity bounds its slots", "[program][csp][distributed][check]") {
    // The cluster states each tile's slots (#635): with 2, the K-split's drain slot 2 is outside.
    platform::DeviceSpecification d = device("kpu_dmm_1x1.json");
    d.cluster->l3_tile.capacity_tiles = 2;
    platform::derive_mesh_views(d);
    std::string got;
    for (const auto& v : check_distributed(parse_distributed(sample("gemm_ksplit_node0.dcsp")), d)) got += v.rule + " " + v.message + "\n";
    CHECK_THAT(got, ContainsSubstring("D4 channel dc: slot 2 of l3[0] is outside its 2 slot(s)"));
    d.cluster->compute_tile.l2.slots = 2;                  // and the compute tile's L2: slot 2 is the call's
    platform::derive_mesh_views(d);
    got.clear();
    for (const auto& v : check_distributed(parse_distributed(sample("gemm_ksplit_node0.dcsp")), d)) got += v.rule + " " + v.message + "\n";
    CHECK_THAT(got, ContainsSubstring("D4 channel cc: slot 2 of l2[0] is outside its 2 slot(s)"));
}

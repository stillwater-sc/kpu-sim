// ============================================================================
// tests/program/test_tile_commands.cpp
// A CSP program lowered to the KPU RTL's tile command stream (#397; tile_commands.hpp). The
// smallest program must give the stream the RTL's own matmul issuer gives (kpugen isa/matmul.py),
// command for command; a reduction's chain and credits follow the RTL's rules; what version 1
// does not lower is refused by name; and the JSON is pinned by a golden file.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/tile_commands.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string read(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

TileCommandStream lower_file(const std::string& path) {
    const std::string text = read(path);
    const lang::Program ast = lang::parse(text);
    const TileProgram in = driver::csp_inputs(ast);
    return lower_tile_commands(lang::compile(ast), TileTarget{}, &in);
}

std::string refusal(const std::string& text) {
    try {
        (void)lower_tile_commands(lang::compile(text), TileTarget{});
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "lowered";
}

}  // namespace

TEST_CASE("tile commands: the 4x4 int8 matmul is the RTL's own stream", "[program][tilecmd]") {
    using namespace tilecmd;
    const auto s = lower_file("tests/program/tilecmd/matmul_int8_4.csp");
    REQUIRE(s.commands.size() == 5);
    const auto& c = s.commands;
    // EXEC CONFIG, output-stationary.
    CHECK((c[0].type == kExec && c[0].opcode == kConfig && c[0].rs1 == 0 && c[0].sync == 0));
    // LOAD A into pair slot 0, taking it: sync 0x400.
    CHECK((c[1].type == kLoad && c[1].opcode == kMemcpy && c[1].local_addr == 0 && c[1].len == 4 && c[1].stride == 4 &&
           c[1].row_bytes == 4 && c[1].sync == 0x400));
    // LOAD B, corner-turned, into B slot buffers + 0 (row 8), marking the pair ready: sync 0x800.
    CHECK((c[2].type == kLoad && c[2].opcode == kTranspose && c[2].local_addr == 8 && c[2].sync == 0x800));
    // COMPUTE: the whole reduction (chain 0), W|R|A|S on pair 0 and accumulator slot 0: 0xF00.
    CHECK((c[3].type == kExec && c[3].opcode == kCompute && c[3].rs1 == 0 && c[3].rs2 == 8 && c[3].rd == 0x80000000u &&
           c[3].k_dim == 4 && c[3].chain == 0 && c[3].sync == 0xF00));
    // STORE the accumulator slot to C's L3 residency: 4 rows of 16 bytes, W|R on slot 0: 0x300.
    CHECK((c[4].type == kStore && c[4].opcode == kMemcpy && c[4].local_addr == 0x80000000u && c[4].len == 4 &&
           c[4].stride == 16 && c[4].row_bytes == 16 && c[4].sync == 0x300));
    for (std::size_t i = 0; i < c.size(); ++i) CHECK(c[i].id == i);
    // L3: the walker's slots at a 64-byte pitch; A and B in, C out.
    REQUIRE(s.l3_init.size() == 2);
    REQUIRE(s.results.size() == 1);
    CHECK(c[1].dram_addr == s.l3_init[0].addr);
    CHECK(c[2].dram_addr == s.l3_init[1].addr);
    CHECK(c[4].dram_addr == s.results[0].addr);
    CHECK(s.l3_init[0].bytes.size() == 16);     // 4x4 int8
}

TEST_CASE("tile commands: a reduction's chain and credits, and the accumulator slots", "[program][tilecmd]") {
    using namespace tilecmd;
    const auto s = lower_file("tests/program/tilecmd/matmul_int8_8_t4.csp");
    std::vector<const TileCommand*> computes, stores;
    for (const auto& c : s.commands) {
        if (c.type == kExec && c.opcode == kCompute) computes.push_back(&c);
        if (c.type == kStore) stores.push_back(&c);
    }
    REQUIRE(computes.size() == 8);
    REQUIRE(stores.size() == 4);
    for (std::size_t r = 0; r < 4; ++r) {
        const TileCommand& first = *computes[2 * r];
        const TileCommand& last = *computes[2 * r + 1];
        const std::uint32_t acc = r % 2;
        CAPTURE(r);
        // The RTL's chains for a two-step reduction: [0b10, 0b01]. The first ACQUIREs the
        // accumulator slot, the last SIGNALs it; both WAIT and RELEASE their pair.
        CHECK(first.chain == 0b10);
        CHECK(last.chain == 0b01);
        CHECK(first.sync == sync(first.rs1 / 4, acc, kWait | kRelease | kAcquire));
        CHECK(last.sync == sync(last.rs1 / 4, acc, kWait | kRelease | kSignal));
        CHECK(first.rd == (kAccumulatorBit | acc * 4));
        CHECK(stores[r]->local_addr == (kAccumulatorBit | acc * 4));
        CHECK(stores[r]->sync == sync(acc, 0, kWait | kRelease));
    }
    // The pair slots alternate call by call.
    for (std::size_t k = 0; k < computes.size(); ++k) CHECK(computes[k]->rs1 == (k % 2) * 4);
    CHECK(s.l3_init.size() == 8);
    CHECK(s.results.size() == 4);
}

TEST_CASE("tile commands: the JSON is pinned", "[program][tilecmd]") {
    const auto s = lower_file("tests/program/tilecmd/matmul_int8_8_t4.csp");
    const std::string json = to_json(s, "kpu-sim");
    const std::string golden = "tests/program/tilecmd/matmul_int8_8_t4.json";
    if (std::getenv("KPU_UPDATE_GOLDEN")) {
        std::ofstream(golden) << json;
        WARN("rewrote " << golden);
    }
    CHECK(json == read(golden));
}

TEST_CASE("tile commands: what version 1 does not lower is refused by name", "[program][tilecmd]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 3) {\n";
    // fp32 operands: the tile computes int8 x int8 -> int32.
    CHECK_THAT(refusal(head + "  tensor A[4,4] tile 4x4 in;\n  tensor B[4,4] tile 4x4 in;\n  tensor C[4,4] tile 4x4 out;\n"
                              "  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                              "  store C[0, 0];\n  release A[0, 0], B[0, 0];\n}\n"),
               ContainsSubstring("element type fp32"));
    // A tile not the array's size.
    CHECK_THAT(refusal(head + "  tensor A[8,8] int8 tile 8x8 in;\n  tensor B[8,8] int8 tile 8x8 in;\n  tensor C[8,8] int32 tile 8x8 out;\n"
                              "  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                              "  store C[0, 0];\n  release A[0, 0], B[0, 0];\n}\n"),
               ContainsSubstring("the 4x4 array's"));
    // An L3 slot reused: the host fills L3 once.
    CHECK_THAT(refusal("csp 1.0\nprogram t machine flat(l3 = 2) {\n"
                       "  tensor A[4,8] int8 tile 4x4 in;\n  tensor B[8,4] int8 tile 4x4 in;\n  tensor C[4,4] int32 tile 4x4 out;\n"
                       "  acc C[0, 0] in fabric {\n    for k in 0..2 {\n      resident A[0, k], B[k, 0];\n"
                       "      call gemm(A[0, k], B[k, 0]) +-> C[0, 0];\n      release A[0, k], B[k, 0];\n    }\n  }\n"
                       "  store C[0, 0];\n}\n"),
               ContainsSubstring("fills L3 once"));
}

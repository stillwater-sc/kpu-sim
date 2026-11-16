// ============================================================================
// tests/program/test_csp_block_format.cpp
// The Combiner's block-format conversion (#427): a result converted into an OCP Microscaling
// format, MXINT8 or MXFP8, as the last of its fused operations -- `via add(b[j]) @ combiner,
// relu @ combiner, mxint8 @ combiner`. The formats are Universal's (mxint8, mxfp8: blocks of 32
// elements sharing one e8m0 scale), through the numeric registry; every level, and the L0 trace's
// replay, computes the same bits.
//
// The test does not include Universal itself: numeric.cpp is the one translation unit that does
// (and Universal v5.17.0's max_elem_exponent specializations are not inline, so a second one
// would not link: stillwater-sc/universal#2083). The registry's conversions are checked against
// values worked by hand from the OCP MX v1.0 rules.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/numeric.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace gen = sw::kpu::program::csp::gen;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::driver::ExecutionLevel;

namespace {

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// S1 with vector units that run the formats.
platform::DeviceSpecification s1_with_formats() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    const std::vector<std::string> ops = {"add", "relu", "gelu", "silu", "atan", "mxint8", "mxfp8"};
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, ops};
    d.movers.bm_vector = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, ops};
    return d;
}

std::string linear(const std::string& dataflow, const std::string& place, const std::string& format) {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dataflow = dataflow;
    o.place = place;
    o.format = format;
    return gen::generate(o);
}

std::vector<float> wide_values(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::ldexp(float((i * 37) % 101) - 50.0f, static_cast<int>(i % 9) - 4) / 7.0f;
    return v;
}

}  // namespace

TEST_CASE("Block formats: the registry converts as Universal's mxint8 and mxfp8 do", "[program][numeric][mx]") {
    CHECK(numeric::is_block_format("mxint8"));
    CHECK(numeric::is_block_format("mxfp8"));
    CHECK_FALSE(numeric::is_block_format("int8"));
    CHECK(numeric::block_format_names() == "mxint8, mxfp8");
    CHECK_THROWS_WITH(numeric::block_convert("mxfp4", nullptr, 0), ContainsSubstring("'mxfp4' is not a block format"));
    for (std::size_t n : {std::size_t{256}, std::size_t{70}}) {     // whole blocks, and a short last one
        CAPTURE(n);
        const std::vector<float> in = wide_values(n);
        std::vector<float> i8 = in, f8 = in;
        numeric::block_convert("mxint8", i8.data(), i8.size());
        numeric::block_convert("mxfp8", f8.data(), f8.size());
        CHECK_FALSE(same_bits(i8, in));                  // the formats lose precision on these values
        CHECK_FALSE(same_bits(f8, in));
        // A converted tile is what the format represents: converting it again changes nothing.
        std::vector<float> again = i8;
        numeric::block_convert("mxint8", again.data(), again.size());
        CHECK(same_bits(again, i8));
    }
    // Worked by hand. A block whose largest magnitude is 100 has shared exponent floor(log2 100) = 6.
    // MXINT8's elements reach 2^6, so its scale is 2^0 and the elements round to integers, ties to
    // even. MXFP8's (e4m3) reach 2^8, so its scale is 2^-2: 100 / 0.25 = 400 lies midway between
    // e4m3's 384 and 416, and ties to the even significand, 384 (so 96); 1.3 / 0.25 = 5.2 rounds
    // to e4m3's 5 (so 1.25); -13 / 0.25 = -52 is exact.
    std::vector<float> i8{100.0f, 1.5f, 2.5f, -3.25f, 0.0f}, f8{100.0f, 1.3f, -13.0f};
    numeric::block_convert("mxint8", i8.data(), i8.size());
    numeric::block_convert("mxfp8", f8.data(), f8.size());
    CHECK(i8 == std::vector<float>{100.0f, 2.0f, 2.0f, -3.0f, 0.0f});
    CHECK(f8 == std::vector<float>{96.0f, 1.25f, -13.0f});
    // The blocks are runs of 32: an element in the second block is scaled by that block's maximum.
    std::vector<float> two(33, 0.0f);
    two[0] = 100.0f;
    two[1] = 0.75f;                                      // block 1: scale 2^0, so 0.75 -> 1 (ties: none)
    two[32] = 0.75f;                                     // block 2, alone: exact
    numeric::block_convert("mxint8", two.data(), two.size());
    CHECK(two[1] == 1.0f);
    CHECK(two[32] == 0.75f);
    // #436's review, at the edges of the scale.
    {   // A block maximum just below 128: floor(log2) is 6 (ilogb, not a rounded log2, which gives
        // 7), so the scale is 2^6 and the largest element saturates at code 127 -- 127, not 128.
        std::vector<float> v{0x1.fffffep+6f};
        numeric::block_convert("mxint8", v.data(), v.size());
        CHECK(v[0] == 127.0f);
    }
    {   // e8m0's smallest scale, 2^-127: MXINT8's implicit 2^-6 makes its spacing 2^-133.
        std::vector<float> v{0x1p-127f, 0x1p-133f};
        numeric::block_convert("mxint8", v.data(), v.size());
        CHECK(v == std::vector<float>{0x1p-127f, 0x1p-133f});
    }
    {   // MXFP8 there too: no reciprocal of a scale below 2^-127, so no inf and no NaN.
        std::vector<float> v{0x1p-126f, 0.0f, 0.0f}, w{0x1p-120f, 0x1p-126f};
        numeric::block_convert("mxfp8", v.data(), v.size());
        numeric::block_convert("mxfp8", w.data(), w.size());
        CHECK(v == std::vector<float>{0x1p-126f, 0.0f, 0.0f});
        CHECK(w == std::vector<float>{0x1p-120f, 0x1p-126f});
    }
    // A NaN or an inf anywhere in a block makes its scale NaN, and the block reads back as NaN --
    // even beside zeros, where the largest magnitude is 0 (#436's review).
    for (const char* f : {"mxint8", "mxfp8"}) {
        CAPTURE(f);
        std::vector<float> v{std::numeric_limits<float>::quiet_NaN(), 0.0f}, w{0.0f, -std::numeric_limits<float>::infinity()};
        numeric::block_convert(f, v.data(), v.size());
        numeric::block_convert(f, w.data(), w.size());
        CHECK((std::isnan(v[0]) && std::isnan(v[1])));
        CHECK((std::isnan(w[0]) && std::isnan(w[1])));
    }
    // An element a byte, a scale a block.
    CHECK(numeric::block_bytes("mxint8", 256) == 256 + 8);
    CHECK(numeric::block_bytes("mxfp8", 70) == 70 + 3);
}

TEST_CASE("Block formats: the language places the conversion last, off the fabric", "[program][csp][lang][mx]") {
    const std::string src = linear("os", "combiner", "mxint8");
    CHECK_THAT(src, ContainsSubstring("via add(b[j]) @ combiner, relu @ combiner, mxint8 @ combiner;"));
    CHECK_THAT(src, ContainsSubstring("into mxint8"));
    (void)lang::validate(src);
    const std::string text = lang::format(lang::parse(src));
    CHECK(lang::format(lang::parse(text)) == text);
    auto refused = [](const std::string& via) {
        const std::string p = "csp 1.0\nprogram t machine flat(l3 = 8) {\n  tensor A[16,16] tile 16x16 in;\n"
                              "  tensor B[16,16] tile 16x16 in;\n  tensor C[16,16] tile 16x16 out;\n"
                              "  resident A[0, 0], B[0, 0];\n"
                              "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                              "  store C[0, 0] via " + via + ";\n  release A[0, 0], B[0, 0];\n}\n";
        try {
            (void)lang::validate(p);
        } catch (const lang::CompileError& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    CHECK(refused("relu @ combiner, mxfp8 @ combiner") == "accepted");
    CHECK(refused("mxint8 @ bm.egress") == "accepted");
    CHECK_THAT(refused("mxint8 @ combiner, relu @ combiner"),
               ContainsSubstring("relu @ combiner follows the block-format conversion mxint8"));
    CHECK_THAT(refused("mxint8 @ fabric"), ContainsSubstring("mxint8 @ fabric: a block-format conversion runs where"));
    // An L2 accumulator's final merge that converts its tile ends its operations: its store takes
    // no stage after it (#436's review).
    {
        const std::string p = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n  tensor A[16,16] tile 16x16 in;\n"
                              "  tensor B[16,16] tile 16x16 in;\n  tensor C[16,16] tile 16x16 out;\n"
                              "  acc C[0, 0] in l2 {\n    resident A[0, 0], B[0, 0];\n"
                              "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0] via mxint8 @ combiner;\n"
                              "    release A[0, 0], B[0, 0];\n  }\n  store C[0, 0] via relu @ bm.egress;\n}\n";
        CHECK_THROWS_WITH(lang::validate(p), ContainsSubstring("csp line 11: store C[0, 0] via ...: its final merge "
                                                               "converted it into a block format"));
        CHECK_THROWS_WITH(lang::compile(p), ContainsSubstring("csp line 11: store C[0,0] via ...: C[0,0]'s final merge "
                                                              "(csp line 8) converted it into a block format"));
    }
    // A target whose Combiner lacks the format refuses it by name.
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu"}};
    const lang::Target t = lang::target_from(d);
    CHECK_THROWS_WITH(lang::validate(src, &t), ContainsSubstring("mxint8 @ combiner: the target's Combiner runs add, relu"));
    // The generator: a format rides the result's move into memory, at the epilogue's place.
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.format = "mxint8";
    o.place = "fabric";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("rides the result's move into memory"));
    o.place = "unfused";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("rides the result's move into memory"));
    o.place = "combiner";
    o.format = "mxfp4";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("format 'mxfp4' (none | mxint8, mxfp8)"));
    o.format = "mxint8";
    o.algo = "matmul";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("format is linear's"));
}

TEST_CASE("Block formats: every placement and level converts the same result, bit for bit",
          "[program][csp][run][mx]") {
    const auto d = s1_with_formats();
    struct Cell { const char* df; const char* place; };
    for (const char* format : {"mxint8", "mxfp8"})
        for (const Cell c : {Cell{"os", "combiner"}, Cell{"os", "bm.egress"}, Cell{"ws", "combiner"},
                             Cell{"ws", "bm.egress"}, Cell{"is", "combiner"}}) {
            CAPTURE(format, c.df, c.place);
            // The same program without the conversion: its result, converted tile by tile, is the
            // oracle -- the conversion is the epilogue's last step, on the tile it would store.
            const lang::Program plain_ast = lang::parse(linear(c.df, c.place, "none"));
            const TileProgram inputs = driver::csp_inputs(plain_ast);
            const auto plain = driver::csp_reference(plain_ast, inputs, 1'000'000);
            REQUIRE(plain.values);
            TileProgram expect = *plain.values;
            TensorOperand& C = expect.operand("C");
            for (Dim ti = 0; ti < C.n_tile_rows(); ++ti)
                for (Dim tj = 0; tj < C.n_tile_cols(); ++tj) {
                    std::vector<float> t;
                    for (Dim r = C.row_begin(ti); r < C.row_end(ti); ++r)
                        for (Dim k = C.col_begin(tj); k < C.col_end(tj); ++k) t.push_back(C.at(r, k));
                    numeric::block_convert(format, t.data(), t.size());
                    std::size_t n = 0;
                    for (Dim r = C.row_begin(ti); r < C.row_end(ti); ++r)
                        for (Dim k = C.col_begin(tj); k < C.col_end(tj); ++k) C.at(r, k) = t[n++];
                }
            const lang::Program ast = lang::parse(linear(c.df, c.place, format));
            driver::CspRunRequest req{ast, &d, inputs,
                                      {ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
            const auto r = driver::run_csp(req);
            REQUIRE(r.reference);
            CHECK(r.reference_note.empty());             // the trace's replay agreed with L-B
            CHECK(same_bits(r.reference->operand("C").values, expect.operand("C").values));
            CHECK_FALSE(same_bits(r.reference->operand("C").values, plain.values->operand("C").values));
            for (const auto& o : r.levels) {
                CAPTURE(to_string(o.level));
                INFO(o.skipped.value_or(""));
                REQUIRE_FALSE(o.skipped);
                CHECK(same_bits(o.values.operand("C").values, r.reference->operand("C").values));
            }
        }
}

TEST_CASE("Block formats: the trace records BLOCK_CONVERT (opset 1.6), and reads it back", "[program][csp][mx]") {
    const lang::Program ast = lang::parse(linear("ws", "combiner", "mxfp8"));
    const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
    REQUIRE(ref.trace);
    std::size_t converts = 0;
    for (const TileOp& op : ref.trace->ops())
        if (op.kind == TileOpKind::BlockConvert) {
            ++converts;
            CHECK(op.format == "mxfp8");
        }
    CHECK(converts == 16);                               // one per C tile, on its final merge
    std::ostringstream os;
    serialize::WriteOptions with_values;
    with_values.include_values = true;
    serialize::write_l0(os, *ref.trace, with_values);
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.6.0"));
    CHECK_THAT(os.str(), ContainsSubstring("kind=BLOCK_CONVERT out=C:0,0 format=mxfp8"));
    std::istringstream is(os.str());
    TileProgram back = serialize::read_l0(is);
    (void)TileProgramReference().run(back);
    CHECK(same_bits(back.operand("C").values, ref.values->operand("C").values));
    // A format the reader does not know is refused by name.
    std::string bad = os.str();
    bad.replace(bad.find("format=mxfp8"), 12, "format=mxfp4");
    std::istringstream bis(bad);
    CHECK_THROWS_WITH(serialize::read_l0(bis), ContainsSubstring("BLOCK_CONVERT format=mxfp4"));
}

TEST_CASE("Combiner: the fused-operation matrix of a real result, bias x activation x format",
          "[program][csp][run][mx][combiner]") {
    // #427's matrix for an fp32 result: every subset of {add(b[j]), relu, mxint8} on the Combiner,
    // under OS, WS and IS (the integer half, with the scale, is in test_csp_requant). The oracle
    // is computed here: the same linear with no stage, then the subset applied tile by tile, in
    // the Combiner's order.
    const auto d = s1_with_formats();
    const std::string full = " via add(b[j]) @ combiner, relu @ combiner, mxint8 @ combiner";
    for (const char* df : {"os", "ws", "is"}) {
        const std::string src = linear(df, "combiner", "mxint8");
        REQUIRE(src.find(full) != std::string::npos);
        auto with = [&](bool bias, bool relu, bool format) {
            std::string via;
            if (bias) via += "add(b[j]) @ combiner";
            if (relu) via += (via.empty() ? "" : ", ") + std::string("relu @ combiner");
            if (format) via += (via.empty() ? "" : ", ") + std::string("mxint8 @ combiner");
            std::string s = src;
            const std::string by = via.empty() ? std::string() : " via " + via;
            for (std::size_t at = 0; (at = s.find(full, at)) != std::string::npos; at += by.size())
                s.replace(at, full.size(), by);          // resume after it: `by` may contain `full`
            return s;
        };
        const lang::Program bare_ast = lang::parse(with(false, false, false));
        const TileProgram inputs = driver::csp_inputs(bare_ast);
        const auto bare = driver::csp_reference(bare_ast, inputs, 1'000'000);
        REQUIRE(bare.values);
        const TensorOperand& b = inputs.operand("b");
        for (const bool bias : {false, true})
            for (const bool relu : {false, true})
                for (const bool format : {false, true}) {
                    if (!bias && !relu && !format) continue;
                    CAPTURE(df, bias, relu, format);
                    TileProgram expect = *bare.values;
                    TensorOperand& C = expect.operand("C");
                    for (Dim ti = 0; ti < C.n_tile_rows(); ++ti)
                        for (Dim tj = 0; tj < C.n_tile_cols(); ++tj) {
                            std::vector<float> t;
                            for (Dim r = C.row_begin(ti); r < C.row_end(ti); ++r)
                                for (Dim k = C.col_begin(tj); k < C.col_end(tj); ++k) {
                                    float v = C.at(r, k);
                                    if (bias) v += b.at(k, 0);
                                    if (relu) v = v > 0.0f ? v : 0.0f;
                                    t.push_back(v);
                                }
                            if (format) numeric::block_convert("mxint8", t.data(), t.size());
                            std::size_t n = 0;
                            for (Dim r = C.row_begin(ti); r < C.row_end(ti); ++r)
                                for (Dim k = C.col_begin(tj); k < C.col_end(tj); ++k) C.at(r, k) = t[n++];
                        }
                    const lang::Program ast = lang::parse(with(bias, relu, format));
                    driver::CspRunRequest req{ast, &d, inputs,
                                              {ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
                    const auto r = driver::run_csp(req);
                    REQUIRE(r.reference);
                    CHECK(r.reference_note.empty());     // the trace's replay agreed with L-B
                    CHECK(same_bits(r.reference->operand("C").values, expect.operand("C").values));
                    for (const auto& o : r.levels) {
                        CAPTURE(to_string(o.level));
                        INFO(o.skipped.value_or(""));
                        REQUIRE_FALSE(o.skipped);
                        CHECK(same_bits(o.values.operand("C").values, expect.operand("C").values));
                    }
                }
    }
}

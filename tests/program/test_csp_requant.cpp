// ============================================================================
// tests/program/test_csp_requant.cpp
// The Combiner's requantization (#427; branes-ai/embodied-ai-soc-generator#68): an int8 x int8
// gemm accumulates in int32, and scale(m, shift) @ combiner takes each result tile into int8 --
// round(x * m / 2^shift), ties to even, saturated -- on its way to memory: on its store (output-
// stationary), or on each tile's final merge (weight- and input-stationary). The result tensor is
// declared in its type with its accumulator: `tensor C[..] int8 tile .. out acc int32`.
//
// The reference is computed here, independently: the same matmul unrequantized (C in int32),
// each element then requantized in double precision (exact for these magnitudes).
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

#include <algorithm>
#include <cmath>
#include <cstring>
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

constexpr long long kM = 3;
constexpr int kShift = 9;

std::string matmul(const std::string& dataflow, bool requant) {
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.dtype = "int8";
    o.dataflow = dataflow;
    o.requant = requant;
    o.requant_m = kM;
    o.requant_shift = kShift;
    return gen::generate(o);
}

// A Combiner that runs the scale (a target unit lists it among its ops, as any stage).
platform::DeviceSpecification s1_with_combiner() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu", "scale"}};
    return d;
}

// The walker's verdict and the validator's, separately: each must refuse on its own.
struct Verdicts {
    std::string walk = "accepted", sym = "accepted";
};
Verdicts refused(const std::string& src) {
    Verdicts v;
    try {
        (void)lang::compile(src);
    } catch (const std::exception& e) {
        v.walk = e.what();
    }
    try {
        (void)lang::validate(src);
    } catch (const std::exception& e) {
        v.sym = e.what();
    }
    return v;
}
#define BOTH_REFUSE(src, text)                                    \
    do {                                                          \
        const Verdicts v_ = refused(src);                         \
        INFO("walker: " << v_.walk);                              \
        CHECK_THAT(v_.walk, ContainsSubstring(text));             \
        INFO("validator: " << v_.sym);                            \
        CHECK_THAT(v_.sym, ContainsSubstring(text));              \
    } while (0)
#define BOTH_ACCEPT(src)                                          \
    do {                                                          \
        const Verdicts v_ = refused(src);                         \
        CHECK(v_.walk == "accepted");                             \
        CHECK(v_.sym == "accepted");                              \
    } while (0)

// x * m / 2^shift in double (exact here), ties to even, clamped to int8.
long long expected(long long x, long long m, int shift) {
    const double q = std::nearbyint(std::ldexp(static_cast<double>(x * m), -shift));
    return static_cast<long long>(std::clamp(q, -128.0, 127.0));
}

}  // namespace

TEST_CASE("Requant: the registry rounds ties to even and saturates", "[program][numeric][requant]") {
    auto rq = [](long long x, long long m, int s) {
        const numeric::Word w = numeric::encode("int32", static_cast<double>(x));
        return static_cast<long long>(numeric::decode("int8", numeric::requantize("int32", "int8", w, m, s)));
    };
    CHECK(rq(100, 3, 2) == 75);
    CHECK(rq(10, 1, 2) == 2);                            // 2.5: ties to even
    CHECK(rq(14, 1, 2) == 4);                            // 3.5
    CHECK(rq(-10, 1, 2) == -2);                          // -2.5
    CHECK(rq(-14, 1, 2) == -4);
    CHECK(rq(11, 1, 2) == 3);                            // 2.75
    CHECK(rq(1000, 1, 0) == 127);                        // saturated
    CHECK(rq(-1000, 1, 0) == -128);
    CHECK(rq(7, -2, 1) == -7);                           // a negative multiplier
    CHECK(rq(2147483647, 2147483647, 62) == 1);          // the widest product: (2^31 - 1)^2 / 2^62 -> 1
    for (long long x = -3000; x <= 3000; x += 7) CHECK(rq(x, kM, kShift) == expected(x, kM, kShift));
    CHECK_THROWS_WITH(numeric::requantize("fp32", "int8", 0, 1, 0), ContainsSubstring("both must be integers"));
    CHECK_THROWS_WITH(numeric::requantize("int32", "int8", 0, 1LL << 31, 0), ContainsSubstring("outside (-2^31, 2^31)"));
    CHECK_THROWS_WITH(numeric::requantize("int32", "int8", 0, 1, 63), ContainsSubstring("outside 0..62"));
}

TEST_CASE("Requant: the programs declare C with its accumulator and carry the scale", "[program][csp][lang][requant]") {
    const std::string os = matmul("os", true), ws = matmul("ws", true);
    CHECK_THAT(os, ContainsSubstring("tensor C[64,64] int8 tile 16x16 out acc int32;"));
    CHECK_THAT(os, ContainsSubstring("store C[i, j] via scale(3, 9) @ combiner;"));
    CHECK_THAT(ws, ContainsSubstring("+-> C[i, j] via scale(3, 9) @ combiner;"));
    CHECK_THAT(ws, ContainsSubstring("requantized into int8 on each final merge"));
    for (const std::string& src : {os, ws, matmul("is", true)}) {
        (void)lang::validate(src);
        const std::string text = lang::format(lang::parse(src));
        CHECK(lang::format(lang::parse(text)) == text);   // `acc int32` and scale(m, shift) round-trip
    }
    // The generator's refusals.
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.requant = true;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("dtype is 'fp32'"));
    o.dtype = "int8";
    o.requant_shift = 63;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("shift in 0..62"));
}

TEST_CASE("Requant: what the language refuses, in the walker and the validator", "[program][csp][lang][requant]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                             "  tensor A[16,16] int8 tile 16x16 in;\n  tensor B[16,16] int8 tile 16x16 in;\n";
    const std::string q = "  tensor C[16,16] int8 tile 16x16 out acc int32;\n";
    const std::string fab = "  resident A[0, 0], B[0, 0];\n"
                            "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n";
    const std::string end = "  release A[0, 0], B[0, 0];\n}\n";
    // Accepted: the scale on the store.
    BOTH_ACCEPT(head + q + fab + "  store C[0, 0] via scale(1, 4) @ combiner;\n" + end);
    // No scale: the accumulator's values would reach memory as int8 words.
    BOTH_REFUSE(head + q + fab + "  store C[0, 0];\n" + end, "holds int32 accumulator values; its context is scale(m, shift) @ combiner");
    // Anything beside it.
    BOTH_REFUSE(head + q + fab + "  store C[0, 0] via scale(1, 4) @ combiner, relu @ combiner;\n" + end, "and nothing else");
    // Off the Combiner, out of order, or twice.
    BOTH_REFUSE(head + q + fab + "  store C[0, 0] via scale(1, 4) @ bm.egress;\n" + end, "scale @ bm.egress: the requantization runs on the Combiner");
    // A scale on a tensor with no accumulator declared.
    BOTH_REFUSE(head + "  tensor C[16,16] int32 tile 16x16 out;\n" + fab + "  store C[0, 0] via scale(1, 4) @ combiner;\n" + end, "declares no acc");
    // A late store carries no stage.
    BOTH_REFUSE(head + q + fab + "  store C[0, 0] late;\n" + end, "is requantized on its way to memory, and a late store carries no stage");
    // Never resident; declared out, integer to integer.
    BOTH_REFUSE(head + q + "  resident C[0, 0];\n  release C[0, 0];\n}\n", "is a requantized result (acc int32), computed and stored, never made resident");
    BOTH_REFUSE(head + "  tensor C[16,16] int8 tile 16x16 in acc int32;\n}\n", "a requantized result is a tensor declared out");
    BOTH_REFUSE(head + "  tensor C[16,16] tile 16x16 out acc int32;\n}\n", "requantization takes an integer accumulator into an integer type");
    // An L2 accumulator: the scale is its final merge's, and its store takes none.
    const std::string l2 = "  acc C[0, 0] in l2 {\n    resident A[0, 0], B[0, 0];\n"
                           "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]";
    const std::string l2end = ";\n    release A[0, 0], B[0, 0];\n  }\n";
    BOTH_ACCEPT(head + q + l2 + " via scale(1, 4) @ combiner" + l2end + "  store C[0, 0];\n}\n");
    BOTH_REFUSE(head + q + l2 + l2end + "  store C[0, 0];\n}\n", "final merge needs `via scale(m, shift) @ combiner`");
    // Every tile of an L2 family must be requantized: C[0, 0] is, C[0, 1] is not (#437's review).
    {
        const std::string wide = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                                 "  tensor A[16,16] int8 tile 16x16 in;\n  tensor B[16,32] int8 tile 16x16 in;\n"
                                 "  tensor C[16,32] int8 tile 16x16 out acc int32;\n"
                                 "  acc C[0, :] in l2 {\n    resident A[0, 0], B[0, :];\n"
                                 "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0] via scale(1, 4) @ combiner;\n"
                                 "    call gemm(A[0, 0], B[0, 1]) +-> C[0, 1];\n"
                                 "    release A[0, 0], B[0, :];\n  }\n  store C[0, :];\n}\n";
        BOTH_REFUSE(wide, "final merge needs `via scale(m, shift) @ combiner`");
        std::string all = wide;
        all.replace(all.find("+-> C[0, 1];"), 12, "+-> C[0, 1] via scale(1, 4) @ combiner;");
        BOTH_ACCEPT(all);
    }
    // An init is in the accumulator's number system (int32), not the result's (int8).
    {
        const std::string init = head + "  tensor D[16,16] int32 tile 16x16 in;\n" + q +
                                 "  resident A[0, 0], B[0, 0], D[0, 0];\n"
                                 "  acc C[0, 0] in fabric init D[0, 0] { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                 "  store C[0, 0] via scale(1, 4) @ combiner;\n  release A[0, 0], B[0, 0], D[0, 0];\n}\n";
        BOTH_ACCEPT(init);
        std::string bad = init;
        bad.replace(bad.find("tensor D[16,16] int32"), 21, "tensor D[16,16] int8 ");
        CHECK_THAT(refused(bad).sym, ContainsSubstring("accumulates in int32; an init is in the accumulator's number system"));
    }
}

TEST_CASE("Requant: every dataflow and level stores the requantized result, bit for bit", "[program][csp][run][requant]") {
    const auto d = s1_with_combiner();
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        // The reference: the same matmul in int32, then each element requantized here.
        const lang::Program plain_ast = lang::parse(matmul(df, false));
        const TileProgram inputs = driver::csp_inputs(plain_ast);
        const auto plain = driver::csp_reference(plain_ast, inputs, 1'000'000);
        REQUIRE(plain.values);
        const TensorOperand& c32 = plain.values->operand("C");
        REQUIRE(c32.type == "int32");
        std::vector<long long> want;
        bool saturated = false, rounded = false;
        for (Dim r = 0; r < c32.rows; ++r)
            for (Dim c = 0; c < c32.cols; ++c) {
                const auto x = static_cast<long long>(c32.value(r, c));
                want.push_back(expected(x, kM, kShift));
                saturated = saturated || std::abs(static_cast<double>(x * kM)) / 512.0 > 127.5;
                rounded = rounded || (x * kM) % 512 != 0;
            }
        CHECK(rounded);                                  // the data reaches the rounding
        const lang::Program ast = lang::parse(matmul(df, true));
        TileProgram in = driver::csp_inputs(ast);
        for (const char* n : {"A", "B"}) in.operand(n).values = inputs.operand(n).values;
        driver::CspRunRequest req{ast, &d, in,
                                  {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
        const auto r = driver::run_csp(req);
        REQUIRE(r.reference);
        CHECK(r.reference_note.empty());                 // the trace's replay agreed with L-B
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            INFO(o.skipped.value_or(""));
            REQUIRE_FALSE(o.skipped);
            const TensorOperand& got = o.values.operand("C");
            CHECK(got.type == "int8");
            std::size_t n = 0, bad = 0;
            for (Dim rr = 0; rr < got.rows; ++rr)
                for (Dim cc = 0; cc < got.cols; ++cc)
                    if (static_cast<long long>(got.value(rr, cc)) != want[n++]) ++bad;
            CHECK(bad == 0);
        }
        (void)saturated;
    }
}

TEST_CASE("Requant: the trace records OPERAND acc= and REQUANT, and reads them back", "[program][csp][requant]") {
    const lang::Program ast = lang::parse(matmul("ws", true));
    const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
    REQUIRE(ref.trace);
    CHECK(ref.trace->operand("C").acc == "int32");
    std::size_t n = 0;
    for (const TileOp& op : ref.trace->ops())
        if (op.kind == TileOpKind::Requant) {
            ++n;
            CHECK(op.scale_m == kM);
            CHECK(op.scale_shift == kShift);
        }
    CHECK(n == 16);                                      // one per C tile, on its final merge
    std::ostringstream os;
    serialize::WriteOptions with_values;
    with_values.include_values = true;
    serialize::write_l0(os, *ref.trace, with_values);
    CHECK_THAT(os.str(), ContainsSubstring("KPUL0 1.4.0"));
    CHECK_THAT(os.str(), ContainsSubstring("MIN_CONSUMER 1.4.0"));
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.7.0"));
    CHECK_THAT(os.str(), ContainsSubstring("type=int8 acc=int32"));
    CHECK_THAT(os.str(), ContainsSubstring("kind=REQUANT out=C:0,0 m=3 shift=9"));
    std::istringstream is(os.str());
    TileProgram back = serialize::read_l0(is);
    CHECK(back.operand("C").acc == "int32");
    (void)TileProgramReference().run(back);
    const auto& a = back.operand("C").values;
    const auto& b = ref.values->operand("C").values;
    CHECK((a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0));
}

TEST_CASE("Requant: the trace-form L-CA driver runs it, and checks the inputs' accumulator", "[program][csp][run][requant]") {
    // The review of #437: the trace form scanned the recorded REQUANT (and BLOCK_CONVERT) as calls,
    // and the structured form did not check an input's accumulator type.
    const auto d = s1_with_combiner();
    const lang::Program ast = lang::parse(matmul("os", true));
    const TileProgram inputs = driver::csp_inputs(ast);
    const auto ref = driver::csp_reference(ast, inputs, 1'000'000);
    REQUIRE(ref.values);
    REQUIRE(ref.trace);
    CspProgram p = lang::compile(matmul("os", true));
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = inputs.operand(name).values;
    auto cfg = sw::kpu::timing::csp_config_from(d);
    REQUIRE(cfg);
    {
        sw::kpu::timing::ConcurrentTimingExecutor exec(cfg->config);
        const auto r = sw::kpu::timing::CspDriver(exec, p).run();
        REQUIRE(r.completed);
        const auto& a = r.values.operand("C").values;
        const auto& b = ref.values->operand("C").values;
        CHECK((a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0));
    }
    {
        TileProgram wrong = inputs;
        wrong.operand("C").acc.clear();                  // int8, accumulating in int8
        sw::kpu::timing::ConcurrentTimingExecutor exec(cfg->config);
        lang::ActionStream s(ast, lang::target_from(d));
        CHECK_THROWS_WITH(sw::kpu::timing::CspDriver(exec, s, wrong),
                          ContainsSubstring("'C' accumulates in int8; the program declares int32"));
    }
    // A REQUANT field with trailing characters is refused, not truncated.
    std::ostringstream os;
    serialize::write_l0(os, *ref.trace);
    std::string bad = os.str();
    bad.replace(bad.find(" m=3 "), 5, " m=3.5 ");
    std::istringstream is(bad);
    CHECK_THROWS_WITH(serialize::read_l0(is), ContainsSubstring("m= and shift= are integers"));
}

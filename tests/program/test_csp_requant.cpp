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
#include <set>
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
    BOTH_REFUSE(head + q + fab + "  store C[0, 0];\n" + end, "holds int32 accumulator values; its context needs scale(m, shift) @ combiner");
    // Beside it (#427's integer epilogue): a bias in the accumulator's type before it, relu after it.
    const std::string vec = "  vector b[16] int32 tile 16 in;\n  vector b8[16] int8 tile 16 in;\n";
    const std::string fab_b = "  resident A[0, 0], B[0, 0], b[0], b8[0];\n"
                              "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n";
    const std::string end_b = "  release A[0, 0], B[0, 0], b[0], b8[0];\n}\n";
    BOTH_ACCEPT(head + q + fab + "  store C[0, 0] via scale(1, 4) @ combiner, relu @ combiner;\n" + end);
    BOTH_ACCEPT(head + vec + q + fab_b + "  store C[0, 0] via add(b[0]) @ combiner, scale(1, 4) @ combiner;\n" + end_b);
    BOTH_ACCEPT(head + vec + q + fab_b +
                "  store C[0, 0] via add(b[0]) @ fabric, scale(1, 4) @ combiner, relu @ bm.egress;\n" + end_b);
    // The bias is in the accumulator's type, and before the scale.
    BOTH_REFUSE(head + vec + q + fab_b + "  store C[0, 0] via add(b8[0]) @ combiner, scale(1, 4) @ combiner;\n" + end_b,
                "is added to its int32 accumulator values; b8 is int8");
    BOTH_REFUSE(head + vec + q + fab_b + "  store C[0, 0] via scale(1, 4) @ combiner, add(b[0]) @ combiner;\n" + end_b,
                "add follows scale");
    // An integer result's activation is relu; a block format converts real values.
    BOTH_REFUSE(head + q + fab + "  store C[0, 0] via scale(1, 4) @ combiner, gelu @ combiner;\n" + end,
                "an integer result's activation is relu");
    BOTH_REFUSE(head + q + fab + "  store C[0, 0] via scale(1, 4) @ combiner, mxint8 @ combiner;\n" + end,
                "a block format converts real values");
    // Still needed on every path into storage, with a bias or relu beside it.
    BOTH_REFUSE(head + vec + q + fab_b + "  store C[0, 0] via add(b[0]) @ combiner;\n" + end_b,
                "its context needs scale(m, shift) @ combiner");
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
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.7.0"));   // REQUANT alone: 1.8.0 is a bias's or relu's
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

namespace {

// A requantized linear (#427): act(scale(A . B + b)), every stage @ combiner, b in int32. Without
// the bias or the relu, by removing it from the context the generator writes.
std::string linear_q(const std::string& dataflow, bool bias, bool relu) {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dtype = "int8";
    o.dataflow = dataflow;
    o.place = "combiner";
    o.requant = true;
    o.requant_m = kM;
    o.requant_shift = kShift;
    std::string src = gen::generate(o);
    auto drop = [&](const std::string& what) {
        for (std::size_t at; (at = src.find(what)) != std::string::npos;) src.erase(at, what.size());
    };
    if (!bias) drop("add(b[j]) @ combiner, ");
    if (!relu) drop(", relu @ combiner");
    return src;
}

platform::DeviceSpecification s1_with_full_combiner() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0,
                                                                  {"add", "relu", "scale", "mxint8", "mxfp8"}};
    return d;
}

}  // namespace

TEST_CASE("Requant: a bias before the scale and relu after it, on every dataflow and level",
          "[program][csp][run][requant]") {
    // The fused-operation matrix of an integer result (#427): bias x scale x relu, under OS, WS
    // and IS. The reference is computed here: the int32 matmul, plus b[c] as an int32 register
    // adds it (wrapping), requantized, then max(0, q).
    const auto d = s1_with_full_combiner();
    CHECK_THAT(linear_q("ws", true, true),
               ContainsSubstring("via add(b[j]) @ combiner, scale(3, 9) @ combiner, relu @ combiner;"));
    CHECK_THAT(linear_q("os", true, true), ContainsSubstring("vector b[64] int32 tile 16 in;"));
    for (const char* df : {"os", "ws", "is"}) {
        const lang::Program full = lang::parse(linear_q(df, true, true));
        const TileProgram in = driver::csp_inputs(full);           // A, B (int8) and b (int32)
        // The int32 matmul on the same A and B.
        const lang::Program plain_ast = lang::parse(matmul(df, false));
        TileProgram plain_in = driver::csp_inputs(plain_ast);
        for (const char* n : {"A", "B"}) plain_in.operand(n).values = in.operand(n).values;
        const auto plain = driver::csp_reference(plain_ast, plain_in, 1'000'000);
        REQUIRE(plain.values);
        const TensorOperand& c32 = plain.values->operand("C");
        const TensorOperand& b = in.operand("b");
        bool biased = false, negative = false;
        for (const bool bias : {false, true})
            for (const bool relu : {false, true}) {
                CAPTURE(df, bias, relu);
                const std::string src = linear_q(df, bias, relu);
                const lang::Program ast = lang::parse(src);
                TileProgram run_in = driver::csp_inputs(ast);
                for (const char* n : {"A", "B", "b"}) run_in.operand(n).values = in.operand(n).values;
                driver::CspRunRequest req{ast, &d, run_in,
                                          {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                           ExecutionLevel::CycleAccurate}};
                const auto r = driver::run_csp(req);
                REQUIRE(r.reference);
                CHECK(r.reference_note.empty());           // the trace replays L-B bit for bit
                std::vector<long long> want;
                for (Dim rr = 0; rr < c32.rows; ++rr)
                    for (Dim cc = 0; cc < c32.cols; ++cc) {
                        long long x = static_cast<long long>(c32.value(rr, cc));
                        if (bias) {
                            const long long bc = static_cast<long long>(b.value(cc, 0));
                            biased = biased || bc != 0;
                            // An int32 register: the sum wraps modulo 2^32.
                            x = static_cast<long long>(static_cast<std::int32_t>(
                                static_cast<std::uint32_t>(static_cast<std::int64_t>(x + bc))));
                        }
                        long long q = expected(x, kM, kShift);
                        negative = negative || q < 0;
                        if (relu) q = std::max(q, 0LL);
                        want.push_back(q);
                    }
                for (const auto& o : r.levels) {
                    CAPTURE(to_string(o.level));
                    INFO(o.skipped.value_or(""));
                    REQUIRE_FALSE(o.skipped);
                    const TensorOperand& got = o.values.operand("C");
                    REQUIRE(got.type == "int8");
                    std::size_t n = 0, bad = 0;
                    for (Dim rr = 0; rr < got.rows; ++rr)
                        for (Dim cc = 0; cc < got.cols; ++cc)
                            if (static_cast<long long>(got.value(rr, cc)) != want[n++]) ++bad;
                    CHECK(bad == 0);
                }
            }
        CHECK(biased);                                    // the data reaches the bias...
        CHECK(negative);                                  // ...and the relu
    }
}

TEST_CASE("Requant: a trace with a bias or relu beside REQUANT is opset 1.8, and replays",
          "[program][csp][requant]") {
    for (const bool bias : {false, true}) {
        CAPTURE(bias);
        const lang::Program ast = lang::parse(linear_q("os", bias, !bias));
        const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
        REQUIRE(ref.trace);
        std::ostringstream os;
        serialize::WriteOptions with_values;
        with_values.include_values = true;
        serialize::write_l0(os, *ref.trace, with_values);
        CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.8.0"));
        CHECK_THAT(os.str(), ContainsSubstring(bias ? "kind=BIAS_ADD" : "kind=ACTIVATION"));
        std::istringstream is(os.str());
        TileProgram back = serialize::read_l0(is);
        (void)TileProgramReference().run(back);
        const auto& a = back.operand("C").values;
        const auto& w = ref.values->operand("C").values;
        CHECK((a.size() == w.size() && std::memcmp(a.data(), w.data(), a.size() * sizeof(float)) == 0));
    }
    // A plain requantized matmul's trace stays 1.7.0: the file says what is in it.
    const lang::Program plain = lang::parse(matmul("os", true));
    const auto ref = driver::csp_reference(plain, driver::csp_inputs(plain), 1'000'000);
    REQUIRE(ref.trace);
    CHECK(serialize::opset_for(*ref.trace).str() == "1.7.0");
}

TEST_CASE("Requant: the generator writes a requantized linear on the Combiner only", "[program][csp][gen][requant]") {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dtype = "int8";
    o.requant = true;
    o.place = "fabric";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("runs on the Combiner, where its scale does"));
    o.place = "combiner";
    o.act = ActivationFn::Gelu;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("an integer result's activation is relu"));
    o.act = ActivationFn::Relu;
    o.format = "mxint8";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("a block format converts real values"));
    o.format = "none";
    o.requant = false;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("dtype is matmul's (and a requantized linear's)"));
}

namespace {

// A requantized linear with a per-channel scale (#427): scale(qm[j], qs[j]) @ combiner.
std::string linear_pc(const std::string& dataflow) {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 14;
    o.dtype = "int8";
    o.dataflow = dataflow;
    o.place = "combiner";
    o.requant = true;
    o.requant_channels = true;
    return gen::generate(o);
}

// Each column's multiplier and shift: valid, and varying, so a level that read the wrong column's
// would be caught.
void set_channels(TileProgram& in) {
    TensorOperand& m = in.operand("qm");
    TensorOperand& s = in.operand("qs");
    for (Dim c = 0; c < m.rows; ++c) {
        m.set_word(c, 0, numeric::encode("int32", double(1 + (c * 5) % 7)));
        s.set_word(c, 0, numeric::encode("int32", double(6 + (c * 3) % 7)));
    }
}

}  // namespace

TEST_CASE("Requant: a per-channel scale, on every dataflow and level", "[program][csp][run][requant]") {
    // Each column c: round((x + b[c]) * qm[c] / 2^qs[c]), saturated, then relu. The reference is
    // computed here from the int32 matmul, as the uniform scale's is.
    const auto d = s1_with_full_combiner();
    CHECK_THAT(linear_pc("os"), ContainsSubstring("via add(b[j]) @ combiner, scale(qm[j], qs[j]) @ combiner, relu @ combiner;"));
    CHECK_THAT(linear_pc("os"), ContainsSubstring("vector qs[64] int32 tile 16 in;"));
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        const lang::Program ast = lang::parse(linear_pc(df));
        TileProgram in = driver::csp_inputs(ast);
        set_channels(in);
        const lang::Program plain_ast = lang::parse(matmul(df, false));
        TileProgram plain_in = driver::csp_inputs(plain_ast);
        for (const char* n : {"A", "B"}) plain_in.operand(n).values = in.operand(n).values;
        const auto plain = driver::csp_reference(plain_ast, plain_in, 1'000'000);
        REQUIRE(plain.values);
        const TensorOperand& c32 = plain.values->operand("C");
        const TensorOperand& b = in.operand("b");
        const TensorOperand& qm = in.operand("qm");
        const TensorOperand& qs = in.operand("qs");
        std::vector<long long> want;
        for (Dim r = 0; r < c32.rows; ++r)
            for (Dim c = 0; c < c32.cols; ++c) {
                const long long x = static_cast<long long>(static_cast<std::int32_t>(static_cast<std::uint32_t>(
                    static_cast<std::int64_t>(c32.value(r, c)) + static_cast<std::int64_t>(b.value(c, 0)))));
                const long long q = expected(x, static_cast<long long>(qm.value(c, 0)), static_cast<int>(qs.value(c, 0)));
                want.push_back(std::max(q, 0LL));
            }
        driver::CspRunRequest req{ast, &d, in,
                                  {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
        const auto r = driver::run_csp(req);
        REQUIRE(r.reference);
        CHECK(r.reference_note.empty());               // the trace (REQUANT_CHANNELS) replays L-B
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            INFO(o.skipped.value_or(""));
            REQUIRE_FALSE(o.skipped);
            const TensorOperand& got = o.values.operand("C");
            std::size_t n = 0, bad = 0;
            for (Dim rr = 0; rr < got.rows; ++rr)
                for (Dim cc = 0; cc < got.cols; ++cc)
                    if (static_cast<long long>(got.value(rr, cc)) != want[n++]) ++bad;
            CHECK(bad == 0);
        }
        // The channels differ: a uniform scale by column 0's values would not give these results.
        std::size_t differ = 0;
        for (Dim c = 1; c < c32.cols; ++c)
            differ += qm.value(c, 0) != qm.value(0, 0) || qs.value(c, 0) != qs.value(0, 0);
        CHECK(differ > 0);
    }
}

TEST_CASE("Requant: synthesized inputs keep a per-channel scale's domains, and every level runs on them",
          "[program][csp][run][requant][inputs]") {
    // #457: the shifts used to be filled across int32's range, and L-B refused the first DRAIN
    // ("the shift -1811849867 is outside 0..62"). The roles come from the program (the scale's
    // vectors), not from operand names.
    const auto d = s1_with_full_combiner();
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        const lang::Program ast = lang::parse(linear_pc(df));
        const auto roles = driver::operand_roles(ast);
        REQUIRE(roles.size() == 2);
        CHECK(roles.at("qm") == driver::OperandRole::Multiplier);
        CHECK(roles.at("qs") == driver::OperandRole::Shift);
        const TileProgram in = driver::csp_inputs(ast);
        const TensorOperand& qm = in.operand("qm");
        const TensorOperand& qs = in.operand("qs");
        std::set<long long> shifts;
        for (Dim r = 0; r < qs.rows; ++r) {
            const auto m = static_cast<long long>(qm.value(r, 0)), sh = static_cast<long long>(qs.value(r, 0));
            CHECK((m >= (1LL << 14) && m < (1LL << 15)));
            CHECK((sh >= 22 && sh <= 28));
            shifts.insert(sh);
        }
        CHECK(shifts.size() > 1);                      // the channels differ
        // The other operands keep fill_operand's values: A is not a role.
        CHECK(roles.count("A") == 0);
        driver::CspRunRequest req{ast, &d, in,
                                  {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
        const auto r = driver::run_csp(req);
        REQUIRE(r.reference);
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            INFO(o.skipped.value_or(""));
            REQUIRE_FALSE(o.skipped);
            CHECK(o.values.operand("C").values == r.reference->operand("C").values);
        }
    }
}

TEST_CASE("Requant: a vector that is a shift anywhere keeps the shift's domain", "[program][csp][requant][inputs]") {
    // #461's review: `scale(qm[j], qs[j])` on one store and `scale(qs[j], qm[j])` on another makes
    // each vector a multiplier in one stage and a shift in the other. Shift wins: its domain is
    // valid in both positions, and the synthesized run would otherwise refuse the first stage.
    const lang::Program ast = lang::parse(
        "csp 1.0\nprogram t machine flat(l3 = 12, l2 = 2) {\n"
        "  tensor A[16,16] int8 tile 16x16 in;\n  tensor B[16,16] int8 tile 16x16 in;\n"
        "  tensor C[16,16] int8 tile 16x16 out acc int32;\n  tensor E[16,16] int8 tile 16x16 out acc int32;\n"
        "  vector qm[16] int32 tile 16 in;\n  vector qs[16] int32 tile 16 in;\n"
        "  resident A[0, 0], B[0, 0], qm[0], qs[0];\n"
        "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
        "  store C[0, 0] via scale(qm[0], qs[0]) @ combiner;\n"
        "  acc E[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> E[0, 0]; }\n"
        "  store E[0, 0] via scale(qs[0], qm[0]) @ combiner;\n"
        "  release A[0, 0], B[0, 0], qm[0], qs[0];\n}\n");
    const auto roles = driver::operand_roles(ast);
    CHECK(roles.at("qm") == driver::OperandRole::Shift);
    CHECK(roles.at("qs") == driver::OperandRole::Shift);
    const TileProgram in = driver::csp_inputs(ast);
    for (const char* n : {"qm", "qs"})
        for (Dim r = 0; r < in.operand(n).rows; ++r) {
            const auto v = static_cast<long long>(in.operand(n).value(r, 0));
            CHECK((v >= 22 && v <= 28));
        }
    CHECK_NOTHROW(driver::csp_reference(ast, in, 1'000'000));
}

TEST_CASE("Requant: a per-channel scale's trace is opset 1.9, and refuses bad values", "[program][csp][requant]") {
    const lang::Program ast = lang::parse(linear_pc("os"));
    TileProgram in = driver::csp_inputs(ast);
    set_channels(in);
    const auto ref = driver::csp_reference(ast, in, 1'000'000);
    REQUIRE(ref.trace);
    std::ostringstream os;
    serialize::WriteOptions with_values;
    with_values.include_values = true;
    serialize::write_l0(os, *ref.trace, with_values);
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.9.0"));
    CHECK_THAT(os.str(), ContainsSubstring("kind=REQUANT_CHANNELS in=qm:0,0;qs:0,0 out=C:0,0"));
    std::istringstream is(os.str());
    TileProgram back = serialize::read_l0(is);
    (void)TileProgramReference().run(back);
    const auto& a = back.operand("C").values;
    const auto& w = ref.values->operand("C").values;
    CHECK((a.size() == w.size() && std::memcmp(a.data(), w.data(), a.size() * sizeof(float)) == 0));
    // A shift outside 0..62 is refused where it is used, naming the stage.
    TileProgram bad = in;
    bad.operand("qs").set_word(3, 0, numeric::encode("int32", 63.0));
    CHECK_THROWS_WITH(driver::csp_reference(ast, bad, 1'000'000), ContainsSubstring("the shift 63 is outside 0..62"));
}

TEST_CASE("Requant: what a per-channel scale refuses", "[program][csp][lang][requant]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8) {\n"
                             "  tensor A[16,16] int8 tile 16x16 in;\n  tensor B[16,16] int8 tile 16x16 in;\n"
                             "  tensor C[16,16] int8 tile 16x16 out acc int32;\n"
                             "  vector qm[16] int32 tile 16 in;\n  vector qs[16] int32 tile 16 in;\n"
                             "  vector q8[16] int8 tile 16 in;\n";
    const std::string fab = "  resident A[0, 0], B[0, 0], qm[0], qs[0], q8[0];\n"
                            "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n";
    const std::string end = "  release A[0, 0], B[0, 0], qm[0], qs[0], q8[0];\n}\n";
    BOTH_ACCEPT(head + fab + "  store C[0, 0] via scale(qm[0], qs[0]) @ combiner;\n" + end);
    BOTH_REFUSE(head + fab + "  store C[0, 0] via scale(qm[0]) @ combiner;\n" + end,
                "a vector of multipliers and one of shifts");
    BOTH_REFUSE(head + fab + "  store C[0, 0] via scale(qm[0], q8[0]) @ combiner;\n" + end,
                "a per-channel scale's multipliers and shifts are int32; q8 is int8");
    BOTH_REFUSE(head + fab + "  store C[0, 0] via scale(qm[0], A[0, 0]) @ combiner;\n" + end, "scale takes a vector operand");
    // Resident, as an add's vector must be.
    BOTH_REFUSE(head + "  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                       "  store C[0, 0] via scale(qm[0], qs[0]) @ combiner;\n  release A[0, 0], B[0, 0];\n}\n",
                "resident");
    // The text round-trips.
    const std::string ok = head + fab + "  store C[0, 0] via scale(qm[0], qs[0]) @ combiner;\n" + end;
    const std::string text = lang::format(lang::parse(ok));
    CHECK_THAT(text, ContainsSubstring("scale(qm[0], qs[0]) @ combiner"));
    CHECK(lang::format(lang::parse(text)) == text);
    // Synthesized inputs keep the shifts in their domain (#457) and run; a shift outside 0..62,
    // given explicitly, is refused, naming the stage and the value.
    const lang::Program ast = lang::parse(linear_pc("os"));
    TileProgram in = driver::csp_inputs(ast);
    CHECK_NOTHROW(driver::csp_reference(ast, in, 1'000'000));
    in.operand("qs").set_word(0, 0, numeric::encode("int32", 70.0));
    CHECK_THROWS_WITH(driver::csp_reference(ast, in, 1'000'000),
                      ContainsSubstring("scale(qm[0,0], qs[0,0]): requantize: the shift 70 is outside 0..62"));
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dtype = "int8";
    o.requant_channels = true;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("it needs --requant"));
}

namespace {

// A requantized matmul with an init (#395): C = scale(A . B + D) or scale(A . B + b), the init in
// the accumulator's type, added before the scale (the fabric's init for os; the Combiner's
// MERGE_INIT on each tile's first merge for ws and is, the scale on its final merge).
gen::Options matmul_qi(const std::string& dataflow, const std::string& init, bool channels, Dim size) {
    gen::Options o;
    o.algo = "matmul";
    o.size = size;
    o.tile = 16;
    o.l3 = 32;
    o.dtype = "int8";
    o.dataflow = dataflow;
    o.init = init;
    o.requant = true;
    o.requant_m = kM;
    o.requant_shift = kShift;
    o.requant_channels = channels;
    return o;
}

}  // namespace

TEST_CASE("Requant: an init before the scale, on every dataflow and level", "[program][csp][run][requant][init]") {
    // The full int8 epilogue the generator used to refuse ("requant with an init is not built
    // yet"): D's tile (init bias) or b's row (init bias-row), int32, added to the int32 matmul as a
    // register adds it (wrapping), then the scale, uniform or per channel. One K block (size 16) is
    // the case where the first merge, which adds the init, is also the final one, which scales.
    const auto d = s1_with_full_combiner();
    std::size_t moved = 0;                                  // results the init changes
    for (const Dim size : {Dim{64}, Dim{16}})
        for (const char* df : {"os", "ws", "is"})
            for (const char* init : {"bias", "bias-row"})
                for (const bool channels : {false, true}) {
                    CAPTURE(size, df, init, channels);
                    const std::string src = gen::generate(matmul_qi(df, init, channels, size));
                    CHECK_THAT(src, ContainsSubstring("int8 tile 16x16 out acc int32"));
                    CHECK_THAT(src, ContainsSubstring(std::string(init) == "bias" ? "tensor D[" : "vector b["));
                    const lang::Program ast = lang::parse(src);
                    TileProgram in = driver::csp_inputs(ast);
                    if (channels) set_channels(in);
                    // The int32 matmul on the same A and B: no init, no scale.
                    gen::Options po = matmul_qi(df, "none", false, size);
                    po.requant = false;
                    const lang::Program plain_ast = lang::parse(gen::generate(po));
                    TileProgram plain_in = driver::csp_inputs(plain_ast);
                    for (const char* n : {"A", "B"}) plain_in.operand(n).values = in.operand(n).values;
                    const auto plain = driver::csp_reference(plain_ast, plain_in, 1'000'000);
                    REQUIRE(plain.values);
                    const TensorOperand& c32 = plain.values->operand("C");
                    const bool tile = std::string(init) == "bias";
                    const TensorOperand& bias = in.operand(tile ? "D" : "b");
                    std::vector<long long> want;
                    for (Dim r = 0; r < c32.rows; ++r)
                        for (Dim c = 0; c < c32.cols; ++c) {
                            const long long x = static_cast<long long>(c32.value(r, c));
                            const long long bv = static_cast<long long>(tile ? bias.value(r, c) : bias.value(c, 0));
                            const long long s = static_cast<long long>(static_cast<std::int32_t>(
                                static_cast<std::uint32_t>(static_cast<std::int64_t>(x + bv))));
                            const long long m = channels ? static_cast<long long>(in.operand("qm").value(c, 0)) : kM;
                            const int sh = channels ? static_cast<int>(in.operand("qs").value(c, 0)) : kShift;
                            want.push_back(expected(s, m, sh));
                            moved += expected(s, m, sh) != expected(x, m, sh);
                        }
                    driver::CspRunRequest req{ast, &d, in,
                                              {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                               ExecutionLevel::CycleAccurate}};
                    const auto r = driver::run_csp(req);
                    REQUIRE(r.reference);
                    CHECK(r.reference_note.empty());       // the trace replays L-B bit for bit
                    REQUIRE(r.levels.size() == 3);
                    for (const auto& o : r.levels) {
                        CAPTURE(to_string(o.level));
                        INFO(o.skipped.value_or(""));
                        REQUIRE_FALSE(o.skipped);
                        const TensorOperand& got = o.values.operand("C");
                        REQUIRE(got.type == "int8");
                        std::size_t n = 0, bad = 0;
                        for (Dim rr = 0; rr < got.rows; ++rr)
                            for (Dim cc = 0; cc < got.cols; ++cc)
                                if (static_cast<long long>(got.value(rr, cc)) != want[n++]) ++bad;
                        CHECK(bad == 0);
                    }
                }
    CHECK(moved > 0);                                       // the data reaches the init
}

TEST_CASE("Requant: an accumulate init is still refused, by name", "[program][csp][gen][requant][init]") {
    // C += A . B onto an int8 result would add the scaled result to an unscaled sum.
    gen::Options o = matmul_qi("os", "accumulate", false, 64);
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("requant with init accumulate is not built yet"));
}

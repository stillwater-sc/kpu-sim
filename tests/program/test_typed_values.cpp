// ============================================================================
// tests/program/test_typed_values.cpp
// The typed value path (#394; docs/plans/typed-value-path.md): an operand declares its number
// system (`tensor A[8,8] int8 tile 4x4 in;`), the arithmetic is stillwater-sc/universal's, and a
// gemm runs the MAC registered for its (input, accumulator) pair -- int8 x int8 summed in int32,
// wrapping at 32 bits, as the KPU RTL's systolic_int8 does.
//
// Values are checked against a reference computed here from the inputs in 64-bit integers and
// wrapped to 32 bits: not against another level, and bit for bit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/numeric.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/program/tile_kernels.hpp>

#include <cmath>
#include <cstdint>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace numeric = sw::kpu::program::numeric;
using Catch::Matchers::ContainsSubstring;

namespace {

gen::Options matmul(const std::string& init = "none", const std::string& orient = "col") {
    gen::Options o;
    o.size = 8;
    o.tile = 4;
    o.l3 = 8;
    o.orient = orient;
    o.init = init;
    o.dtype = "int8";
    return o;
}

// x modulo 2^32, as two's complement: what an int32 register holds.
std::int32_t wrap32(long long x) { return static_cast<std::int32_t>(static_cast<std::uint32_t>(x)); }

std::int32_t ival(const TensorOperand& t, Dim r, Dim c) { return static_cast<std::int32_t>(t.value(r, c)); }

// C[r][j] = init[r][j] + sum_k A[r][k] B[k][j], in 64-bit integers, wrapped to int32 at the end --
// the same as wrapping after every step, since addition and multiplication commute with mod 2^32.
std::vector<std::int32_t> reference(const TileProgram& in, const std::string& init) {
    const TensorOperand& A = in.operand("A");
    const TensorOperand& B = in.operand("B");
    const Dim n = A.rows;
    std::vector<std::int32_t> c(static_cast<std::size_t>(n) * n);
    for (Dim r = 0; r < n; ++r)
        for (Dim j = 0; j < n; ++j) {
            long long v = 0;
            if (init == "bias") v = ival(in.operand("D"), r, j);
            if (init == "bias-row") v = ival(in.operand("b"), j, 0);
            if (init == "accumulate") v = ival(in.operand("C"), r, j);
            for (Dim k = 0; k < n; ++k) v += static_cast<long long>(ival(A, r, k)) * ival(B, k, j);
            c[static_cast<std::size_t>(r) * n + j] = wrap32(v);
        }
    return c;
}

std::vector<std::int32_t> ints(const TensorOperand& t) {
    std::vector<std::int32_t> v;
    for (Dim r = 0; r < t.rows; ++r)
        for (Dim c = 0; c < t.cols; ++c) v.push_back(ival(t, r, c));
    return v;
}

struct LbRun {
    TileProgram in, out, trace;
};

LbRun lb(const std::string& src) {
    const lang::Program ast = lang::parse(src);
    LbRun r;
    r.in = driver::csp_inputs(ast);
    BehavioralInterpreter b;
    b.begin(r.in, /*record=*/true, lang::l3_geometry(ast), static_cast<std::uint32_t>(ast.l2));
    lang::ActionStream s(ast);
    while (auto e = s.next()) b.step(e->action, e->op ? &*e->op : nullptr, &e->l0);
    (void)b.finish();
    r.out = b.result();
    r.trace = b.trace();
    return r;
}

std::string error_of(const std::string& src, bool symbolic) {
    try {
        if (symbolic) (void)lang::validate(src);
        else (void)lang::compile(src);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "accepted";
}

const char* const kHead = "csp 1.0\nprogram t machine flat(l3 = 6) {\n"
                          "  tensor A[4,4] int8 tile 4x4 in;\n  tensor B[4,4] int8 tile 4x4 in;\n"
                          "  tensor F[4,4] tile 4x4 in;\n  tensor C[4,4] int32 tile 4x4 out;\n"
                          "  tensor G[4,4] tile 4x4 out;\n  vector b[4] int32 tile 4 in;\n";
const char* const kBody = "  resident A[0, 0], B[0, 0], F[0, 0], b[0];\n";
const char* const kTail = "  release A[0, 0], B[0, 0], F[0, 0], b[0];\n}\n";

}  // namespace

TEST_CASE("typed values: the registry's number systems, on Universal", "[program][numeric]") {
    CHECK(numeric::names() == "fp32, int8, int32");
    CHECK(numeric::mac_names() == "fp32 -> fp32, int8 -> int32");
    CHECK(numeric::find("posit16") == nullptr);
    CHECK_THROWS_WITH(numeric::get("posit16"), ContainsSubstring("'posit16' is not a registered number system"));
    // int8's word is its two's-complement byte, zero-extended.
    CHECK(numeric::encode("int8", -1) == 0xFFu);
    CHECK(numeric::encode("int8", -128) == 0x80u);
    CHECK(numeric::encode("int8", 127) == 0x7Fu);
    CHECK(numeric::decode("int8", 0xFFu) == -1.0);
    CHECK(numeric::decode("int8", 0x80u) == -128.0);
    CHECK(numeric::encode("int8", 200) == numeric::encode("int8", -56));   // modulo 2^8
    CHECK(numeric::decode("int32", numeric::encode("int32", -7)) == -7.0);
    CHECK(numeric::decode("int32", 0x80000000u) == -2147483648.0);
    // fp32's word is the float.
    CHECK(numeric::encode("fp32", 1.5) == numeric::word_of(1.5f));
    for (std::int32_t v : {-128, -1, 0, 1, 127})
        CHECK(numeric::decode("int8", numeric::encode("int8", v)) == double(v));
    // Modulo 2^N for any finite value, even past long long's range; a non-finite one is refused.
    CHECK(numeric::encode("int8", 18446744073709551616.0) == 0u);           // 2^64
    CHECK(numeric::encode("int8", 1099511627781.0) == 5u);                  // 2^40 + 5
    CHECK(numeric::encode("int32", -18446744073709551616.0) == 0u);
    CHECK(numeric::encode("int8", -3.7) == numeric::encode("int8", -3));    // truncated
    CHECK_THROWS_WITH(numeric::encode("int8", std::nan("")), ContainsSubstring("has no integer<8> encoding"));
    CHECK_THROWS_WITH(numeric::encode("int32", HUGE_VAL), ContainsSubstring("has no integer<32> encoding"));
}

TEST_CASE("typed values: the int8 -> int32 MAC wraps at 32 bits", "[program][numeric]") {
    const numeric::Mac mac = numeric::mac("int8", "int32");
    REQUIRE(mac);
    CHECK(numeric::mac("int8", "fp32") == nullptr);
    CHECK(numeric::mac("int32", "int32") == nullptr);
    // 1x1x2: c = 2147483600 + (-128)(-128) + 127*127, past INT32_MAX: it wraps.
    const numeric::Word a[2] = {numeric::encode("int8", -128), numeric::encode("int8", 127)};
    const numeric::Word b[2] = {numeric::encode("int8", -128), numeric::encode("int8", 127)};
    numeric::Word c[1] = {numeric::encode("int32", 2147483600)};
    mac(a, b, c, 1, 1, 2);
    CHECK(numeric::decode("int32", c[0]) == double(wrap32(2147483600LL + 16384 + 16129)));
    CHECK(numeric::decode("int32", c[0]) < 0);
}

TEST_CASE("typed values: declared, formatted, refused by name", "[program][numeric][lang]") {
    const std::string src = gen::generate(matmul());
    CHECK_THAT(src, ContainsSubstring("tensor A[8,8] int8 tile 4x4 in;"));
    CHECK_THAT(src, ContainsSubstring("tensor C[8,8] int32 tile 4x4 out;"));
    const lang::Program ast = lang::parse(src);
    CHECK(lang::format(lang::parse(lang::format(ast))) == lang::format(ast));
    CHECK_THAT(lang::format(ast), ContainsSubstring("tensor B[8,8] int8 tile 4x4 in;"));
    // fp32 is the default, and is not written.
    gen::Options f = matmul();
    f.dtype = "fp32";
    CHECK_THAT(lang::format(lang::parse(gen::generate(f))), ContainsSubstring("tensor C[8,8] tile 4x4 out;"));
    CHECK(lang::parse(gen::generate(f)).decls.at(0).type == "fp32");
    // An unknown number system is a syntax error naming the registered ones.
    std::string bad = src;
    bad.replace(bad.find("int8"), 4, "posit16");
    CHECK_THROWS_WITH(lang::parse(bad), ContainsSubstring("'posit16' is not a number system (fp32, int8, int32)"));
}

TEST_CASE("typed values: L-B computes int8 x int8 in int32, bit for bit", "[program][numeric][lb]") {
    for (const char* orient : {"col", "row"})
        for (const char* init : {"none", "bias", "bias-row", "accumulate"}) {
            CAPTURE(orient, init);
            const std::string src = gen::generate(matmul(init, orient));
            const LbRun r = lb(src);
            REQUIRE(r.out.operand("C").type == "int32");
            CHECK(ints(r.out.operand("C")) == reference(r.in, init));
            // The inputs are untouched, and span int8's range.
            CHECK(r.out.operand("A").values == r.in.operand("A").values);
            int lo = 0, hi = 0;
            for (std::int32_t v : ints(r.in.operand("A"))) { lo = std::min(lo, v); hi = std::max(hi, v); }
            CHECK(lo >= -128);
            CHECK(hi <= 127);
            CHECK(hi - lo > 200);
        }
    // Accumulating onto a C near INT32_MAX: every element passes it, and wraps as the RTL's does.
    const lang::Program ast = lang::parse(gen::generate(matmul("accumulate")));
    TileProgram in = driver::csp_inputs(ast);
    for (Dim r = 0; r < 8; ++r)
        for (Dim c = 0; c < 8; ++c) {
            in.operand("A").set_value(r, c, 127);
            in.operand("B").set_value(r, c, 127);
            in.operand("C").set_value(r, c, 2147483647.0 - (r * 8 + c));
        }
    const driver::CspReference ref = driver::csp_reference(ast, in, 0);
    REQUIRE(ref.values);
    const auto want = reference(in, "accumulate");
    CHECK(ints(ref.values->operand("C")) == want);
    for (std::int32_t v : want) CHECK(v < 0);                 // 127 * 127 * 8 past INT32_MAX: negative
}

TEST_CASE("typed values: the L0 trace replays bit for bit and round-trips as .l0", "[program][numeric][l0]") {
    const lang::Program ast = lang::parse(gen::generate(matmul("bias")));
    const TileProgram in = driver::csp_inputs(ast);
    const driver::CspReference ref = driver::csp_reference(ast, in, 1u << 20);
    REQUIRE(ref.replay_mismatch.empty());
    REQUIRE(ref.trace);
    std::ostringstream os;
    serialize::WriteOptions w;
    w.include_values = true;
    serialize::write_l0(os, *ref.trace, w);
    const std::string text = os.str();
    CHECK_THAT(text, ContainsSubstring("KPUL0 1.3.0"));
    CHECK_THAT(text, ContainsSubstring("MIN_CONSUMER 1.3.0"));
    CHECK_THAT(text, ContainsSubstring("type=int8"));
    CHECK_THAT(text, ContainsSubstring("type=int32"));
    std::istringstream is(text);
    const TileProgram back = serialize::read_l0(is);
    for (const auto& name : ref.trace->operand_order()) {
        CAPTURE(name);
        CHECK(back.operand(name).type == ref.trace->operand(name).type);
        CHECK(ints(back.operand(name)) == ints(ref.trace->operand(name)));
    }
    // The replay of the read-back trace computes L-B's C.
    TileProgram replay = back;
    TileProgramReference().run(replay);
    CHECK(ints(replay.operand("C")) == ints(ref.values->operand("C")));
    // An fp32 program's file is what it was: container 1.2.0, no type attribute.
    gen::Options f = matmul();
    f.dtype = "fp32";
    const lang::Program fa = lang::parse(gen::generate(f));
    const driver::CspReference fr = driver::csp_reference(fa, driver::csp_inputs(fa), 1u << 20);
    std::ostringstream fs;
    serialize::write_l0(fs, *fr.trace, w);
    CHECK_THAT(fs.str(), ContainsSubstring("KPUL0 1.2.0"));
    CHECK_THAT(fs.str(), !ContainsSubstring("type="));
    // A value out of the type's range is refused, not wrapped.
    std::string out_of_range = text;
    const std::size_t row = out_of_range.find("VALUES_ROW \"A\" 0 ");
    REQUIRE(row != std::string::npos);
    out_of_range.insert(row + std::string("VALUES_ROW \"A\" 0 ").size(), "300 ");
    const std::size_t eol = out_of_range.find('\n', row);
    out_of_range.erase(out_of_range.rfind(' ', eol), eol - out_of_range.rfind(' ', eol));
    std::istringstream bad(out_of_range);
    CHECK_THROWS_WITH(serialize::read_l0(bad), ContainsSubstring("'300' is not a int8 value"));
    // A typed file must demand 1.3.0 even without values: a 1.0.0 reader would read its words as fp32.
    serialize::WriteOptions nv;
    std::ostringstream ns;
    serialize::write_l0(ns, *ref.trace, nv);
    std::string under = ns.str();
    const std::size_t mc = under.find("MIN_CONSUMER 1.3.0");
    REQUIRE(mc != std::string::npos);
    under.replace(mc, 18, "MIN_CONSUMER 1.0.0");
    std::istringstream us(under);
    CHECK_THROWS_WITH(serialize::read_l0(us), ContainsSubstring("but its records need 1.3.0"));
    // Integer values are written without the stream locale's digit grouping.
    struct Grouped : std::numpunct<char> {
        char do_thousands_sep() const override { return ','; }
        std::string do_grouping() const override { return "\3"; }
    };
    TileProgram big = *ref.trace;
    big.operand("C").set_value(0, 0, 1234567);
    std::ostringstream gs;
    gs.imbue(std::locale(std::locale::classic(), new Grouped));
    serialize::write_l0(gs, big, w);
    CHECK_THAT(gs.str(), ContainsSubstring(" 1234567"));
    CHECK_THAT(gs.str(), !ContainsSubstring("1,234,567"));
    std::istringstream gi(gs.str());
    CHECK(ival(serialize::read_l0(gi).operand("C"), 0, 0) == 1234567);
}

TEST_CASE("typed values: what is refused, and where", "[program][numeric][refuse]") {
    struct Case { std::string stmts, why; };
    const std::vector<Case> cases = {
        {"  acc G[0, 0] in fabric { call gemm(A[0, 0], F[0, 0]) +-> G[0, 0]; }\n  store G[0, 0];\n",
         "A is int8 and F is fp32; a gemm's inputs share a number system"},
        {"  acc G[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> G[0, 0]; }\n  store G[0, 0];\n",
         "no MAC for int8 -> fp32"},
        {"  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n  store C[0, 0] via relu @ fabric;\n",
         "a tile context computes in fp32; C is int32"},
        {"  acc G[0, 0] in fabric { call gemm(F[0, 0], F[0, 0]) +-> G[0, 0]; }\n  store G[0, 0] via add(b[0]) @ fabric;\n",
         "add(b[0]): a tile context computes in fp32; b is int32"},
        {"  acc C[0, 0] in fabric init F[0, 0] { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n  store C[0, 0];\n",
         "F is fp32 and C is int32; an init is in the accumulator's number system"},
    };
    for (const Case& c : cases) {
        const std::string src = std::string(kHead) + kBody + c.stmts + kTail;
        CAPTURE(src);
        CHECK_THAT(error_of(src, true), ContainsSubstring(c.why));
    }
    // A typed gemm with its bias row as the init is accepted, and computes.
    const std::string ok = std::string(kHead) + kBody +
                           "  acc C[0, 0] in fabric init b[0] { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                           "  store C[0, 0];\n" + kTail;
    CHECK(error_of(ok, true) == "accepted");
    // The kernels refuse a typed operand where they compute in fp32, by name.
    TileProgram p("k");
    p.add_operand(TensorOperand("Y", 4, 4, 4, 4)).type = "int32";
    p.add_operand(TensorOperand("v", 4, 1, 4, 1));
    TileOp add;
    add.kind = TileOpKind::BiasAdd;
    add.inputs = {{"v", 0, 0}};
    add.outputs = {{"Y", 0, 0}};
    TileKernelState st;
    CHECK_THROWS_WITH(apply(p, add, st), ContainsSubstring("operand Y is int32; this kernel computes in fp32 only"));
}

TEST_CASE("typed values: L-T1 carries L-B's; L-CA and the .dflow writer refuse until their steps",
          "[program][numeric][lt1]") {
    const auto spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    const auto dev = spec.device(0);
    const std::string src = gen::generate(matmul("bias-row"));
    const lang::Program ast = lang::parse(src);
    const TileProgram in = driver::csp_inputs(ast);
    const auto lt = driver::csp_run_level(driver::ExecutionLevel::BlockSequential, ast, &dev, in);
    REQUIRE_FALSE(lt.skipped);
    const LbRun r = lb(src);
    CHECK(ints(lt.values.operand("C")) == reference(r.in, "bias-row"));
    // The per-tile check reads C as int32, exactly: one unit off is a miss, whatever the bound.
    TileProgram off = lt.values;
    off.operand("C").set_value(3, 5, off.operand("C").value(3, 5) + 1);
    const TileCheck exact = check_tiles(off, lt.values, "C", ElementFormat{}, Proximity{1e6, 1.0});
    CHECK_FALSE(exact.pass);
    CHECK(exact.misses.size() == 1);
    CHECK(check_tiles(lt.values, lt.values, "C").pass);
    // L-CA: step 2.
    const auto why = driver::csp_level_supported(driver::ExecutionLevel::CycleAccurate, ast, &dev);
    REQUIRE(why);
    CHECK_THAT(*why, ContainsSubstring("L-CA computes in fp32 until docs/plans/typed-value-path.md step 2"));
    // The .dflow writer: step 3.
    CHECK_THROWS_WITH(record::DomainFlowRecorder(ast, src, 0),
                      ContainsSubstring("A is int8; the .dflow writer carries fp32 operands"));
}

TEST_CASE("typed values: kpu-csp-gen's dtype", "[program][numeric][gen]") {
    gen::Options o = matmul();
    o.dtype = "int32";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("dtype 'int32' has no registered gemm"));
    o = matmul();
    o.algo = "lu";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("dtype is matmul's"));
}

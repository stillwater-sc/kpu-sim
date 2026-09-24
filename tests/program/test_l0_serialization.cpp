// ============================================================================
// tests/program/test_l0_serialization.cpp
// The L0 portable program as a file (#265 increment 1).
//
// The claim being tested is not "it parses" but "it is PORTABLE": a program
// written to a file and read back must EXECUTE IDENTICALLY. Structural equality
// is necessary and not sufficient -- a field silently dropped can leave two
// programs that look alike and compute differently.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
using namespace sw::kpu::program::serialize;

namespace {

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

} // namespace

TEST_CASE("a program round-trips and then executes bit-identically",
          "[program][serialize]") {
    // This is the whole point of the format. Values are increment 2, so the reloaded
    // program is filled the same way afterwards -- what must survive the file is the
    // STRUCTURE that decides the computation.
    for (const char* algo : {"matmul", "lu"}) {
        ProgramSpec ps;
        ps.algo = algo;
        ps.size = 48;
        ps.tile = 16;

        TileProgram original = derive(ps);
        const TileProgram reloaded = from_string(to_string(original));

        // Structure first, because a difference here localises a bug far better than a
        // value mismatch at the end does.
        REQUIRE(reloaded.ops().size() == original.ops().size());
        REQUIRE(reloaded.operand_order() == original.operand_order());
        for (const std::string& key : original.operand_order()) {
            const auto& a = original.operand(key);
            const auto& b = reloaded.operand(key);
            CHECK(b.rows == a.rows);
            CHECK(b.cols == a.cols);
            CHECK(b.tile_rows == a.tile_rows);
            CHECK(b.tile_cols == a.tile_cols);
            CHECK(b.values.size() == a.values.size());
        }
        for (std::size_t i = 0; i < original.ops().size(); ++i) {
            const TileOp& a = original.ops()[i];
            const TileOp& b = reloaded.ops()[i];
            CHECK(b.kind == a.kind);
            REQUIRE(b.inputs.size() == a.inputs.size());
            for (std::size_t k = 0; k < a.inputs.size(); ++k) {
                CHECK(b.inputs[k].operand == a.inputs[k].operand);
                CHECK(b.inputs[k].ti == a.inputs[k].ti);
                CHECK(b.inputs[k].tj == a.inputs[k].tj);
            }
            REQUIRE(b.outputs.size() == a.outputs.size());
            for (std::size_t k = 0; k < a.outputs.size(); ++k) {
                CHECK(b.outputs[k].operand == a.outputs[k].operand);
                CHECK(b.outputs[k].ti == a.outputs[k].ti);
                CHECK(b.outputs[k].tj == a.outputs[k].tj);
            }
            CHECK(b.alpha == a.alpha);
            CHECK(b.pivot_slot == a.pivot_slot);
            CHECK(b.port == a.port);
            CHECK(b.port_kind == a.port_kind);
            CHECK(b.label == a.label);
        }

        // Then the claim that matters: same answers, at BOTH levels that exist. A format
        // that round-trips structurally but executes differently is not portable.
        const DeviceSpec ds;
        const auto device = make_device(ds);
        for (ExecutionLevel level : all_levels()) {
            if (!level_implemented(level)) continue;
            TileProgram a = derive(ps);
            TileProgram b = from_string(to_string(a));
            fill(a, ps);
            fill(b, ps);
            const auto ra = run_at(level, a, device, Placement::single(device.compute_tiles));
            const auto rb = run_at(level, b, device, Placement::single(device.compute_tiles));
            CHECK(bits_equal(a.operand(result_operand(ps)).values,
                             b.operand(result_operand(ps)).values));
            CHECK(rb.makespan == ra.makespan);          // and identical timing
            CHECK(rb.summary.row_swaps == ra.summary.row_swaps);
            CHECK(rb.summary.permutation == ra.summary.permutation);
        }
    }
}

TEST_CASE("the file is text a human can read and diff", "[program][serialize]") {
    // §5's reason for text: a golden corpus whose rot is visible in a diff. That only
    // holds if the content is actually legible, so this pins the shape.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    const std::string text = to_string(derive(ps));

    // Pinned literally, so a version bump has to be DELIBERATE: an accidental one would
    // silently change what older readers accept, which is the failure this axis exists to
    // prevent. 1.2.0 added the STREAMS record; 1.1.0 added VALUES_ROW.
    CHECK(text.rfind("KPUL0 1.2.0", 0) == 0);           // magic first, version with it
    // A kernel is still readable by a 1.0.0 reader, because nothing was added to it.
    CHECK(text.find("MIN_CONSUMER 1.0.0\n") != std::string::npos);
    // The op set did NOT change: a new container record is not a new operator.
    CHECK(text.find("OPSET tile 1.0.0\n") != std::string::npos);
    CHECK(text.find("VALUES none\n") != std::string::npos);   // a kernel, not a test case
    CHECK(text.find("OPERAND \"A\" rows=32 cols=32 tile_rows=16 tile_cols=16\n") !=
          std::string::npos);
    CHECK(text.find("OP kind=MATMUL_ACCUM in=A:0,0;B:0,0 out=C:0,0\n") != std::string::npos);
    CHECK(text.find("\nEND\n") != std::string::npos);

    // alpha is omitted at its default, so a diff shows what differs rather than what is
    // merely present.
    CHECK(text.find("alpha=") == std::string::npos);
}

TEST_CASE("LU's pivot slots and labels survive the file", "[program][serialize]") {
    // The fields most likely to be dropped silently, because GEMM does not use them.
    ProgramSpec ps;
    ps.algo = "lu";
    ps.size = 64;
    ps.tile = 16;
    TileProgram original = derive(ps);
    const std::string text = to_string(original);
    CHECK(text.find("pivot=") != std::string::npos);
    CHECK(text.find("label=") != std::string::npos);
    CHECK(text.find("alpha=-1") != std::string::npos);     // the trailing update

    const TileProgram reloaded = from_string(text);
    std::size_t pivots = 0, labels = 0, negative_alpha = 0;
    for (std::size_t i = 0; i < original.ops().size(); ++i) {
        if (original.ops()[i].pivot_slot >= 0) { ++pivots;
            CHECK(reloaded.ops()[i].pivot_slot == original.ops()[i].pivot_slot); }
        if (!original.ops()[i].label.empty()) { ++labels;
            CHECK(reloaded.ops()[i].label == original.ops()[i].label); }
        if (original.ops()[i].alpha != 1.0f) { ++negative_alpha;
            CHECK(reloaded.ops()[i].alpha == original.ops()[i].alpha); }
    }
    CHECK(pivots > 0);
    CHECK(labels > 0);
    CHECK(negative_alpha > 0);
}

TEST_CASE("a name with quotes and backslashes round-trips", "[program][serialize]") {
    // Escaping is where text formats rot quietly: a name that breaks quoting would make
    // the rest of the record unparseable, and a corpus file unreadable.
    TileProgram p(R"(odd "name" with \ backslash)");
    p.add_operand(TensorOperand("A", 16, 16, 16, 16));
    TileOp f;
    f.kind = TileOpKind::Feed;
    f.port_kind = PortKind::Input;
    f.port = R"(port "West")";
    f.inputs = {TileCoord{"A", 0, 0}};
    f.label = R"(a label with "quotes" and \)";
    p.push(std::move(f));

    const TileProgram back = from_string(to_string(p));
    CHECK(back.name() == p.name());
    REQUIRE(back.ops().size() == 1);
    CHECK(back.ops()[0].port == p.ops()[0].port);
    CHECK(back.ops()[0].label == p.ops()[0].label);
}

// ----------------------------------------------------------------------------
// Refusals. R4's whole point is that a file this build cannot read fails with a
// diagnostic rather than a crash or a mis-parse: kernels/bin/*.kpubin renumbered
// opcodes with no version bump and now abort with "std::get: wrong index for
// variant", which tells a user nothing.
// ----------------------------------------------------------------------------
TEST_CASE("an unreadable file is refused with a cause, never mis-parsed",
          "[program][serialize]") {
    auto cause_of = [](const std::string& text) {
        try {
            from_string(text);
        } catch (const FormatError& e) {
            return e.cause();
        }
        FAIL("expected a FormatError");
        return FormatError::Cause::MalformedRecord;
    };

    SECTION("not an L0 file at all") {
        CHECK(cause_of("# just a comment\n") == FormatError::Cause::NotAnL0File);
        CHECK(cause_of("DFG 1.0.0\nEND\n") == FormatError::Cause::NotAnL0File);
    }
    SECTION("a version this reader is too old for") {
        // A MAJOR bump means the container changed shape, so trying would mis-parse.
        CHECK(cause_of("KPUL0 2.0.0\nMIN_CONSUMER 2.0.0\nEND\n") ==
              FormatError::Cause::UnsupportedVersion);
        // And MIN_CONSUMER is the gate even when the container version looks fine.
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.5.0\nEND\n") ==
              FormatError::Cause::UnsupportedVersion);
    }
    SECTION("a preamble that cannot be trusted") {
        CHECK(cause_of("KPUL0 1.0.0\nEND\n") == FormatError::Cause::MalformedPreamble);
        CHECK(cause_of("KPUL0 1.x\nEND\n") == FormatError::Cause::MalformedPreamble);
    }
    SECTION("an op this build does not implement") {
        // Refused rather than skipped: executing a program with an op you do not
        // understand computes the wrong answer silently.
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n"
                       "OP kind=CONV2D_DIRECT out=C:0,0\nEND\n") ==
              FormatError::Cause::UnknownOp);
    }
    SECTION("a record missing a required field") {
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n"
                       "OPERAND \"A\" rows=16 cols=16\nEND\n") ==
              FormatError::Cause::MalformedRecord);
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\nOP out=C:0,0\nEND\n") ==
              FormatError::Cause::MalformedRecord);
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n"
                       "OP kind=FEED in=A-0-0\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a truncated file") {
        // A partial program would execute a partial answer, which is the worst outcome.
        CHECK(cause_of("KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\nOPERAND \"A\" rows=16 cols=16 "
                       "tile_rows=16 tile_cols=16\n") == FormatError::Cause::Truncated);
    }
}

TEST_CASE("an unknown optional field is ignored, so a minor bump stays readable",
          "[program][serialize]") {
    // The other half of R8. An added attribute that carries no semantics this build needs
    // must not break the load, or every producer bump becomes a breaking change.
    const TileProgram p = from_string(
        "KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\nOPSET tile 1.0.0\n"
        "FUTURE_RECORD something=1\n"
        "PROGRAM \"forward-compatible\"\n"
        "OPERAND \"A\" rows=16 cols=16 tile_rows=16 tile_cols=16 future_attr=7\n"
        "OP kind=FEED in=A:0,0 port_kind=input port=\"West\" annotation=\"ignored\"\n"
        "END\n");
    CHECK(p.name() == "forward-compatible");
    REQUIRE(p.ops().size() == 1);
    CHECK(p.ops()[0].kind == TileOpKind::Feed);
    CHECK(p.ops()[0].port == "West");
    CHECK(p.operand("A").rows == 16);
}

// ============================================================================
// Adversarial input. The first version of this file tested the happy path and the
// version gates thoroughly, and barely tested hostile content -- its one escaping
// case covered the two characters I happened to think of. Every case below was a
// real defect found in review.
// ============================================================================

TEST_CASE("control characters in a value cannot forge records",
          "[program][serialize][adversarial]") {
    // The reader is getline()-based, so a literal newline inside a value splits one
    // record into two -- and a value containing a line reading END would terminate the
    // read early, silently dropping every op after it. That is a program that loads
    // successfully and computes something else.
    TileProgram p("name\nEND\nOP kind=FEED in=A:0,0");
    p.add_operand(TensorOperand("A", 16, 16, 16, 16));
    TileOp f;
    f.kind = TileOpKind::Feed;
    f.port_kind = PortKind::Input;
    f.port = "West";
    f.inputs = {TileCoord{"A", 0, 0}};
    f.label = "tab\there\r\nand a newline";
    p.push(std::move(f));

    const std::string text = to_string(p);
    // The forged END must not appear as a record of its own.
    CHECK(text.find("\nEND\nOP kind=FEED") == std::string::npos);
    CHECK(text.find("\\n") != std::string::npos);        // encoded, not literal

    const TileProgram back = from_string(text);
    CHECK(back.name() == p.name());                      // and decoded symmetrically
    REQUIRE(back.ops().size() == 1);                     // nothing was dropped or forged
    CHECK(back.ops()[0].label == p.ops()[0].label);
}

TEST_CASE("an operand name containing the coordinate grammar round-trips",
          "[program][serialize][adversarial]") {
    // TensorOperand names are free strings and coord() used to write them raw, so a name
    // with a space, ':' or ';' produced a file write_l0() emitted and read_l0() refused.
    const std::string awkward = "odd name:with;grammar,chars";
    TileProgram p("awkward operands");
    p.add_operand(TensorOperand(awkward, 32, 32, 16, 16));
    TileOp f;
    f.kind = TileOpKind::Feed;
    f.port_kind = PortKind::Input;
    f.port = "West";
    f.inputs = {TileCoord{awkward, 1, 1}};
    p.push(std::move(f));

    const TileProgram back = from_string(to_string(p));
    REQUIRE(back.ops().size() == 1);
    REQUIRE(back.ops()[0].inputs.size() == 1);
    CHECK(back.ops()[0].inputs[0].operand == awkward);
    CHECK(back.ops()[0].inputs[0].ti == 1);
    CHECK(back.ops()[0].inputs[0].tj == 1);
    CHECK(back.operand(awkward).rows == 32);
}

TEST_CASE("alpha survives with enough digits to compute the same answer",
          "[program][serialize][adversarial]") {
    // The default 6 significant digits do not round-trip a float: 1.0000001f writes as
    // "1" and reads back as 1.0f, so the reloaded program computes a DIFFERENT RESULT --
    // breaking the bit-identical claim the format rests on. The original tests used only
    // alpha=-1, which is exact, so they could not see this.
    for (float alpha : {1.0000001f, -0.5f, 3.14159265f, 1e-7f, -1.0f}) {
        TileProgram p("alpha fidelity");
        p.add_operand(TensorOperand("A", 16, 16, 16, 16));
        p.add_operand(TensorOperand("B", 16, 16, 16, 16));
        p.add_operand(TensorOperand("C", 16, 16, 16, 16));
        TileOp m;
        m.kind = TileOpKind::MatMulAccum;
        m.inputs = {TileCoord{"A", 0, 0}, TileCoord{"B", 0, 0}};
        m.outputs = {TileCoord{"C", 0, 0}};
        m.alpha = alpha;
        p.push(std::move(m));

        const TileProgram back = from_string(to_string(p));
        REQUIRE(back.ops().size() == 1);
        // Bit-exact, not approximately equal: a coefficient that differs in the last bit
        // makes the two programs different programs.
        CHECK(std::memcmp(&back.ops()[0].alpha, &alpha, sizeof(float)) == 0);
    }
}

TEST_CASE("a hostile number cannot become an enormous allocation",
          "[program][serialize][adversarial]") {
    auto cause_of = [](const std::string& text) {
        try {
            from_string(text);
        } catch (const FormatError& e) {
            return e.cause();
        }
        FAIL("expected a FormatError");
        return FormatError::Cause::Truncated;
    };
    const std::string pre = "KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n";

    // std::stoul accepts a sign: "-1" would become ULONG_MAX, the cast to Dim would make
    // it UINT32_MAX, and TensorOperand would then try to allocate rows*cols floats -- an
    // oversized allocation or a length_error thrown from OUTSIDE read_l0, which a caller
    // catching FormatError would not catch.
    CHECK(cause_of(pre + "OPERAND \"A\" rows=-1 cols=16 tile_rows=16 tile_cols=16\nEND\n") ==
          FormatError::Cause::MalformedRecord);
    CHECK(cause_of(pre + "OPERAND \"A\" rows= 16 cols=16 tile_rows=16 tile_cols=16\nEND\n") ==
          FormatError::Cause::MalformedRecord);
    // Above Dim's range, which the cast would otherwise truncate silently.
    CHECK(cause_of(pre + "OPERAND \"A\" rows=4294967296 cols=16 tile_rows=16 "
                         "tile_cols=16\nEND\n") == FormatError::Cause::MalformedRecord);
    // pivot is cast to int, so it needs its own bound.
    CHECK(cause_of(pre + "OPERAND \"A\" rows=16 cols=16 tile_rows=16 tile_cols=16\n"
                         "OP kind=LU_DIAG_FACTOR out=A:0,0 pivot=99999999999\nEND\n") ==
          FormatError::Cause::MalformedRecord);
}

TEST_CASE("the reader validates a program against its own registry",
          "[program][serialize][adversarial]") {
    auto cause_of = [](const std::string& text) {
        try {
            from_string(text);
        } catch (const FormatError& e) {
            return e.cause();
        }
        FAIL("expected a FormatError");
        return FormatError::Cause::Truncated;
    };
    const std::string pre = "KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n";
    const std::string a16 = "OPERAND \"A\" rows=16 cols=16 tile_rows=16 tile_cols=16\n";

    SECTION("a duplicate operand is a FormatError, not a foreign exception") {
        // add_operand() throws std::invalid_argument, which a caller catching FormatError
        // would not catch -- so the reader has to check first.
        CHECK(cause_of(pre + a16 + a16 + "END\n") == FormatError::Cause::MalformedRecord);
    }
    SECTION("an op naming an undeclared operand fails at LOAD, not at execution") {
        CHECK(cause_of(pre + a16 + "OP kind=FEED in=Z:0,0\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a tile outside the operand's grid is refused, not left to index past the end") {
        // This is the one that mattered most: unchecked, execution indexes
        // TensorOperand::values beyond its length -- undefined behaviour from a file the
        // loader accepted.
        CHECK(cause_of(pre + a16 + "OP kind=FEED in=A:1,0\nEND\n") ==
              FormatError::Cause::MalformedRecord);
        CHECK(cause_of(pre + a16 + "OP kind=DRAIN out=A:0,7\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a valid in-range coordinate still loads") {
        CHECK_NOTHROW(from_string(pre + "OPERAND \"A\" rows=32 cols=32 tile_rows=16 "
                                        "tile_cols=16\nOP kind=FEED in=A:1,1\nEND\n"));
    }
}

// ============================================================================
// Increment 2 — values, optionally
//
// A program WITH values is a test case; one without is a kernel. The file says
// which, so a reader never infers it from zeros -- an all-zero operand is a
// legitimate kernel input, and guessing would make the two indistinguishable.
// ============================================================================

TEST_CASE("a test case carries its inputs, and executes without being re-filled",
          "[program][serialize][values]") {
    // The whole point of carrying values: the file is self-contained. If the reloaded
    // program needed fill() to compute the right answer, the file would not be a test case
    // at all -- it would be a kernel with a misleading preamble.
    for (const char* algo : {"matmul", "lu"}) {
        ProgramSpec ps;
        ps.algo = algo;
        ps.size = 48;
        ps.tile = 16;

        TileProgram original = derive(ps);
        fill(original, ps);

        LoadInfo info;
        TileProgram reloaded = from_string(to_test_case(original), &info);
        CHECK(info.has_values);

        // Inputs are bit-identical before anything runs.
        for (const std::string& key : original.operand_order())
            CHECK(bits_equal(reloaded.operand(key).values, original.operand(key).values));

        // And the reloaded program computes the same answer with NO fill() call.
        const DeviceSpec ds;
        const auto device = make_device(ds);
        TileProgram ref = derive(ps);
        fill(ref, ps);
        run_at(ExecutionLevel::BlockSequential, ref, device,
               Placement::single(device.compute_tiles));
        run_at(ExecutionLevel::BlockSequential, reloaded, device,
               Placement::single(device.compute_tiles));
        CHECK(bits_equal(ref.operand(result_operand(ps)).values,
                         reloaded.operand(result_operand(ps)).values));
    }
}

TEST_CASE("a kernel says it has no values, and carries none",
          "[program][serialize][values]") {
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    TileProgram filled = derive(ps);
    fill(filled, ps);

    // Written as a kernel, the values are deliberately dropped -- and the file says so,
    // which is what lets a reader tell this from a test case whose inputs are all zero.
    const std::string kernel = to_string(filled);
    CHECK(kernel.find("VALUES none\n") != std::string::npos);
    CHECK(kernel.find("VALUES_ROW") == std::string::npos);

    LoadInfo info;
    const TileProgram back = from_string(kernel, &info);
    CHECK_FALSE(info.has_values);
    for (float v : back.operand("A").values) CHECK(v == 0.0f);
}

TEST_CASE("values are exact, including the ones a stream cannot parse",
          "[program][serialize][values]") {
    // Measured before this was written: hexfloat does not round-trip through iostreams,
    // and inf/-inf/nan parse in NEITHER encoding, so they need explicit tokens. A masked
    // attention value is -inf, so this is a real case rather than a curiosity.
    TileProgram p("awkward values");
    p.add_operand(TensorOperand("A", 2, 6, 2, 6));
    auto& A = p.operand("A");
    A.at(0, 0) = 1.0000001f;                                   // needs 9 digits
    A.at(0, 1) = -0.0f;                                        // sign of zero matters
    A.at(0, 2) = std::numeric_limits<float>::denorm_min();
    A.at(0, 3) = std::numeric_limits<float>::max();
    A.at(0, 4) = -std::numeric_limits<float>::infinity();      // an attention mask
    A.at(0, 5) = std::numeric_limits<float>::infinity();
    A.at(1, 0) = std::numeric_limits<float>::quiet_NaN();
    A.at(1, 1) = 3.14159265f;
    A.at(1, 2) = 1e-7f;
    A.at(1, 3) = std::numeric_limits<float>::lowest();
    A.at(1, 4) = std::numeric_limits<float>::min();
    A.at(1, 5) = -1.0f;

    const std::string text = to_test_case(p);
    CHECK(text.find("-inf") != std::string::npos);
    CHECK(text.find("nan") != std::string::npos);

    const TileProgram back = from_string(text);
    const auto& B = back.operand("A");
    for (Dim r = 0; r < 2; ++r)
        for (Dim c = 0; c < 6; ++c) {
            const float want = A.at(r, c), got = B.at(r, c);
            if (std::isnan(want)) { CHECK(std::isnan(got)); continue; }
            // memcmp, not ==, so -0.0f is distinguished from 0.0f.
            CHECK(std::memcmp(&got, &want, sizeof(float)) == 0);
        }
}

TEST_CASE("values are written one row per record, so a diff is readable",
          "[program][serialize][values]") {
    // §5's justification for text only holds if a change shows up small. One record per
    // row means a changed row is one changed line; a whole operand per line would make
    // every change look like a rewrite.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    TileProgram a = derive(ps);
    fill(a, ps);
    TileProgram b = derive(ps);
    fill(b, ps);
    b.operand("A").at(7, 3) = 42.5f;          // one element differs

    const std::string ta = to_test_case(a), tb = to_test_case(b);
    std::istringstream sa(ta), sb(tb);
    std::string la, lb;
    std::size_t differing = 0, total = 0;
    while (std::getline(sa, la) && std::getline(sb, lb)) {
        ++total;
        if (la != lb) ++differing;
    }
    CHECK(total > 32);                         // there are per-row records
    CHECK(differing == 1);                     // and a one-element change moves one line
}

TEST_CASE("a partially or inconsistently valued file is refused",
          "[program][serialize][values]") {
    auto cause_of = [](const std::string& text) {
        try {
            from_string(text);
        } catch (const FormatError& e) {
            return e.cause();
        }
        FAIL("expected a FormatError");
        return FormatError::Cause::Truncated;
    };
    const std::string pre = "KPUL0 1.0.0\nMIN_CONSUMER 1.0.0\n";
    const std::string a2 = "OPERAND \"A\" rows=2 cols=2 tile_rows=2 tile_cols=2\n";

    SECTION("a missing row: a partial test case would compute a partial answer") {
        CHECK(cause_of(pre + "VALUES inline\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("the wrong number of values in a row") {
        CHECK(cause_of(pre + "VALUES inline\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2 3\nVALUES_ROW \"A\" 1 4 5\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a duplicated row, where the later one would silently win") {
        CHECK(cause_of(pre + "VALUES inline\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2\nVALUES_ROW \"A\" 0 3 4\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a row outside the operand") {
        CHECK(cause_of(pre + "VALUES inline\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2\nVALUES_ROW \"A\" 5 3 4\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a row for an operand that does not exist") {
        CHECK(cause_of(pre + "VALUES inline\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2\nVALUES_ROW \"A\" 1 3 4\n"
                       "VALUES_ROW \"Z\" 0 9 9\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("a file that contradicts itself about carrying inputs") {
        // VALUES none plus VALUES_ROW records: one of the two is a lie, and guessing which
        // would mean either dropping inputs or claiming a kernel is a test case.
        CHECK(cause_of(pre + "VALUES none\n" + a2 +
                       "VALUES_ROW \"A\" 0 1 2\nEND\n") ==
              FormatError::Cause::MalformedRecord);
    }
    SECTION("an unrecognised VALUES mode") {
        CHECK(cause_of(pre + "VALUES sideband\n" + a2 + "END\n") ==
              FormatError::Cause::MalformedPreamble);
    }
    SECTION("a complete, consistent test case loads") {
        LoadInfo info;
        CHECK_NOTHROW(from_string(pre + "VALUES inline\n" + a2 +
                                  "VALUES_ROW \"A\" 0 1 2\nVALUES_ROW \"A\" 1 3 4\nEND\n",
                                  &info));
    }
}

TEST_CASE("a test case demands a reader that understands values, a kernel does not",
          "[program][serialize][values]") {
    // R4's mechanism, and getting it wrong is silent rather than loud: a 1.0.0 reader does
    // not know VALUES_ROW, so it would SKIP every value record, accept the file, and
    // execute a test case with zero-initialised inputs. A confident wrong answer is the
    // worst possible outcome for a format whose purpose is reproducibility.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);

    const std::string kernel = to_string(p);
    const std::string test_case = to_test_case(p);

    // The demand depends on WHAT IS IN THE FILE, not on who wrote it.
    CHECK(kernel.find("MIN_CONSUMER 1.0.0\n") != std::string::npos);
    CHECK(test_case.find("MIN_CONSUMER 1.1.0\n") != std::string::npos);
    // A kernel stays readable by the older reader, because nothing was added to it -- a
    // blanket bump would orphan files that are still perfectly readable.
    CHECK(min_consumer_for(false).str() == "1.0.0");
    CHECK(min_consumer_for(true).str() == "1.1.0");

    // Both still load here, since this build is the newer reader.
    LoadInfo k, t;
    CHECK_NOTHROW(from_string(kernel, &k));
    CHECK_NOTHROW(from_string(test_case, &t));
    CHECK_FALSE(k.has_values);
    CHECK(t.has_values);

    // And the gate bites in the direction that protects this reader: a file needing a newer
    // one is refused rather than partially understood.
    //
    // Expressed RELATIVE to the reader rather than as a literal. A hardcoded "future"
    // version goes stale the moment the format moves -- this case said 1.2.0 and silently
    // stopped testing anything when the reader BECAME 1.2.0, which is the same staleness
    // trap the corpus refusal fixture hit.
    const Version newer{reader_version().major, reader_version().minor + 1, 0};
    try {
        from_string("KPUL0 " + reader_version().str() + "\nMIN_CONSUMER " + newer.str() +
                    "\nEND\n");
        FAIL("expected a refusal for MIN_CONSUMER " + newer.str());
    } catch (const FormatError& e) {
        CHECK(e.cause() == FormatError::Cause::UnsupportedVersion);
    }
}

TEST_CASE("write_l0 keeps its two-argument form", "[program][serialize]") {
    // Removing the overload would break callers compiled against increment 1 for no
    // benefit -- the default is what makes the new option additive.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 16;
    ps.tile = 16;
    const TileProgram p = derive(ps);
    std::ostringstream two_arg;
    write_l0(two_arg, p);                       // must compile and mean "no values"
    CHECK(two_arg.str().find("VALUES none\n") != std::string::npos);
    CHECK(two_arg.str() == to_string(p));
}

// ============================================================================
// Increment 4 — the L1 stream annotation (ADR §7.4, optional at TRANSACTIONAL)
//
// The file records the DATAFLOW CHOICE, not the derived annotations. A
// StreamProgram's signatures, network, wavefront timings and array extents are all
// functions of the space-time map and the program, so storing them would be
// caching a pure function -- and a cache can contradict its input. A file that
// records the choice cannot be internally inconsistent.
// ============================================================================

TEST_CASE("the stream annotation round-trips as a choice, and re-derives",
          "[program][serialize][streams]") {
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 48;
    ps.tile = 16;

    for (const char* alias : {"output-stationary", "weight-stationary", "a-stationary",
                              "fully-streaming"}) {
        TileProgram p = derive(ps);
        fill(p, ps);
        const auto map = map_for(alias);

        LoadInfo info;
        const TileProgram back = from_string(to_test_case(p, map.name), &info);
        CHECK(info.has_streams);
        CHECK(info.dataflow == map.name);

        // The point of storing the choice: the consumer re-derives, and gets the same
        // annotation the writer had. Compared through the disassembly, which is the
        // StreamProgram's own account of itself.
        TileProgram reloaded = back;
        const auto original_sp = stream::derive_matmul_streams(p, map);
        const auto rederived_sp = stream::derive_matmul_streams(reloaded, map_for(alias));
        CHECK(rederived_sp.disassemble() == original_sp.disassemble());
    }
}

TEST_CASE("a program with no annotation says so, and none is invented",
          "[program][serialize][streams]") {
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);

    const std::string text = to_test_case(p);          // no dataflow
    CHECK(text.find("STREAMS") == std::string::npos);

    LoadInfo info;
    from_string(text, &info);
    CHECK_FALSE(info.has_streams);
    CHECK(info.dataflow.empty());
}

TEST_CASE("an annotated file demands a reader that understands annotations",
          "[program][serialize][streams]") {
    // The rule this format already states, applied to the record that just arrived: a
    // STREAMS record carries semantics, so a 1.1.0 reader -- which would SKIP it and execute
    // with no L1 timing while reporting success -- must refuse. Same silent-wrong-answer
    // shape as VALUES_ROW on a 1.0.0 reader, one level down.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);

    CHECK(to_test_case(p).find("MIN_CONSUMER 1.1.0\n") != std::string::npos);
    CHECK(to_test_case(p, "output-stationary").find("MIN_CONSUMER 1.2.0\n") !=
          std::string::npos);
    // And a kernel with an annotation but no values still needs 1.2.0: the demand follows
    // the RECORDS PRESENT, not a single flag.
    WriteOptions kernel_with_streams;
    kernel_with_streams.dataflow = "output-stationary";
    const std::string text = to_string(p, kernel_with_streams);
    CHECK(text.find("VALUES none\n") != std::string::npos);
    CHECK(text.find("MIN_CONSUMER 1.2.0\n") != std::string::npos);

    CHECK(min_consumer_for(false, false).str() == "1.0.0");
    CHECK(min_consumer_for(true, false).str() == "1.1.0");
    CHECK(min_consumer_for(false, true).str() == "1.2.0");
    CHECK(min_consumer_for(true, true).str() == "1.2.0");
}

TEST_CASE("a dataflow this build cannot reconstruct is refused, not ignored",
          "[program][serialize][streams]") {
    // Ignoring it would execute with different L1 timing than the file describes while
    // reporting success -- so it is refused, and the diagnostic lists what IS reconstructible
    // rather than leaving the writer to guess.
    auto cause_of = [](const std::string& text) {
        try {
            from_string(text);
        } catch (const FormatError& e) {
            return e.cause();
        }
        FAIL("expected a FormatError");
        return FormatError::Cause::Truncated;
    };
    const std::string pre = "KPUL0 1.2.0\nMIN_CONSUMER 1.2.0\n";
    CHECK(cause_of(pre + "STREAMS dataflow=\"diagonal-hopping\"\nEND\n") ==
          FormatError::Cause::MalformedRecord);
    CHECK(cause_of(pre + "STREAMS\nEND\n") == FormatError::Cause::MalformedRecord);
    CHECK(cause_of(pre + "STREAMS dataflow=\"\"\nEND\n") ==
          FormatError::Cause::MalformedRecord);

    // Every name the format claims to know must actually be reconstructible, or the list is
    // a lie.
    for (const std::string& name : known_dataflows()) {
        CHECK(is_known_dataflow(name));
        CHECK_NOTHROW(from_string(pre + "STREAMS dataflow=" + "\"" + name + "\"\nEND\n"));
        // and it must be a name map_for() round-trips, not merely a string in a list
        CHECK(map_for(name).name == name);
    }
}

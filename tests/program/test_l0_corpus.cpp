// ============================================================================
// tests/program/test_l0_corpus.cpp
// The golden corpus (#265 increment 3): checked-in .l0 files that CI LOADS AND
// EXECUTES.
//
// This exists because kernels/bin/*.kpubin rotted -- opcodes renumbered with no
// version bump, and those files now abort with "std::get: wrong index for
// variant". A version policy that is not tested is a version policy that will be
// wrong, so the checked-in files are the evidence that old files still work.
//
// Paths are relative to the project root; the test's WORKING_DIRECTORY is set to
// CMAKE_SOURCE_DIR, following tests/compiler's convention.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
using namespace sw::kpu::program::serialize;

namespace {

const char* kCorpus = "tests/program/corpus/";

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());                       // a missing corpus file is a failure, not a skip
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

// The spec lives WITH the case, so a new corpus entry cannot be added to one check and
// forgotten by another -- which is exactly what happened: derivation equivalence covered
// matmul only, so a change to derive_lu_tile_program would have left the LU input stale
// while every other check still passed.
struct Case {
    const char* name;
    const char* input;
    const char* expected;
    const char* algo;
    Dim size;
    Dim tile;
};

const std::vector<Case>& cases() {
    static const std::vector<Case> c = {
        {"matmul 48^3 t16", "matmul_48x48x48_t16.l0", "matmul_48x48x48_t16.result.l0",
         "matmul", 48, 16},
        {"tile LU 64 t16",  "lu_64_t16.l0",           "lu_64_t16.result.l0",
         "lu", 64, 16},
    };
    return c;
}

ProgramSpec spec_of(const Case& c) {
    ProgramSpec ps;
    ps.algo = c.algo;
    ps.size = c.size;
    ps.tile = c.tile;
    return ps;
}

} // namespace

// The recorded answers are compared WITHIN A TOLERANCE, not bit-exactly, and the reason is
// specific rather than defensive: this project builds Release with `-march=native
// -mtune=native`, so instruction selection -- FMA contraction, vectorisation width,
// reduction order -- follows the HOST CPU. Two machines therefore compute different last
// bits for the same program, and no checked-in file can promise otherwise.
//
// The first version of this corpus compared bit-exactly and passed locally, which was luck
// rather than evidence: fill_matmul produces exact quarter-integers, so matmul's arithmetic
// stays exactly representable whatever the compiler emits. Tile LU divides, its
// intermediates are not representable, and CI failed on LU alone -- on both levels
// identically, which is what distinguishes a machine difference from a model disagreement.
//
// So the corpus makes TWO SEPARATE claims, and conflating them is what produced the
// overclaim:
//   - CROSS-MACHINE: the recorded answer is still the answer, within tolerance. This
//     catches an op that starts computing something else.
//   - SAME-MACHINE: executing the corpus file is BIT-IDENTICAL to executing a freshly
//     derived program. This catches executor and derivation drift, where bit-exactness is
//     genuinely promised.
namespace {

// Chosen for cross-machine FP variation in a float accumulation, not borrowed from the
// L-CA timing band: a differing last bit amplified through a 64-wide factorisation is
// still small in relative terms, while a wrong kernel is not.
constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-4;

bool close_enough(const std::vector<float>& got, const std::vector<float>& want,
                  std::string& why) {
    if (got.size() != want.size()) {
        why = "size " + std::to_string(got.size()) + " vs " + std::to_string(want.size());
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = want[i];
        if (std::isnan(a) && std::isnan(b)) continue;
        const double tol = kAtol + kRtol * std::abs(b);
        if (!(std::abs(a - b) <= tol)) {
            std::ostringstream ss;
            ss << "element " << i << ": got " << a << ", expected " << b
               << " (tolerance " << tol << ")";
            why = ss.str();
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("every corpus program loads and computes its recorded answer",
          "[program][serialize][corpus]") {
    // Two files per case, because LU factors A IN PLACE -- a single post-execution snapshot
    // would have overwritten the input it was meant to preserve.
    for (const Case& c : cases()) {
        const std::string input_path = std::string(kCorpus) + c.input;
        const std::string expected_path = std::string(kCorpus) + c.expected;

        LoadInfo info;
        TileProgram expected = from_string(read_file(expected_path));
        REQUIRE(expected.operand_order().size() > 0);

        for (ExecutionLevel level : all_levels()) {
            if (!level_implemented(level)) continue;
            TileProgram prog = from_string(read_file(input_path), &info);
            // A corpus entry must be SELF-CONTAINED: it carries its inputs, so no fill()
            // call appears anywhere in this test.
            CHECK(info.has_values);

            const DeviceSpec ds;
            const auto device = make_device(ds);
            run_at(level, prog, device, Placement::single(device.compute_tiles));

            for (const std::string& key : expected.operand_order()) {
                REQUIRE(prog.has_operand(key));
                std::string why;
                const bool ok = close_enough(prog.operand(key).values,
                                             expected.operand(key).values, why);
                INFO(std::string(c.name) + " at " + short_name(level) + ", operand " + key
                     << ": " << why);
                CHECK(ok);
            }
        }
    }
}

TEST_CASE("executing a corpus file is bit-identical to executing a fresh derivation",
          "[program][serialize][corpus]") {
    // Where bit-exactness IS promised: one machine, one build, two routes to the same
    // program. This is the check that catches executor or derivation drift, and it is kept
    // separate from the recorded-answer comparison because the two fail for different
    // reasons and conflating them hides which one moved.
    for (const Case& sp : cases()) {
        const ProgramSpec ps = spec_of(sp);
        const DeviceSpec ds;
        const auto device = make_device(ds);

        for (ExecutionLevel level : all_levels()) {
            if (!level_implemented(level)) continue;
            TileProgram from_file = from_string(read_file(std::string(kCorpus) + sp.input));
            TileProgram fresh = derive(ps);
            fill(fresh, ps);
            run_at(level, from_file, device, Placement::single(device.compute_tiles));
            run_at(level, fresh, device, Placement::single(device.compute_tiles));
            for (const std::string& key : fresh.operand_order()) {
                INFO(std::string(sp.input) + " at " + short_name(level) + ", operand " + key);
                CHECK(bits_equal(from_file.operand(key).values, fresh.operand(key).values));
            }
        }
    }
}

TEST_CASE("re-serializing a corpus file reproduces it byte for byte",
          "[program][serialize][corpus]") {
    // Deliberately strict, and it WILL fail on any format change -- that is the point.
    // The checked-in bytes are the evidence; regenerating them is a decision that has to
    // answer "does this need a version bump?", not a chore to route around.
    for (const Case& c : cases()) {
        for (const char* file : {c.input, c.expected}) {
            const std::string path = std::string(kCorpus) + file;
            const std::string on_disk = read_file(path);
            const TileProgram loaded = from_string(on_disk);
            const std::string rewritten = to_test_case(loaded);
            INFO("corpus file " << path);
            CHECK(rewritten == on_disk);
        }
    }
}

TEST_CASE("the corpus files declare the versions they actually need",
          "[program][serialize][corpus]") {
    // A test case carries VALUES_ROW records, so it must demand a reader that understands
    // them. If this ever reads 1.0.0 again, an older reader would skip every value record
    // and execute with zero inputs.
    for (const Case& c : cases()) {
        for (const char* file : {c.input, c.expected}) {
            const std::string text = read_file(std::string(kCorpus) + file);
            INFO("corpus file " << file);
            CHECK(text.find("VALUES inline\n") != std::string::npos);
            CHECK(text.find("MIN_CONSUMER 1.1.0\n") != std::string::npos);
        }
    }
}

TEST_CASE("both version gates refuse, and each is exercised on its own",
          "[program][serialize][corpus]") {
    // Two fixtures, because one cannot test both gates. An earlier single fixture declared
    // KPUL0 9.0.0 AND MIN_CONSUMER 9.0.0, and the reader rejects the container major BEFORE
    // reading MIN_CONSUMER -- so it passed with UnsupportedVersion even if the min_consumer
    // gate were broken. A test that passes for the wrong reason guards nothing.
    //
    // Both are hand-written and never regenerated, so no tool can quietly bring them in
    // line with the current version.
    auto refuses = [](const char* file) {
        const std::string text = read_file(std::string(kCorpus) + file);
        try {
            from_string(text);
            FAIL(std::string(file) + " must not load");
        } catch (const FormatError& e) {
            CHECK(e.cause() == FormatError::Cause::UnsupportedVersion);
        }
    };
    // MIN_CONSUMER 9.0.0 with a container version this reader DOES support, so the only
    // thing that can refuse it is the min_consumer gate itself.
    refuses("needs_a_newer_reader.l0");
    // And the container-major check, on its own.
    refuses("needs_a_newer_container.l0");

    // The first fixture must really be past the container gate, or it is the old test again.
    const std::string text = read_file(std::string(kCorpus) + "needs_a_newer_reader.l0");
    CHECK(text.rfind("KPUL0 1.1.0", 0) == 0);
    CHECK(text.find("MIN_CONSUMER 9.0.0\n") != std::string::npos);
}

TEST_CASE("every corpus program still matches what the derivation produces today",
          "[program][serialize][corpus]") {
    // Separate from the execution check, and it answers a different question: that one asks
    // "does this file still compute its answer", this asks "is the file still what our
    // derivation would emit". They diverge -- a derivation change keeps the corpus
    // executing correctly while silently making it stale -- so conflating them would hide
    // which of the two moved.
    //
    // EVERY case, driven by the case list. This covered matmul only, which meant a change
    // to the LU derivation would have gone unnoticed while every check still passed.
    for (const Case& c : cases()) {
        const ProgramSpec ps = spec_of(c);
        TileProgram fresh = derive(ps);
        fill(fresh, ps);
        INFO("corpus input " << c.input);
        CHECK(to_test_case(fresh) == read_file(std::string(kCorpus) + c.input));
    }
}

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

struct Case {
    const char* name;
    const char* input;
    const char* expected;
};

const std::vector<Case>& cases() {
    static const std::vector<Case> c = {
        {"matmul 48^3 t16", "matmul_48x48x48_t16.l0", "matmul_48x48x48_t16.result.l0"},
        {"tile LU 64 t16",  "lu_64_t16.l0",           "lu_64_t16.result.l0"},
    };
    return c;
}

} // namespace

TEST_CASE("every corpus program loads and computes its recorded answer",
          "[program][serialize][corpus]") {
    // The claim the corpus exists to defend: a file written earlier still executes to the
    // same values today. Two files per case, because LU factors A IN PLACE -- a single
    // post-execution snapshot would have overwritten the input it was meant to preserve.
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
                INFO(std::string(c.name) + " at " + short_name(level) + ", operand " + key);
                REQUIRE(prog.has_operand(key));
                CHECK(bits_equal(prog.operand(key).values, expected.operand(key).values));
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

TEST_CASE("the hand-written future file still refuses to load",
          "[program][serialize][corpus]") {
    // This fixture is never regenerated -- it is hand-written precisely so no tool can
    // quietly bring it in line with the current version. If it starts loading, the
    // min_consumer gate has stopped working, which is exactly how the .kpubin corpus
    // rotted.
    const std::string text = read_file(std::string(kCorpus) + "needs_a_newer_reader.l0");
    try {
        from_string(text);
        FAIL("a file declaring MIN_CONSUMER 9.0.0 must not load");
    } catch (const FormatError& e) {
        CHECK(e.cause() == FormatError::Cause::UnsupportedVersion);
    }
}

TEST_CASE("a corpus program still matches what the derivation produces today",
          "[program][serialize][corpus]") {
    // Separate from the execution check, and it answers a different question: the one
    // above asks "does this file still compute its answer", this asks "is the file still
    // what our derivation would emit". Those can diverge -- a derivation change keeps the
    // corpus executing correctly while silently making it stale -- and conflating them
    // would hide which of the two moved.
    ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = 48;
    ps.tile = 16;
    TileProgram fresh = derive(ps);
    fill(fresh, ps);
    const std::string corpus = read_file(std::string(kCorpus) + cases()[0].input);
    CHECK(to_test_case(fresh) == corpus);
}

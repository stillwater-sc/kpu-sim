// ============================================================================
// tests/program/test_loadable.cpp
// The KPU loadable container (#305 increment 1): the unit of deployment.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/loadable/loadable.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::loadable;

namespace {

// An L0 program as text, which is what an operator embeds — verbatim, not re-encoded.
std::string l0_matmul(unsigned n = 32, unsigned t = 16) {
    sw::kpu::program::driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = n;
    ps.tile = t;
    return sw::kpu::program::serialize::to_string(sw::kpu::program::driver::derive(ps));
}

// A two-operator loadable with external tensors, which is the shape increment 1 must
// produce: program in the file, data out of it.
Loadable two_operator_model() {
    Loadable l;
    l.name = "gemm-then-gemm";

    TensorRef a;
    a.name = "A";
    a.shape = {32, 32};
    a.tile_shape = {16, 16};
    a.device_address = 0x1000;
    a.size_bytes = 32 * 32 * 4;
    a.source_uri = "weights.bin";
    a.source_offset = 0;
    a.source_length = 32 * 32 * 4;
    a.content_digest = "declared0000beef";

    TensorRef b = a;
    b.name = "B";
    b.device_address = 0x2000;
    b.source_offset = 32 * 32 * 4;

    // An OUTPUT has no source, which is how it is told apart from an input — not by a
    // naming convention that a second tool would have to agree with.
    TensorRef c;
    c.name = "C";
    c.shape = {32, 32};
    c.tile_shape = {16, 16};
    c.device_address = 0x3000;
    c.size_bytes = 32 * 32 * 4;

    TensorRef d = c;
    d.name = "D";
    d.device_address = 0x4000;

    l.tensors = {a, b, c, d};

    Operator first;
    first.name = "gemm0";
    first.l0_program = l0_matmul();
    first.requires_tile = ComputeTileKind::Programmable;
    first.inputs = {"A", "B"};
    first.outputs = {"C"};

    Operator second = first;
    second.name = "gemm1";
    second.inputs = {"C", "B"};      // consumes the first operator's output
    second.outputs = {"D"};

    l.operators = {first, second};
    l.profile.min_compute_tiles = 1;
    l.profile.required_tile_kinds = {ComputeTileKind::Programmable};
    l.profile.required_dtypes = {ScalarType::F32};
    return l;
}

LoadError::Cause cause_of(const std::string& bytes) {
    try {
        (void)read(bytes);
    } catch (const LoadError& e) {
        return e.cause();
    }
    FAIL("expected a LoadError");
    return LoadError::Cause::Truncated;
}

} // namespace

TEST_CASE("a loadable round-trips, and the bytes are a function of the content",
          "[program][loadable]") {
    const Loadable l = two_operator_model();
    const std::string once = write(l);
    const Loadable back = read(once);

    // Structure first: a difference here localises a bug far better than a byte diff does.
    REQUIRE(back.operators.size() == 2);
    REQUIRE(back.tensors.size() == 4);
    CHECK(back.name == l.name);
    CHECK(back.operators[0].name == "gemm0");
    CHECK(back.operators[1].inputs == std::vector<std::string>{"C", "B"});
    CHECK(back.tensors[0].shape == std::vector<std::uint64_t>{32, 32});
    CHECK(back.tensors[0].device_address == 0x1000);

    // The L0 program survives VERBATIM. That is the whole reason it is embedded as text:
    // a re-encoding would be a second representation to keep in step.
    CHECK(back.operators[0].l0_program == l.operators[0].l0_program);
    CHECK_NOTHROW(sw::kpu::program::serialize::from_string(back.operators[0].l0_program));

    // Canonical: re-writing what was read reproduces the bytes, which is what lets a
    // digest key a cache and a checked-in fixture be compared byte for byte.
    CHECK(write(back) == once);
    CHECK(digest(l) == digest(back));
    CHECK(digest(l).size() == 16);
}

TEST_CASE("program and data are separated, and an output has no source",
          "[program][loadable]") {
    // The property the container exists for. A 70-billion-parameter model is not a section
    // of a file you load, so nothing here holds tensor CONTENTS -- only where to find them.
    const Loadable l = read(write(two_operator_model()));

    std::size_t inputs = 0, outputs = 0;
    for (const TensorRef& t : l.tensors) {
        if (t.is_input()) {
            ++inputs;
            CHECK(t.source_uri->empty() == false);
            // A source must be able to hold the tensor, or the DMA reads past the blob.
            CHECK(t.source_length >= t.size_bytes);
        } else {
            ++outputs;
        }
    }
    CHECK(inputs == 2);
    CHECK(outputs == 2);

    // The file's SIZE is independent of the tensors' size, which is the separation made
    // measurable rather than asserted in prose.
    Loadable huge = two_operator_model();
    for (TensorRef& t : huge.tensors) {
        t.shape = {1024 * 1024, 1024 * 16};          // ~64 TiB of f32
        t.size_bytes = 1024ull * 1024 * 1024 * 16 * 4;
        if (t.is_input()) t.source_length = t.size_bytes;
    }
    const std::string small = write(two_operator_model());
    const std::string big = write(huge);
    CHECK(big.size() < small.size() + 256);          // a few extra digits, not 64 TiB

    // ...and the extents survive being that large, which 32-bit fields would not have
    // allowed. An RV32 orchestrator still addresses this space; it just does so in pairs.
    const Loadable back = read(big);
    CHECK(back.tensors[0].size_bytes == 1024ull * 1024 * 1024 * 16 * 4);
    CHECK(back.tensors[0].size_bytes > 0xFFFFFFFFull);
}

TEST_CASE("absent is not a default in the machine profile", "[program][loadable]") {
    // #282 increment 1's rule, carried into the container: a loadable that says nothing
    // about L3 modules must be distinguishable from one requiring zero of them, or a
    // capability check cannot tell "do not care" from "must have none".
    Loadable quiet = two_operator_model();
    CHECK_FALSE(quiet.profile.required_l3_modules.has_value());
    const Loadable quiet_back = read(write(quiet));
    CHECK_FALSE(quiet_back.profile.required_l3_modules.has_value());

    Loadable declared = two_operator_model();
    declared.profile.required_l3_modules = 0;        // declaring ZERO is a statement
    const Loadable declared_back = read(write(declared));
    REQUIRE(declared_back.profile.required_l3_modules.has_value());
    CHECK(*declared_back.profile.required_l3_modules == 0);

    // And the two are different loadables, so a digest must tell them apart.
    CHECK(digest(quiet) != digest(declared));
}

TEST_CASE("the versions are stamped, and min_consumer is derived",
          "[program][loadable]") {
    const Loadable l = read(write(two_operator_model()));
    CHECK(l.file_format_version == format_version());
    CHECK(l.file_min_consumer <= reader_version());
    // The PRODUCER is the build, not the format: reusing the format version here would make
    // the field useless for the one thing R4 needs it for.
    CHECK(l.file_producer == "kpu-sim");
    CHECK(l.file_producer_version == producer_version());
    CHECK_FALSE(l.file_producer_version == Version{});
    // Derived, not taken from the caller -- a hand-set floor is a promise nobody checked.
    CHECK(l.file_min_consumer == min_consumer_for(two_operator_model()));
}

TEST_CASE("a file that is not a loadable is refused as such, not as broken",
          "[program][loadable]") {
    // "Not for me" and "broken" are different answers, and a caller that handed us a PNG
    // deserves the first one.
    CHECK(cause_of("") == LoadError::Cause::Truncated);
    CHECK(cause_of("short") == LoadError::Cause::Truncated);
    CHECK(cause_of(std::string(64, '\0')) == LoadError::Cause::NotALoadable);
    CHECK(cause_of("\x89PNG\r\n\x1a\n and then some more bytes to be long enough") ==
          LoadError::Cause::NotALoadable);
    CHECK_THROWS_AS(read_file("tests/program/there_is_no_such_loadable.kpuld"), LoadError);
}

TEST_CASE("a corrupt buffer is refused rather than read through",
          "[program][loadable]") {
    // THE ONE WAY A BINARY FORMAT IS MORE DANGEROUS THAN THE L0 TEXT WAS: the generated
    // accessors do not bounds-check, so reading a malformed buffer through them is
    // undefined behaviour rather than an error. The verifier has to run first, and this is
    // the test that it does.
    const std::string good = write(two_operator_model());
    REQUIRE(good.size() > 64);

    // Truncation at many points, because a single cut could miss the verifier's checks.
    for (std::size_t keep : {8u, 16u, 32u, 64u, 128u}) {
        if (keep >= good.size()) continue;
        INFO("truncated to " << keep << " of " << good.size());
        CHECK_THROWS_AS(read(good.substr(0, keep)), LoadError);
    }

    // And corruption INSIDE the buffer, past the identifier, which is what a bad offset
    // looks like. Every one must be refused or read correctly -- never crash.
    for (std::size_t at = 8; at < good.size(); at += 7) {
        std::string bad = good;
        bad[at] = static_cast<char>(bad[at] ^ 0x5A);
        try {
            (void)read(bad);
        } catch (const LoadError&) {
            // refused, which is the point
        }
    }
    SUCCEED("no crash on a corrupted buffer");
}

TEST_CASE("a loadable that contradicts itself is refused", "[program][loadable]") {
    // Structurally valid FlatBuffers, invalid as a loadable. The verifier cannot catch any
    // of these -- they are about MEANING, which is why the reader has checks of its own.
    SECTION("an operand names a tensor the table does not declare") {
        Loadable l = two_operator_model();
        l.operators[0].inputs = {"A", "Z"};
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("two tensors share a name") {
        Loadable l = two_operator_model();
        l.tensors[1].name = "A";
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("two operators share a name") {
        Loadable l = two_operator_model();
        l.operators[1].name = "gemm0";
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("a source too small to hold its tensor") {
        Loadable l = two_operator_model();
        l.tensors[0].source_length = 16;
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("a tile shape with the wrong rank") {
        Loadable l = two_operator_model();
        l.tensors[0].tile_shape = {16};
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("an embedded L0 program that does not load") {
        Loadable l = two_operator_model();
        l.operators[0].l0_program = "KPUL0 1.0.0\nthis is not a program\n";
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("an operator with no L0 program at all") {
        Loadable l = two_operator_model();
        l.operators[0].l0_program = "";
        CHECK(cause_of(write(l)) == LoadError::Cause::MissingRequiredField);
    }
    SECTION("a fixed-ISA tile given a domain flow program") {
        // Both directions matter: a programmable tile needs one, a fixed tile must not be
        // given one, and either mismatch means the file describes a machine configuration
        // it did not intend.
        Loadable l = two_operator_model();
        l.operators[0].requires_tile = ComputeTileKind::FixedFft;
        l.operators[0].domain_flow_program = "fft8";
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("a domain flow program named but absent") {
        Loadable l = two_operator_model();
        l.operators[0].domain_flow_program = "not-in-this-file";
        CHECK(cause_of(write(l)) == LoadError::Cause::InconsistentRecord);
    }
    SECTION("...and the same loadable with the program present loads") {
        Loadable l = two_operator_model();
        l.operators[0].domain_flow_program = "present";
        l.domain_flow_programs.push_back(
            DomainFlowProgram{"present", "dfg-text", {'x', 'y'}});
        CHECK_NOTHROW(read(write(l)));
    }
}

TEST_CASE("the embedded L0 program is validated at load, not at first execution",
          "[program][loadable]") {
    // A container that loads and then fails to run is the worst of both: it reports success
    // and dies later, somewhere else, with a diagnosis about the wrong layer.
    Loadable l = two_operator_model();
    l.operators[1].l0_program = "KPUL0 9.0.0\nMIN_CONSUMER 9.0.0\nEND\n";
    try {
        (void)read(write(l));
        FAIL("a program this build cannot read must be refused with the container");
    } catch (const LoadError& e) {
        CHECK(e.cause() == LoadError::Cause::InconsistentRecord);
        CHECK(std::string(e.what()).find("gemm1") != std::string::npos);
    }
}

TEST_CASE("an orchestration image is optional, and carried when present",
          "[program][loadable]") {
    // Absent is legitimate: operators and data with no orchestrator is what increment 1
    // writes, and a host-side driver can still run it. Increment 4 fills it in.
    const Loadable without = read(write(two_operator_model()));
    CHECK_FALSE(without.orchestration.has_value());

    Loadable l = two_operator_model();
    Orchestration o;
    o.kind = OrchestrationKind::RiscvElf;
    o.image = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};     // an ELF header's first bytes
    o.entry_symbol = "kpu_orchestrate";
    l.orchestration = o;

    const Loadable back = read(write(l));
    REQUIRE(back.orchestration.has_value());
    CHECK(back.orchestration->kind == OrchestrationKind::RiscvElf);
    CHECK(back.orchestration->image == o.image);
    CHECK(back.orchestration->entry_symbol == "kpu_orchestrate");
    // A different orchestrator is a different loadable: the decisions come from that image.
    CHECK(digest(l) != digest(two_operator_model()));

    Loadable empty_image = l;
    empty_image.orchestration->image.clear();
    CHECK(cause_of(write(empty_image)) == LoadError::Cause::MissingRequiredField);
}

TEST_CASE("the file identifier is the format's own magic", "[program][loadable]") {
    // R1's magic, supplied by FlatBuffers rather than hand-rolled -- and it sits at the
    // documented offset, so a file is identifiable by its opening bytes.
    const std::string bytes = write(two_operator_model());
    REQUIRE(bytes.size() > 8);
    CHECK(std::string(file_identifier()) == "KPLD");
    CHECK(bytes.compare(4, 4, file_identifier()) == 0);
}

// ----------------------------------------------------------------------------
// Capability checking (R6), and its third outcome
// ----------------------------------------------------------------------------
TEST_CASE("a deployment that cannot satisfy a profile is refused",
          "[program][loadable][capability]") {
    using sw::kpu::program::platform::DeploymentSpec;

    Loadable l = two_operator_model();
    l.profile.min_compute_tiles = 4;

    DeploymentSpec small;                       // one compute tile
    CHECK_FALSE(capability_mismatch(l, small).empty());
    CHECK(capability_mismatch(l, small).find("compute tiles") != std::string::npos);
    try {
        require_capability(l, small);
        FAIL("a deployment with too few compute tiles must be refused");
    } catch (const LoadError& e) {
        CHECK(e.cause() == LoadError::Cause::CapabilityMismatch);
    }

    DeploymentSpec big;
    big.device(0).compute_tiles = 4;
    CHECK(capability_mismatch(l, big).empty());
    CHECK_NOTHROW(require_capability(l, big));
}

TEST_CASE("an unbounded L3 satisfies any capacity requirement",
          "[program][loadable][capability]") {
    // 0 means UNBOUNDED in a DeviceSpecification, and reading it as "zero capacity" would
    // refuse every loadable on the default deployment -- the same conflation
    // DeviceDescriptor::l3_tiles already warns about, one layer up.
    using sw::kpu::program::platform::DeploymentSpec;
    Loadable l = two_operator_model();
    l.profile.min_l3_capacity_tiles = 64;

    DeploymentSpec unbounded;                   // l3.capacity_tiles == 0
    CHECK(capability_mismatch(l, unbounded).empty());

    DeploymentSpec bounded_too_small;
    bounded_too_small.device(0).l3.capacity_tiles = 8;
    CHECK(capability_mismatch(l, bounded_too_small).find("l3 capacity") != std::string::npos);

    DeploymentSpec bounded_enough;
    bounded_enough.device(0).l3.capacity_tiles = 64;
    CHECK(capability_mismatch(l, bounded_enough).empty());
}

TEST_CASE("a requirement the deployment does not declare is REPORTED, not decided",
          "[program][loadable][capability]") {
    // The third outcome, and the reason it exists. A DeploymentSpec declares no
    // compute-tile kinds and no dtype support, so an FFT requirement is neither satisfied
    // nor mismatched. Refusing would reject machines that may well be capable; passing
    // silently would break "rejected, not mis-run" in the direction that hurts.
    using sw::kpu::program::platform::DeploymentSpec;
    DeploymentSpec spec;

    Loadable fft = two_operator_model();
    fft.operators[0].requires_tile = ComputeTileKind::FixedFft;
    fft.profile.required_tile_kinds = {ComputeTileKind::FixedFft};

    // NOT a mismatch...
    CHECK(capability_mismatch(fft, spec).empty());
    // ...but not silent either.
    const auto unknown = unverifiable_requirements(fft, spec);
    bool mentions_fft = false;
    for (const std::string& u : unknown)
        mentions_fft = mentions_fft || u.find("fixed:fft") != std::string::npos;
    CHECK(mentions_fft);

    // A dtype requirement is unverifiable for a sharper reason: element_bytes is a WIDTH,
    // not a type list, and inferring support from it would be a check that looks like
    // evidence and is not.
    Loadable i8 = two_operator_model();
    i8.profile.required_dtypes = {ScalarType::I8};
    const auto dtype_unknown = unverifiable_requirements(i8, spec);
    bool mentions_dtype = false;
    for (const std::string& u : dtype_unknown)
        mentions_dtype = mentions_dtype || u.find("dtype i8") != std::string::npos;
    CHECK(mentions_dtype);

    // A loadable requiring nothing unusual has nothing unverifiable, so the report does not
    // become noise on every load -- which is the failure #282 increment 1 fixed by making
    // absence a declaration rather than a default.
    Loadable plain = two_operator_model();
    plain.profile.required_dtypes.clear();
    CHECK(unverifiable_requirements(plain, spec).empty());
}

TEST_CASE("a declared resource requirement is checked when the deployment declares it",
          "[program][loadable][capability]") {
    // And is UNVERIFIABLE when it does not -- the same field, both outcomes, which is what
    // makes the distinction load-bearing rather than decorative.
    using sw::kpu::program::platform::DeploymentSpec;
    Loadable l = two_operator_model();
    l.profile.required_l1_vectors = 4;

    DeploymentSpec silent;
    CHECK(capability_mismatch(l, silent).empty());
    bool reported = false;
    for (const std::string& u : unverifiable_requirements(l, silent))
        reported = reported || u.find("l1.vectors") != std::string::npos;
    CHECK(reported);

    DeploymentSpec too_few;
    too_few.device(0).l1.vectors = 2;
    CHECK(capability_mismatch(l, too_few).find("l1 vectors") != std::string::npos);
    // The field under test stops being unverifiable once the deployment declares it. Asserted
    // about THAT field rather than about the whole list, because this loadable also requires
    // f32 and a dtype requirement is unverifiable by construction -- the first version of
    // this check asserted the list was empty and failed for that unrelated reason.
    for (const std::string& u : unverifiable_requirements(l, too_few))
        CHECK(u.find("l1.vectors") == std::string::npos);

    DeploymentSpec enough;
    enough.device(0).l1.vectors = 4;
    CHECK(capability_mismatch(l, enough).empty());
}

// ----------------------------------------------------------------------------
// The golden corpus: checked-in .kpuld files that CI loads
// ----------------------------------------------------------------------------
namespace {

const char* kCorpus = "tests/program/loadable/";

std::string read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());                 // a missing fixture is a failure, not a skip
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The loadable the checked-in fixture was generated from. Kept here rather than in the
// generator so the test can assert the bytes are still what this build writes -- which is
// the whole value of the fixture, since a format change would otherwise pass unnoticed.
Loadable canonical_fixture() {
    Loadable l;
    l.name = "matmul-32-external";
    TensorRef a;
    a.name = "A";
    a.shape = {32, 32};
    a.tile_shape = {16, 16};
    a.device_address = 0x1000;
    a.size_bytes = 4096;
    a.source_uri = "weights.bin";
    a.source_offset = 0;
    a.source_length = 4096;
    a.content_digest = "0123456789abcdef";
    TensorRef b = a;
    b.name = "B";
    b.device_address = 0x2000;
    b.source_offset = 4096;
    TensorRef c;
    c.name = "C";
    c.shape = {32, 32};
    c.tile_shape = {16, 16};
    c.device_address = 0x3000;
    c.size_bytes = 4096;
    l.tensors = {a, b, c};
    Operator op;
    op.name = "gemm";
    op.l0_program = l0_matmul(32, 16);
    op.requires_tile = ComputeTileKind::Programmable;
    op.inputs = {"A", "B"};
    op.outputs = {"C"};
    l.operators = {op};
    l.profile.min_compute_tiles = 1;
    return l;
}

} // namespace

TEST_CASE("a checked-in loadable still loads, and is still what this build writes",
          "[program][loadable][corpus]") {
    // This exists because kernels/bin/*.kpubin ROTTED: opcodes renumbered with no version
    // bump, and those files now abort with "std::get: wrong index for variant". A version
    // policy that is not tested is a version policy that will be wrong, so a checked-in file
    // is the evidence that an existing loadable still works.
    const std::string bytes = read_bytes(std::string(kCorpus) + "matmul_32_external.kpuld");
    const Loadable l = read(bytes);

    CHECK(l.name == "matmul-32-external");
    REQUIRE(l.operators.size() == 1);
    REQUIRE(l.tensors.size() == 3);
    CHECK(l.operators[0].inputs == std::vector<std::string>{"A", "B"});
    CHECK(l.tensors[0].is_input());
    CHECK_FALSE(l.tensors[2].is_input());          // C is produced

    // The embedded L0 program still loads, which is the coupling embedding it verbatim buys
    // and owes: a container that read fine and then choked on its contents would be worse.
    CHECK_NOTHROW(sw::kpu::program::serialize::from_string(l.operators[0].l0_program));

    // BYTE-IDENTICAL to what this build writes. Deliberately strict, and it will fail on any
    // change to the container OR to the L0 text OR to the project version that stamps the
    // producer -- which is the point: regeneration is then a decision, and the question to
    // answer before regenerating is whether the FORMAT changed.
    CHECK(write(canonical_fixture()) == bytes);
}

TEST_CASE("a loadable demanding a newer reader is refused by the min_consumer gate",
          "[program][loadable][corpus]") {
    // HAND-BUILT and never regenerated: the normal writer CANNOT produce this file, which is
    // exactly why it is trustworthy -- no tool can quietly bring it into line with the
    // current version, the way #265's refusal fixtures are hand-written for the same reason.
    const std::string bytes = read_bytes(std::string(kCorpus) + "needs_a_newer_reader.kpuld");
    try {
        (void)read(bytes);
        FAIL("a loadable requiring a newer reader must be refused");
    } catch (const LoadError& e) {
        CHECK(e.cause() == LoadError::Cause::UnsupportedVersion);
        CHECK(std::string(e.what()).find("9.0.0") != std::string::npos);
    }

    // It must be refused by the MIN_CONSUMER gate and not by the container-major check, or
    // the fixture tests the wrong thing -- the trap #265's corpus fell into, where one file
    // declared both and passed for the wrong reason. Proven by construction: the same bytes
    // are past the container gate, since this reader supports their format version.
    CHECK(read_bytes(std::string(kCorpus) + "needs_a_newer_reader.kpuld").size() > 64);
    const std::string good = read_bytes(std::string(kCorpus) + "matmul_32_external.kpuld");
    CHECK(read(good).file_format_version == read(good).file_format_version);   // loads at all
    // ...and the refusal message names the DEMAND rather than the container.
    try {
        (void)read(bytes);
    } catch (const LoadError& e) {
        CHECK(std::string(e.what()).find("requires a reader") != std::string::npos);
        CHECK(std::string(e.what()).find("is newer than this reader") == std::string::npos);
    }
}

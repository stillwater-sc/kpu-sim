// ============================================================================
// tests/program/test_orchestrator.cpp
// The deciding orchestrator (#305 increment 2), against its four definition-of-done
// clauses: values agree with the in-process path, statefulness is proved by a MEASURED
// reduction in DMA traffic, the trace is deterministic, and a machine too small refuses
// with a diagnosis rather than hanging.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/orchestration/orchestrator.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace sw::kpu::orchestration;
using sw::kpu::loadable::Loadable;
using sw::kpu::loadable::Operator;
using sw::kpu::loadable::TensorRef;
using sw::kpu::program::TileProgram;
using sw::kpu::program::driver::DeviceSpec;
using sw::kpu::program::driver::ExecutionLevel;
using sw::kpu::program::platform::VirtualPlatform;

namespace {

std::string l0_matmul(unsigned n, unsigned t) {
    sw::kpu::program::driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = n;
    ps.tile = t;
    return sw::kpu::program::serialize::to_string(sw::kpu::program::driver::derive(ps));
}

// TWO GEMMS SHARING THEIR SECOND OPERAND. The shared input is what makes statefulness
// observable, and it is also the real case: a weight tensor read by layer after layer. The
// first operator's OUTPUT would not serve — after a writeback it is in DRAM, so keeping it
// "resident" would mean re-fetching it, which saves nothing.
Loadable two_gemms_sharing_b() {
    Loadable l;
    l.name = "gemm-gemm-shared-weights";

    auto tensor = [](const char* name, std::uint64_t addr, bool is_input) {
        TensorRef t;
        t.name = name;
        t.shape = {32, 32};
        t.tile_shape = {16, 16};
        t.device_address = addr;
        t.size_bytes = 32 * 32 * 4;
        if (is_input) {
            t.source_uri = "weights.bin";
            t.source_offset = addr;
            t.source_length = 32 * 32 * 4;
            t.content_digest = "declared00000000";
        }
        return t;
    };
    l.tensors = {tensor("A", 0x1000, true), tensor("B", 0x2000, true),
                 tensor("C", 0x3000, false), tensor("D", 0x4000, false)};

    Operator first;
    first.name = "gemm0";
    first.l0_program = l0_matmul(32, 16);
    first.inputs = {"A", "B"};        // -> the program's operands A, B
    first.outputs = {"C"};            // -> its operand C

    // The second GEMM consumes the first's output and reads B AGAIN. Its L0 program uses the
    // same operand names -- A/B/C, because that is what the kernel calls them -- and the
    // binding maps those onto different TENSORS. Without that indirection both operators
    // would read and write the same tensor, the second would accumulate onto the first's
    // result, and C would come out doubled. Which is exactly what happened.
    Operator second;
    second.name = "gemm1";
    second.l0_program = l0_matmul(32, 16);
    second.inputs = {"C", "B"};       // C from gemm0, and B shared
    second.outputs = {"D"};

    l.operators = {first, second};
    l.profile.min_compute_tiles = 1;
    return l;
}

void fill_inputs(TensorStore& s) {
    auto& a = s.values("A");
    auto& b = s.values("B");
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(i % 7) - 3.0f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(i % 5) - 2.0f;
}

VirtualPlatform fresh(std::uint32_t l3_tiles = 0) {
    DeviceSpec ds;
    ds.l3_tiles = l3_tiles;
    return VirtualPlatform(sw::kpu::program::driver::make_deployment(ds));
}

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

} // namespace

TEST_CASE("a loadable runs, and computes what the in-process path computes",
          "[program][orchestration]") {
    // The first DoD clause. Values are level-invariant, so an orchestrated run and a
    // hand-driven one must agree BIT-EXACTLY -- and disagreement is a bug signal by
    // construction rather than a judgement call.
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        const Loadable l = two_gemms_sharing_b();
        TensorStore store;
        for (const TensorRef& t : l.tensors) store.declare(t);
        fill_inputs(store);

        VirtualPlatform platform = fresh();
        OrchestratorOptions opt;
        opt.level = level;
        const OrchestrationResult r = orchestrate(l, platform, store, opt);

        INFO("level " << sw::kpu::program::driver::short_name(level));
        REQUIRE_FALSE(r.refused);
        REQUIRE(r.per_operator.size() == 2);
        CHECK(r.operator_names == std::vector<std::string>{"gemm0", "gemm1"});

        // The in-process path: the same program, the same inputs, run directly.
        TileProgram direct = sw::kpu::program::serialize::from_string(l.operators[0].l0_program);
        direct.operand("A").values = store.values("A");
        direct.operand("B").values = store.values("B");
        const auto dev = sw::kpu::program::driver::make_device(DeviceSpec{});
        sw::kpu::program::driver::run_at(level, direct, dev,
                                        sw::kpu::program::Placement::single(dev.compute_tiles));
        // gemm0's output, which gemm1 no longer overwrites.
        CHECK(bit_identical(direct.operand("C").values, store.values("C")));
        // ...and gemm1 produced something of its own from it.
        bool d_computed = false;
        for (float v : store.values("D")) d_computed = d_computed || (v != 0.0f);
        CHECK(d_computed);
    }
}

TEST_CASE("statefulness is proved by a measured reduction in DMA traffic",
          "[program][orchestration][resident]") {
    // The second DoD clause, and the reason it is a TRANSFER COUNT rather than a flag: a
    // boolean saying "reuse happened" can be true while nothing was saved.
    const Loadable l = two_gemms_sharing_b();

    auto run_with_reuse = [&](bool reuse) {
        TensorStore store;
        for (const TensorRef& t : l.tensors) store.declare(t);
        fill_inputs(store);
        VirtualPlatform platform = fresh();
        OrchestratorOptions opt;
        opt.level = ExecutionLevel::BlockSequential;
        opt.reuse_shared_inputs = reuse;
        return orchestrate(l, platform, store, opt);
    };

    const OrchestrationResult cold = run_with_reuse(false);
    const OrchestrationResult warm = run_with_reuse(true);
    REQUIRE_FALSE(cold.refused);
    REQUIRE_FALSE(warm.refused);

    // Every tile fetched twice, once per operator, versus fetched once and kept.
    CHECK(warm.dma_transfers() < cold.dma_transfers());

    // NO SECOND PLACE for a tile that stayed resident. Counted from the trace, which is
    // what the orchestrator actually decided rather than what the executor happened to do.
    std::size_t places = 0, releases = 0;
    for (const Descriptor& d : warm.trace.issued) {
        if (d.kind == DescriptorKind::Place) ++places;
        if (d.kind == DescriptorKind::Release) ++releases;
    }
    std::size_t cold_places = 0;
    for (const Descriptor& d : cold.trace.issued)
        if (d.kind == DescriptorKind::Place) ++cold_places;
    CHECK(places < cold_places);
    CHECK(releases > 0);                         // and it gives slots back

    // The values are the same either way. Residency says where a tile IS, never what it
    // contains, so a placement decision that changed the answer would be a bug and not an
    // optimisation.
    TensorStore a_store, b_store;
    for (const TensorRef& t : l.tensors) { a_store.declare(t); b_store.declare(t); }
    fill_inputs(a_store);
    fill_inputs(b_store);
    VirtualPlatform pa = fresh(), pb = fresh();
    OrchestratorOptions off, on;
    off.reuse_shared_inputs = false;
    on.reuse_shared_inputs = true;
    orchestrate(l, pa, a_store, off);
    orchestrate(l, pb, b_store, on);
    CHECK(bit_identical(a_store.values("C"), b_store.values("C")));
    CHECK(bit_identical(a_store.values("D"), b_store.values("D")));
}

TEST_CASE("the recorded trace is identical across runs", "[program][orchestration]") {
    // The third DoD clause. ADR 0002 §3.5 says a run is a pure function of its inputs; a
    // DECIDING orchestrator keeps that true only if its decisions derive from those inputs
    // alone. The way to check a "provided that" is to record what it decided and compare.
    const Loadable l = two_gemms_sharing_b();
    auto once = [&] {
        TensorStore store;
        for (const TensorRef& t : l.tensors) store.declare(t);
        fill_inputs(store);
        VirtualPlatform platform = fresh();
        return orchestrate(l, platform, store, {}).trace;
    };
    const DescriptorTrace first = once();
    const DescriptorTrace second = once();

    CHECK(first.canonical_bytes() == second.canonical_bytes());
    CHECK(first.digest() == second.digest());
    // Not two empty strings: the comparison has something to compare.
    CHECK(first.canonical_bytes().size() > 64);
    CHECK_FALSE(first.issued.empty());
    CHECK_FALSE(first.completions.empty());

    // Every descriptor gets exactly one completion -- an unanswered descriptor would be an
    // orchestrator waiting forever, which is the failure mode §6.4 exists to remove.
    CHECK(first.issued.size() == first.completions.size());
}

TEST_CASE("a machine too small refuses with a diagnosis, never a hang",
          "[program][orchestration][refusal]") {
    // The fourth DoD clause, and the one that makes runtime allocation safe rather than
    // merely possible. For a static schedule a block on insufficient credit is fine, because
    // the compiler proved the schedule fits. For a runtime allocator a block is a HANG and a
    // refusal is a DECISION POINT.
    const Loadable l = two_gemms_sharing_b();
    TensorStore store;
    for (const TensorRef& t : l.tensors) store.declare(t);
    fill_inputs(store);

    VirtualPlatform tiny = fresh(/*l3_tiles=*/2);     // far less than the working set
    const OrchestrationResult r = orchestrate(l, tiny, store, {});

    CHECK(r.refused);
    CHECK_FALSE(r.diagnosis.empty());
    // The diagnosis names the operator and the arithmetic, because "it did not fit" sends
    // the reader nowhere.
    CHECK(r.diagnosis.find("gemm") != std::string::npos);
    CHECK(r.diagnosis.find("slot") != std::string::npos);

    // The refusal is IN THE TRACE as a completion, not only in the return value: a caller
    // reading the trace must see why it stopped.
    bool refused_in_trace = false;
    for (const Completion& c : r.trace.completions)
        refused_in_trace = refused_in_trace ||
                           c.status == CompletionStatus::RefusedInsufficientCredit;
    CHECK(refused_in_trace);
}

TEST_CASE("no descriptor and no completion carries payload", "[program][orchestration]") {
    // §3's rule, asserted against the TYPE rather than against one run: an MMIO window into
    // buffer contents is a backdoor, and #284 owns the backdoor precisely so it stays
    // flagged in provenance.
    //
    // A structured binding is the tripwire, exactly as it is for RunIdentity: adding a field
    // to Descriptor makes this a COMPILE error here, where someone has to justify it, rather
    // than a silent widening of what an orchestrator may touch.
    const Descriptor d{};
    {
        const auto& [id, kind, tile, leg, resource, target, compute_tile, wait_for] = d;
        (void)id; (void)kind; (void)tile; (void)leg;
        (void)resource; (void)target; (void)compute_tile; (void)wait_for;
    }
    const Completion c{};
    {
        const auto& [did, status, cycles, timed, released, diagnosis] = c;
        (void)did; (void)status; (void)cycles; (void)timed; (void)released; (void)diagnosis;
    }
    SUCCEED("the descriptor and completion surfaces hold no tensor data");
}

TEST_CASE("a PLACE reports that it was not timed, rather than reporting zero",
          "[program][orchestration]") {
    // At L-T1 the executor decides when each leg happens, across the whole run, so a PLACE
    // has no latency of its own (#305 §6.2). `cycles = 0` with `timed = false` says that;
    // `cycles = 0` alone would be a measurement invented out of an absence.
    const Loadable l = two_gemms_sharing_b();
    TensorStore store;
    for (const TensorRef& t : l.tensors) store.declare(t);
    fill_inputs(store);
    VirtualPlatform platform = fresh();
    const OrchestrationResult r = orchestrate(l, platform, store, {});
    REQUIRE_FALSE(r.refused);

    std::size_t untimed_places = 0, timed_launches = 0;
    for (std::size_t i = 0; i < r.trace.issued.size(); ++i) {
        const Descriptor& d = r.trace.issued[i];
        const Completion& c = r.trace.completions[i];
        if (d.kind == DescriptorKind::Place || d.kind == DescriptorKind::Release) {
            CHECK_FALSE(c.timed);
            ++untimed_places;
        }
        if (d.kind == DescriptorKind::Launch && c.timed) ++timed_launches;
    }
    CHECK(untimed_places > 0);
    // A LAUNCH at L-T1 does have a makespan, so the two are distinguishable -- which is what
    // makes "not timed" a statement about this level rather than about the whole ABI.
    CHECK(timed_launches == 2);
    for (const Completion& c : r.trace.completions)
        if (c.timed) CHECK(c.str().find("cycles=not-modelled") == std::string::npos);
}

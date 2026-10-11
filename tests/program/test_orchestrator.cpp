// ============================================================================
// tests/program/test_orchestrator.cpp
// The deciding orchestrator (#305 increment 2) on CSP operators (kpu-run-csp-programs step
// 4d.2), against its four definition-of-done clauses: values agree with the in-process path,
// statefulness is proved by a MEASURED reduction in DMA traffic, the trace is deterministic,
// and a machine too small refuses with a diagnosis rather than hanging. Plus what 4d.2 adds:
// a result kept in L3 and never stored by its producer reaches the next operator, and the
// chain checks refuse what would lose or misread a held tile.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "orchestration_fixtures.hpp"

#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace orchestration_fixtures;
namespace lang = sw::kpu::program::csp::lang;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

namespace {

struct Run {
    OrchestrationResult result;
    TensorStore store;
};

Run run(const Loadable& l, OrchestratorOptions opt = {}, std::uint32_t l3_tiles = 0) {
    Run r;
    for (const TensorRef& t : l.tensors) r.store.declare(t);
    fill_inputs(r.store);
    VirtualPlatform platform = fresh(l3_tiles);
    r.result = orchestrate(l, platform, r.store, opt);
    return r;
}

// The smallest L3 the whole chain runs in.
std::uint32_t min_l3(const Loadable& l) {
    for (std::uint32_t cap = 1; cap <= 40; ++cap)
        if (!run(l, {}, cap).result.refused) return cap;
    return 0;                                    // nothing worked, which would be a bug
}

// One operator's program run directly on the platform, on the values given.
driver::CspLevelOutcome direct(const Loadable& l, std::size_t op, ExecutionLevel level,
                              const std::map<std::string, std::vector<float>>& operands) {
    const lang::Program ast = lang::parse(l.operators[op].csp_program);
    TileProgram inputs = driver::csp_inputs(ast);
    for (const auto& [name, values] : operands) inputs.operand(name).values = values;
    VirtualPlatform platform = fresh();
    const auto h = platform.load_csp(ast, inputs);
    return platform.run_csp(h, level).outcome;
}

std::size_t count(const DescriptorTrace& t, DescriptorKind k) {
    std::size_t n = 0;
    for (const Descriptor& d : t.issued) n += d.kind == k;
    return n;
}

} // namespace

TEST_CASE("a loadable runs, and computes what the in-process path computes",
          "[program][orchestration]") {
    // The first DoD clause. Values are level-invariant, so an orchestrated run and a
    // hand-driven one must agree BIT-EXACTLY -- and disagreement is a bug signal by
    // construction rather than a judgement call.
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        INFO("level " << driver::short_name(level));
        const Loadable l = two_gemms_sharing_weights();
        const Run r = run(l, OrchestratorOptions{level});
        REQUIRE_FALSE(r.result.refused);
        REQUIRE(r.result.per_operator.size() == 2);
        CHECK(r.result.operator_names == std::vector<std::string>{"gemm0", "gemm1"});

        // The in-process path: each program, run directly on what its tensors held.
        const auto first = direct(l, 0, level, {{"A", r.store.values("X")}, {"B", r.store.values("W")}});
        CHECK(bit_identical(first.values.operand("C").values, r.store.values("H")));
        const auto second = direct(l, 1, level, {{"A", r.store.values("H")}, {"B", r.store.values("W")}});
        CHECK(bit_identical(second.values.operand("C").values, r.store.values("Y")));
        bool y_computed = false;
        for (float v : r.store.values("Y")) y_computed = y_computed || (v != 0.0f);
        CHECK(y_computed);
    }
}

TEST_CASE("statefulness is proved by a measured reduction in DMA traffic",
          "[program][orchestration][resident]") {
    // The second DoD clause, and the reason it is a TRANSFER COUNT rather than a flag: a
    // boolean saying "reuse happened" can be true while nothing was saved. Residency is written
    // in the programs now, so the comparison is between two loadables: the same chain with the
    // shared W retained and inherited, and with every operator cold.
    const Run cold = run(two_gemms_sharing_weights(false));
    const Run warm = run(two_gemms_sharing_weights(true));
    REQUIRE_FALSE(cold.result.refused);
    REQUIRE_FALSE(warm.result.refused);
    REQUIRE(cold.result.per_operator.size() == 2);
    REQUIRE(warm.result.per_operator.size() == 2);

    CHECK(warm.result.dma_transfers() < cold.result.dma_transfers());
    // gemm0 runs FIRST, so nothing can be resident for it: it pays in full either way.
    CHECK(warm.result.per_operator[0].dram_loads == cold.result.per_operator[0].dram_loads);
    CHECK(warm.result.per_operator[0].dram_loads == 12);
    // AND THE SAVING IS EXACTLY THE SHARED TILES -- four, W's 2x2 tiling -- which is what makes
    // this a reuse measurement instead of a number that merely got smaller. gemm1's other
    // input, H, was STORED by gemm0, so it is fetched.
    CHECK(cold.result.per_operator[1].dram_loads - warm.result.per_operator[1].dram_loads == 4);

    // The orchestrator decided admission only: no PLACE and no RELEASE, in either chain.
    for (const Run* r : {&cold, &warm}) {
        CHECK(count(r->result.trace, DescriptorKind::Place) == 0);
        CHECK(count(r->result.trace, DescriptorKind::Release) == 0);
        CHECK(count(r->result.trace, DescriptorKind::Reserve) == 2);
        CHECK(count(r->result.trace, DescriptorKind::Launch) == 2);
    }

    // The values are the same either way. Residency says where a tile IS, never what it
    // contains, so a residency decision that changed the answer would be a bug and not an
    // optimisation.
    CHECK(bit_identical(cold.store.values("H"), warm.store.values("H")));
    CHECK(bit_identical(cold.store.values("Y"), warm.store.values("Y")));
}

TEST_CASE("a tile held across a run that never names it still occupies L3",
          "[program][orchestration][resident]") {
    // THE CASE A TWO-OPERATOR CHAIN CANNOT PRODUCE. W is read by gemm0 and gemm2, not by gemm1,
    // so during gemm1's run the device holds four W tiles that gemm1's program never mentions.
    // gemm1 runs in what is left: its reservation must fit beside them.
    const Loadable warm = three_gemms_with_a_gap(true);
    const Loadable cold = three_gemms_with_a_gap(false);

    // The smallest machine: cold, every operator's own L3 (5); warm, gemm1's 5 beside the four
    // held W tiles -- the largest need of the chain, above gemm0's 7 and gemm2's 3 + 4 held.
    // Holding W across the gap costs exactly its four tiles.
    CHECK(min_l3(cold) == 5);
    CHECK(min_l3(warm) == 9);

    // One short of that, gemm1's RESERVE is the refusal, and it names what is held.
    const Run short_one = run(warm, {}, 8);
    REQUIRE(short_one.result.refused);
    CHECK_THAT(short_one.result.diagnosis, ContainsSubstring("gemm1"));
    CHECK_THAT(short_one.result.diagnosis, ContainsSubstring("4 held"));

    // The reuse survives the gap: gemm2 loads no W tile.
    const Run w = run(warm), c = run(cold);
    REQUIRE_FALSE(w.result.refused);
    REQUIRE_FALSE(c.result.refused);
    REQUIRE(w.result.per_operator.size() == 3);
    CHECK(c.result.per_operator[2].dram_loads - w.result.per_operator[2].dram_loads == 4);
    CHECK(w.result.per_operator[1].dram_loads == c.result.per_operator[1].dram_loads);
    CHECK(bit_identical(w.store.values("Y"), c.store.values("Y")));
}

TEST_CASE("a result kept in L3 and never stored reaches the next operator",
          "[program][orchestration][resident]") {
    // H = relu(X W) is retained by its producer and never stored by it; the next operator
    // inherits it, adds a bias in place, and stores it. The answer is the two oracles composed,
    // at every level that computes values.
    const Loadable l = relu_then_bias();
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        INFO("level " << driver::short_name(level));
        const Run r = run(l, OrchestratorOptions{level});
        REQUIRE_FALSE(r.result.refused);

        // The composed oracle: the first's L0 on X, W; its C is the second's inherited C.
        const lang::Program first = lang::parse(l.operators[0].csp_program);
        const lang::Program second = lang::parse(l.operators[1].csp_program);
        TileProgram in1 = driver::csp_inputs(first);
        in1.operand("A").values = r.store.values("X");
        in1.operand("B").values = r.store.values("W");
        const auto ref1 = driver::csp_reference(first, in1, 1'000'000);
        REQUIRE(ref1.values.has_value());
        TileProgram in2 = driver::csp_inputs(second);
        in2.operand("C").values = ref1.values->operand("C").values;
        in2.operand("b").values = r.store.values("bias");
        const auto ref2 = driver::csp_reference(second, in2, 1'000'000);
        REQUIRE(ref2.values.has_value());
        CHECK(bit_identical(r.store.values("H"), ref2.values->operand("C").values));
    }

    // Between the two launches H is in L3 and NOT in DRAM: driven by hand, so the state between
    // them can be seen.
    TensorStore store;
    for (const TensorRef& t : l.tensors) store.declare(t);
    fill_inputs(store);
    VirtualPlatform platform = fresh();
    KpuDevice device(l, platform, store, ExecutionLevel::BlockSequential);
    DirectPort port(device);
    CHECK(port.manifest(0).retains.size() == 4);
    CHECK(port.manifest(1).inherits.size() == 4);
    CHECK(port.manifest(1).bound() == 16 - 4);
    std::uint64_t id = 1;
    auto send = [&](DescriptorKind k, const char* op, std::uint32_t slots = 0) {
        Descriptor d;
        d.id = id++;
        d.kind = k;
        d.target = op;
        d.slots = slots;
        port.submit(d);
        Completion c;
        REQUIRE(port.poll_completion(c));
        return c;
    };
    REQUIRE(send(DescriptorKind::Reserve, "matmul_relu", 16).status == CompletionStatus::Done);
    REQUIRE(send(DescriptorKind::Launch, "matmul_relu").status == CompletionStatus::Done);
    CHECK(port.read_status().held == 4);
    CHECK(port.is_resident(TileRef{"H", 1, 1}));
    for (float v : store.values("H")) CHECK(v == 0.0f);             // never stored
    REQUIRE(send(DescriptorKind::Reserve, "bias", 12).status == CompletionStatus::Done);
    REQUIRE(send(DescriptorKind::Launch, "bias").status == CompletionStatus::Done);
    CHECK(port.read_status().held == 0);
    CHECK_FALSE(port.is_resident(TileRef{"H", 1, 1}));
}

TEST_CASE("the chain is checked at load: a held tile is never lost or misread",
          "[program][orchestration][resident][refusal]") {
    auto refusal = [](const Loadable& l) {
        const Run r = run(l);
        REQUIRE(r.result.refused);
        return r.result.diagnosis;
    };
    SECTION("a retained tile nothing inherits") {
        Loadable l = two_gemms_sharing_weights(true);
        l.operators.pop_back();
        const std::string why = refusal(l);
        CHECK_THAT(why, ContainsSubstring("gemm0"));
        CHECK_THAT(why, ContainsSubstring("no later operator inherits"));
    }
    SECTION("an inherited tile nothing retained") {
        Loadable l = two_gemms_sharing_weights(true);
        l.operators[0] = gemm("gemm0", "cold", {"X", "W"}, {"H"});
        const std::string why = refusal(l);
        CHECK_THAT(why, ContainsSubstring("gemm1"));
        CHECK_THAT(why, ContainsSubstring("no earlier operator retains"));
    }
    SECTION("a tile retained twice") {
        Loadable l = three_gemms_with_a_gap(true);
        l.operators[1] = gemm("gemm1", "retain", {"H", "W"}, {"G"});
        CHECK_THAT(refusal(l), ContainsSubstring("already retains"));
    }
    SECTION("a retained, unstored tile loaded from DRAM before it is claimed") {
        // H is retained by matmul_relu and never stored; a GEMM between it and its claimant
        // would load H's stale DRAM copy.
        Loadable l = relu_then_bias();
        l.tensors.push_back(tensor("G", 0x6000, false));
        l.operators.insert(l.operators.begin() + 1, gemm("between", "cold", {"H", "W"}, {"G"}));
        const std::string why = refusal(l);
        CHECK_THAT(why, ContainsSubstring("between"));
        CHECK_THAT(why, ContainsSubstring("has not stored"));
    }
    SECTION("a tile stored, written again, then retained") {
        // DRAM holds the value at the store and L3 a later one; a level reports only the later
        // one, so writing the tile back would put a value in DRAM the program never stored.
        Loadable l;
        l.name = "store-then-write";
        // X and W are unused; fill_inputs expects them.
        l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true), tensor("H", 0x3000, true),
                     tensor("bias", 0x5000, true, {32}, {16})};
        l.operators = {sw::kpu::loadable::csp_operator(
            "twice", "csp 1.0\nprogram twice machine flat(l3 = 4) {\n"
                     "  tensor C[32,32] tile 16x16 inout;\n  vector b[32] tile 16 in;\n"
                     "  resident C[0, 0];\n  resident b[0];\n"
                     "  call add(C[0, 0], b[0]) -> C[0, 0];\n  store C[0, 0];\n"
                     "  call add(C[0, 0], b[0]) -> C[0, 0];\n"
                     "  retain C[0, 0];\n  release b[0];\n}\n",
            {"H", "bias"}, {})};
        const std::string why = refusal(l);
        CHECK_THAT(why, ContainsSubstring("twice"));
        CHECK_THAT(why, ContainsSubstring("writes it again before retaining it"));
    }
    SECTION("an operand tiled differently from its tensor") {
        Loadable l = two_gemms_sharing_weights(false);
        l.tensors[1].tile_shape = {8, 8};
        CHECK_THAT(refusal(l), ContainsSubstring("tiled"));
    }
    SECTION("an operator binding the wrong number of tensors") {
        Loadable l = two_gemms_sharing_weights(false);
        l.operators[0].inputs = {"X"};
        CHECK_THAT(refusal(l), ContainsSubstring("input operand"));
    }
}

TEST_CASE("the recorded trace is identical across runs", "[program][orchestration]") {
    // The third DoD clause. ADR 0002 §3.5 says a run is a pure function of its inputs; a
    // DECIDING orchestrator keeps that true only if its decisions derive from those inputs
    // alone. The way to check a "provided that" is to record what it decided and compare.
    const Loadable l = two_gemms_sharing_weights();
    const DescriptorTrace first = run(l).result.trace;
    const DescriptorTrace second = run(l).result.trace;

    CHECK(first.canonical_bytes() == second.canonical_bytes());
    CHECK(first.digest() == second.digest());
    // Not two empty strings: the comparison has something to compare.
    CHECK(first.canonical_bytes().size() > 32);
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
    const Run r = run(two_gemms_sharing_weights(), {}, /*l3_tiles=*/2);

    CHECK(r.result.refused);
    CHECK_FALSE(r.result.diagnosis.empty());
    // The diagnosis names the operator and the arithmetic, because "it did not fit" sends
    // the reader nowhere.
    CHECK_THAT(r.result.diagnosis, ContainsSubstring("gemm"));
    CHECK_THAT(r.result.diagnosis, ContainsSubstring("slot"));

    // The refusal is IN THE TRACE as a completion, not only in the return value: a caller
    // reading the trace must see why it stopped.
    bool refused_in_trace = false;
    for (const Completion& c : r.result.trace.completions)
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
    //
    // Increment 3 widened both, and each new field is an index, a count or a flag: `slots` and
    // `flags` on the descriptor; on the completion the refusal's CAUSE and its arithmetic
    // (needed / available / capacity / blocking operator), which cross the MMIO ABI as fixed
    // 32-bit fields. None of them can hold a tensor element.
    const Descriptor d{};
    {
        const auto& [id, kind, tile, leg, resource, target, compute_tile, wait_for, slots,
                     flags] = d;
        (void)id; (void)kind; (void)tile; (void)leg; (void)resource; (void)target;
        (void)compute_tile; (void)wait_for; (void)slots; (void)flags;
    }
    const Completion c{};
    {
        const auto& [did, status, cycles, timed, released, diagnosis, cause, needed, available,
                     capacity, blocking_op] = c;
        (void)did; (void)status; (void)cycles; (void)timed; (void)released; (void)diagnosis;
        (void)cause; (void)needed; (void)available; (void)capacity; (void)blocking_op;
    }
    SUCCEED("the descriptor and completion surfaces hold no tensor data");
}

TEST_CASE("a completion says whether it was timed, rather than reporting zero",
          "[program][orchestration]") {
    // A RESERVE takes no time of its own, and at L-B nothing is timed at all. `cycles = 0` with
    // `timed = false` says that; `cycles = 0` alone would be a measurement invented out of an
    // absence.
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        INFO("level " << driver::short_name(level));
        const Run r = run(two_gemms_sharing_weights(), OrchestratorOptions{level});
        REQUIRE_FALSE(r.result.refused);
        std::size_t untimed_reserves = 0, timed_launches = 0;
        for (const Descriptor& d : r.result.trace.issued)
            for (const Completion& c : r.result.trace.completions) {
                if (c.descriptor_id != d.id) continue;
                if (d.kind == DescriptorKind::Reserve && !c.timed) ++untimed_reserves;
                if (d.kind == DescriptorKind::Launch && c.timed) {
                    ++timed_launches;
                    CHECK(c.cycles > 0);
                }
            }
        CHECK(untimed_reserves == 2);
        // A LAUNCH at L-T1 has a makespan, so the two are distinguishable -- which is what makes
        // "not timed" a statement about the descriptor and the level rather than the whole ABI.
        CHECK(timed_launches == (level == ExecutionLevel::BlockSequential ? 2u : 0u));
        for (const Completion& c : r.result.trace.completions)
            if (c.timed) CHECK(c.str().find("cycles=not-modelled") == std::string::npos);
    }
}

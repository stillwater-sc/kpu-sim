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

// TWO GEMMS SHARING A WEIGHT TENSOR. The shared input is what makes statefulness observable,
// and it is also the real case: a weight tensor read by layer after layer. The first operator's
// OUTPUT would not serve — after a writeback it is in DRAM, so keeping it "resident" would mean
// re-fetching it, which saves nothing.
//
// NO TENSOR IS NAMED LIKE AN OPERAND, deliberately. The kernel's operands are A/B/C whatever
// the model calls them, so a fixture whose tensors are also A/B/C cannot tell a tensor-keyed
// bug from an operand-keyed one: both spellings agree by coincidence, and a review finding
// about exactly that confusion reproduced as "all tests pass". X/W/H/Y are disjoint from
// A/B/C, so any place the two vocabularies are mixed up now shows up as a wrong answer or a
// missed reuse.
Loadable two_gemms_sharing_weights() {
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
    // X = activations in, W = the shared weights, H = the hidden result, Y = the output.
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("H", 0x3000, false), tensor("Y", 0x4000, false)};

    Operator first;
    first.name = "gemm0";
    first.l0_program = l0_matmul(32, 16);
    first.inputs = {"X", "W"};        // -> the program's operands A, B
    first.outputs = {"H"};            // -> its operand C

    // The second GEMM consumes the first's output and reads W AGAIN. Its L0 program uses the
    // same operand names -- A/B/C, because that is what the kernel calls them -- and the
    // binding maps those onto different TENSORS. Without that indirection both operators
    // would read and write the same tensor, the second would accumulate onto the first's
    // result, and the hidden result would come out doubled. Which is exactly what happened.
    Operator second;
    second.name = "gemm1";
    second.l0_program = l0_matmul(32, 16);
    second.inputs = {"H", "W"};       // H from gemm0, and W shared -> operands A, B
    second.outputs = {"Y"};

    l.operators = {first, second};
    l.profile.min_compute_tiles = 1;
    return l;
}

// THREE OPERATORS WITH A GAP: W is read by the first and the THIRD, and not by the second.
// That gap is the case a two-operator chain cannot produce -- during the middle run the
// orchestrator holds four W tiles that the middle program never mentions, so they have no
// operand name and therefore no tile key in its vocabulary. They still occupy L3.
Loadable three_gemms_with_a_gap() {
    Loadable l;
    l.name = "gemm-gemm-gemm-reuse-across-a-gap";

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
    l.tensors = {tensor("X", 0x1000, true), tensor("W", 0x2000, true),
                 tensor("V", 0x5000, true), tensor("H", 0x3000, false),
                 tensor("G", 0x6000, false), tensor("Y", 0x4000, false)};

    auto gemm = [](const char* name, std::vector<std::string> in, std::vector<std::string> out) {
        Operator o;
        o.name = name;
        o.l0_program = l0_matmul(32, 16);
        o.inputs = std::move(in);
        o.outputs = std::move(out);
        return o;
    };
    l.operators = {gemm("gemm0", {"X", "W"}, {"H"}),
                   gemm("gemm1", {"H", "V"}, {"G"}),     // does not mention W
                   gemm("gemm2", {"G", "W"}, {"Y"})};    // wants it again
    l.profile.min_compute_tiles = 1;
    return l;
}

void fill_inputs(TensorStore& s) {
    auto& a = s.values("X");
    auto& b = s.values("W");
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(i % 7) - 3.0f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(i % 5) - 2.0f;
    if (s.has("V")) {
        auto& v = s.values("V");
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = float(i % 3) - 1.0f;
    }
}

// The final output of a whole orchestrated run, for "the answer does not depend on the
// placement decisions" -- which is the level-invariance claim applied to residency.
std::vector<float> final_output(const Loadable& l, const char* tensor, bool reuse);

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

std::vector<float> final_output(const Loadable& l, const char* tensor, bool reuse) {
    TensorStore store;
    for (const TensorRef& t : l.tensors) store.declare(t);
    fill_inputs(store);
    VirtualPlatform platform = fresh();
    OrchestratorOptions opt;
    opt.level = ExecutionLevel::BlockSequential;
    opt.reuse_shared_inputs = reuse;
    orchestrate(l, platform, store, opt);
    return store.values(tensor);
}

} // namespace

TEST_CASE("a loadable runs, and computes what the in-process path computes",
          "[program][orchestration]") {
    // The first DoD clause. Values are level-invariant, so an orchestrated run and a
    // hand-driven one must agree BIT-EXACTLY -- and disagreement is a bug signal by
    // construction rather than a judgement call.
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        const Loadable l = two_gemms_sharing_weights();
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
        direct.operand("A").values = store.values("X");
        direct.operand("B").values = store.values("W");
        const auto dev = sw::kpu::program::driver::make_device(DeviceSpec{});
        sw::kpu::program::driver::run_at(level, direct, dev,
                                        sw::kpu::program::Placement::single(dev.compute_tiles));
        // gemm0's output, which gemm1 no longer overwrites.
        CHECK(bit_identical(direct.operand("C").values, store.values("H")));
        // ...and gemm1 produced something of its own from it.
        bool y_computed = false;
        for (float v : store.values("Y")) y_computed = y_computed || (v != 0.0f);
        CHECK(y_computed);
    }
}

TEST_CASE("statefulness is proved by a measured reduction in DMA traffic",
          "[program][orchestration][resident]") {
    // The second DoD clause, and the reason it is a TRANSFER COUNT rather than a flag: a
    // boolean saying "reuse happened" can be true while nothing was saved.
    const Loadable l = two_gemms_sharing_weights();

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

    // AND THE SAVING IS EXACTLY THE SHARED TILES -- four, tensor B's 2x2 tiling -- which is
    // the assertion that makes this a reuse measurement instead of a number that merely got
    // smaller. Two ways to be wrong were both live before review:
    //
    //   * seeding the tiles this operator's own PLACEs just asked for. Every tile then looks
    //     resident, gemm0 pays no DMA at all, and the "saving" is the whole traffic.
    //   * seeding TENSOR keys where the executor compares OPERAND keys. The shared weights are
    //     tensor W read through operand B, so a tensor-keyed seed ("W#0#0") matches nothing the
    //     executor knows and the saving drops to ZERO. It reads as "reuse does not work", which
    //     is why this fixture's tensors are X/W/H/Y: while they were A/B/C/D the two spellings
    //     agreed by coincidence and the bug was invisible.
    //
    // Both inflate this number, so pinning it down catches both.
    auto dma_of = [](const RunOutcome& o) {
        if (!o.stats) return std::size_t{0};
        const auto it = o.stats->hop_transfers.find(sw::kpu::program::Hop::DmaDramToL3);
        return it == o.stats->hop_transfers.end() ? std::size_t{0} : it->second;
    };
    REQUIRE(cold.per_operator.size() == 2);
    REQUIRE(warm.per_operator.size() == 2);
    // gemm0 runs FIRST, so nothing can be resident for it: it pays in full either way.
    CHECK(dma_of(warm.per_operator[0]) == dma_of(cold.per_operator[0]));
    CHECK(dma_of(warm.per_operator[0]) > 0);
    // gemm1 saves the four shared W tiles, and nothing else. Its other input is tensor H,
    // which gemm0 WROTE rather than placed -- it is in DRAM, so it is fetched.
    CHECK(dma_of(cold.per_operator[1]) - dma_of(warm.per_operator[1]) == 4);

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
    CHECK(bit_identical(a_store.values("H"), b_store.values("H")));
    CHECK(bit_identical(a_store.values("Y"), b_store.values("Y")));
}

TEST_CASE("a tile held across a run that never names it still occupies L3",
          "[program][orchestration][resident]") {
    // THE CASE A TWO-OPERATOR CHAIN CANNOT PRODUCE. W is read by gemm0 and gemm2, not by gemm1,
    // so during gemm1's run the orchestrator holds four W tiles that gemm1's program never
    // mentions. They have no operand there, so they have no tile key either -- and a slot with
    // no key is a slot the executor cannot be told about by name.
    //
    // Left uncounted it is a quiet overstatement of capacity: gemm1 would place up to the full
    // L3 while four slots were already gone, and report a peak residency the machine could not
    // have delivered. `foreign_held_slots` is a COUNT for exactly this reason -- a synthetic key
    // could collide with a real operand name, and a collision would mark a real tile resident
    // and skip its DMA leg.
    const Loadable l = three_gemms_with_a_gap();

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
    REQUIRE(warm.per_operator.size() == 3);

    auto peak_of = [](const RunOutcome& o) {
        return o.stats ? o.stats->peak_l3_residency : std::size_t{0};
    };
    // gemm1's own working set is identical either way -- same program, same inputs. The whole
    // difference is the four W tiles held through it, and they show up in the peak.
    CHECK(peak_of(warm.per_operator[1]) == peak_of(cold.per_operator[1]) + 4);

    // And the reuse survives the gap: gemm2 reads W without a second PLACE for any of its tiles.
    std::size_t w_places = 0;
    for (const Descriptor& d : warm.trace.issued)
        if (d.kind == DescriptorKind::Place && d.tile.tensor == "W") ++w_places;
    CHECK(w_places == 4);                        // placed once, by gemm0, and never again

    // Values do not depend on any of it.
    CHECK(bit_identical(final_output(l, "Y", false), final_output(l, "Y", true)));
}

TEST_CASE("reuse costs only the tiles it actually keeps", "[program][orchestration][resident]") {
    // RETAINING A TILE NOBODY WILL READ AGAIN buys nothing and costs a slot for the whole run,
    // which can refuse a run that fits -- the same argument as releasing before asking, applied
    // to retention. So retention is filtered by what a LATER operator reads.
    //
    // Measured on the smallest L3 the whole chain fits in, which is where the difference is
    // visible as a refusal rather than as a number in a stats block:
    //
    //   reuse off                    8 slots   (each operator's own live set, nothing held)
    //   reuse on, filtered          12 slots   (+4: tensor W, held across gemm1 for gemm2)
    //   reuse on, UNFILTERED        21 slots   (every read tile of every operator held)
    //
    // The last row is what this assertion exists to keep out: 21 slots to save four fetches.
    const Loadable l = three_gemms_with_a_gap();
    auto min_cap = [&](bool reuse) {
        for (std::uint32_t cap = 1; cap <= 60; ++cap) {
            TensorStore store;
            for (const TensorRef& t : l.tensors) store.declare(t);
            fill_inputs(store);
            VirtualPlatform platform = fresh(cap);
            OrchestratorOptions opt;
            opt.level = ExecutionLevel::BlockSequential;
            opt.reuse_shared_inputs = reuse;
            if (!orchestrate(l, platform, store, opt).refused) return std::size_t(cap);
        }
        return std::size_t(0);                   // nothing worked, which would be a bug
    };
    const std::size_t cold = min_cap(false);
    const std::size_t warm = min_cap(true);
    REQUIRE(cold > 0);
    REQUIRE(warm > 0);
    CHECK(warm == cold + 4);                     // exactly the four tiles it keeps
}

TEST_CASE("the recorded trace is identical across runs", "[program][orchestration]") {
    // The third DoD clause. ADR 0002 §3.5 says a run is a pure function of its inputs; a
    // DECIDING orchestrator keeps that true only if its decisions derive from those inputs
    // alone. The way to check a "provided that" is to record what it decided and compare.
    const Loadable l = two_gemms_sharing_weights();
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
    const Loadable l = two_gemms_sharing_weights();
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
    const Loadable l = two_gemms_sharing_weights();
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

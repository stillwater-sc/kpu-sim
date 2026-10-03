// ============================================================================
// tests/program/test_reservation.cpp
// Reserve-then-launch (#305 increment 3): the rules R1-R4 one at a time, the freedom they
// buy, the wedge they prevent, and the claim the whole scheme rests on -- a granted
// reservation completes.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include "orchestration_fixtures.hpp"

using namespace orchestration_fixtures;

namespace {

struct Run {
    OrchestrationResult result;
    TensorStore store;
};

Run run(const Loadable& l, OrchestratorOptions opt, std::uint32_t l3_tiles) {
    Run r;
    for (const TensorRef& t : l.tensors) r.store.declare(t);
    fill_inputs(r.store);
    VirtualPlatform platform = fresh(l3_tiles);
    r.result = orchestrate(l, platform, r.store, opt);
    return r;
}

// Index of the first issued descriptor matching, or npos.
std::size_t first_issued(const DescriptorTrace& t, DescriptorKind k, const std::string& target) {
    for (std::size_t i = 0; i < t.issued.size(); ++i)
        if (t.issued[i].kind == k && t.issued[i].target == target) return i;
    return static_cast<std::size_t>(-1);
}

const Descriptor* issued_by_id(const DescriptorTrace& t, std::uint64_t id) {
    for (const Descriptor& d : t.issued)
        if (d.id == id) return &d;
    return nullptr;
}

// Hand-driven: one descriptor, its completion.
struct Hand {
    Loadable l = three_gemms_with_a_gap();
    TensorStore store;
    VirtualPlatform platform;
    KpuDevice device;
    DirectPort port;
    std::uint64_t next = 1;

    explicit Hand(std::uint32_t cap)
        : platform(fresh(cap)),
          device(l, platform, store, ExecutionLevel::BlockSequential),
          port(device) {}

    // Completions are matched BY ID and the rest are kept: a RELEASE at last read completes
    // after the LAUNCH it governs, so a completion for an earlier descriptor can arrive while
    // waiting for a later one, and dropping it would lose it silently.
    std::vector<Completion> backlog;
    Completion send(Descriptor d) {
        d.id = next++;
        port.submit(d);
        Completion c;
        while (port.poll_completion(c)) backlog.push_back(c);
        for (auto it = backlog.begin(); it != backlog.end(); ++it)
            if (it->descriptor_id == d.id) {
                Completion out = *it;
                backlog.erase(it);
                return out;
            }
        FAIL("no completion for descriptor " << d.id);
        return c;
    }
    Completion reserve(const char* op, std::uint32_t slots) {
        Descriptor d;
        d.kind = DescriptorKind::Reserve;
        d.target = op;
        d.slots = slots;
        return send(d);
    }
    Completion place(const char* op, TileRef t) {
        Descriptor d;
        d.kind = DescriptorKind::Place;
        d.target = op;
        d.tile = std::move(t);
        return send(d);
    }
    Completion launch(const char* op) {
        Descriptor d;
        d.kind = DescriptorKind::Launch;
        d.target = op;
        return send(d);
    }
};

} // namespace

// ---- the rules, one at a time -------------------------------------------------------
TEST_CASE("R1: reservations are granted in operator order", "[program][orchestration][reserve]") {
    Hand h(64);
    const Completion c = h.reserve("gemm1", 6);
    CHECK(c.status == CompletionStatus::RefusedInsufficientCredit);
    CHECK(c.cause == RefusalCause::ReservationOutOfOrder);
    CHECK(c.blocking_op == 0);                       // gemm0 stands in the way
    CHECK(c.diagnosis.find("gemm0") != std::string::npos);

    // Once gemm0 holds one, gemm1 may reserve AHEAD of gemm0's launch -- the freedom -- and
    // gemm2 behind it, since every earlier uncompleted operator now holds a reservation.
    CHECK(h.reserve("gemm0", 10).status == CompletionStatus::Done);
    CHECK(h.reserve("gemm1", 6).status == CompletionStatus::Done);
    CHECK(h.reserve("gemm2", 6).status == CompletionStatus::Done);
}

TEST_CASE("R2: a reservation is all or none, and refused rather than queued",
          "[program][orchestration][reserve]") {
    Hand h(12);
    CHECK(h.reserve("gemm0", 10).status == CompletionStatus::Done);
    const Completion c = h.reserve("gemm1", 6);
    CHECK(c.status == CompletionStatus::RefusedInsufficientCredit);
    CHECK(c.cause == RefusalCause::InsufficientCredit);
    CHECK(c.needed == 6);
    CHECK(c.available == 2);
    CHECK(c.capacity == 12);
    // Nothing was partially granted: the status still shows only gemm0's claim.
    CHECK(h.port.read_status().reserved == 10);
}

TEST_CASE("R3: a PLACE needs a reservation; a prefetch must fit inside it",
          "[program][orchestration][reserve]") {
    Hand h(64);
    const Completion none = h.place("gemm0", TileRef{"X", 0, 0});
    CHECK(none.cause == RefusalCause::NoReservation);

    CHECK(h.reserve("gemm0", 10).status == CompletionStatus::Done);
    CHECK(h.reserve("gemm1", 2).status == CompletionStatus::Done);
    // gemm1 is not next to launch, so its tiles occupy slots while gemm0 runs: two fit...
    CHECK(h.place("gemm1", TileRef{"V", 0, 0}).status == CompletionStatus::Done);
    CHECK(h.place("gemm1", TileRef{"V", 0, 1}).status == CompletionStatus::Done);
    // ...a third does not.
    const Completion over = h.place("gemm1", TileRef{"V", 1, 0});
    CHECK(over.cause == RefusalCause::ReservationExceeded);
}

TEST_CASE("R4: launches are in order and within what was reserved",
          "[program][orchestration][reserve]") {
    Hand h(64);
    CHECK(h.reserve("gemm0", 10).status == CompletionStatus::Done);
    CHECK(h.reserve("gemm1", 6).status == CompletionStatus::Done);
    const Completion early = h.launch("gemm1");
    CHECK(early.cause == RefusalCause::ReservationOutOfOrder);
    CHECK(early.blocking_op == 0);

    // A reservation too small for the run is caught BEFORE the run, not inside it: a granted
    // reservation that ended in an in-run refusal is the failure this increment removes.
    Hand small(64);
    CHECK(small.reserve("gemm0", 1).status == CompletionStatus::Done);
    const Completion c = small.launch("gemm0");
    CHECK(c.cause == RefusalCause::ReservationExceeded);
    CHECK(c.needed == small.port.manifest(0).peak_live_tiles);
    CHECK(c.available == 1);
}

// ---- DoD 3: the freedom ---------------------------------------------------------------
// three_gemms_with_a_gap: gemm1 reads V, which nothing earlier writes or reads. That is the
// tile a reserve-then-launch orchestrator can place AHEAD of gemm0's launch. Bounds, from the
// manifests: gemm0 reserves 6 live + 4 retained (W, kept for gemm2) = 10, gemm1 reserves 6.
TEST_CASE("a placement order program-order acquisition forbids completes, where it is sound",
          "[program][orchestration][reserve]") {
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions order, rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;

    const std::uint32_t cap = 16;                    // 10 + 6: both reservations fit
    const Run a = run(l, order, cap);
    const Run b = run(l, rtl, cap);
    REQUIRE_FALSE(a.result.refused);
    REQUIRE_FALSE(b.result.refused);

    // THE ORDER: gemm1 reserved and V placed for it before gemm0 launches. Program-order
    // acquisition cannot produce this -- gemm0 is an earlier, unfired operator still needing
    // slots.
    const std::size_t launch0 = first_issued(b.result.trace, DescriptorKind::Launch, "gemm0");
    const std::size_t reserve1 = first_issued(b.result.trace, DescriptorKind::Reserve, "gemm1");
    const std::size_t place1 = first_issued(b.result.trace, DescriptorKind::Place, "gemm1");
    REQUIRE(launch0 != static_cast<std::size_t>(-1));
    CHECK(reserve1 < launch0);
    CHECK(place1 < launch0);
    CHECK(b.result.trace.issued[place1].tile.tensor == "V");
    // ...and the program-order run does not.
    CHECK(first_issued(a.result.trace, DescriptorKind::Place, "gemm1") >
          first_issued(a.result.trace, DescriptorKind::Launch, "gemm0"));

    // Same answer: placement says where a tile is, never what it holds.
    CHECK(bit_identical(a.store.values("Y"), b.store.values("Y")));
    CHECK(bit_identical(a.store.values("H"), b.store.values("H")));
    // And, at L-T1, the SAME DMA traffic: a PLACE has no leg of its own at this level, so the
    // prefetched V tiles are fetched inside gemm1's run either way (plan §3.5). Prefetch buys
    // allocation freedom here, not speed; #283 is where it can buy speed.
    CHECK(a.result.dma_transfers() == b.result.dma_transfers());
    // gemm0 ran beside gemm1's claim, and its peak residency says so.
    REQUIRE(b.result.per_operator.size() == 3);
    CHECK(b.result.per_operator[0].stats->peak_l3_residency <= cap);
}

TEST_CASE("a refused reservation is a decision point, not a stall",
          "[program][orchestration][reserve]") {
    // One slot short of both reservations: gemm1's RESERVE is refused, the orchestrator runs
    // gemm0 without prefetching, and gemm1 reserves on its own turn.
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;
    const Run r = run(l, rtl, 15);
    REQUIRE_FALSE(r.result.refused);

    bool refused_then_granted = false;
    bool saw_refusal = false;
    for (const Completion& c : r.result.trace.completions) {
        const Descriptor* d = issued_by_id(r.result.trace, c.descriptor_id);
        REQUIRE(d != nullptr);
        if (d->kind != DescriptorKind::Reserve || d->target != "gemm1") continue;
        if (c.status != CompletionStatus::Done) {
            CHECK(c.cause == RefusalCause::InsufficientCredit);
            saw_refusal = true;
        } else if (saw_refusal) {
            refused_then_granted = true;
        }
    }
    CHECK(refused_then_granted);
    CHECK(first_issued(r.result.trace, DescriptorKind::Place, "gemm1") >
          first_issued(r.result.trace, DescriptorKind::Launch, "gemm0"));
}

TEST_CASE("without reservations the same freedom wedges, and the device refuses it",
          "[program][orchestration][reserve][refusal]") {
    // The WRONG shape of plan §3.3: place gemm1's tiles first, then gemm0's, one slot per PLACE,
    // nothing ordered. At 11 slots, V's 4 then X's 4 and W's 4 is 12: gemm0 cannot place, and
    // gemm1 cannot launch before gemm0. Hold-and-wait.
    //
    // The device must REFUSE, naming who holds the credits -- never hang. `orchestrate` returning
    // at all is the no-hang assertion: the device never blocks, and the orchestrator reports a
    // missing completion as a protocol error rather than spinning.
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions greedy;
    greedy.policy = AllocationPolicy::GreedyPrefetch;
    greedy.device.enforce_reservations = false;
    const Run g = run(l, greedy, 11);
    REQUIRE(g.result.refused);
    const Completion& last = g.result.trace.completions.back();
    CHECK(last.cause == RefusalCause::InsufficientCredit);
    CHECK(last.blocking_op == 1);                    // gemm1, a LATER operator
    CHECK(g.result.diagnosis.find("gemm1") != std::string::npos);
    const Descriptor* d = issued_by_id(g.result.trace, last.descriptor_id);
    REQUIRE(d != nullptr);
    CHECK(d->kind == DescriptorKind::Place);
    CHECK(d->target == "gemm0");

    // The same capacity under reserve-then-launch completes: the reservation for gemm1 is
    // refused up front instead of half-granted.
    OrchestratorOptions rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;
    CHECK_FALSE(run(l, rtl, 11).result.refused);

    // And with reservations ENFORCED, the greedy shape cannot even start.
    OrchestratorOptions greedy_enforced;
    greedy_enforced.policy = AllocationPolicy::GreedyPrefetch;
    const Run e = run(l, greedy_enforced, 64);
    REQUIRE(e.result.refused);
    CHECK(e.result.trace.completions.back().cause == RefusalCause::NoReservation);
}

// ---- the claim the scheme rests on ----------------------------------------------------
TEST_CASE("a granted reservation completes, at every capacity",
          "[program][orchestration][reserve]") {
    // The deadlock-freedom argument in kpu_device.hpp leans on one empirical premise: the
    // reservation bound (peak live tiles + retained) is SUFFICIENT for the executor. So sweep
    // capacity and assert the only refusals are at a RESERVE -- the decision point -- and never
    // from a LAUNCH, which would mean a granted reservation ended in an in-run refusal.
    const std::vector<Loadable> loadables = {two_gemms_sharing_weights(),
                                             three_gemms_with_a_gap()};
    for (const Loadable& l : loadables)
        for (AllocationPolicy policy :
             {AllocationPolicy::ProgramOrder, AllocationPolicy::ReserveThenLaunch})
            for (bool reuse : {false, true}) {
                bool completed_before = false;
                for (std::uint32_t cap = 1; cap <= 24; ++cap) {
                    INFO(l.name << " policy " << to_string(policy) << " reuse " << reuse
                                << " L3 " << cap);
                    OrchestratorOptions opt;
                    opt.policy = policy;
                    opt.reuse_shared_inputs = reuse;
                    const Run r = run(l, opt, cap);
                    if (!r.result.refused) {
                        completed_before = true;
                        continue;
                    }
                    // Monotone: more L3 never turns a completion into a refusal.
                    CHECK_FALSE(completed_before);
                    const Completion& last = r.result.trace.completions.back();
                    const Descriptor* d = issued_by_id(r.result.trace, last.descriptor_id);
                    REQUIRE(d != nullptr);
                    CHECK(d->kind == DescriptorKind::Reserve);
                    CHECK(last.cause == RefusalCause::InsufficientCredit);
                }
                CHECK(completed_before);
            }
}

TEST_CASE("the minimum L3 is the reservation bound, and reuse costs exactly what it keeps",
          "[program][orchestration][reserve]") {
    // Increment 2 measured a cold minimum of 8 slots for this chain. That 8 was its own
    // check -- |tiles to place| -- not the machine's: the matmul's live set is 6, and the
    // executor completes there. The reservation bound is that live set, so the minimum drops
    // to it, and holding W across the gap still costs exactly its four tiles.
    const Loadable l = three_gemms_with_a_gap();
    auto min_cap = [&](bool reuse) {
        for (std::uint32_t cap = 1; cap <= 40; ++cap) {
            OrchestratorOptions opt;
            opt.reuse_shared_inputs = reuse;
            if (!run(l, opt, cap).result.refused) return cap;
        }
        return 0u;
    };
    CHECK(min_cap(false) == 6);
    CHECK(min_cap(true) == 10);
}

TEST_CASE("reserve-then-launch is deterministic", "[program][orchestration][reserve]") {
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;
    for (std::uint32_t cap : {0u, 11u, 16u}) {
        INFO("L3 " << cap);
        const Run a = run(l, rtl, cap);
        const Run b = run(l, rtl, cap);
        CHECK(a.result.trace.digest() == b.result.trace.digest());
        CHECK(a.result.trace.issued.size() == a.result.trace.completions.size());
    }
}

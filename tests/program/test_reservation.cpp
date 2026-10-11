// ============================================================================
// tests/program/test_reservation.cpp
// Reserve-then-launch (#305 increment 3), on CSP operators (kpu-run-csp-programs step 4d.2):
// the rules R1-R4 one at a time, the freedom they buy, and the claim the whole scheme rests
// on -- a granted reservation completes.
//
// The fixture is three_gemms_with_a_gap: gemm0 retains W (program L3 7), gemm1 does not name
// it (L3 5), gemm2 inherits it (L3 7, 4 of them held). Reservations: 7, 5 and 3.
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

    // Completions are matched BY ID and the rest are kept, so one posted out of order would
    // not be lost silently.
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

TEST_CASE("R3: a PLACE is refused: a program loads its own tiles",
          "[program][orchestration][reserve]") {
    // Before step 4d.2 a PLACE moved a tile to L3 ahead of the run that read it. A CSP program
    // sequences its own Loads, so a PLACE would be a DMA no program sequences; the device
    // refuses it by name, with or without a reservation.
    Hand h(64);
    const Completion none = h.place("gemm0", TileRef{"X", 0, 0});
    CHECK(none.status == CompletionStatus::RefusedUnsupported);
    CHECK(none.cause == RefusalCause::Unsupported);
    CHECK(none.diagnosis.find("loads its own tiles") != std::string::npos);
    CHECK(h.reserve("gemm0", 7).status == CompletionStatus::Done);
    CHECK(h.place("gemm0", TileRef{"X", 0, 0}).cause == RefusalCause::Unsupported);
    CHECK(h.port.read_status().held == 0);
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
    CHECK(c.needed == small.port.manifest(0).bound());
    CHECK(c.needed == 7);                            // gemm0's program L3, nothing inherited
    CHECK(c.available == 1);

    // gemm2 inherits W, so its need is its program's L3 less the four held slots -- and it
    // cannot launch unless W is held.
    Hand chain(64);
    CHECK(chain.port.manifest(2).bound() == 3);
    CHECK(chain.port.manifest(2).inherits.size() == 4);
    CHECK(chain.reserve("gemm0", 7).status == CompletionStatus::Done);
    CHECK(chain.launch("gemm0").status == CompletionStatus::Done);
    CHECK(chain.port.read_status().held == 4);
    // Dropping W before its claimant runs: the claimant's LAUNCH names the tile.
    Descriptor drop;
    drop.kind = DescriptorKind::Release;
    drop.tile = TileRef{"W", 1, 0};
    CHECK(chain.send(drop).status == CompletionStatus::Done);
    CHECK(chain.reserve("gemm1", 5).status == CompletionStatus::Done);
    CHECK(chain.launch("gemm1").status == CompletionStatus::Done);
    CHECK(chain.reserve("gemm2", 3).status == CompletionStatus::Done);
    const Completion lost = chain.launch("gemm2");
    CHECK(lost.cause == RefusalCause::Unsupported);
    CHECK(lost.diagnosis.find("W[1,0]") != std::string::npos);
}

// ---- DoD 3: the freedom ---------------------------------------------------------------
// Reserve-then-launch reserves gemm1 BEFORE gemm0 launches -- an order program-order
// acquisition forbids, since gemm0 is an earlier, unfired operator still needing slots. Whether
// gemm1 fits beside what gemm0 will leave is then decided while gemm0 runs.
TEST_CASE("a reservation order program-order acquisition forbids completes, where it is sound",
          "[program][orchestration][reserve]") {
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions order, rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;

    const std::uint32_t cap = 12;                    // 7 + 5: both reservations fit
    const Run a = run(l, order, cap);
    const Run b = run(l, rtl, cap);
    REQUIRE_FALSE(a.result.refused);
    REQUIRE_FALSE(b.result.refused);

    const std::size_t launch0 = first_issued(b.result.trace, DescriptorKind::Launch, "gemm0");
    const std::size_t reserve1 = first_issued(b.result.trace, DescriptorKind::Reserve, "gemm1");
    REQUIRE(launch0 != static_cast<std::size_t>(-1));
    CHECK(reserve1 < launch0);
    // ...and the program-order run does not.
    CHECK(first_issued(a.result.trace, DescriptorKind::Reserve, "gemm1") >
          first_issued(a.result.trace, DescriptorKind::Launch, "gemm0"));

    // Same answer, and the same DMA traffic: admission says when a program runs, never what it
    // moves.
    CHECK(bit_identical(a.store.values("Y"), b.store.values("Y")));
    CHECK(bit_identical(a.store.values("H"), b.store.values("H")));
    CHECK(a.result.dma_transfers() == b.result.dma_transfers());
    REQUIRE(b.result.per_operator.size() == 3);
}

TEST_CASE("a refused reservation is a decision point, not a stall",
          "[program][orchestration][reserve]") {
    // One slot short of both reservations: gemm1's early RESERVE is refused, the orchestrator
    // runs gemm0 alone, and gemm1 reserves on its own turn -- beside the four W tiles gemm0
    // retained (4 + 5 = 9 of 11).
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;
    const Run r = run(l, rtl, 11);
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
}

// ---- the claim the scheme rests on ----------------------------------------------------
TEST_CASE("a granted reservation completes, at every capacity",
          "[program][orchestration][reserve]") {
    // The deadlock-freedom argument in kpu_device.hpp leans on one premise: the reservation
    // bound (the program's L3, less what it inherits) is SUFFICIENT for its run. So sweep
    // capacity and assert the only refusals are at a RESERVE -- the decision point -- and never
    // from a LAUNCH, which would mean a granted reservation ended in an in-run refusal.
    std::vector<Loadable> loadables;
    for (bool warm : {false, true}) {
        loadables.push_back(two_gemms_sharing_weights(warm));
        loadables.push_back(three_gemms_with_a_gap(warm));
    }
    loadables.push_back(relu_then_bias());
    for (const Loadable& l : loadables)
        for (AllocationPolicy policy :
             {AllocationPolicy::ProgramOrder, AllocationPolicy::ReserveThenLaunch}) {
            bool completed_before = false;
            for (std::uint32_t cap = 1; cap <= 24; ++cap) {
                INFO(l.name << " policy " << to_string(policy) << " L3 " << cap);
                OrchestratorOptions opt;
                opt.policy = policy;
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
    // Cold, the largest program L3 of the chain (5). Warm, gemm1's 5 beside the four W tiles
    // held across it: holding W costs exactly its four tiles.
    auto min_cap = [&](const Loadable& l) {
        for (std::uint32_t cap = 1; cap <= 40; ++cap)
            if (!run(l, {}, cap).result.refused) return cap;
        return 0u;
    };
    CHECK(min_cap(three_gemms_with_a_gap(false)) == 5);
    CHECK(min_cap(three_gemms_with_a_gap(true)) == 9);
    // H = relu(X W) held between its producer and its consumer: the producer's 16 is the most.
    CHECK(min_cap(relu_then_bias()) == 16);
}

TEST_CASE("reserve-then-launch is deterministic", "[program][orchestration][reserve]") {
    const Loadable l = three_gemms_with_a_gap();
    OrchestratorOptions rtl;
    rtl.policy = AllocationPolicy::ReserveThenLaunch;
    for (std::uint32_t cap : {0u, 11u, 12u}) {
        INFO("L3 " << cap);
        const Run a = run(l, rtl, cap);
        const Run b = run(l, rtl, cap);
        CHECK(a.result.trace.digest() == b.result.trace.digest());
        CHECK(a.result.trace.issued.size() == a.result.trace.completions.size());
    }
}

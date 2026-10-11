// ============================================================================
// tests/program/test_orchestration_mmio.cpp
// The call ABI as MMIO (#305 increment 3), against two of its definition-of-done clauses:
//
//   1. identical values AND identical residency behaviour through MMIO as through direct
//      calls -- one orchestrator, one machine, two transports, compared byte for byte;
//   2. no descriptor and no status read carries payload -- asserted on the WIRE (record
//      layouts) and on the RUN (tensor DRAM faults; a run's bus log does not depend on
//      tensor values).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "orchestration_fixtures.hpp"

#include <sw/kpu/orchestration/mmio.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>

#include <memory>

using namespace orchestration_fixtures;
namespace abi = sw::kpu::orchestration::abi;

namespace {

struct Run {
    OrchestrationResult result;
    TensorStore store;
};

Run run(const Loadable& l, OrchestratorOptions opt, std::uint32_t l3_tiles = 0,
        float input_offset = 0.0f) {
    Run r;
    for (const TensorRef& t : l.tensors) r.store.declare(t);
    fill_inputs(r.store);
    // A uniform offset changes every input VALUE and no shape -- the lever the
    // non-interference test pulls.
    if (input_offset != 0.0f)
        for (const char* n : {"X", "W", "V"})
            if (r.store.has(n))
                for (float& v : r.store.values(n)) v += input_offset;
    VirtualPlatform platform = fresh(l3_tiles);
    r.result = orchestrate(l, platform, r.store, opt);
    return r;
}

std::size_t peak_of(const sw::kpu::program::driver::CspLevelOutcome& o) { return o.peak_l3; }

} // namespace

TEST_CASE("the MMIO transport agrees with the direct one, byte for byte",
          "[program][orchestration][mmio]") {
    // DoD 1. Both transports drive the SAME KpuDevice with the SAME orchestrator, so every
    // difference here is a transport bug by definition -- an encoding that lost a field, a
    // ring that reordered, a diagnosis that did not survive the trip.
    const std::vector<std::pair<const char*, Loadable>> loadables = {
        {"two-gemms", two_gemms_sharing_weights()}, {"three-gemms", three_gemms_with_a_gap()},
        {"relu-then-bias", relu_then_bias()}};
    const char* outputs[] = {"H", "Y"};
    auto has = [](const Loadable& l, const char* t) {
        for (const TensorRef& r : l.tensors)
            if (r.name == t) return true;
        return false;
    };

    for (const auto& [label, l] : loadables)
        for (ExecutionLevel level :
             {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential})
            for (AllocationPolicy policy :
                 {AllocationPolicy::ProgramOrder, AllocationPolicy::ReserveThenLaunch})
                for (std::uint32_t cap : {0u, 16u}) {
                    INFO(label << " level " << sw::kpu::program::driver::short_name(level)
                               << " policy " << to_string(policy) << " L3 " << cap);
                    OrchestratorOptions direct, mmio;
                    direct.level = mmio.level = level;
                    direct.policy = mmio.policy = policy;
                    mmio.transport = Transport::Mmio;
                    const Run d = run(l, direct, cap);
                    const Run m = run(l, mmio, cap);

                    REQUIRE_FALSE(d.result.refused);
                    REQUIRE_FALSE(m.result.refused);
                    // The decisions and their completions, as recorded.
                    CHECK(d.result.trace.canonical_bytes() == m.result.trace.canonical_bytes());
                    // The values.
                    for (const char* out : outputs)
                        if (has(l, out)) CHECK(bit_identical(d.store.values(out), m.store.values(out)));
                    // The residency behaviour: what was fetched, and how full L3 got.
                    CHECK(d.result.dma_transfers() == m.result.dma_transfers());
                    REQUIRE(d.result.per_operator.size() == m.result.per_operator.size());
                    for (std::size_t i = 0; i < d.result.per_operator.size(); ++i)
                        CHECK(peak_of(d.result.per_operator[i]) ==
                              peak_of(m.result.per_operator[i]));
                    // And the MMIO run really went over the bus.
                    CHECK(m.result.bus_accesses > 0);
                    CHECK(d.result.bus_accesses == 0);
                }
}

TEST_CASE("an MMIO run computes what the in-process path computes",
          "[program][orchestration][mmio]") {
    // Increment 2's first DoD clause, repeated through the ABI: values are level-invariant
    // and transport-invariant, so disagreement with a hand-driven run_at is a bug signal.
    for (ExecutionLevel level : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}) {
        const Loadable l = two_gemms_sharing_weights();
        OrchestratorOptions opt;
        opt.level = level;
        opt.transport = Transport::Mmio;
        const Run m = run(l, opt);
        REQUIRE_FALSE(m.result.refused);

        namespace lang = sw::kpu::program::csp::lang;
        const lang::Program ast = lang::parse(l.operators[0].csp_program);
        TileProgram inputs = sw::kpu::program::driver::csp_inputs(ast);
        inputs.operand("A").values = m.store.values("X");
        inputs.operand("B").values = m.store.values("W");
        VirtualPlatform platform = fresh();
        const auto direct = platform.run_csp(platform.load_csp(ast, inputs), level).outcome;
        CHECK(bit_identical(direct.values.operand("C").values, m.store.values("H")));
    }
}

TEST_CASE("a refusal crosses the ABI with its cause, its arithmetic and its words",
          "[program][orchestration][mmio][refusal]") {
    // The diagnosis is written by the device into its DIAG area and read back by the port; the
    // numbers ride in the completion record. A refusal that arrived as "refused" alone would
    // send the reader nowhere.
    const Loadable l = two_gemms_sharing_weights();
    OrchestratorOptions direct, mmio;
    mmio.transport = Transport::Mmio;
    const Run d = run(l, direct, 2);
    const Run m = run(l, mmio, 2);
    REQUIRE(d.result.refused);
    REQUIRE(m.result.refused);
    CHECK(m.result.diagnosis == d.result.diagnosis);
    CHECK(m.result.diagnosis.find("gemm0") != std::string::npos);
    CHECK(m.result.trace.canonical_bytes() == d.result.trace.canonical_bytes());

    const Completion& last = m.result.trace.completions.back();
    CHECK(last.status == CompletionStatus::RefusedInsufficientCredit);
    CHECK(last.cause == RefusalCause::InsufficientCredit);
    CHECK(last.capacity == 2);
    CHECK(last.available == 2);
    CHECK(last.needed > last.available);
}

TEST_CASE("tensor DRAM is a fault on the orchestrator's bus", "[program][orchestration][mmio]") {
    // DoD 2, ISOLATION. The property is an absence -- the orchestrator cannot read a tensor --
    // so the test has to try. Every tensor's DRAM range is mapped, and mapped as a trap.
    const Loadable l = two_gemms_sharing_weights();
    TensorStore store;
    VirtualPlatform platform = fresh();
    KpuDevice device(l, platform, store, ExecutionLevel::BlockSequential);
    MmioSystem system(device);

    // The fault must NAME the tensor: an address that is merely unmapped faults too, and a
    // test that accepted either would pass with the tensor mapping deleted -- which it did,
    // before this check.
    using Catch::Matchers::ContainsSubstring;
    for (const TensorRef& t : l.tensors) {
        INFO("tensor " << t.name);
        const std::string named = "tensor DRAM \"" + t.name + "\"";
        CHECK_THROWS_WITH(system.bus().read64(t.device_address), ContainsSubstring(named));
        CHECK_THROWS_WITH(system.bus().read64(t.device_address + t.size_bytes - 8),
                          ContainsSubstring(named));
        CHECK_THROWS_WITH(system.bus().write64(t.device_address, 0), ContainsSubstring(named));
    }
    // Nothing mapped is a fault too, rather than a zero that reads like data.
    CHECK_THROWS_AS(system.bus().read64(0x10'0000'0000ull), BusFault);
    // The control window is ordinary: the same bus does work.
    CHECK(system.bus().read64(MmioSystem::kMmioBase + abi::reg::ID) == abi::kMagic);
}

TEST_CASE("what the orchestrator observes does not depend on tensor values",
          "[program][orchestration][mmio]") {
    // DoD 2, NON-INTERFERENCE: the general form of "no status read carries payload". Two runs
    // whose inputs differ in every element produce the same bus log -- every address, every
    // direction, every value the orchestrator read or wrote. If any register, ring entry or
    // diagnosis reflected tensor contents, the logs would differ. This covers registers that
    // do not exist yet, which a review of today's register map cannot.
    for (AllocationPolicy policy :
         {AllocationPolicy::ProgramOrder, AllocationPolicy::ReserveThenLaunch}) {
        INFO("policy " << to_string(policy));
        const Loadable l = three_gemms_with_a_gap();
        OrchestratorOptions opt;
        opt.transport = Transport::Mmio;
        opt.policy = policy;
        const Run a = run(l, opt, 0, 0.0f);
        const Run b = run(l, opt, 0, 0.5f);
        REQUIRE_FALSE(a.result.refused);
        REQUIRE_FALSE(b.result.refused);

        // The inputs really differed: the outputs do.
        CHECK_FALSE(bit_identical(a.store.values("Y"), b.store.values("Y")));
        // ...and the orchestrator could not tell.
        CHECK(a.result.bus_accesses == b.result.bus_accesses);
        CHECK(a.result.bus_log_digest == b.result.bus_log_digest);
    }
}

TEST_CASE("the wire records are fixed, and hold indices rather than names",
          "[program][orchestration][mmio][abi]") {
    // DoD 2 on the WIRE. A size change is a compile error here, where it has to be justified.
    static_assert(sizeof(abi::DescriptorRecord) == 64);
    static_assert(sizeof(abi::CompletionRecord) == 64);

    const Loadable l = three_gemms_with_a_gap();
    VirtualPlatform platform = fresh();
    const abi::NameTable names(l, platform.deployment());

    SECTION("every descriptor kind round-trips") {
        Descriptor d;
        d.id = 0x1122334455667788ull;
        d.tile = TileRef{"W", 1, 0};
        d.leg = sw::kpu::program::Hop::DmaDramToL3;
        d.resource.device = platform.deployment().device(0).name;
        d.resource.kind = sw::kpu::program::platform::ResourceKind::L3Tile;
        d.resource.path = {0};
        d.target = "gemm2";
        d.slots = 10;
        d.flags = kReleaseAtLastRead;
        d.wait_for = 7;
        for (DescriptorKind k : {DescriptorKind::Place, DescriptorKind::Release,
                                 DescriptorKind::Launch, DescriptorKind::Fence,
                                 DescriptorKind::Reserve}) {
            d.kind = k;
            const Descriptor back = abi::decode(abi::encode(d, names), names);
            CHECK(back.str() == d.str());
            CHECK(back.slots == d.slots);
            CHECK(back.flags == d.flags);
            CHECK(back.wait_for == d.wait_for);
            CHECK(back.resource == d.resource);
        }
        // A name the tables do not hold cannot be encoded -- there is no string field to
        // smuggle it in.
        d.target = "not-an-operator";
        CHECK_THROWS(abi::encode(d, names));
    }

    SECTION("a completion releasing several tiles spans records, and decodes whole") {
        Completion c;
        c.descriptor_id = 42;
        c.released = {TileRef{"W", 0, 0}, TileRef{"W", 0, 1}, TileRef{"V", 1, 1}};
        const auto records = abi::encode(c, names, 0, 0);
        REQUIRE(records.size() == 3);
        bool more = false;
        std::uint32_t off = 0, len = 0;
        Completion back = abi::decode(records[0], names, more, off, len);
        for (std::size_t i = 1; i < records.size(); ++i) {
            REQUIRE(more);
            abi::decode_continuation(records[i], names, back, more);
        }
        CHECK_FALSE(more);
        CHECK(back.str() == c.str());
    }
}

TEST_CASE("the status surface reports placement through MMIO as it does directly",
          "[program][orchestration][mmio]") {
    // Residency, credits and inventory are METADATA the orchestrator may read (§6.4). Driven
    // by hand so the state is known: gemm0 reserves, a PLACE is refused (its program loads its
    // own tiles), and the launch leaves the four W tiles its program retains.
    const Loadable l = three_gemms_with_a_gap();
    auto drive = [&](KpuPort& port) {
        Descriptor r;
        r.id = 1;
        r.kind = DescriptorKind::Reserve;
        r.target = "gemm0";
        r.slots = 7;
        port.submit(r);
        Descriptor p;
        p.id = 2;
        p.kind = DescriptorKind::Place;
        p.tile = TileRef{"W", 0, 0};
        p.target = "gemm0";
        port.submit(p);
        Descriptor launch;
        launch.id = 3;
        launch.kind = DescriptorKind::Launch;
        launch.target = "gemm0";
        port.submit(launch);
        std::vector<std::string> seen;
        Completion c;
        while (port.poll_completion(c)) seen.push_back(c.str());
        return seen;
    };

    TensorStore sd, sm;
    VirtualPlatform pd = fresh(16), pm = fresh(16);
    KpuDevice dd(l, pd, sd, ExecutionLevel::BlockSequential);
    KpuDevice dm(l, pm, sm, ExecutionLevel::BlockSequential);
    DirectPort direct(dd);
    MmioSystem system(dm);
    KpuPort& mmio = system.port();

    CHECK(drive(direct) == drive(mmio));
    for (KpuPort* port : {static_cast<KpuPort*>(&direct), &mmio}) {
        CHECK(port->is_resident(TileRef{"W", 0, 0}));       // retained by the program
        CHECK(port->is_resident(TileRef{"W", 1, 1}));
        CHECK_FALSE(port->is_resident(TileRef{"X", 0, 0})); // read, released
        const StatusSnapshot s = port->read_status();
        CHECK(s.held == 4);
        CHECK(s.reserved == 0);                             // gemm0's returned at completion
        CHECK(s.completed == 1);
        CHECK(s.l3_capacity == 16);
        CHECK(s.credits_free() == 12);
        // The manifests cross the bus whole: the lists a decider plans admission with.
        for (std::uint32_t op = 0; op < 3; ++op) {
            const OperatorManifest a = direct.manifest(op), b = mmio.manifest(op);
            CHECK(a.l3_slots == b.l3_slots);
            CHECK(a.reads == b.reads);
            CHECK(a.inherits == b.inherits);
            CHECK(a.retains == b.retains);
        }
        CHECK(port->manifest(2).inherits.size() == 4);
    }
    CHECK(direct.inventory() == mmio.inventory());
    CHECK_FALSE(mmio.inventory().empty());
}

TEST_CASE("a descriptor kind outside the vocabulary is refused, not dropped",
          "[program][orchestration][mmio]") {
    // Every descriptor gets exactly one completion. A kind byte the device does not know --
    // possible on the wire -- must come back as a refusal naming it, not as silence that the
    // orchestrator would misreport as "no completion".
    const Loadable l = two_gemms_sharing_weights();
    Descriptor bad;
    bad.id = 7;
    bad.kind = static_cast<DescriptorKind>(0x7F);

    TensorStore sd, sm;
    VirtualPlatform pd = fresh(), pm = fresh();
    KpuDevice dd(l, pd, sd, ExecutionLevel::BlockSequential);
    KpuDevice dm(l, pm, sm, ExecutionLevel::BlockSequential);
    DirectPort direct(dd);
    MmioSystem system(dm);
    for (KpuPort* port : {static_cast<KpuPort*>(&direct), &system.port()}) {
        port->submit(bad);
        Completion c;
        REQUIRE(port->poll_completion(c));
        CHECK(c.descriptor_id == 7);
        CHECK(c.status == CompletionStatus::RefusedUnsupported);
        CHECK(c.cause == RefusalCause::Unsupported);
        CHECK(c.diagnosis.find("127") != std::string::npos);
        CHECK_FALSE(port->poll_completion(c));
    }
}

TEST_CASE("a doorbell rung before the completion ring exists waits, rather than crashing",
          "[program][orchestration][mmio]") {
    // Increment 4's guest can ring the doorbell in any order it likes. Completions posted before
    // the completion ring is programmed wait in the device's backlog and appear once it is.
    const Loadable l = two_gemms_sharing_weights();
    TensorStore store;
    VirtualPlatform platform = fresh();
    KpuDevice device(l, platform, store, ExecutionLevel::BlockSequential);
    const abi::NameTable names(l, platform.deployment());
    const std::uint64_t base = MmioSystem::kCtrlBase;
    std::vector<std::uint8_t> ctrl(0x4000, 0);
    KpuMmioDevice regs(device, ctrl, base, names);

    regs.write_reg(abi::reg::DRING_BASE, base);
    regs.write_reg(abi::reg::DRING_SIZE, 4);
    Descriptor fence;
    fence.id = 1;
    fence.kind = DescriptorKind::Fence;
    const abi::DescriptorRecord rec = abi::encode(fence, names);
    std::copy(rec.begin(), rec.end(), ctrl.begin());
    CHECK_NOTHROW(regs.write_reg(abi::reg::DRING_TAIL, 1));        // no completion ring yet
    CHECK(regs.read_reg(abi::reg::IRQ_STATUS) == 0);

    // A REFUSAL rung early too: its diagnosis must survive the wait, not be dropped because
    // the DIAG area did not exist yet when it was posted.
    Descriptor configure;
    configure.id = 2;
    configure.kind = DescriptorKind::Configure;
    const abi::DescriptorRecord rec2 = abi::encode(configure, names);
    std::copy(rec2.begin(), rec2.end(), ctrl.begin() + abi::kDescriptorBytes);
    CHECK_NOTHROW(regs.write_reg(abi::reg::DRING_TAIL, 2));

    // Size BEFORE base -- the ABI does not order them -- must not post into address 0.
    CHECK_NOTHROW(regs.write_reg(abi::reg::CRING_SIZE, 4));
    CHECK(regs.read_reg(abi::reg::IRQ_STATUS) == 0);
    regs.write_reg(abi::reg::CRING_BASE, base + 0x1000);             // now it exists
    CHECK(regs.read_reg(abi::reg::CRING_HEAD) == 1);                 // the fence; the refusal waits for DIAG
    regs.write_reg(abi::reg::DIAG_BASE, base + 0x2000);
    regs.write_reg(abi::reg::DIAG_SIZE, 0x1000);
    CHECK(regs.read_reg(abi::reg::CRING_HEAD) == 2);
    CHECK(regs.read_reg(abi::reg::IRQ_STATUS) == 1);

    abi::CompletionRecord out{};
    std::copy(ctrl.begin() + 0x1000 + abi::kCompletionBytes,
              ctrl.begin() + 0x1000 + 2 * abi::kCompletionBytes, out.begin());
    bool more = false;
    std::uint32_t off = 0, len = 0;
    const Completion refused = abi::decode(out, names, more, off, len);
    CHECK(refused.descriptor_id == 2);
    CHECK(refused.status == CompletionStatus::RefusedUnsupported);
    REQUIRE(len > 0);
    const std::string text(reinterpret_cast<const char*>(ctrl.data()) + 0x2000 + off, len);
    CHECK(text.find("CONFIGURE") != std::string::npos);
}

TEST_CASE("aliased tensors still map, and still fault", "[program][orchestration][mmio]") {
    // A loadable may alias two tensors (an in-place operator). Overlapping FAULT regions are
    // allowed; the address traps either way. Overlap with RAM or MMIO stays a refusal.
    Loadable l = two_gemms_sharing_weights();
    l.tensors[3].device_address = l.tensors[2].device_address + 0x100;   // Y overlaps H
    TensorStore store;
    VirtualPlatform platform = fresh();
    KpuDevice device(l, platform, store, ExecutionLevel::BlockSequential);
    std::unique_ptr<MmioSystem> system;
    REQUIRE_NOTHROW(system = std::make_unique<MmioSystem>(device));
    CHECK_THROWS_AS(system->bus().read64(l.tensors[3].device_address), BusFault);

    Loadable clash = two_gemms_sharing_weights();
    clash.tensors[0].device_address = MmioSystem::kCtrlBase;            // over control memory
    TensorStore store2;
    VirtualPlatform platform2 = fresh();
    KpuDevice device2(clash, platform2, store2, ExecutionLevel::BlockSequential);
    CHECK_THROWS_AS(MmioSystem(device2), std::invalid_argument);
}

// ============================================================================
// tests/timing/test_memory_side_harness.cpp
// Memory-side debugger step 2: DRAM, controllers, DMA engines and store buffers on their own,
// the NoC ports stubbed by request models and sinks (docs/plans/memory-side-debugger.md §3.2).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/memory_flow_record.hpp>
#include <sw/kpu/timing/memory_side_harness.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::memside;
using sw::kpu::program::platform::DeviceSpecification;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;
using Kind = RequestModel::Kind;

namespace {

DeviceSpecification t4(unsigned engines = 8) {
    DeviceSpecification d = read_spec_file("tests/program/deploy/kpu_t4.json").device(0);
    d.dma.engines = engines;
    d.dma.window = 32;
    return d;
}

RequestModel stream(std::uint32_t n, bool load = true, std::uint64_t base = 0x100000) {
    RequestModel m;
    m.kind = Kind::Stream;
    m.is_load = load;
    m.base = base;
    m.count = n;
    m.bytes = 4096;
    return m;
}

// The ceiling of the T4's controller in bytes per executor cycle.
double ceiling(const DeviceSpecification& d) {
    const auto& m = *d.memory.dram;
    return m.channels * (m.channel_width_bits / 8.0) * m.data_rate_mtps * 1e6 / 1e9;
}

}  // namespace

TEST_CASE("Request models generate the documented addresses", "[timing][memside][models]") {
    RequestModel s = stream(3);
    CHECK(s.generate() == std::vector<std::pair<std::uint64_t, std::uint32_t>>{
                              {0x100000, 4096}, {0x101000, 4096}, {0x102000, 4096}});

    RequestModel st;
    st.kind = Kind::Strided;
    st.base = 0;
    st.count = 3;
    st.bytes = 256;
    st.stride = 8192;
    CHECK(st.generate() == std::vector<std::pair<std::uint64_t, std::uint32_t>>{
                               {0, 256}, {8192, 256}, {16384, 256}});

    RequestModel r;
    r.kind = Kind::Random;
    r.base = 0x10000;
    r.region = 1 << 20;
    r.count = 100;
    r.bytes = 4096;
    r.seed = 7;
    const auto a = r.generate();
    CHECK(a == r.generate());                           // deterministic
    for (const auto& [addr, n] : a) {
        CHECK(addr >= 0x10000);
        CHECK(addr + n <= 0x10000 + (1u << 20));
        CHECK(addr % 4096 == 0);
    }

    // A 64 x 64 fp32 matrix with a 512-byte pitch, cut into 32 x 32 tiles: each tile row is a
    // 128-byte request, rows a pitch apart.
    RequestModel mt;
    mt.kind = Kind::MatrixTiles;
    mt.base = 0;
    mt.rows = mt.cols = 64;
    mt.tile_rows = mt.tile_cols = 32;
    mt.pitch = 512;
    const auto segs = mt.generate();
    REQUIRE(segs.size() == 4 * 32);
    CHECK(segs[0] == std::make_pair<std::uint64_t, std::uint32_t>(0, 128));
    CHECK(segs[1] == std::make_pair<std::uint64_t, std::uint32_t>(512, 128));
    CHECK(segs[32] == std::make_pair<std::uint64_t, std::uint32_t>(128, 128));   // tile (0, 1)
    CHECK(segs[64] == std::make_pair<std::uint64_t, std::uint32_t>(32 * 512, 128));   // tile (1, 0)

    // Replay: the LOAD and STORE tiles of a real schedule.
    schedule::MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = 64;
    g.Ti = g.Tj = g.Tk = 32;
    const auto sched = schedule::MatMulScheduleGenerator(g).generate();
    REQUIRE(sched.valid);
    CHECK(RequestModel::replay_of(sched, true).generate().size() ==
          sched.count_ops(schedule::ScheduleOpType::LOAD));
    CHECK(RequestModel::replay_of(sched, false).generate().size() ==
          sched.count_ops(schedule::ScheduleOpType::STORE));
}

TEST_CASE("Harness: every load lands and every ejection is written; every credit returns",
          "[timing][memside]") {
    const DeviceSpecification d = t4();
    MemorySideHarness::Config c;
    MemorySideHarness probe(d, c);
    const unsigned port = probe.port_of(0);
    c.streams = {{port, stream(64), 0}, {port, stream(32, false, 0x800000), 0}};
    MemorySideHarness h(d, c);
    REQUIRE(h.run());

    std::size_t loads = 0, stores = 0;
    for (const auto& r : h.requests()) {
        CHECK(r.done);
        (r.is_load ? loads : stores)++;
        // A request's life in order.
        CHECK(r.offered <= r.credit);
        CHECK(r.credit <= r.first_burst);
        CHECK(r.first_burst <= r.last_burst);
        CHECK(r.last_burst <= r.retired);
    }
    CHECK(loads == 64);
    CHECK(stores == 32);
    CHECK(h.bytes_moved() == 96u * 4096);
    CHECK(h.l3_credits().available() == h.l3_credits().capacity());
    for (std::size_t e = 0; e < h.engines(); ++e) CHECK(h.engine(e).store_buffer().held() == 0);

    // Every burst the controllers saw finished, and there are exactly as many as the bytes span.
    std::size_t bursts = 0;
    for (const auto& mc : h.controllers())
        for (const auto& b : mc->recorded_bursts()) {
            CHECK(b.finished);
            ++bursts;
        }
    CHECK(bursts == 96u * 4096 / 64);
}

TEST_CASE("Harness: a slow injection bus paces the loads and refuses the engines",
          "[timing][memside][backpressure]") {
    const DeviceSpecification d = t4();
    MemorySideHarness::Config c;
    MemorySideHarness probe(d, c);
    c.streams = {{probe.port_of(0), stream(32), 0}};

    c.ports.block_cycles = 400;                         // one block per 400 cycles
    MemorySideHarness slow(d, c);
    REQUIRE(slow.run());
    CHECK(slow.now() >= 32 * 400);                      // the bus, not DRAM, sets the pace
    std::size_t refusals = 0;
    for (const auto& e : slow.port_events()) refusals += e.kind == MemorySideHarness::PortEvent::Kind::Refuse;
    CHECK(refusals > 0);

    // Without the bus, DRAM sets the pace: the same loads run at the controller's ceiling.
    c.ports.infinite = true;
    MemorySideHarness fast(d, c);
    REQUIRE(fast.run());
    CHECK(fast.now() * 2 < slow.now());
    CHECK(static_cast<double>(fast.bytes_moved()) / static_cast<double>(fast.now()) >= 0.85 * ceiling(d));
}

TEST_CASE("Harness: the step-3 table reproduces -- one engine at W = 32 saturates, 32 at W = 1 do not",
          "[timing][memside][window]") {
    auto run = [](unsigned engines, std::size_t window) {
        const DeviceSpecification d = t4(engines);
        MemorySideHarness::Config c;
        c.window = window;
        c.l3_slots = 4096;
        c.ports.infinite = true;
        MemorySideHarness probe(d, c);
        c.streams = {{probe.port_of(0), stream(128), 0}};
        MemorySideHarness h(d, c);
        REQUIRE(h.run());
        return static_cast<double>(h.bytes_moved()) / static_cast<double>(h.now());
    };
    const double peak = ceiling(t4());
    const double one = run(1, 32), many = run(32, 1);
    CAPTURE(peak, one, many);
    CHECK(one >= 0.90 * peak);
    CHECK(many < 0.80 * peak);
}

TEST_CASE("Harness: a device it cannot run is refused by name", "[timing][memside]") {
    DeviceSpecification d = t4();
    d.dma.window.reset();
    CHECK_THROWS_WITH(MemorySideHarness(d, {}), ContainsSubstring("no burst window"));
    d = t4();
    d.memory.dram.reset();
    d.dma.window.reset();
    CHECK_THROWS_WITH(MemorySideHarness(d, {}), ContainsSubstring("memory.dram"));
    MemorySideHarness::Config c;
    c.streams = {{99, stream(1), 0}};
    CHECK_THROWS_WITH(MemorySideHarness(t4(), c), ContainsSubstring("port 99 has no attached DMA engine"));
}

TEST_CASE("Harness: requests are offered when due, whichever stream they came from",
          "[timing][memside]") {
    // One engine, two streams: one paced at 1000 cycles, one all at once. The second stream's
    // requests are due at cycle 0 and must not wait behind the first stream's later ones.
    const DeviceSpecification d = t4(1);
    MemorySideHarness::Config c;
    c.ports.infinite = true;
    MemorySideHarness probe(d, c);
    const unsigned port = probe.port_of(0);
    c.streams = {{port, stream(3, true, 0x100000), 1000}, {port, stream(3, true, 0x200000), 0}};
    MemorySideHarness h(d, c);
    REQUIRE(h.run());
    for (const auto& r : h.requests()) {
        CAPTURE(r.id, r.address);
        if (r.address >= 0x200000) CHECK(r.offered == 0);
        else CHECK(r.offered == 1000 * ((r.address - 0x100000) / 4096));
    }
}

// ============================================================================
// Step 3: the run as a .mflow record
// ============================================================================
TEST_CASE("Record: a harness run round-trips through .mflow, and every row names what it should",
          "[timing][memside][record]") {
    namespace rec = sw::kpu::program::record;
    const DeviceSpecification d = t4();
    MemorySideHarness::Config c;
    MemorySideHarness probe(d, c);
    const unsigned port = probe.port_of(0);
    c.streams = {{port, stream(32), 0}, {port, stream(16, false, 0x800000), 0}};
    MemorySideHarness h(d, c);
    REQUIRE(h.run());
    const rec::MemoryFlowRecord r = to_record(h, d.name);

    // Stations: 2 channels x 16 banks, 2 buses, 8 engines, 8 store buffers, 1 port.
    std::map<std::string, std::size_t> kinds;
    for (const auto& s : r.stations) ++kinds[s.kind];
    CHECK(kinds["dram_bank"] == 32);
    CHECK(kinds["dram_bus"] == 2);
    CHECK(kinds["dma"] == 8);
    CHECK(kinds["dmabuf"] == 8);
    CHECK(kinds["port"] == 1);

    // Integrity: commands name bursts (a refresh names none), bursts name requests, and each
    // request's bursts are exactly the ones its bytes span.
    std::vector<std::size_t> per_request(r.requests.size(), 0);
    for (const auto& b : r.bursts) {
        REQUIRE(b.request < r.requests.size());
        ++per_request[b.request];
        CHECK(b.engine == r.requests[b.request].engine);
        CHECK(b.is_load == r.requests[b.request].is_load);
    }
    for (std::size_t i = 0; i < r.requests.size(); ++i) CHECK(per_request[i] == r.requests[i].bytes / 64);
    for (const auto& cmd : r.commands) {
        if (cmd.kind == rec::MemoryFlowRecord::CommandKind::Refresh) {
            CHECK(cmd.burst == rec::kNone);
            continue;
        }
        REQUIRE(cmd.burst < r.bursts.size());
        const auto& b = r.bursts[cmd.burst];
        CHECK((b.mc == cmd.mc && b.channel == cmd.channel && b.bank_group == cmd.bank_group && b.bank == cmd.bank));
        if (cmd.kind != rec::MemoryFlowRecord::CommandKind::Precharge) CHECK(b.row == cmd.row);
    }

    const std::string dir = "memside_record_roundtrip.mflow";
    rec::write_mflow(r, dir);
    const rec::MemoryFlowRecord back = rec::read_mflow(dir);
    CHECK(back.device == r.device);
    CHECK(back.makespan == r.makespan);
    CHECK(back.window == r.window);
    CHECK(back.stations.size() == r.stations.size());
    REQUIRE(back.bursts.size() == r.bursts.size());
    REQUIRE(back.commands.size() == r.commands.size());
    REQUIRE(back.requests.size() == r.requests.size());
    CHECK(back.buffers.size() == r.buffers.size());
    CHECK(back.ports.size() == r.ports.size());
    for (std::size_t i = 0; i < r.bursts.size(); ++i) {
        CHECK(back.bursts[i].row == r.bursts[i].row);
        CHECK(back.bursts[i].outcome == r.bursts[i].outcome);
        CHECK(back.bursts[i].done == r.bursts[i].done);
    }
    for (std::size_t i = 0; i < r.commands.size(); ++i) {
        CHECK(back.commands[i].kind == r.commands[i].kind);
        CHECK(back.commands[i].burst == r.commands[i].burst);
        CHECK(back.commands[i].issue == r.commands[i].issue);
    }
    for (std::size_t i = 0; i < r.requests.size(); ++i) CHECK(back.requests[i].address == r.requests[i].address);
    CHECK(std::filesystem::exists(dir + "/lod.json"));

    // A bundle of another format or version is refused by name.
    {
        std::ofstream m(dir + "/manifest.json");
        m << R"({"format":"kpu-mflow","version":99})";
    }
    CHECK_THROWS_WITH(rec::read_mflow(dir), ContainsSubstring("version 99"));
    {
        std::ofstream m(dir + "/manifest.json");
        m << R"({"format":"kpu-tflow","version":3})";
    }
    CHECK_THROWS_WITH(rec::read_mflow(dir), ContainsSubstring("not a .mflow bundle"));
    std::filesystem::remove_all(dir);
}


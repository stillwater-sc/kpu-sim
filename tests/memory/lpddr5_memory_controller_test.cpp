// ============================================================================
// tests/memory/lpddr5_memory_controller_test.cpp
// Progressive test suite for LPDDR5 Memory Controller
//
// Tests are organized in levels of increasing complexity:
// Level 1: Single bank operations
// Level 2: Two bank operations
// Level 3: Three bank operations
// Level 4: Four banks (full bank group)
// Level 5: Read sequences
// Level 6: Write sequences
// Level 7: Mixed read/write with turnaround
// Level 8: Multi-bank-group concurrency
// Level 9: Full state space exploration
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/components/lpddr5_memory_controller.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/trace_exporter.hpp>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace sw::kpu::lpddr5;

// ============================================================================
// Test Fixture Helpers
// ============================================================================

struct TestContext {
    LPDDR5MemoryController::Config config;
    std::unique_ptr<LPDDR5MemoryController> mc;

    TestContext(uint8_t num_channels = 1, BurstLength bl = BurstLength::BL16) {
        config.num_channels = num_channels;
        config.burst_length = bl;
        config.queue_depth = 64;
        mc = std::make_unique<LPDDR5MemoryController>(config);
    }

    // Run until completion or timeout
    bool run_until_complete(uint64_t max_cycles = 10000) {
        uint64_t start = mc->current_cycle();
        while (mc->has_pending() && (mc->current_cycle() - start) < max_cycles) {
            mc->tick();
            if (mc->has_violations()) {
                return false;
            }
        }
        return !mc->has_pending();
    }

    // Run a specific number of cycles
    void run_cycles(uint64_t cycles) {
        for (uint64_t i = 0; i < cycles; ++i) {
            mc->tick();
        }
    }

    // Generate address for specific bank/row
    uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col = 0) {
        // Address format: [row | bank | col | byte_offset]
        uint64_t addr = 0;
        addr |= static_cast<uint64_t>(row) << (6 + 10 + 4);  // row at top
        addr |= static_cast<uint64_t>(bank) << (6 + 10);      // bank
        addr |= static_cast<uint64_t>(col) << 6;              // column
        return addr;
    }

    // Generate address for dual channel
    uint64_t make_address_dc(uint8_t channel, uint8_t bank, uint32_t row) {
        // [row | bank | col | channel | byte_offset]
        uint64_t addr = 0;
        addr |= static_cast<uint64_t>(row) << (6 + 1 + 10 + 4);
        addr |= static_cast<uint64_t>(bank) << (6 + 1 + 10);
        addr |= static_cast<uint64_t>(channel) << 6;
        return addr;
    }

    // Check for violations and print them
    void check_violations() {
        if (mc->has_violations()) {
            std::cerr << "\n=== INVARIANT VIOLATIONS DETECTED ===" << std::endl;
            for (const auto& v : mc->violations()) {
                std::cerr << "  Cycle " << v.cycle
                          << ": [" << v.invariant_id << "] "
                          << v.message
                          << " (channel=" << (int)v.channel
                          << ", bank=" << (int)v.bank << ")"
                          << std::endl;
            }
            std::cerr << "======================================\n" << std::endl;
        }
        REQUIRE_FALSE(mc->has_violations());
    }

    // Print statistics
    void print_stats() {
        const auto& s = mc->stats();
        std::cout << "\n=== Memory Controller Statistics ===" << std::endl;
        std::cout << "  Reads:          " << s.reads << std::endl;
        std::cout << "  Writes:         " << s.writes << std::endl;
        std::cout << "  Page hits:      " << s.page_hits << std::endl;
        std::cout << "  Page empty:     " << s.page_empty << std::endl;
        std::cout << "  Page conflicts: " << s.page_conflicts << std::endl;
        std::cout << "  Refreshes:      " << s.refreshes << std::endl;
        std::cout << "  Avg latency:    " << std::fixed << std::setprecision(2)
                  << s.avg_latency() << " cycles" << std::endl;
        std::cout << "  Stall cycles:   " << s.stall_cycles << std::endl;
        std::cout << "  R->W turnarounds: " << s.read_to_write_turnarounds << std::endl;
        std::cout << "  W->R turnarounds: " << s.write_to_read_turnarounds << std::endl;
        std::cout << "  Total cycles:   " << mc->current_cycle() << std::endl;
        std::cout << "====================================\n" << std::endl;
    }
};

// ============================================================================
// Level 1: Single Bank Operations
// ============================================================================

TEST_CASE("Level1: Single bank operations", "[lpddr5][level1]") {
    TestContext ctx;

    SECTION("Single read") {
        uint64_t addr = ctx.make_address(0, 100, 0);
        auto id = ctx.mc->submit_read(addr, 64);
        REQUIRE(id.has_value());

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 1);
        CHECK(s.page_empty == 1);  // First access is always page empty
    }

    SECTION("Single write") {
        uint64_t addr = ctx.make_address(0, 100, 0);
        std::vector<uint8_t> data(64, 0xAB);
        auto id = ctx.mc->submit_write(addr, data.data(), 64);
        REQUIRE(id.has_value());

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.writes == 1);
        CHECK(s.page_empty == 1);
    }

    SECTION("Page hit - same row") {
        uint64_t addr1 = ctx.make_address(0, 100, 0);
        uint64_t addr2 = ctx.make_address(0, 100, 64);  // Same row, different column

        ctx.mc->submit_read(addr1, 64);
        REQUIRE(ctx.run_until_complete());

        ctx.mc->submit_read(addr2, 64);
        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 2);
        CHECK(s.page_empty == 1);
        CHECK(s.page_hits == 1);
    }

    SECTION("Page conflict - different row") {
        uint64_t addr1 = ctx.make_address(0, 100, 0);
        uint64_t addr2 = ctx.make_address(0, 200, 0);  // Different row

        ctx.mc->submit_read(addr1, 64);
        REQUIRE(ctx.run_until_complete());

        ctx.mc->submit_read(addr2, 64);
        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 2);
        CHECK(s.page_empty == 1);
        CHECK(s.page_conflicts == 1);
    }

    SECTION("Multiple page hits") {
        uint64_t base = ctx.make_address(0, 100, 0);

        for (int i = 0; i < 10; ++i) {
            ctx.mc->submit_read(base + i * 64, 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 10);
        CHECK(s.page_empty == 1);
        CHECK(s.page_hits == 9);
    }
}

// ============================================================================
// Level 2: Two Bank Operations
// ============================================================================

TEST_CASE("Level2: Two bank operations", "[lpddr5][level2]") {
    TestContext ctx;

    SECTION("Sequential access to two banks") {
        uint64_t addr0 = ctx.make_address(0, 100, 0);
        uint64_t addr1 = ctx.make_address(1, 100, 0);

        ctx.mc->submit_read(addr0, 64);
        ctx.mc->submit_read(addr1, 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 2);
        CHECK(s.page_empty == 2);
    }

    SECTION("Two banks same group (tRRD_L)") {
        // Banks 0 and 1 are in the same bank group
        uint64_t addr0 = ctx.make_address(0, 100, 0);
        uint64_t addr1 = ctx.make_address(1, 100, 0);

        ctx.mc->submit_read(addr0, 64);
        ctx.mc->submit_read(addr1, 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();  // Should respect tRRD_L
    }

    SECTION("Two banks different groups (tRRD_S)") {
        // Banks 0 and 4 are in different bank groups
        uint64_t addr0 = ctx.make_address(0, 100, 0);
        uint64_t addr4 = ctx.make_address(4, 100, 0);

        ctx.mc->submit_read(addr0, 64);
        ctx.mc->submit_read(addr4, 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();  // Should use shorter tRRD_S
    }

    SECTION("Interleaved accesses") {
        for (int i = 0; i < 5; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, 100, i * 64), 64);
            ctx.mc->submit_read(ctx.make_address(1, 100, i * 64), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 10);
    }
}

// ============================================================================
// Level 3: Three Bank Operations
// ============================================================================

TEST_CASE("Level3: Three bank operations", "[lpddr5][level3]") {
    TestContext ctx;

    SECTION("Three banks sequential") {
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);
        ctx.mc->submit_read(ctx.make_address(1, 100, 0), 64);
        ctx.mc->submit_read(ctx.make_address(2, 100, 0), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 3);
        CHECK(s.page_empty == 3);
    }

    SECTION("Three banks mixed groups") {
        // Banks 0, 4, 8 are in different bank groups
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);
        ctx.mc->submit_read(ctx.make_address(4, 100, 0), 64);
        ctx.mc->submit_read(ctx.make_address(8, 100, 0), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
    }
}

// ============================================================================
// Level 4: Four Bank Operations (Full Bank Group)
// ============================================================================

TEST_CASE("Level4: Four bank operations", "[lpddr5][level4]") {
    TestContext ctx;

    SECTION("Full bank group") {
        // Banks 0, 1, 2, 3 are all in bank group 0
        for (int b = 0; b < 4; ++b) {
            ctx.mc->submit_read(ctx.make_address(b, 100, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 4);
    }

    SECTION("tFAW constraint") {
        // Issue 4 activates, then 5th should wait for tFAW
        for (int b = 0; b < 5; ++b) {
            ctx.mc->submit_read(ctx.make_address(b, 100, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 5);
    }

    SECTION("All bank groups") {
        // One bank from each group
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);   // BG0
        ctx.mc->submit_read(ctx.make_address(4, 100, 0), 64);   // BG1
        ctx.mc->submit_read(ctx.make_address(8, 100, 0), 64);   // BG2
        ctx.mc->submit_read(ctx.make_address(12, 100, 0), 64);  // BG3

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
    }
}

// ============================================================================
// Level 5: Read Sequences
// ============================================================================

TEST_CASE("Level5: Read sequences", "[lpddr5][level5]") {
    TestContext ctx;

    SECTION("Read burst to same row") {
        for (int i = 0; i < 16; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, 100, i * 64), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 16);
        CHECK(s.page_empty == 1);
        CHECK(s.page_hits == 15);
    }

    SECTION("Read strided - different rows") {
        for (int i = 0; i < 8; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, i * 10, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 8);
        CHECK(s.page_empty == 1);
        CHECK(s.page_conflicts == 7);
    }

    SECTION("Read multi-bank burst") {
        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < 4; ++i) {
                ctx.mc->submit_read(ctx.make_address(b, 100, i * 64), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 16);
    }
}

// ============================================================================
// Level 6: Write Sequences
// ============================================================================

TEST_CASE("Level6: Write sequences", "[lpddr5][level6]") {
    TestContext ctx;
    std::vector<uint8_t> data(64, 0xCD);

    SECTION("Write burst to same row") {
        for (int i = 0; i < 16; ++i) {
            ctx.mc->submit_write(ctx.make_address(0, 100, i * 64), data.data(), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.writes == 16);
        CHECK(s.page_empty == 1);
        CHECK(s.page_hits == 15);
    }

    SECTION("Write strided - different rows") {
        for (int i = 0; i < 8; ++i) {
            ctx.mc->submit_write(ctx.make_address(0, i * 10, 0), data.data(), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.writes == 8);
    }
}

// ============================================================================
// Level 7: Mixed Read/Write with Turnaround
// ============================================================================

TEST_CASE("Level7: Mixed read/write with turnaround", "[lpddr5][level7]") {
    TestContext ctx;
    std::vector<uint8_t> data(64, 0x12);

    SECTION("Read then write (tRTW)") {
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);
        ctx.mc->submit_write(ctx.make_address(0, 100, 64), data.data(), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 1);
        CHECK(s.writes == 1);
        CHECK(s.read_to_write_turnarounds == 1);
    }

    SECTION("Write then read (tWTR)") {
        ctx.mc->submit_write(ctx.make_address(0, 100, 0), data.data(), 64);
        ctx.mc->submit_read(ctx.make_address(0, 100, 64), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 1);
        CHECK(s.writes == 1);
        CHECK(s.write_to_read_turnarounds == 1);
    }

    SECTION("Alternating read/write") {
        for (int i = 0; i < 8; ++i) {
            if (i % 2 == 0) {
                ctx.mc->submit_read(ctx.make_address(0, 100, i * 64), 64);
            } else {
                ctx.mc->submit_write(ctx.make_address(0, 100, i * 64), data.data(), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 4);
        CHECK(s.writes == 4);
        CHECK(s.read_to_write_turnarounds + s.write_to_read_turnarounds > 0);
    }

    SECTION("Mixed across banks") {
        // Reads to bank 0, writes to bank 1
        for (int i = 0; i < 4; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, 100, i * 64), 64);
            ctx.mc->submit_write(ctx.make_address(1, 100, i * 64), data.data(), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 4);
        CHECK(s.writes == 4);
    }
}

// ============================================================================
// Level 8: Multi-Bank-Group Concurrency
// ============================================================================

TEST_CASE("Level8: Multi-bank-group concurrency", "[lpddr5][level8]") {
    TestContext ctx;

    SECTION("All bank groups concurrent") {
        for (int bg = 0; bg < 4; ++bg) {
            uint8_t bank = bg * 4;  // First bank of each group
            ctx.mc->submit_read(ctx.make_address(bank, 100, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
    }

    SECTION("High concurrency - all 16 banks") {
        for (int b = 0; b < 16; ++b) {
            ctx.mc->submit_read(ctx.make_address(b, 100, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 16);
        CHECK(s.page_empty == 16);
    }

    SECTION("Sustained throughput") {
        for (int round = 0; round < 4; ++round) {
            for (int b = 0; b < 16; ++b) {
                ctx.mc->submit_read(ctx.make_address(b, 100 + round, 0), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 64);

        INFO("Avg latency: " << s.avg_latency() << " cycles");
        INFO("Total cycles: " << ctx.mc->current_cycle());
    }
}

// ============================================================================
// Level 9: Full State Space Exploration
// ============================================================================

TEST_CASE("Level9: State space exploration", "[lpddr5][level9]") {
    TestContext ctx;

    SECTION("Random access pattern") {
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_int_distribution<uint8_t> bank_dist(0, 15);
        std::uniform_int_distribution<uint32_t> row_dist(0, 1000);
        std::uniform_int_distribution<int> type_dist(0, 1);

        std::vector<uint8_t> data(64, 0x99);

        for (int i = 0; i < 100; ++i) {
            uint8_t bank = bank_dist(rng);
            uint32_t row = row_dist(rng);
            uint64_t addr = ctx.make_address(bank, row, 0);

            if (type_dist(rng) == 0) {
                ctx.mc->submit_read(addr, 64);
            } else {
                ctx.mc->submit_write(addr, data.data(), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
        ctx.print_stats();
    }

    SECTION("Stress test - 500 random requests") {
        std::mt19937 rng(123);
        std::uniform_int_distribution<uint8_t> bank_dist(0, 15);
        std::uniform_int_distribution<uint32_t> row_dist(0, 100);  // Limited for conflicts
        std::uniform_int_distribution<int> type_dist(0, 1);

        std::vector<uint8_t> data(64, 0xAA);

        for (int i = 0; i < 500; ++i) {
            uint8_t bank = bank_dist(rng);
            uint32_t row = row_dist(rng);
            uint64_t addr = ctx.make_address(bank, row, 0);

            if (type_dist(rng) == 0) {
                ctx.mc->submit_read(addr, 64);
            } else {
                ctx.mc->submit_write(addr, data.data(), 64);
            }
        }

        REQUIRE(ctx.run_until_complete(50000));
        ctx.check_violations();
        ctx.print_stats();
    }

    SECTION("Worst case page conflicts") {
        // Every access is to a different row - maximum conflicts
        for (int i = 0; i < 32; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, i, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.page_conflicts == 31);  // All but first are conflicts
        ctx.print_stats();
    }

    SECTION("Bank group timing stress") {
        // Rapid activates within same bank group
        for (int round = 0; round < 10; ++round) {
            for (int b = 0; b < 4; ++b) {  // All in BG0
                ctx.mc->submit_read(ctx.make_address(b, round * 10, 0), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
        ctx.print_stats();
    }

    SECTION("Turnaround stress") {
        std::vector<uint8_t> data(64, 0xBB);

        // Rapid read-write-read-write pattern
        for (int i = 0; i < 50; ++i) {
            uint8_t bank = i % 16;
            if (i % 2 == 0) {
                ctx.mc->submit_read(ctx.make_address(bank, 100, 0), 64);
            } else {
                ctx.mc->submit_write(ctx.make_address(bank, 100, 0), data.data(), 64);
            }
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();
        ctx.print_stats();
    }
}

// ============================================================================
// Dual Channel Tests
// ============================================================================

TEST_CASE("Dual channel operations", "[lpddr5][dualchannel]") {
    TestContext ctx(2);  // 2 channels

    SECTION("Both channels") {
        ctx.mc->submit_read(ctx.make_address_dc(0, 0, 100), 64);
        ctx.mc->submit_read(ctx.make_address_dc(1, 0, 100), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 2);
    }

    SECTION("Channel interleaved") {
        for (int i = 0; i < 16; ++i) {
            uint8_t ch = i % 2;
            ctx.mc->submit_read(ctx.make_address_dc(ch, i / 2, 100), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 16);
    }
}

// ============================================================================
// BL32 Tests
// ============================================================================

TEST_CASE("BL32 operations", "[lpddr5][bl32]") {
    TestContext ctx(1, BurstLength::BL32);

    SECTION("Single BL32 read") {
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 128);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 1);
    }

    SECTION("BL32 burst sequence") {
        for (int i = 0; i < 8; ++i) {
            ctx.mc->submit_read(ctx.make_address(0, 100, i * 128), 128);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        const auto& s = ctx.mc->stats();
        CHECK(s.reads == 8);
    }
}

// ============================================================================
// Tracing and Chrome Trace Export Tests
// ============================================================================

TEST_CASE("Tracing: Resource tracking and Chrome Trace export", "[lpddr5][trace]") {
    TestContext ctx;

    // Create a resource tracker
    sw::trace::ResourceTracker tracker;

    // Enable tracing and set the tracker
    ctx.mc->enable_tracing(true);
    ctx.mc->set_resource_tracker(&tracker);

    SECTION("Basic trace collection") {
        // Submit a few requests to generate trace data
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);
        ctx.mc->submit_read(ctx.make_address(0, 100, 64), 64);  // Page hit
        ctx.mc->submit_read(ctx.make_address(1, 200, 0), 64);   // Different bank

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        // Check that trace entries were collected
        const auto& entries = ctx.mc->trace_entries();
        CHECK(!entries.empty());
        INFO("Collected " << entries.size() << " trace entries");

        // Check that resource tracker has data
        auto tracks = tracker.get_all_tracks();
        CHECK(!tracks.empty());
        INFO("Tracked " << tracks.size() << " resources");

        // Verify we have bank tracks
        bool has_bank_track = false;
        for (const auto& [res_id, track] : tracks) {
            if (res_id.type == sw::trace::ComponentType::LPDDR5_BANK) {
                has_bank_track = true;
                break;
            }
        }
        CHECK(has_bank_track);
    }

    SECTION("Chrome Trace export") {
        // Submit a workload that exercises multiple operations
        std::vector<uint8_t> data(64, 0xAB);

        // Page empty (activate + read)
        ctx.mc->submit_read(ctx.make_address(0, 100, 0), 64);

        // Page hit (just read)
        ctx.mc->submit_read(ctx.make_address(0, 100, 64), 64);

        // Page conflict (precharge + activate + read)
        ctx.mc->submit_read(ctx.make_address(0, 200, 0), 64);

        // Write to same row
        ctx.mc->submit_write(ctx.make_address(0, 200, 64), data.data(), 64);

        // Read from different bank
        ctx.mc->submit_read(ctx.make_address(4, 300, 0), 64);

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        // Finalize the tracker
        tracker.finalize(ctx.mc->current_cycle());

        // Export trace entries to Chrome Trace format
        const std::string trace_file = "/tmp/lpddr5_trace_entries.json";
        bool exported = sw::trace::ChromeTraceExporter::export_traces(
            trace_file,
            ctx.mc->trace_entries(),
            3.2  // 3.2 GHz clock
        );
        CHECK(exported);

        // Export resource tracker data
        const std::string resource_file = "/tmp/lpddr5_resources.json";
        auto tracks = tracker.get_all_tracks();
        bool resource_exported = sw::trace::ResourceTrackerExporter::export_to_chrome_trace(
            resource_file,
            tracks,
            3.2  // 3.2 GHz clock
        );
        CHECK(resource_exported);

        // Verify files exist and have content
        std::ifstream trace_in(trace_file);
        REQUIRE(trace_in.good());
        std::string content((std::istreambuf_iterator<char>(trace_in)),
                            std::istreambuf_iterator<char>());
        CHECK(content.size() > 100);  // Non-trivial content
        INFO("Trace file size: " << content.size() << " bytes");

        std::ifstream resource_in(resource_file);
        REQUIRE(resource_in.good());
        std::string resource_content((std::istreambuf_iterator<char>(resource_in)),
                                      std::istreambuf_iterator<char>());
        CHECK(resource_content.size() > 100);
        INFO("Resource file size: " << resource_content.size() << " bytes");

        // Print summary
        std::cout << "\n=== Chrome Trace Export ===" << std::endl;
        std::cout << "  Trace entries: " << ctx.mc->trace_entries().size() << std::endl;
        std::cout << "  Resources tracked: " << tracks.size() << std::endl;
        std::cout << "  Trace file: " << trace_file << std::endl;
        std::cout << "  Resource file: " << resource_file << std::endl;
        std::cout << "  Open with: chrome://tracing or https://ui.perfetto.dev" << std::endl;
        std::cout << "===========================\n" << std::endl;
    }

    SECTION("Resource utilization statistics") {
        // Submit a workload
        for (int i = 0; i < 20; ++i) {
            ctx.mc->submit_read(ctx.make_address(i % 4, 100 + (i / 4) * 10, 0), 64);
        }

        REQUIRE(ctx.run_until_complete());
        ctx.check_violations();

        // Finalize and get stats
        tracker.finalize(ctx.mc->current_cycle());
        auto stats = tracker.get_aggregate_stats();

        CHECK(stats.total_resources > 0);
        CHECK(stats.total_cycles > 0);

        std::cout << "\n=== Resource Utilization ===" << std::endl;
        std::cout << "  Total resources: " << stats.total_resources << std::endl;
        std::cout << "  Total cycles: " << stats.total_cycles << std::endl;
        std::cout << "  Overall utilization: "
                  << std::fixed << std::setprecision(1)
                  << (stats.overall_utilization * 100) << "%" << std::endl;
        std::cout << "  Max concurrency: " << stats.max_concurrency << std::endl;
        std::cout << "  Avg concurrency: "
                  << std::fixed << std::setprecision(2)
                  << stats.avg_concurrency << std::endl;
        std::cout << "=============================\n" << std::endl;
    }
}

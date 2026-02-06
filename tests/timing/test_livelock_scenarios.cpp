// ============================================================================
// tests/timing/test_livelock_scenarios.cpp
// Livelock detection and avoidance test suite
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/livelock_detector.hpp>
#include <sw/kpu/timing/tag_cam.hpp>
#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <atomic>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using namespace sw::kpu::isa;

// ============================================================================
// LivelockDetector Basic Tests
// ============================================================================

TEST_CASE("LivelockDetector construction", "[timing][livelock]") {
    SECTION("Default configuration") {
        LivelockDetector detector;
        REQUIRE(detector.config().stall_threshold == 1000);
        REQUIRE(detector.config().tile_stuck_threshold == 5000);
        REQUIRE(detector.livelock_count() == 0);
    }

    SECTION("Custom configuration") {
        LivelockDetector::Config config;
        config.stall_threshold = 500;
        config.tile_stuck_threshold = 2000;
        config.dump_state_on_detect = false;

        LivelockDetector detector(config);
        REQUIRE(detector.config().stall_threshold == 500);
        REQUIRE(detector.config().tile_stuck_threshold == 2000);
    }
}

TEST_CASE("LivelockDetector progress tracking", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.stall_threshold = 100;

    LivelockDetector detector(config);

    SECTION("No livelock when progress is made") {
        LivelockDetector::ProgressMetrics metrics;
        metrics.tiles_dma_completed = 0;

        auto result = detector.check(0, metrics);
        REQUIRE_FALSE(result.livelock_detected);

        // Make progress
        metrics.tiles_dma_completed = 10;
        result = detector.check(50, metrics);
        REQUIRE_FALSE(result.livelock_detected);

        // More progress
        metrics.tiles_dma_completed = 20;
        result = detector.check(100, metrics);
        REQUIRE_FALSE(result.livelock_detected);
    }

    SECTION("Livelock detected on stall") {
        LivelockDetector::ProgressMetrics metrics;
        metrics.tiles_dma_completed = 10;

        // Initial check
        auto result = detector.check(0, metrics);
        REQUIRE_FALSE(result.livelock_detected);

        // No progress for threshold cycles
        result = detector.check(50, metrics);
        REQUIRE_FALSE(result.livelock_detected);

        result = detector.check(100, metrics);
        REQUIRE_FALSE(result.livelock_detected);

        // Still no progress - should trigger
        result = detector.check(150, metrics);
        REQUIRE(result.livelock_detected);
        REQUIRE(result.stall_duration >= 100);
        REQUIRE(detector.livelock_count() == 1);
    }
}

TEST_CASE("LivelockDetector callback invocation", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.stall_threshold = 50;

    LivelockDetector detector(config);

    bool callback_invoked = false;
    Cycle callback_stall_duration = 0;

    detector.set_callback([&](const LivelockDetector::DetectionResult& result) {
        callback_invoked = true;
        callback_stall_duration = result.stall_duration;
    });

    LivelockDetector::ProgressMetrics metrics;
    metrics.tiles_dma_completed = 5;

    // Initial state
    detector.check(0, metrics);

    // Stall until livelock
    detector.check(60, metrics);

    REQUIRE(callback_invoked);
    REQUIRE(callback_stall_duration >= 50);
}

TEST_CASE("LivelockDetector reset", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.stall_threshold = 50;

    LivelockDetector detector(config);

    LivelockDetector::ProgressMetrics metrics;
    metrics.tiles_dma_completed = 5;

    // Trigger livelock
    detector.check(0, metrics);
    detector.check(60, metrics);
    REQUIRE(detector.livelock_count() == 1);

    // Reset
    detector.reset();
    REQUIRE(detector.livelock_count() == 0);

    // Should not trigger immediately after reset
    auto result = detector.check(0, metrics);
    REQUIRE_FALSE(result.livelock_detected);
}

TEST_CASE("LivelockDetector abort_on_detect", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.stall_threshold = 50;
    config.abort_on_detect = true;

    LivelockDetector detector(config);

    LivelockDetector::ProgressMetrics metrics;
    metrics.tiles_dma_completed = 5;

    detector.check(0, metrics);

    REQUIRE_THROWS_AS(detector.check(60, metrics), std::runtime_error);
}

// ============================================================================
// LivelockDetector Stuck Tile Detection
// ============================================================================

TEST_CASE("LivelockDetector find_stuck_tiles", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.tile_stuck_threshold = 100;

    LivelockDetector detector(config);

    TagCAM cam(8);

    TileID tile1{MatrixID::A, 0, 0, 0};
    TileID tile2{MatrixID::B, 0, 0, 0};

    SECTION("No stuck tiles when CAM empty") {
        auto stuck = detector.find_stuck_tiles(cam, 1000);
        REQUIRE(stuck.empty());
    }

    SECTION("Detect stuck tile") {
        // Add tile at cycle 0
        cam.insert(tile1, 0, 0);
        cam.insert(tile2, 0, 50);  // Added later

        // Check at cycle 200 - tile1 should be stuck
        auto stuck = detector.find_stuck_tiles(cam, 200);
        REQUIRE(stuck.size() >= 1);
    }

    SECTION("No stuck tiles when recently arrived") {
        cam.insert(tile1, 0, 90);

        auto stuck = detector.find_stuck_tiles(cam, 100);
        REQUIRE(stuck.empty());
    }
}

// ============================================================================
// LivelockDetector Circular Dependency Detection
// ============================================================================

TEST_CASE("LivelockDetector circular dependency detection", "[timing][livelock]") {
    LivelockDetector detector;

    CreditPool l3_credits(8);
    CreditPool l2_credits(8);
    TagCAM l3_cam(8);

    SECTION("No circular dependency when credits available") {
        // L3 and L2 have credits
        bool circular = detector.check_circular_dependency(
            l3_credits, l2_credits, l3_cam, 0);
        REQUIRE_FALSE(circular);
    }

    SECTION("No circular dependency when CAM empty") {
        // Exhaust credits but CAM is empty
        while (l3_credits.acquire()) {}
        while (l2_credits.acquire()) {}

        bool circular = detector.check_circular_dependency(
            l3_credits, l2_credits, l3_cam, 1);
        REQUIRE_FALSE(circular);
    }

    SECTION("Circular dependency when all blocked") {
        // Fill L3 CAM
        TileID tile{MatrixID::A, 0, 0, 0};
        l3_cam.insert(tile, 0, 0);

        // Exhaust all credits
        while (l3_credits.acquire()) {}
        while (l2_credits.acquire()) {}

        // BlockMovers waiting for credit
        bool circular = detector.check_circular_dependency(
            l3_credits, l2_credits, l3_cam, 1);
        REQUIRE(circular);
    }
}

// ============================================================================
// PartitionedCreditPool Livelock Prevention Tests
// ============================================================================

TEST_CASE("PartitionedCreditPool prevents A/B starvation", "[timing][livelock][credit_pool]") {
    // Scenario: Without partitioning, A tiles could fill all buffers
    // waiting for B tiles, causing livelock

    PartitionedCreditPool pool(4, 4, 4);  // 12 total, 4 per matrix

    SECTION("A cannot monopolize buffers") {
        // Try to fill all with A
        int a_acquired = 0;
        while (pool.acquire(PartitionedCreditPool::Matrix::A)) {
            a_acquired++;
        }

        // Should only get A's partition
        REQUIRE(a_acquired == 4);

        // B still has credits!
        REQUIRE(pool.available(PartitionedCreditPool::Matrix::B) == 4);
        REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    }

    SECTION("B cannot monopolize buffers") {
        // Try to fill all with B
        int b_acquired = 0;
        while (pool.acquire(PartitionedCreditPool::Matrix::B)) {
            b_acquired++;
        }

        REQUIRE(b_acquired == 4);
        REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 4);
    }
}

TEST_CASE("PartitionedCreditPool A partition full doesn't block B acquisition", "[timing][livelock][credit_pool]") {
    PartitionedCreditPool pool(2, 2, 2);

    // Fill A partition
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE_FALSE(pool.acquire(PartitionedCreditPool::Matrix::A));  // Full

    // B should still work
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::B) == 1);
}

TEST_CASE("PartitionedCreditPool custom A/B/C ratios", "[timing][livelock][credit_pool]") {
    // Custom ratio: more A buffers for A-heavy workloads
    PartitionedCreditPool pool(6, 2, 4);  // 12 total

    REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::A) == 6);
    REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::B) == 2);
    REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::C) == 4);

    // Can acquire up to partition capacity
    for (int i = 0; i < 6; ++i) {
        REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    }
    REQUIRE_FALSE(pool.acquire(PartitionedCreditPool::Matrix::A));

    // B still works with smaller partition
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE_FALSE(pool.acquire(PartitionedCreditPool::Matrix::B));
}

// ============================================================================
// Interleaved Scheduling Tests
// ============================================================================

TEST_CASE("Interleaved A-B scheduling prevents livelock", "[timing][livelock][schedule]") {
    MatMulScheduleGenerator::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 64;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);

    // Verify interleaved pattern
    auto analysis = ScheduleAnalysis::analyze(result);

    // Max consecutive same-matrix ops should be small
    REQUIRE(analysis.max_consecutive_a <= 3);
    REQUIRE(analysis.max_consecutive_b <= 3);
    REQUIRE(analysis.is_interleaved);
}

TEST_CASE("Blocked A-A-A-B-B-B with limited credits may stall", "[timing][livelock][schedule]") {
    // This test demonstrates why blocked scheduling is dangerous
    MatMulScheduleGenerator::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 128;  // 8 K tiles
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);

    auto analysis = ScheduleAnalysis::analyze(result);

    // Should NOT be interleaved
    REQUIRE_FALSE(analysis.is_interleaved);

    // Should have large consecutive runs
    REQUIRE(analysis.max_consecutive_a > 3);
    REQUIRE(analysis.max_consecutive_b > 3);
}

// ============================================================================
// Work-Conserving Scheduling Tests
// ============================================================================

TEST_CASE("Work-conserving scheduling makes progress", "[timing][livelock]") {
    // Test concept: work-conserving means if ANY tile can proceed, it does
    // This prevents head-of-line blocking

    // Simulate a work queue with multiple tiles
    struct WorkItem {
        TileID tile;
        bool ready;
    };

    std::vector<WorkItem> queue = {
        {{MatrixID::A, 0, 0, 0}, false},  // Not ready
        {{MatrixID::A, 1, 0, 0}, false},  // Not ready
        {{MatrixID::B, 0, 0, 0}, true},   // Ready!
        {{MatrixID::A, 2, 0, 0}, false},  // Not ready
    };

    // Work-conserving: find any ready item, not just head
    auto find_ready = [&queue]() -> int {
        for (size_t i = 0; i < queue.size(); ++i) {
            if (queue[i].ready) return static_cast<int>(i);
        }
        return -1;
    };

    // Non-work-conserving would only check head
    auto check_head_only = [&queue]() -> bool {
        return !queue.empty() && queue[0].ready;
    };

    // Work-conserving finds the ready item
    REQUIRE(find_ready() == 2);

    // Non-work-conserving would report stuck
    REQUIRE_FALSE(check_head_only());
}

// ============================================================================
// Priority Aging Tests
// ============================================================================

TEST_CASE("Priority aging prevents starvation", "[timing][livelock]") {
    // Test concept: older tiles get higher priority to prevent starvation

    TileDescriptor old_tile;
    old_tile.tile_id = {MatrixID::A, 0, 0, 0};
    old_tile.enqueue_cycle = 0;  // Enqueued at cycle 0

    TileDescriptor new_tile;
    new_tile.tile_id = {MatrixID::A, 1, 0, 0};
    new_tile.enqueue_cycle = 100;  // Enqueued at cycle 100

    Cycle current_cycle = 150;

    // Age-based priority
    Size old_priority = old_tile.age_priority(current_cycle);
    Size new_priority = new_tile.age_priority(current_cycle);

    // Old tile should have higher priority (more age)
    REQUIRE(old_priority > new_priority);
    REQUIRE(old_priority == 150);
    REQUIRE(new_priority == 50);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("Pathological 10:1 A/B ratio schedule analysis", "[timing][livelock][edge]") {
    // Create a schedule with severely unbalanced A/B ratio
    ScheduleResult result;
    result.valid = true;

    TileDescriptor a_tile, b_tile;
    a_tile.tile_id.matrix = MatrixID::A;
    b_tile.tile_id.matrix = MatrixID::B;

    // 10 A tiles for every 1 B tile
    for (int i = 0; i < 10; ++i) {
        for (int a = 0; a < 10; ++a) {
            result.operations.push_back(ScheduleOperation::load(a_tile));
        }
        result.operations.push_back(ScheduleOperation::load(b_tile));
    }

    auto analysis = ScheduleAnalysis::analyze(result);

    // Should detect this as NOT interleaved
    REQUIRE_FALSE(analysis.is_interleaved);
    REQUIRE(analysis.max_consecutive_a >= 10);

    // Ratio check
    REQUIRE(analysis.a_ops == 100);
    REQUIRE(analysis.b_ops == 10);
}

TEST_CASE("Credit pool exhaustion recovery", "[timing][livelock][credit_pool]") {
    CreditPool pool(4);

    // Exhaust pool
    REQUIRE(pool.acquire());
    REQUIRE(pool.acquire());
    REQUIRE(pool.acquire());
    REQUIRE(pool.acquire());
    REQUIRE(pool.empty());

    // Cannot acquire more
    REQUIRE_FALSE(pool.acquire());
    REQUIRE_FALSE(pool.acquire());

    // Release one - should allow acquire
    pool.release();
    REQUIRE(pool.acquire());
    REQUIRE(pool.empty());

    // Incremental release and acquire
    for (int i = 0; i < 10; ++i) {
        pool.release();
        REQUIRE(pool.available() == 1);
        REQUIRE(pool.acquire());
        REQUIRE(pool.available() == 0);
    }
}

TEST_CASE("TagCAM saturation handling", "[timing][livelock]") {
    TagCAM cam(4);  // Small CAM

    // Fill the CAM
    TileID t1{MatrixID::A, 0, 0, 0};
    TileID t2{MatrixID::A, 1, 0, 0};
    TileID t3{MatrixID::A, 2, 0, 0};
    TileID t4{MatrixID::A, 3, 0, 0};
    TileID t5{MatrixID::A, 4, 0, 0};  // Won't fit

    REQUIRE(cam.insert(t1, 0, 0));
    REQUIRE(cam.insert(t2, 1, 0));
    REQUIRE(cam.insert(t3, 2, 0));
    REQUIRE(cam.insert(t4, 3, 0));

    // CAM is full
    REQUIRE(cam.size() == 4);
    REQUIRE_FALSE(cam.insert(t5, 4, 0));  // Should fail

    // Remove one - should allow new insert
    cam.invalidate(t2);
    REQUIRE(cam.size() == 3);
    REQUIRE(cam.insert(t5, 4, 0));
}

TEST_CASE("LivelockDetector invariant verification", "[timing][livelock]") {
    LivelockDetector::Config config;
    config.tile_stuck_threshold = 100;

    LivelockDetector detector(config);

    CreditPool l3_credits(8);
    CreditPool l2_credits(8);
    TagCAM l3_cam(8);
    TagCAM l2_cam(8);

    SECTION("Invariants hold when idle") {
        bool valid = detector.verify_invariants(
            l3_credits, l2_credits, l3_cam, l2_cam, 0, false);
        REQUIRE(valid);  // Empty CAMs means idle is OK
    }

    SECTION("Invariants fail when work pending but no progress") {
        // Add tile to CAM
        TileID tile{MatrixID::A, 0, 0, 0};
        l3_cam.insert(tile, 0, 0);

        bool valid = detector.verify_invariants(
            l3_credits, l2_credits, l3_cam, l2_cam, 0, false);
        REQUIRE_FALSE(valid);  // Work pending but not progressing
    }

    SECTION("Invariants hold when work pending and progressing") {
        TileID tile{MatrixID::A, 0, 0, 0};
        l3_cam.insert(tile, 0, 0);

        bool valid = detector.verify_invariants(
            l3_credits, l2_credits, l3_cam, l2_cam, 0, true);
        REQUIRE(valid);  // Progressing is OK
    }

    SECTION("Invariants fail when tile stuck too long") {
        TileID tile{MatrixID::A, 0, 0, 0};
        l3_cam.insert(tile, 0, 0);  // Arrived at cycle 0

        // Check at cycle 200 - tile has been there too long
        bool valid = detector.verify_invariants(
            l3_credits, l2_credits, l3_cam, l2_cam, 200, true);
        REQUIRE_FALSE(valid);
    }
}

// ============================================================================
// SystemState Diagnostics Tests
// ============================================================================

TEST_CASE("SystemState to_string diagnostics", "[timing][livelock]") {
    LivelockDetector::SystemState state;
    state.current_cycle = 12345;
    state.l3_credits_available = 4;
    state.l3_credits_capacity = 8;
    state.l2_credits_available = 30;
    state.l2_credits_capacity = 64;
    state.l3_cam_entries = 10;
    state.l2_cam_entries = 20;
    state.dma_queue_depth = 5;
    state.bm_queue_depth = 3;
    state.str_queue_depth = 2;
    state.dma_idle = false;
    state.bm_idle = false;
    state.str_idle = true;

    std::string diagnostic = state.to_string();

    // Should contain key information
    REQUIRE(diagnostic.find("12345") != std::string::npos);  // Cycle
    REQUIRE(diagnostic.find("4/8") != std::string::npos);    // L3 credits
    REQUIRE(diagnostic.find("busy") != std::string::npos);   // DMA state
    REQUIRE(diagnostic.find("idle") != std::string::npos);   // STR state
}

// ============================================================================
// ProgressMetrics Tests
// ============================================================================

TEST_CASE("ProgressMetrics comparison", "[timing][livelock]") {
    LivelockDetector::ProgressMetrics m1, m2;

    SECTION("Equal metrics") {
        m1.tiles_dma_completed = 10;
        m2.tiles_dma_completed = 10;
        REQUIRE_FALSE(m1 > m2);
        REQUIRE_FALSE(m2 > m1);
    }

    SECTION("Different metrics") {
        m1.tiles_dma_completed = 10;
        m2.tiles_dma_completed = 15;
        REQUIRE(m2 > m1);
        REQUIRE_FALSE(m1 > m2);
    }

    SECTION("Total aggregation") {
        m1.tiles_dma_completed = 5;
        m1.tiles_moved = 3;
        m1.tiles_streamed = 2;
        REQUIRE(m1.total() == 10);
    }
}

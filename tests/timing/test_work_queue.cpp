// ============================================================================
// tests/timing/test_work_queue.cpp
// Unit tests for WorkQueue, PriorityWorkQueue, and TileWorkQueue
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/work_queue.hpp>
#include <sw/kpu/timing/tile_descriptor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileDescriptor make_tile_desc(MatrixID matrix, Size ti, Size tj, Size tk = 0,
                                      Cycle enqueue_cycle = 0) {
    TileDescriptor desc;
    desc.tile_id = {matrix, ti, tj, tk};
    desc.enqueue_cycle = enqueue_cycle;
    return desc;
}

// ============================================================================
// WorkQueue<int> Basic Tests
// ============================================================================

TEST_CASE("WorkQueue default constructor creates unbounded", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE(queue.empty());
    REQUIRE(queue.size() == 0);
    REQUIRE(queue.max_size() == 0);  // 0 = unbounded
    REQUIRE_FALSE(queue.full());
}

TEST_CASE("WorkQueue bounded constructor", "[timing][work_queue]") {
    WorkQueue<int> queue(5);

    REQUIRE(queue.max_size() == 5);
    REQUIRE_FALSE(queue.full());
}

TEST_CASE("WorkQueue enqueue adds items", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE(queue.enqueue(1));
    REQUIRE(queue.enqueue(2));
    REQUIRE(queue.enqueue(3));

    REQUIRE(queue.size() == 3);
    REQUIRE_FALSE(queue.empty());
}

TEST_CASE("WorkQueue enqueue respects max_size", "[timing][work_queue]") {
    WorkQueue<int> queue(2);

    REQUIRE(queue.enqueue(1));
    REQUIRE(queue.enqueue(2));
    REQUIRE(queue.full());

    REQUIRE_FALSE(queue.enqueue(3));
    REQUIRE(queue.size() == 2);
}

TEST_CASE("WorkQueue unbounded queue never full", "[timing][work_queue]") {
    WorkQueue<int> queue;

    for (int i = 0; i < 1000; ++i) {
        REQUIRE(queue.enqueue(i));
        REQUIRE_FALSE(queue.full());
    }
    REQUIRE(queue.size() == 1000);
}

TEST_CASE("WorkQueue peek returns front", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);

    REQUIRE(queue.peek() == 10);
    REQUIRE(queue.size() == 3);  // Peek doesn't remove
}

TEST_CASE("WorkQueue peek throws when empty", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE_THROWS_AS(queue.peek(), std::out_of_range);
}

TEST_CASE("WorkQueue try_peek returns optional", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE_FALSE(queue.try_peek().has_value());

    queue.enqueue(42);
    auto result = queue.try_peek();
    REQUIRE(result.has_value());
    REQUIRE(*result == 42);
}

TEST_CASE("WorkQueue dequeue removes front", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);

    REQUIRE(queue.dequeue() == 1);
    REQUIRE(queue.dequeue() == 2);
    REQUIRE(queue.dequeue() == 3);
    REQUIRE(queue.empty());
}

TEST_CASE("WorkQueue dequeue throws when empty", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE_THROWS_AS(queue.dequeue(), std::out_of_range);
}

TEST_CASE("WorkQueue try_dequeue returns optional", "[timing][work_queue]") {
    WorkQueue<int> queue;

    REQUIRE_FALSE(queue.try_dequeue().has_value());

    queue.enqueue(42);
    auto result = queue.try_dequeue();
    REQUIRE(result.has_value());
    REQUIRE(*result == 42);
    REQUIRE(queue.empty());
}

TEST_CASE("WorkQueue FIFO ordering", "[timing][work_queue]") {
    WorkQueue<int> queue;

    for (int i = 0; i < 10; ++i) {
        queue.enqueue(i);
    }

    for (int i = 0; i < 10; ++i) {
        REQUIRE(queue.dequeue() == i);
    }
}

TEST_CASE("WorkQueue reset", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);

    queue.reset();

    REQUIRE(queue.empty());
    REQUIRE(queue.size() == 0);
}

TEST_CASE("WorkQueue items returns all", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);

    auto items = queue.items();
    REQUIRE(items.size() == 3);
    REQUIRE(items[0] == 1);
    REQUIRE(items[1] == 2);
    REQUIRE(items[2] == 3);
}

TEST_CASE("WorkQueue find_if locates matching", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(5);
    queue.enqueue(10);
    queue.enqueue(15);

    auto result = queue.find_if([](int x) { return x > 7; });
    REQUIRE(result.has_value());
    REQUIRE(*result == 10);
}

TEST_CASE("WorkQueue find_if returns nullopt when not found", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);

    auto result = queue.find_if([](int x) { return x > 100; });
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("WorkQueue remove_if removes first", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(5);
    queue.enqueue(10);
    queue.enqueue(5);

    REQUIRE(queue.remove_if([](int x) { return x == 5; }));
    REQUIRE(queue.size() == 3);

    // Verify FIFO order maintained, first 5 removed
    REQUIRE(queue.dequeue() == 1);
    REQUIRE(queue.dequeue() == 10);
    REQUIRE(queue.dequeue() == 5);
}

TEST_CASE("WorkQueue remove_if returns false when not found", "[timing][work_queue]") {
    WorkQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);

    REQUIRE_FALSE(queue.remove_if([](int x) { return x == 100; }));
    REQUIRE(queue.size() == 2);
}

// ============================================================================
// PriorityWorkQueue Tests
// ============================================================================

TEST_CASE("PriorityWorkQueue default constructor creates unbounded", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    REQUIRE(queue.empty());
    REQUIRE(queue.size() == 0);
}

TEST_CASE("PriorityWorkQueue enqueue with priority", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    REQUIRE(queue.enqueue(1, 10));
    REQUIRE(queue.enqueue(2, 20));
    REQUIRE(queue.enqueue(3, 5));

    REQUIRE(queue.size() == 3);
}

TEST_CASE("PriorityWorkQueue bounded rejects excess", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue(2);

    REQUIRE(queue.enqueue(1, 10));
    REQUIRE(queue.enqueue(2, 20));
    REQUIRE(queue.full());

    REQUIRE_FALSE(queue.enqueue(3, 30));
}

TEST_CASE("PriorityWorkQueue highest priority first", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    queue.enqueue(1, 10);  // Low priority
    queue.enqueue(2, 100); // High priority
    queue.enqueue(3, 50);  // Medium priority

    // Should dequeue in priority order: 2, 3, 1
    REQUIRE(*queue.try_dequeue() == 2);
    REQUIRE(*queue.try_dequeue() == 3);
    REQUIRE(*queue.try_dequeue() == 1);
}

TEST_CASE("PriorityWorkQueue try_peek returns highest", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    queue.enqueue(1, 10);
    queue.enqueue(2, 100);
    queue.enqueue(3, 50);

    auto top = queue.try_peek();
    REQUIRE(top.has_value());
    REQUIRE(top->item == 2);
    REQUIRE(top->priority == 100);

    // Peek doesn't remove
    REQUIRE(queue.size() == 3);
}

TEST_CASE("PriorityWorkQueue try_dequeue on empty", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    REQUIRE_FALSE(queue.try_dequeue().has_value());
}

TEST_CASE("PriorityWorkQueue reset", "[timing][work_queue]") {
    PriorityWorkQueue<int> queue;

    queue.enqueue(1, 10);
    queue.enqueue(2, 20);

    queue.reset();

    REQUIRE(queue.empty());
}

// ============================================================================
// TileWorkQueue Tests
// ============================================================================

TEST_CASE("TileWorkQueue default constructor", "[timing][work_queue]") {
    TileWorkQueue queue;

    REQUIRE(queue.empty());
    REQUIRE(queue.total_size() == 0);
}

TEST_CASE("TileWorkQueue enqueue partitions by matrix", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 1));
    queue.enqueue(make_tile_desc(MatrixID::C, 0, 0));

    REQUIRE(queue.size(MatrixID::A) == 2);
    REQUIRE(queue.size(MatrixID::B) == 1);
    REQUIRE(queue.size(MatrixID::C) == 1);
    REQUIRE(queue.total_size() == 4);
}

TEST_CASE("TileWorkQueue bounded partitions", "[timing][work_queue]") {
    TileWorkQueue queue(2);  // Max 2 per matrix

    // Fill A partition
    REQUIRE(queue.enqueue(make_tile_desc(MatrixID::A, 0, 0)));
    REQUIRE(queue.enqueue(make_tile_desc(MatrixID::A, 0, 1)));
    REQUIRE_FALSE(queue.enqueue(make_tile_desc(MatrixID::A, 0, 2)));

    // B partition still available
    REQUIRE(queue.enqueue(make_tile_desc(MatrixID::B, 0, 0)));
}

TEST_CASE("TileWorkQueue peek by matrix", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 1, 2));
    queue.enqueue(make_tile_desc(MatrixID::B, 3, 4));

    auto a = queue.peek(MatrixID::A);
    REQUIRE(a.has_value());
    REQUIRE(a->tile_id.ti == 1);
    REQUIRE(a->tile_id.tj == 2);

    auto b = queue.peek(MatrixID::B);
    REQUIRE(b.has_value());
    REQUIRE(b->tile_id.ti == 3);
    REQUIRE(b->tile_id.tj == 4);

    auto c = queue.peek(MatrixID::C);
    REQUIRE_FALSE(c.has_value());
}

TEST_CASE("TileWorkQueue dequeue by matrix", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 1));

    auto first = queue.dequeue(MatrixID::A);
    REQUIRE(first.has_value());
    REQUIRE(first->tile_id.tj == 0);

    auto second = queue.dequeue(MatrixID::A);
    REQUIRE(second.has_value());
    REQUIRE(second->tile_id.tj == 1);

    auto third = queue.dequeue(MatrixID::A);
    REQUIRE_FALSE(third.has_value());
}

TEST_CASE("TileWorkQueue find_ready", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 1));
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0));

    // Find tile with tj == 1
    auto result = queue.find_ready([](const TileDescriptor& t) {
        return t.tile_id.tj == 1;
    });

    REQUIRE(result.has_value());
    REQUIRE(result->tile_id.tj == 1);
}

TEST_CASE("TileWorkQueue find_ready returns nullopt when none match", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0));

    auto result = queue.find_ready([](const TileDescriptor& t) {
        return t.tile_id.ti == 999;
    });

    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("TileWorkQueue dequeue_oldest_ready", "[timing][work_queue]") {
    TileWorkQueue queue;

    // Enqueue with different enqueue cycles
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0, 0, 300));  // Newest
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0, 0, 100));  // Oldest
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 1, 0, 200));  // Middle

    // All tiles ready
    auto oldest = queue.dequeue_oldest_ready(
        [](const TileDescriptor&) { return true; },
        1000
    );

    REQUIRE(oldest.has_value());
    REQUIRE(oldest->tile_id.matrix == MatrixID::B);
    REQUIRE(oldest->enqueue_cycle == 100);

    // Next oldest
    auto next = queue.dequeue_oldest_ready(
        [](const TileDescriptor&) { return true; },
        1000
    );

    REQUIRE(next.has_value());
    REQUIRE(next->enqueue_cycle == 200);
}

TEST_CASE("TileWorkQueue dequeue_oldest_ready with predicate", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0, 0, 100));  // Oldest but not ready
    queue.enqueue(make_tile_desc(MatrixID::A, 0, 1, 0, 200));  // Ready
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0, 0, 150));  // Ready, oldest ready

    // Only tiles with tj > 0 are "ready"
    auto oldest = queue.dequeue_oldest_ready(
        [](const TileDescriptor& t) { return t.tile_id.tj > 0 || t.tile_id.matrix == MatrixID::B; },
        1000
    );

    REQUIRE(oldest.has_value());
    REQUIRE(oldest->tile_id.matrix == MatrixID::B);  // Oldest among ready
    REQUIRE(oldest->enqueue_cycle == 150);
}

TEST_CASE("TileWorkQueue reset", "[timing][work_queue]") {
    TileWorkQueue queue;

    queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::B, 0, 0));
    queue.enqueue(make_tile_desc(MatrixID::C, 0, 0));

    queue.reset();

    REQUIRE(queue.empty());
    REQUIRE(queue.total_size() == 0);
    REQUIRE(queue.size(MatrixID::A) == 0);
    REQUIRE(queue.size(MatrixID::B) == 0);
    REQUIRE(queue.size(MatrixID::C) == 0);
}

// ============================================================================
// Dataflow Scenario Tests
// ============================================================================

TEST_CASE("TileWorkQueue DMA scheduling scenario", "[timing][work_queue]") {
    // Simulate DMA work queue receiving tile load requests

    TileWorkQueue dma_queue(4);  // Max 4 per matrix

    // Schedule loads for A and B tiles
    for (Size i = 0; i < 3; ++i) {
        dma_queue.enqueue(make_tile_desc(MatrixID::A, 0, i, 0, 100 + i));
        dma_queue.enqueue(make_tile_desc(MatrixID::B, i, 0, 0, 100 + i));
    }

    REQUIRE(dma_queue.total_size() == 6);

    // Work-conserving: process any ready tile
    int processed = 0;
    while (!dma_queue.empty()) {
        auto tile = dma_queue.dequeue_oldest_ready(
            [](const TileDescriptor&) { return true; },
            1000
        );
        if (tile) {
            ++processed;
        }
    }

    REQUIRE(processed == 6);
}

TEST_CASE("TileWorkQueue BlockMover waiting for data", "[timing][work_queue]") {
    // BlockMover has tiles in queue but waits for data in L3

    TileWorkQueue bm_queue;

    // Queue tiles to move
    bm_queue.enqueue(make_tile_desc(MatrixID::A, 0, 0));
    bm_queue.enqueue(make_tile_desc(MatrixID::A, 0, 1));
    bm_queue.enqueue(make_tile_desc(MatrixID::B, 0, 0));

    // Simulate: only A[0,1] is in L3
    auto ready = bm_queue.find_ready([](const TileDescriptor& t) {
        // Only A[0,1] is "ready" (in L3)
        return t.tile_id.matrix == MatrixID::A && t.tile_id.tj == 1;
    });

    REQUIRE(ready.has_value());
    REQUIRE(ready->tile_id.tj == 1);
}

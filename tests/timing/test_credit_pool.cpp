// ============================================================================
// tests/timing/test_credit_pool.cpp
// Unit tests for CreditPool and PartitionedCreditPool
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/credit_pool.hpp>
#include <stdexcept>

using namespace sw::kpu::timing;

// ============================================================================
// CreditPool Tests
// ============================================================================

TEST_CASE("CreditPool construction", "[timing][credit_pool]") {
    SECTION("Constructor sets capacity and available") {
        CreditPool pool(8);

        REQUIRE(pool.capacity() == 8);
        REQUIRE(pool.available() == 8);
        REQUIRE(pool.full());
        REQUIRE_FALSE(pool.empty());
    }

    SECTION("Constructor rejects zero capacity") {
        REQUIRE_THROWS_AS(CreditPool(0), std::invalid_argument);
    }
}

TEST_CASE("CreditPool acquire", "[timing][credit_pool]") {
    SECTION("Acquire decrements available") {
        CreditPool pool(4);

        REQUIRE(pool.acquire());
        REQUIRE(pool.available() == 3);
        REQUIRE(pool.outstanding() == 1);

        REQUIRE(pool.acquire());
        REQUIRE(pool.available() == 2);
        REQUIRE(pool.outstanding() == 2);
    }

    SECTION("Acquire returns false when empty") {
        CreditPool pool(2);

        REQUIRE(pool.acquire());
        REQUIRE(pool.acquire());
        REQUIRE(pool.empty());

        // Should fail when empty
        REQUIRE_FALSE(pool.acquire());
        REQUIRE_FALSE(pool.acquire());

        // Available should still be 0
        REQUIRE(pool.available() == 0);
    }
}

TEST_CASE("CreditPool release", "[timing][credit_pool]") {
    SECTION("Release increments available") {
        CreditPool pool(4);

        // Acquire all credits
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(pool.acquire());
        }
        REQUIRE(pool.empty());

        // Release them back
        pool.release();
        REQUIRE(pool.available() == 1);
        REQUIRE_FALSE(pool.empty());

        pool.release();
        REQUIRE(pool.available() == 2);
    }

    SECTION("Release throws when full") {
        CreditPool pool(2);

        // Pool starts full
        REQUIRE(pool.full());

        // Releasing when full should throw
        REQUIRE_THROWS_AS(pool.release(), std::overflow_error);
    }

    SECTION("Release after acquire does not throw") {
        CreditPool pool(2);

        pool.acquire();
        REQUIRE_NOTHROW(pool.release());
        REQUIRE(pool.full());
    }
}

TEST_CASE("CreditPool has_credit", "[timing][credit_pool]") {
    CreditPool pool(1);

    REQUIRE(pool.has_credit());
    pool.acquire();
    REQUIRE_FALSE(pool.has_credit());
    pool.release();
    REQUIRE(pool.has_credit());
}

TEST_CASE("CreditPool reset", "[timing][credit_pool]") {
    CreditPool pool(5);

    // Acquire some credits
    pool.acquire();
    pool.acquire();
    pool.acquire();
    REQUIRE(pool.available() == 2);
    REQUIRE(pool.outstanding() == 3);

    // Reset should restore to full
    pool.reset();
    REQUIRE(pool.available() == 5);
    REQUIRE(pool.outstanding() == 0);
    REQUIRE(pool.full());
}

TEST_CASE("CreditPool acquire_blocking throws", "[timing][credit_pool]") {
    CreditPool pool(2);

    REQUIRE_THROWS_AS(pool.acquire_blocking(), std::runtime_error);
}

TEST_CASE("CreditPool stress test", "[timing][credit_pool]") {
    CreditPool pool(8);

    // Simulate many acquire/release cycles
    for (int cycle = 0; cycle < 1000; ++cycle) {
        // Acquire all
        for (size_t i = 0; i < 8; ++i) {
            REQUIRE(pool.acquire());
        }
        REQUIRE(pool.empty());

        // Release all
        for (size_t i = 0; i < 8; ++i) {
            pool.release();
        }
        REQUIRE(pool.full());
    }
}

TEST_CASE("CreditPool partial acquire/release", "[timing][credit_pool]") {
    CreditPool pool(10);

    // Acquire 7
    for (int i = 0; i < 7; ++i) {
        REQUIRE(pool.acquire());
    }
    REQUIRE(pool.available() == 3);

    // Release 4
    for (int i = 0; i < 4; ++i) {
        pool.release();
    }
    REQUIRE(pool.available() == 7);

    // Acquire 5 more
    for (int i = 0; i < 5; ++i) {
        REQUIRE(pool.acquire());
    }
    REQUIRE(pool.available() == 2);
}

// ============================================================================
// PartitionedCreditPool Tests
// ============================================================================

TEST_CASE("PartitionedCreditPool construction", "[timing][credit_pool]") {
    SECTION("Equal partition constructor") {
        PartitionedCreditPool pool(9);

        // Should partition equally: 3, 3, 3
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::A) == 3);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::B) == 3);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::C) == 3);
        REQUIRE(pool.total_capacity() == 9);
    }

    SECTION("Equal partition with remainder") {
        PartitionedCreditPool pool(10);

        // Should partition: 3, 3, 4 (remainder goes to C)
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::A) == 3);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::B) == 3);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::C) == 4);
        REQUIRE(pool.total_capacity() == 10);
    }

    SECTION("Custom partition constructor") {
        PartitionedCreditPool pool(4, 2, 6);

        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::A) == 4);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::B) == 2);
        REQUIRE(pool.capacity(PartitionedCreditPool::Matrix::C) == 6);
        REQUIRE(pool.total_capacity() == 12);
    }

    SECTION("Constructor rejects too small") {
        REQUIRE_THROWS_AS(PartitionedCreditPool(2), std::invalid_argument);
        REQUIRE_THROWS_AS(PartitionedCreditPool(1), std::invalid_argument);
    }

    SECTION("Constructor rejects zero partition") {
        REQUIRE_THROWS_AS(PartitionedCreditPool(0, 2, 2), std::invalid_argument);
        REQUIRE_THROWS_AS(PartitionedCreditPool(2, 0, 2), std::invalid_argument);
        REQUIRE_THROWS_AS(PartitionedCreditPool(2, 2, 0), std::invalid_argument);
    }
}

TEST_CASE("PartitionedCreditPool acquire from partition", "[timing][credit_pool]") {
    PartitionedCreditPool pool(3, 3, 3);

    // Acquire from A
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 2);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::B) == 3);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::C) == 3);

    // Acquire from B
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 2);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::B) == 2);
}

TEST_CASE("PartitionedCreditPool acquire fails when partition empty", "[timing][credit_pool]") {
    PartitionedCreditPool pool(1, 1, 1);

    // Empty partition A
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE_FALSE(pool.acquire(PartitionedCreditPool::Matrix::A));

    // B should still have credits
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
}

TEST_CASE("PartitionedCreditPool release to partition", "[timing][credit_pool]") {
    PartitionedCreditPool pool(2, 2, 2);

    // Acquire and release from A
    pool.acquire(PartitionedCreditPool::Matrix::A);
    pool.acquire(PartitionedCreditPool::Matrix::A);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 0);

    pool.release(PartitionedCreditPool::Matrix::A);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 1);
}

TEST_CASE("PartitionedCreditPool release throws when partition full", "[timing][credit_pool]") {
    PartitionedCreditPool pool(2, 2, 2);

    // All partitions start full
    REQUIRE_THROWS_AS(pool.release(PartitionedCreditPool::Matrix::A), std::overflow_error);
    REQUIRE_THROWS_AS(pool.release(PartitionedCreditPool::Matrix::B), std::overflow_error);
    REQUIRE_THROWS_AS(pool.release(PartitionedCreditPool::Matrix::C), std::overflow_error);
}

TEST_CASE("PartitionedCreditPool total_available", "[timing][credit_pool]") {
    PartitionedCreditPool pool(3, 3, 3);

    REQUIRE(pool.total_available() == 9);

    pool.acquire(PartitionedCreditPool::Matrix::A);
    pool.acquire(PartitionedCreditPool::Matrix::B);
    REQUIRE(pool.total_available() == 7);
}

TEST_CASE("PartitionedCreditPool reset", "[timing][credit_pool]") {
    PartitionedCreditPool pool(2, 3, 4);

    // Acquire some from each
    pool.acquire(PartitionedCreditPool::Matrix::A);
    pool.acquire(PartitionedCreditPool::Matrix::B);
    pool.acquire(PartitionedCreditPool::Matrix::B);
    pool.acquire(PartitionedCreditPool::Matrix::C);

    REQUIRE(pool.total_available() == 5);

    pool.reset();

    REQUIRE(pool.available(PartitionedCreditPool::Matrix::A) == 2);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::B) == 3);
    REQUIRE(pool.available(PartitionedCreditPool::Matrix::C) == 4);
    REQUIRE(pool.total_available() == 9);
}

TEST_CASE("PartitionedCreditPool livelock prevention scenario", "[timing][credit_pool]") {
    // This test demonstrates why partitioning prevents livelock
    // Scenario: 8 total buffers, if not partitioned, all could fill with A tiles
    // waiting for B tiles, causing livelock.

    PartitionedCreditPool pool(3, 3, 2);  // 8 total

    // Fill A partition completely
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
    REQUIRE_FALSE(pool.acquire(PartitionedCreditPool::Matrix::A));

    // B partition is still available - progress can be made!
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::B));

    // C partition still available for outputs
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::C));
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::C));

    // Now release some to make progress
    pool.release(PartitionedCreditPool::Matrix::A);
    REQUIRE(pool.acquire(PartitionedCreditPool::Matrix::A));
}

// ============================================================================
// CreditPool partition mode (issue #89): PartitionedCreditPool wired behind
// the CreditPool interface via indexed acquire/release
// ============================================================================

TEST_CASE("CreditPool partition mode splits capacity per matrix", "[timing][credit_pool][partition]") {
    CreditPool pool(10);
    REQUIRE_FALSE(pool.partitioned());
    pool.partition_equal();  // 10 -> A=3, B=3, C=4 (remainder to C)
    REQUIRE(pool.partitioned());

    REQUIRE(pool.available(0) == 3);   // A
    REQUIRE(pool.available(1) == 3);   // B
    REQUIRE(pool.available(2) == 4);   // C
    REQUIRE(pool.available() == 10);   // total
    REQUIRE(pool.capacity() == 10);
}

TEST_CASE("CreditPool partition isolation: exhausting A leaves B and C usable", "[timing][credit_pool][partition]") {
    CreditPool pool(9);
    pool.partition_equal();  // 3/3/3

    REQUIRE(pool.acquire(0));
    REQUIRE(pool.acquire(0));
    REQUIRE(pool.acquire(0));
    REQUIRE_FALSE(pool.acquire(0));    // A exhausted

    REQUIRE(pool.acquire(1));          // B unaffected
    REQUIRE(pool.acquire(2));          // C unaffected
    REQUIRE(pool.available() == 4);
    REQUIRE(pool.outstanding() == 5);

    pool.release(0);
    REQUIRE(pool.acquire(0));
}

TEST_CASE("CreditPool partition mode rejects un-indexed acquire/release", "[timing][credit_pool][partition]") {
    CreditPool pool(6);
    pool.partition_equal();
    REQUIRE_THROWS_AS(pool.acquire(), std::logic_error);
    REQUIRE_THROWS_AS(pool.release(), std::logic_error);
}

TEST_CASE("CreditPool indexed calls behave as shared when not partitioned", "[timing][credit_pool][partition]") {
    CreditPool pool(2);
    REQUIRE(pool.acquire(0));
    REQUIRE(pool.acquire(1));          // same shared pool
    REQUIRE_FALSE(pool.acquire(2));    // exhausted regardless of index
    pool.release(2);
    REQUIRE(pool.available() == 1);
    REQUIRE(pool.acquire());           // un-indexed still valid in shared mode
}

TEST_CASE("CreditPool custom partition validates shares", "[timing][credit_pool][partition]") {
    CreditPool pool(8);
    REQUIRE_THROWS_AS(pool.partition(4, 4, 4), std::invalid_argument);  // sums to 12
    pool.partition(2, 2, 4);
    REQUIRE(pool.available(2) == 4);

    // Per-partition over-release throws
    REQUIRE(pool.acquire(2));
    pool.release(2);
    REQUIRE_THROWS_AS(pool.release(2), std::overflow_error);
}

TEST_CASE("CreditPool partitioning requires a quiescent pool; reset keeps partitions", "[timing][credit_pool][partition]") {
    CreditPool pool(6);
    REQUIRE(pool.acquire());
    REQUIRE_THROWS_AS(pool.partition_equal(), std::logic_error);
    pool.release();
    pool.partition_equal();

    REQUIRE(pool.acquire(0));
    REQUIRE(pool.acquire(1));
    pool.reset();
    REQUIRE(pool.partitioned());
    REQUIRE(pool.available() == 6);
    REQUIRE(pool.available(0) == 2);
}

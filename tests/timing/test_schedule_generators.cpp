// ============================================================================
// tests/timing/test_schedule_generators.cpp
// Unit tests for schedule generators
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <set>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using namespace sw::kpu::isa;

// ============================================================================
// ScheduleOperation Tests
// ============================================================================

TEST_CASE("ScheduleOperation factory methods", "[timing][schedule]") {
    TileDescriptor tile;
    tile.tile_id.matrix = MatrixID::A;
    tile.tile_id.ti = 1;
    tile.tile_id.tj = 2;
    tile.tile_id.tk = 3;
    tile.size_bytes = 1024;

    SECTION("load operation") {
        auto op = ScheduleOperation::load(tile, 2);
        REQUIRE(op.type == ScheduleOpType::LOAD);
        REQUIRE(op.tile.tile_id == tile.tile_id);
        REQUIRE(op.engine_id == 2);
        REQUIRE(op.matrix() == MatrixID::A);
    }

    SECTION("store operation") {
        auto op = ScheduleOperation::store(tile);
        REQUIRE(op.type == ScheduleOpType::STORE);
        REQUIRE(op.engine_id == -1);  // auto-select
    }

    SECTION("move operation") {
        auto op = ScheduleOperation::move(tile, true, 1);
        REQUIRE(op.type == ScheduleOpType::MOVE);
        REQUIRE(op.transpose == true);
        REQUIRE(op.mover_id == 1);
    }

    SECTION("writeback operation") {
        auto op = ScheduleOperation::writeback(tile);
        REQUIRE(op.type == ScheduleOpType::WRITEBACK);
    }

    SECTION("feed operation") {
        auto op = ScheduleOperation::feed(tile);
        REQUIRE(op.type == ScheduleOpType::FEED);
    }

    SECTION("drain operation") {
        auto op = ScheduleOperation::drain(tile);
        REQUIRE(op.type == ScheduleOpType::DRAIN);
    }

    SECTION("to_string") {
        auto op = ScheduleOperation::load(tile);
        std::string str = op.to_string();
        REQUIRE(str.find("LOAD") != std::string::npos);
        REQUIRE(str.find("A[") != std::string::npos);
    }
}

// ============================================================================
// ScheduleMetadata Tests
// ============================================================================

TEST_CASE("ScheduleMetadata calculations", "[timing][schedule]") {
    ScheduleMetadata meta;
    meta.a_tiles = 10;
    meta.b_tiles = 10;
    meta.c_tiles = 5;

    REQUIRE(meta.total_tiles() == 25);
    REQUIRE(meta.estimated_ops() == 75);  // (10+10)*3 + 5*3
}

// ============================================================================
// ScheduleResult Tests
// ============================================================================

TEST_CASE("ScheduleResult counting", "[timing][schedule]") {
    ScheduleResult result;
    result.valid = true;

    TileDescriptor a_tile, b_tile, c_tile;
    a_tile.tile_id.matrix = MatrixID::A;
    b_tile.tile_id.matrix = MatrixID::B;
    c_tile.tile_id.matrix = MatrixID::C;

    result.operations.push_back(ScheduleOperation::load(a_tile));
    result.operations.push_back(ScheduleOperation::load(b_tile));
    result.operations.push_back(ScheduleOperation::move(a_tile));
    result.operations.push_back(ScheduleOperation::move(b_tile));
    result.operations.push_back(ScheduleOperation::feed(a_tile));
    result.operations.push_back(ScheduleOperation::feed(b_tile));
    result.operations.push_back(ScheduleOperation::drain(c_tile));
    result.operations.push_back(ScheduleOperation::store(c_tile));

    REQUIRE(result.size() == 8);
    REQUIRE(result.count_ops(ScheduleOpType::LOAD) == 2);
    REQUIRE(result.count_ops(ScheduleOpType::MOVE) == 2);
    REQUIRE(result.count_ops(ScheduleOpType::FEED) == 2);
    REQUIRE(result.count_ops(ScheduleOpType::DRAIN) == 1);
    REQUIRE(result.count_ops(ScheduleOpType::STORE) == 1);
    REQUIRE(result.count_matrix_ops(MatrixID::A) == 3);
    REQUIRE(result.count_matrix_ops(MatrixID::B) == 3);
    REQUIRE(result.count_matrix_ops(MatrixID::C) == 2);
}

// ============================================================================
// ScheduleAnalysis Tests
// ============================================================================

TEST_CASE("ScheduleAnalysis interleaving detection", "[timing][schedule]") {
    ScheduleResult result;
    result.valid = true;

    TileDescriptor a_tile, b_tile;
    a_tile.tile_id.matrix = MatrixID::A;
    b_tile.tile_id.matrix = MatrixID::B;

    SECTION("Interleaved schedule detected") {
        // A-B-A-B pattern
        for (int i = 0; i < 4; ++i) {
            result.operations.push_back(ScheduleOperation::load(a_tile));
            result.operations.push_back(ScheduleOperation::load(b_tile));
        }

        auto analysis = ScheduleAnalysis::analyze(result);
        REQUIRE(analysis.is_interleaved);
        REQUIRE(analysis.max_consecutive_a <= 1);
        REQUIRE(analysis.max_consecutive_b <= 1);
    }

    SECTION("Non-interleaved schedule detected") {
        // A-A-A-A-B-B-B-B pattern
        for (int i = 0; i < 4; ++i) {
            result.operations.push_back(ScheduleOperation::load(a_tile));
        }
        for (int i = 0; i < 4; ++i) {
            result.operations.push_back(ScheduleOperation::load(b_tile));
        }

        auto analysis = ScheduleAnalysis::analyze(result);
        REQUIRE_FALSE(analysis.is_interleaved);
        REQUIRE(analysis.max_consecutive_a == 4);
        REQUIRE(analysis.max_consecutive_b == 4);
    }
}

// ============================================================================
// MatMulScheduleGenerator Tests
// ============================================================================

TEST_CASE("MatMulScheduleGenerator configuration", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 64;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;

    REQUIRE(config.m_tiles() == 4);
    REQUIRE(config.n_tiles() == 4);
    REQUIRE(config.k_tiles() == 4);
    REQUIRE(config.total_a_tiles() == 16);  // 4 * 4
    REQUIRE(config.total_b_tiles() == 16);  // 4 * 4
    REQUIRE(config.total_c_tiles() == 16);  // 4 * 4
    REQUIRE(config.tile_size_bytes() == 16 * 16 * 4);  // 1024 bytes
}

TEST_CASE("MatMulScheduleGenerator invalid config", "[timing][schedule][matmul]") {
    SECTION("Zero matrix dimensions") {
        MatMulScheduleGenerator::Config config;
        config.M = 0;  // Invalid
        config.N = 64;
        config.K = 64;

        MatMulScheduleGenerator gen(config);
        auto result = gen.generate();

        REQUIRE_FALSE(result.valid);
        REQUIRE(result.error_message.find("non-zero") != std::string::npos);
    }

    SECTION("Zero tile dimensions") {
        MatMulScheduleGenerator::Config config;
        config.M = 64;
        config.N = 64;
        config.K = 64;
        config.Ti = 0;  // Invalid

        MatMulScheduleGenerator gen(config);
        auto result = gen.generate();

        REQUIRE_FALSE(result.valid);
    }
}

TEST_CASE("MatMulScheduleGenerator output_stationary", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);
    REQUIRE(result.size() > 0);

    // 2x2x2 tiles = 4 A tiles, 4 B tiles, 4 C tiles
    REQUIRE(result.metadata.a_tiles == 4);
    REQUIRE(result.metadata.b_tiles == 4);
    REQUIRE(result.metadata.c_tiles == 4);
    REQUIRE(result.metadata.strategy == "output_stationary");

    // Check we have load, move, feed, drain, writeback, store ops
    REQUIRE(result.count_ops(ScheduleOpType::LOAD) > 0);
    REQUIRE(result.count_ops(ScheduleOpType::MOVE) > 0);
    REQUIRE(result.count_ops(ScheduleOpType::FEED) > 0);
    REQUIRE(result.count_ops(ScheduleOpType::DRAIN) > 0);
    REQUIRE(result.count_ops(ScheduleOpType::WRITEBACK) > 0);
    REQUIRE(result.count_ops(ScheduleOpType::STORE) > 0);
}

TEST_CASE("MatMulScheduleGenerator interleaved_ab", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);
    REQUIRE(result.metadata.strategy == "interleaved_ab");

    // Verify interleaving
    auto analysis = ScheduleAnalysis::analyze(result);
    REQUIRE(analysis.is_interleaved);

    // Max consecutive A/B should be small (at most 1 in perfect interleaving)
    REQUIRE(analysis.max_consecutive_a <= 3);
    REQUIRE(analysis.max_consecutive_b <= 3);
}

TEST_CASE("MatMulScheduleGenerator blocked_ab", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 64;  // More K tiles
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);
    REQUIRE(result.metadata.strategy == "blocked_ab");

    // Blocked schedule should NOT be interleaved
    auto analysis = ScheduleAnalysis::analyze(result);
    REQUIRE_FALSE(analysis.is_interleaved);

    // Should have consecutive blocks of A and B
    REQUIRE(analysis.max_consecutive_a > 3);
}

TEST_CASE("MatMulScheduleGenerator prefetch_next", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 64;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::PREFETCH_NEXT;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);
    REQUIRE(result.metadata.strategy == "prefetch_next");

    // Prefetch strategy has more loads due to prefetching
    Size expected_k_tiles = 4;  // 64/16
    Size expected_c_tiles = 4;  // 2x2

    // Each C tile has k_tiles iterations, and each loads A+B
    // Plus prefetch loads for all but the last iteration
    size_t load_ops = result.count_ops(ScheduleOpType::LOAD);
    REQUIRE(load_ops > expected_k_tiles * 2 * expected_c_tiles);  // More due to prefetch
}

TEST_CASE("MatMulScheduleGenerator address calculation", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.a_base = 0x1000;
    config.b_base = 0x2000;
    config.c_base = 0x3000;
    config.strategy = MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);

    // Find A and B tile addresses
    std::set<Address> a_addrs, b_addrs, c_addrs;

    for (const auto& op : result.operations) {
        if (op.type == ScheduleOpType::LOAD || op.type == ScheduleOpType::STORE) {
            switch (op.matrix()) {
                case MatrixID::A: a_addrs.insert(op.tile.dram_address); break;
                case MatrixID::B: b_addrs.insert(op.tile.dram_address); break;
                case MatrixID::C: c_addrs.insert(op.tile.dram_address); break;
            }
        }
    }

    // A addresses should start at a_base
    REQUIRE(*a_addrs.begin() >= config.a_base);
    // B addresses should start at b_base
    REQUIRE(*b_addrs.begin() >= config.b_base);
    // C addresses should start at c_base
    REQUIRE(*c_addrs.begin() >= config.c_base);
}

TEST_CASE("MatMulScheduleGenerator name and description", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 128;
    config.N = 256;
    config.K = 64;

    MatMulScheduleGenerator gen(config);

    REQUIRE(gen.name() == "MatMulScheduleGenerator");
    REQUIRE(gen.description().find("128") != std::string::npos);
    REQUIRE(gen.description().find("256") != std::string::npos);
    REQUIRE(gen.description().find("64") != std::string::npos);
}

TEST_CASE("MatMulScheduleGenerator large problem", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 256;
    config.N = 256;
    config.K = 256;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);

    // 16x16x16 = 4096 output tiles
    Size m_tiles = 256 / 16;  // 16
    Size n_tiles = 256 / 16;  // 16
    [[maybe_unused]] Size k_tiles = 256 / 16;  // 16

    REQUIRE(result.metadata.c_tiles == m_tiles * n_tiles);
    REQUIRE(result.size() > 10000);  // Large schedule

    // Still should be interleaved
    auto analysis = ScheduleAnalysis::analyze(result);
    REQUIRE(analysis.is_interleaved);
}

TEST_CASE("MatMulScheduleGenerator interleaved is livelock-free", "[timing][schedule][matmul][livelock]") {
    // This test verifies that the interleaved schedule has good balance
    MatMulScheduleGenerator::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 128;  // More K = more A/B tiles
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);

    auto analysis = ScheduleAnalysis::analyze(result);

    // A and B ops should be roughly balanced
    size_t a_ops = analysis.a_ops;
    size_t b_ops = analysis.b_ops;

    // Ratio should be close to 1:1 (within 2x)
    double ratio = static_cast<double>(a_ops) / static_cast<double>(b_ops);
    REQUIRE(ratio >= 0.5);
    REQUIRE(ratio <= 2.0);

    // Should be interleaved
    REQUIRE(analysis.is_interleaved);
}

TEST_CASE("MatMulScheduleGenerator default strategy is interleaved", "[timing][schedule][matmul]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    // Don't set strategy - should default to INTERLEAVED_AB

    MatMulScheduleGenerator gen(config);
    auto result = gen.generate();

    REQUIRE(result.valid);
    REQUIRE(result.metadata.strategy == "interleaved_ab");
}

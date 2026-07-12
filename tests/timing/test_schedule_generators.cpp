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
#include <sw/kpu/timing/schedule/elementwise_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/softmax_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/layernorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>
#include <sw/kpu/timing/schedule/schedule_validator.hpp>

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

    // Prefetch shifts loads one iteration earlier but must NOT duplicate
    // them: every LOAD pairs 1:1 with a MOVE (each load inserts an L3
    // TagCAM reference, each move consumes one; duplicate loads strand
    // references and L3 credits -> livelock at scale, see #61/#64)
    Size expected_k_tiles = 4;  // 64/16
    Size expected_c_tiles = 4;  // 2x2

    size_t load_ops = result.count_ops(ScheduleOpType::LOAD);
    size_t move_ops = result.count_ops(ScheduleOpType::MOVE);
    REQUIRE(load_ops == expected_k_tiles * 2 * expected_c_tiles);
    REQUIRE(load_ops == move_ops);
}

TEST_CASE("MatMulScheduleGenerator blocked_ab derives bursts from the resource envelope",
          "[timing][schedule][matmul][envelope]") {
    // 128^3 at 16^3 tiles: k_tiles = 8
    MatMulScheduleGenerator::Config config;
    config.M = 128;
    config.N = 128;
    config.K = 128;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;

    SECTION("constrained envelope chunks the K loop") {
        // share = min(8/4, 16/4) = 2 tiles -> bursts of 2 tiles
        // (= at most 4 consecutive same-matrix ops: 2 x (LOAD + MOVE))
        config.l3_buffer_count = 8;
        config.l2_bank_count = 16;
        REQUIRE(config.max_burst_tiles() == 2);

        MatMulScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        auto analysis = ScheduleAnalysis::analyze(schedule);
        REQUIRE(analysis.max_consecutive_a <= 4);
        REQUIRE(analysis.max_consecutive_b <= 4);
        REQUIRE(is_livelock_safe(schedule, 8, 16));

        // Blocking must not change total work: LOAD:MOVE:FEED stay 1:1:1
        size_t loads = schedule.count_ops(ScheduleOpType::LOAD);
        REQUIRE(loads == schedule.count_ops(ScheduleOpType::MOVE));
        REQUIRE(loads == schedule.count_ops(ScheduleOpType::FEED));
        REQUIRE(loads == 2u * 8u * 64u);  // 2 matrices x k_tiles x c_tiles
    }

    SECTION("large envelope degenerates to the full K loop") {
        // share = min(32/4, 64/4) = 8 tiles >= k_tiles -> single block,
        // identical structure to the historical all-A-then-all-B ordering
        config.l3_buffer_count = 32;
        config.l2_bank_count = 64;
        REQUIRE(config.max_burst_tiles() == 8);

        MatMulScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        auto analysis = ScheduleAnalysis::analyze(schedule);
        REQUIRE(analysis.max_consecutive_a == 16);  // 8 tiles x (LOAD + MOVE)
        REQUIRE(is_livelock_safe(schedule, 32, 64));
    }

    SECTION("degenerate envelope still makes progress") {
        // share clamps to 1 tile
        config.l3_buffer_count = 2;
        config.l2_bank_count = 2;
        REQUIRE(config.max_burst_tiles() == 1);

        MatMulScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        auto analysis = ScheduleAnalysis::analyze(schedule);
        REQUIRE(analysis.max_consecutive_a <= 2);
        REQUIRE(is_livelock_safe(schedule, 2, 2));
    }
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

// ============================================================================
// Resource envelope on the non-matmul generators (issue #90): generation
// refuses a priori when the schedule's implied working set exceeds the
// envelope share, instead of wedging at runtime
// ============================================================================

TEST_CASE("SoftmaxScheduleGenerator enforces its multi-pass working set against the envelope",
          "[timing][schedule][softmax][envelope]") {
    SoftmaxScheduleGenerator::Config config;
    config.batch_size = 32;
    config.Ti = 16;
    config.Tj = 16;

    SECTION("small reduction fits the default envelope") {
        config.reduction_dim = 96;   // 6 tiles + 2 scratch = 8 <= share 8
        REQUIRE(config.required_working_set() == 8);
        SoftmaxScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }

    SECTION("transformer-scale reduction is refused under the default envelope") {
        config.reduction_dim = 1024;  // 64 tiles + 2 scratch = 66 > share 8
        SoftmaxScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }

    SECTION("the same reduction generates under an adequate envelope") {
        config.reduction_dim = 1024;
        config.l3_buffer_count = 512;
        config.l2_bank_count = 512;   // share 128 >= 66
        SoftmaxScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }
}

TEST_CASE("LayerNormScheduleGenerator enforces resident affine params against the envelope",
          "[timing][schedule][layernorm][envelope]") {
    LayerNormScheduleGenerator::Config config;
    config.batch_size = 4;
    config.sequence_length = 64;
    config.hidden_size = 768;   // 48 hidden tiles
    config.Ti = 16;
    config.Tj = 16;

    SECTION("affine params exceed the default envelope share") {
        config.affine = true;   // 2*48 + 3 = 99 > share 8
        LayerNormScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }

    SECTION("non-affine streaming fits the default envelope") {
        config.affine = false;  // working set 3
        LayerNormScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }

    SECTION("affine generates under an adequate envelope") {
        config.affine = true;
        config.l3_buffer_count = 512;
        config.l2_bank_count = 512;   // share 128 >= 98
        LayerNormScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }
}

TEST_CASE("BatchNorm and Conv2D carry the envelope and refuse degenerate shares",
          "[timing][schedule][envelope]") {
    SECTION("batchnorm training fits the default envelope") {
        BatchNormScheduleGenerator::Config config;
        config.N = 4; config.C = 8; config.H = 16; config.W = 16;
        config.training = true;
        REQUIRE(config.required_working_set() == 5);
        BatchNormScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }

    SECTION("batchnorm inference all-channel preload is refused at default envelope") {
        BatchNormScheduleGenerator::Config config;
        config.N = 4; config.C = 8; config.H = 16; config.W = 16;
        config.training = false;   // preloads 4*C params: 33 > share 8
        REQUIRE(config.required_working_set() == 33);
        BatchNormScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }

    SECTION("batchnorm inference generates under an adequate envelope") {
        BatchNormScheduleGenerator::Config config;
        config.N = 4; config.C = 8; config.H = 16; config.W = 16;
        config.training = false;
        config.l3_buffer_count = 256;
        config.l2_bank_count = 256;   // share 64 >= 33
        BatchNormScheduleGenerator gen(config);
        REQUIRE(gen.generate().valid);
    }

    SECTION("batchnorm training refused under a degenerate envelope") {
        BatchNormScheduleGenerator::Config config;
        config.N = 4; config.C = 8; config.H = 16; config.W = 16;
        config.training = true;
        config.l3_buffer_count = 8;
        config.l2_bank_count = 8;    // share 2 < 5
        BatchNormScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }

    SECTION("conv2d streams within the default envelope") {
        Conv2DScheduleGenerator::Config config;
        config.N = 1; config.H_in = 32; config.W_in = 32; config.C_in = 16;
        config.C_out = 16; config.Kh = 3; config.Kw = 3;
        config.padding_h = 1; config.padding_w = 1;
        Conv2DScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);
        // Fine-grained interleaving: capacity-aware safety holds at defaults
        REQUIRE(is_livelock_safe(schedule, 32, 64));
    }

    SECTION("conv2d refused under a degenerate envelope") {
        Conv2DScheduleGenerator::Config config;
        config.l3_buffer_count = 4;
        config.l2_bank_count = 4;    // share 1 < 3
        Conv2DScheduleGenerator gen(config);
        REQUIRE_FALSE(gen.generate().valid);
    }
}

// ============================================================================
// Envelope-mismatch detection (issue #91): generators stamp their envelope
// into the metadata; ScheduleExecutor and VAL-007 surface disagreements
// ============================================================================

TEST_CASE("Generators record their envelope in schedule metadata", "[timing][schedule][envelope]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32; config.N = 32; config.K = 32;
    config.Ti = 16; config.Tj = 16; config.Tk = 16;
    config.l3_buffer_count = 16;
    config.l2_bank_count = 24;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);
    REQUIRE(schedule.metadata.l3_buffer_count == 16);
    REQUIRE(schedule.metadata.l2_bank_count == 24);
}

TEST_CASE("ScheduleExecutor warns when the execution envelope differs", "[timing][schedule][envelope]") {
    MatMulScheduleGenerator::Config gen_config;
    gen_config.M = 32; gen_config.N = 32; gen_config.K = 32;
    gen_config.Ti = 16; gen_config.Tj = 16; gen_config.Tk = 16;
    gen_config.l3_buffer_count = 32;
    gen_config.l2_bank_count = 64;

    MatMulScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config exec_config;
    exec_config.max_cycles = 1'000'000;

    SECTION("matching envelope executes without warnings") {
        exec_config.l3_buffer_count = 32;
        exec_config.l2_bank_count = 64;
        ConcurrentTimingExecutor executor(exec_config);
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(schedule);
        REQUIRE(result.success);
        REQUIRE(result.warnings.empty());
    }

    SECTION("smaller executor pools produce a may-wedge warning") {
        exec_config.l3_buffer_count = 8;
        exec_config.l2_bank_count = 16;
        ConcurrentTimingExecutor executor(exec_config);
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(schedule);
        REQUIRE(result.warnings.size() == 1);
        REQUIRE(result.warnings[0].find("SMALLER") != std::string::npos);
        REQUIRE(result.warnings[0].find("may wedge") != std::string::npos);
    }

    SECTION("larger executor pools produce a benign mismatch warning") {
        exec_config.l3_buffer_count = 128;
        exec_config.l2_bank_count = 128;
        ConcurrentTimingExecutor executor(exec_config);
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(schedule);
        REQUIRE(result.success);
        REQUIRE(result.warnings.size() == 1);
        REQUIRE(result.warnings[0].find("larger") != std::string::npos);
    }

    SECTION("hand-built schedules without a recorded envelope are not flagged") {
        ScheduleResult hand_built = schedule;
        hand_built.metadata.l3_buffer_count = 0;
        hand_built.metadata.l2_bank_count = 0;
        exec_config.l3_buffer_count = 8;
        exec_config.l2_bank_count = 16;
        ConcurrentTimingExecutor executor(exec_config);
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(hand_built);
        REQUIRE(result.warnings.empty());
    }
}

TEST_CASE("VAL-007 flags envelope disagreement in validation", "[timing][schedule][envelope][validation]") {
    MatMulScheduleGenerator::Config config;
    config.M = 32; config.N = 32; config.K = 32;
    config.Ti = 16; config.Tj = 16; config.Tk = 16;
    config.l3_buffer_count = 32;
    config.l2_bank_count = 64;

    MatMulScheduleGenerator gen(config);
    auto schedule = gen.generate();

    ScheduleValidator validator;

    SECTION("matching pools raise no VAL-007") {
        auto result = validator.validate_livelock_safety(schedule, 32, 64);
        for (const auto& issue : result.issues) {
            REQUIRE(issue.rule_id != "VAL-007");
        }
    }

    SECTION("smaller pools raise VAL-007 with the guarantees-void note") {
        auto result = validator.validate_livelock_safety(schedule, 8, 16);
        bool found = false;
        for (const auto& issue : result.issues) {
            if (issue.rule_id == "VAL-007") {
                found = true;
                REQUIRE(issue.message.find("smaller") != std::string::npos);
            }
        }
        REQUIRE(found);
    }
}

// ============================================================================
// ElementwiseScheduleGenerator (issue #101, epic E2): paired two-stream
// emission, broadcast delivery, and executable COMPUTEs (resolves #139
// for the elementwise family)
// ============================================================================

TEST_CASE("ElementwiseScheduleGenerator emits paired executable schedules",
          "[timing][schedule][elementwise]") {
    ElementwiseScheduleGenerator::Config config;
    config.num_elements = 1024;
    config.tile_elems = 256;   // 4 data tiles

    SECTION("binary form pairs the streams and carries both-operand deps") {
        config.form = ElementwiseScheduleGenerator::Form::BINARY;
        ElementwiseScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == 8);   // 4 A + 4 B
        REQUIRE(schedule.count_ops(ScheduleOpType::MOVE) == 8);
        REQUIRE(schedule.count_ops(ScheduleOpType::FEED) == 8);
        REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == 4);

        // Paired interleave: no same-matrix monopolization
        auto analysis = ScheduleAnalysis::analyze(schedule);
        REQUIRE(analysis.max_consecutive_a <= 2);
        REQUIRE(analysis.max_consecutive_b <= 2);
        REQUIRE(is_livelock_safe(schedule, config.l3_buffer_count,
                                 config.l2_bank_count));

        // Every COMPUTE depends on exactly its A and B pair (executable,
        // unlike the #139-affected generators)
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::COMPUTE) continue;
            REQUIRE(op.dependency_tiles.size() == 2);
            REQUIRE(op.dependency_tiles[0].matrix == MatrixID::A);
            REQUIRE(op.dependency_tiles[1].matrix == MatrixID::B);
            REQUIRE(op.dependency_tiles[0].ti == op.tile.tile_id.ti);
        }

        // Envelope stamped (issue #91)
        REQUIRE(schedule.metadata.l3_buffer_count == 32);
        REQUIRE(schedule.metadata.l2_bank_count == 64);
    }

    SECTION("broadcast form delivers B once with a seeded consumer count") {
        config.form = ElementwiseScheduleGenerator::Form::BROADCAST_B;
        ElementwiseScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        // One B load/move; four B feeds (one per consumer)
        size_t b_loads = 0, b_moves = 0, b_feeds = 0;
        for (const auto& op : schedule.operations) {
            if (op.tile.tile_id.matrix != MatrixID::B) continue;
            if (op.type == ScheduleOpType::LOAD) ++b_loads;
            if (op.type == ScheduleOpType::MOVE) {
                ++b_moves;
                REQUIRE(op.tile.consumer_count == 4);  // 1:1:k discipline
            }
            if (op.type == ScheduleOpType::FEED) ++b_feeds;
        }
        REQUIRE(b_loads == 1);
        REQUIRE(b_moves == 1);
        REQUIRE(b_feeds == 4);

        // Every COMPUTE depends on its A tile AND the broadcast tile
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::COMPUTE) continue;
            REQUIRE(op.dependency_tiles.size() == 2);
            REQUIRE(op.dependency_tiles[1].matrix == MatrixID::B);
        }
    }

    SECTION("unary form streams a single operand") {
        config.form = ElementwiseScheduleGenerator::Form::UNARY;
        ElementwiseScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);
        REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == 4);
        REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == 4);
    }

    SECTION("degenerate envelope is refused a priori") {
        config.form = ElementwiseScheduleGenerator::Form::BINARY;
        config.l3_buffer_count = 4;
        config.l2_bank_count = 4;   // share 1 < working set 3
        ElementwiseScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }
}

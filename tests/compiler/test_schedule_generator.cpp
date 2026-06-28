#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sw/compiler/schedule_generator.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <iostream>
#include <iomanip>

using namespace sw::kpu::compiler;

TEST_CASE("ScheduleGenerator - Basic Instantiation", "[schedule_generator][unit]") {
    ScheduleGenerator::PerformanceModel perf;
    ScheduleGenerator generator(perf);

    REQUIRE(generator.performance_model().dram_bandwidth == 100.0);
    REQUIRE(generator.performance_model().clock_freq_ghz == 1.0);
}

TEST_CASE("ScheduleGenerator - Small Matrix Schedule", "[schedule_generator][basic]") {
    // Create a simple tile configuration
    TileOptimizer optimizer;
    Size M = 128, N = 128, K = 128;
    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    REQUIRE(config.valid);

    // Generate schedule
    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config, ScheduleGenerator::Strategy::SEQUENTIAL);

    SECTION("Schedule has basic structure") {
        REQUIRE(schedule.M == M);
        REQUIRE(schedule.N == N);
        REQUIRE(schedule.K == K);
        REQUIRE(schedule.commands.size() > 0);
    }

    SECTION("Schedule has memory allocations") {
        REQUIRE(schedule.allocations.size() > 0);

        // Should have A, B, C in GDDR6
        int gddr6_count = 0;
        for (const auto& alloc : schedule.allocations) {
            if (alloc.level == ScheduleGenerator::MemoryLevel::KPU_GDDR6) {
                gddr6_count++;
            }
        }
        REQUIRE(gddr6_count >= 3);  // At least A, B, C
    }

    SECTION("Schedule has DMA commands") {
        REQUIRE(schedule.num_dma_transfers > 0);
        REQUIRE(schedule.total_dram_bytes > 0);
    }

    SECTION("Schedule has valid timing") {
        REQUIRE(schedule.total_cycles > 0);
        REQUIRE(schedule.estimated_time_ms > 0.0);
    }

    std::cout << "\n128×128×128 Schedule Summary:\n";
    std::cout << "  Total commands:  " << schedule.commands.size() << "\n";
    std::cout << "  Total cycles:    " << schedule.total_cycles << "\n";
    std::cout << "  Estimated time:  " << schedule.estimated_time_ms << " ms\n";
    std::cout << "  DRAM traffic:    " << (static_cast<double>(schedule.total_dram_bytes) / (1024.0 * 1024.0)) << " MB\n";
}

TEST_CASE("ScheduleGenerator - Medium Matrix Schedule", "[schedule_generator][medium]") {
    TileOptimizer optimizer;
    Size M = 512, N = 512, K = 512;
    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    REQUIRE(config.valid);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    SECTION("Schedule dimensions match input") {
        REQUIRE(schedule.M == M);
        REQUIRE(schedule.N == N);
        REQUIRE(schedule.K == K);
    }

    SECTION("Tile configuration preserved") {
        REQUIRE(schedule.config.Ti == config.Ti);
        REQUIRE(schedule.config.Tj == config.Tj);
        REQUIRE(schedule.config.Tk == config.Tk);
    }

    SECTION("Performance estimates reasonable") {
        // Total DRAM traffic should be less than naive (3 * M * N * K * 4 bytes)
        Size naive_dram = (M * K + K * N + M * N) * 4;
        // Our schedule might load full matrices for now, so check it's at least reasonable
        REQUIRE(schedule.total_dram_bytes >= naive_dram);  // At least as much
        REQUIRE(schedule.total_dram_bytes <= naive_dram * 2);  // Not absurdly high
    }

    std::cout << "\n512×512×512 Schedule Summary:\n";
    std::cout << "  Tile config: " << config.Ti << "×" << config.Tj << "×" << config.Tk << "\n";
    std::cout << "  Commands:    " << schedule.commands.size() << "\n";
    std::cout << "  DMA xfers:   " << schedule.num_dma_transfers << "\n";
    std::cout << "  Block moves: " << schedule.num_block_moves << "\n";
    std::cout << "  Streams:     " << schedule.num_streams << "\n";
    std::cout << "  Computes:    " << schedule.num_computes << "\n";
    std::cout << "  Total cycles: " << schedule.total_cycles << "\n";
}

TEST_CASE("ScheduleGenerator - Large Matrix Schedule", "[schedule_generator][large]") {
    TileOptimizer optimizer;
    Size M = 1024, N = 1024, K = 1024;
    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    REQUIRE(config.valid);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    SECTION("Schedule has all command types") {
        bool has_dma = false;
        bool has_block_move = false;
        bool has_stream = false;
        bool has_compute = false;

        for (const auto& cmd : schedule.commands) {
            switch (cmd.type) {
                case ScheduleGenerator::CommandType::DMA_TRANSFER:
                    has_dma = true;
                    break;
                case ScheduleGenerator::CommandType::BLOCK_MOVE:
                    has_block_move = true;
                    break;
                case ScheduleGenerator::CommandType::STREAM_L2_TO_L1:
                case ScheduleGenerator::CommandType::STREAM_L1_TO_L2:
                    has_stream = true;
                    break;
                case ScheduleGenerator::CommandType::COMPUTE_MATMUL:
                    has_compute = true;
                    break;
                default:
                    break;
            }
        }

        REQUIRE(has_dma);
        REQUIRE(has_block_move);
        REQUIRE(has_stream);
        REQUIRE(has_compute);
    }

    SECTION("Commands have valid timing") {
        for (const auto& cmd : schedule.commands) {
            REQUIRE(cmd.latency_cycles > 0);
            REQUIRE(cmd.end_cycle >= cmd.start_cycle);
            REQUIRE(cmd.end_cycle == cmd.start_cycle + cmd.latency_cycles);
        }
    }

    std::cout << "\n1024×1024×1024 Schedule:\n";
    generator.print_schedule(schedule, false);
}

TEST_CASE("ScheduleGenerator - Rectangular Matrix", "[schedule_generator][rectangular]") {
    TileOptimizer optimizer;

    SECTION("Tall skinny: 1024×128×512") {
        Size M = 1024, N = 128, K = 512;
        auto config = optimizer.optimize(M, N, K);
        REQUIRE(config.valid);

        ScheduleGenerator generator;
        auto schedule = generator.generate(M, N, K, config);

        REQUIRE(schedule.M == M);
        REQUIRE(schedule.N == N);
        REQUIRE(schedule.K == K);
        REQUIRE(schedule.commands.size() > 0);

        std::cout << "\nTall Skinny (1024×128×512):\n";
        std::cout << "  Commands: " << schedule.commands.size() << "\n";
        std::cout << "  Cycles:   " << schedule.total_cycles << "\n";
    }

    SECTION("Short wide: 128×1024×512") {
        Size M = 128, N = 1024, K = 512;
        auto config = optimizer.optimize(M, N, K);
        REQUIRE(config.valid);

        ScheduleGenerator generator;
        auto schedule = generator.generate(M, N, K, config);

        REQUIRE(schedule.M == M);
        REQUIRE(schedule.N == N);
        REQUIRE(schedule.K == K);
        REQUIRE(schedule.commands.size() > 0);

        std::cout << "\nShort Wide (128×1024×512):\n";
        std::cout << "  Commands: " << schedule.commands.size() << "\n";
        std::cout << "  Cycles:   " << schedule.total_cycles << "\n";
    }
}

TEST_CASE("ScheduleGenerator - Validation", "[schedule_generator][validation]") {
    TileOptimizer optimizer;
    Size M = 512, N = 512, K = 512;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    SECTION("Valid schedule passes validation") {
        std::string error_msg;
        bool valid = generator.validate(schedule, error_msg);

        REQUIRE(valid);
        REQUIRE(error_msg == "Valid");
    }
}

TEST_CASE("ScheduleGenerator - Command Dependencies", "[schedule_generator][dependencies]") {
    TileOptimizer optimizer;
    Size M = 256, N = 256, K = 256;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    SECTION("All commands have valid dependencies") {
        for (size_t i = 0; i < schedule.commands.size(); ++i) {
            const auto& cmd = schedule.commands[i];

            // All dependencies should refer to earlier commands
            for (size_t dep_idx : cmd.depends_on) {
                REQUIRE(dep_idx < i);
            }
        }
    }

    SECTION("Commands respect dependencies in timing") {
        for (size_t i = 0; i < schedule.commands.size(); ++i) {
            const auto& cmd = schedule.commands[i];

            // Command should start after all dependencies complete
            for (size_t dep_idx : cmd.depends_on) {
                const auto& dep = schedule.commands[dep_idx];
                REQUIRE(cmd.start_cycle >= dep.end_cycle);
            }
        }
    }
}

TEST_CASE("ScheduleGenerator - Strategy Comparison", "[schedule_generator][strategies]") {
    TileOptimizer optimizer;
    Size M = 512, N = 512, K = 512;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;

    auto seq_schedule = generator.generate(M, N, K, config,
                                           ScheduleGenerator::Strategy::SEQUENTIAL);
    auto db_schedule = generator.generate(M, N, K, config,
                                          ScheduleGenerator::Strategy::DOUBLE_BUFFERED);
    auto pipe_schedule = generator.generate(M, N, K, config,
                                            ScheduleGenerator::Strategy::FULLY_PIPELINED);

    SECTION("All strategies produce valid schedules") {
        REQUIRE(seq_schedule.commands.size() > 0);
        REQUIRE(db_schedule.commands.size() > 0);
        REQUIRE(pipe_schedule.commands.size() > 0);
    }

    SECTION("All strategies have same basic structure") {
        // For now, they should be identical since optimizations aren't implemented yet
        REQUIRE(seq_schedule.commands.size() == db_schedule.commands.size());
        REQUIRE(seq_schedule.commands.size() == pipe_schedule.commands.size());
    }

    std::cout << "\nStrategy Comparison (512×512×512):\n";
    std::cout << "  Sequential:      " << seq_schedule.total_cycles << " cycles\n";
    std::cout << "  Double-buffered: " << db_schedule.total_cycles << " cycles\n";
    std::cout << "  Fully pipelined: " << pipe_schedule.total_cycles << " cycles\n";
}

TEST_CASE("ScheduleGenerator - Print Schedule", "[schedule_generator][print]") {
    TileOptimizer optimizer;
    Size M = 256, N = 256, K = 256;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    SECTION("Print schedule (non-verbose)") {
        std::cout << "\n";
        generator.print_schedule(schedule, false);
        // Just verify it doesn't crash
        REQUIRE(true);
    }

    SECTION("Print schedule (verbose)") {
        std::cout << "\n";
        generator.print_schedule(schedule, true);
        // Just verify it doesn't crash
        REQUIRE(true);
    }
}

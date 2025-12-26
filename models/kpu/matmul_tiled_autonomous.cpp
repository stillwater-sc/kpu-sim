/**
 * @file matmul_tiled_autonomous.cpp
 * @brief Configurable tiled matrix multiplication on autonomous KPU
 *
 * Features:
 * - Configurable matrix dimensions (M×K) × (K×N) = (M×N)
 * - Automatic tiling for 16×16 systolic array
 * - Support for square, rectangular, and skinny tensors
 * - Autonomous execution with signal-based orchestration
 * - Performance metrics and validation
 */

#include <sw/system/toplevel.hpp>
#include <sw/system/config_loader.hpp>
#include <sw/system/config_formatter.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include "autonomous_orchestrator.hpp"
#include "kpu_profiler.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace sw::sim;
using namespace sw::kpu;

/**
 * @brief Configuration for tiled matrix multiplication
 */
struct MatMulConfig {
    size_t M;  // Rows of A, rows of C
    size_t K;  // Cols of A, rows of B
    size_t N;  // Cols of B, cols of C

    size_t tile_size;  // Systolic array dimension (16 for 16×16)

    bool verbose;
    bool validate;
    bool profile;
    bool show_timeline;

    // Performance tracking
    size_t total_cycles;
    double execution_time_ms;
    mutable double gflops;  // mutable to allow calculation in const print method

    MatMulConfig(size_t m, size_t k, size_t n, size_t tile = 16,
                 bool v = false, bool val = true, bool prof = false, bool timeline = false)
        : M(m), K(k), N(n), tile_size(tile), verbose(v), validate(val),
          profile(prof), show_timeline(timeline),
          total_cycles(0), execution_time_ms(0.0), gflops(0.0) {}

    void print() const {
        std::cout << "\n========================================\n";
        std::cout << "  Tiled Matrix Multiplication Config\n";
        std::cout << "========================================\n";
        std::cout << "Matrix dimensions:\n";
        std::cout << "  A: " << M << " x " << K << "\n";
        std::cout << "  B: " << K << " x " << N << "\n";
        std::cout << "  C: " << M << " x " << N << "\n";
        std::cout << "\nTiling:\n";
        std::cout << "  Tile size: " << tile_size << " × " << tile_size << "\n";
        std::cout << "  M tiles: " << (M + tile_size - 1) / tile_size << "\n";
        std::cout << "  K tiles: " << (K + tile_size - 1) / tile_size << "\n";
        std::cout << "  N tiles: " << (N + tile_size - 1) / tile_size << "\n";
        std::cout << "  Total tiles: " <<
            ((M + tile_size - 1) / tile_size) *
            ((K + tile_size - 1) / tile_size) *
            ((N + tile_size - 1) / tile_size) << "\n";
        std::cout << "\nOptions:\n";
        std::cout << "  Verbose: " << (verbose ? "ON" : "OFF") << "\n";
        std::cout << "  Validate: " << (validate ? "ON" : "OFF") << "\n";
        std::cout << "  Profile: " << (profile ? "ON" : "OFF") << "\n";
        std::cout << "  Timeline: " << (show_timeline ? "ON" : "OFF") << "\n";
        std::cout << "========================================\n";
    }

    void print_performance() const {
        std::cout << "\n========================================\n";
        std::cout << "  Performance Metrics\n";
        std::cout << "========================================\n";
        std::cout << "Total cycles: " << total_cycles << "\n";
        std::cout << "Execution time: " << std::fixed << std::setprecision(3)
                  << execution_time_ms << " ms\n";

        // Calculate theoretical FLOPs: 2*M*N*K (multiply-add counts as 2 ops)
        double flops = 2.0 * M * N * K;
        if (execution_time_ms > 0) {
            gflops = (flops / 1e9) / (execution_time_ms / 1000.0);
            std::cout << "Performance: " << std::fixed << std::setprecision(2)
                      << gflops << " GFLOPS\n";
        }

        // Calculate utilization
        double theoretical_cycles = static_cast<double>(std::max(M, N) * std::max(K, N));
        double utilization = (theoretical_cycles / total_cycles) * 100.0;
        std::cout << "Array utilization: " << std::fixed << std::setprecision(1)
                  << utilization << "%\n";
        std::cout << "========================================\n";
    }
};

/**
 * @brief Initialize matrix with test pattern
 */
void initialize_matrix(std::vector<float>& matrix, size_t rows, size_t cols,
                      const std::string& pattern = "sequential") {
    if (pattern == "sequential") {
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = static_cast<float>(i % 100) * 0.01f;
        }
    } else if (pattern == "identity") {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i * cols + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    } else if (pattern == "ones") {
        std::fill(matrix.begin(), matrix.end(), 1.0f);
    } else if (pattern == "random") {
        for (auto& val : matrix) {
            val = static_cast<float>(rand() % 100) / 100.0f;
        }
    }
}

/**
 * @brief CPU reference implementation for validation
 */
void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Validate KPU results against CPU reference
 */
bool validate_results(const std::vector<float>& kpu_result,
                     const std::vector<float>& cpu_result,
                     float tolerance = 1e-3f) {
    if (kpu_result.size() != cpu_result.size()) {
        std::cerr << "ERROR: Size mismatch: KPU=" << kpu_result.size()
                  << ", CPU=" << cpu_result.size() << "\n";
        return false;
    }

    size_t error_count = 0;
    float max_error = 0.0f;
    size_t first_error_idx = 0;

    for (size_t i = 0; i < kpu_result.size(); ++i) {
        float error = std::abs(kpu_result[i] - cpu_result[i]);
        if (error > tolerance) {
            if (error_count == 0) {
                first_error_idx = i;
            }
            error_count++;
            max_error = std::max(max_error, error);
        }
    }

    if (error_count > 0) {
        std::cerr << "VALIDATION FAILED:\n";
        std::cerr << "  Errors: " << error_count << " / " << kpu_result.size() << "\n";
        std::cerr << "  Max error: " << max_error << "\n";
        std::cerr << "  First error at index " << first_error_idx << ":\n";
        std::cerr << "    Expected: " << cpu_result[first_error_idx] << "\n";
        std::cerr << "    Got: " << kpu_result[first_error_idx] << "\n";
        return false;
    }

    std::cout << " VALIDATION PASSED: All " << kpu_result.size()
              << " elements within tolerance (" << tolerance << ")\n";
    return true;
}

/**
 * @brief Execute tiled matrix multiplication using autonomous KPU
 */
bool execute_tiled_matmul(KPUSimulator* kpu, MatMulConfig& config) {
    config.print();

    // Initialize profiler
    KPUProfiler profiler(config.profile);

    // Allocate host matrices
    std::cout << "\n[1] Allocating matrices...\n";
    std::vector<float> A(config.M * config.K);
    std::vector<float> B(config.K * config.N);
    std::vector<float> C(config.M * config.N, 0.0f);

    initialize_matrix(A, config.M, config.K, "sequential");
    initialize_matrix(B, config.K, config.N, "sequential");

    std::cout << "  A: " << (A.size() * sizeof(float) / 1024.0f) << " KB\n";
    std::cout << "  B: " << (B.size() * sizeof(float) / 1024.0f) << " KB\n";
    std::cout << "  C: " << (C.size() * sizeof(float) / 1024.0f) << " KB\n";

    // CPU reference (if validation enabled)
    std::vector<float> C_ref;
    if (config.validate) {
        std::cout << "\n[2] Computing CPU reference...\n";
        C_ref.resize(config.M * config.N);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(A, B, C_ref, config.M, config.K, config.N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << "  CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " ms\n";
    }

    // KPU execution
    std::cout << "\n[3] Loading data to KPU...\n";
    const size_t bank_id = 0;
    const Address bank_a_addr = 0x0000;
    const Address bank_b_addr = bank_a_addr + A.size() * sizeof(float);
    const Address bank_c_addr = bank_b_addr + B.size() * sizeof(float);

    kpu->write_memory_bank(bank_id, bank_a_addr, A.data(), A.size() * sizeof(float));
    kpu->write_memory_bank(bank_id, bank_b_addr, B.data(), B.size() * sizeof(float));
    std::cout << "  Data loaded to memory bank " << bank_id << "\n";

    // TODO: Implement tiled execution with autonomous orchestration
    // For now, implement single-tile or simple tiling approach

    std::cout << "\n[4] KPU Execution\n";
    std::cout << "  NOTE: Full tiled implementation in progress\n";
    std::cout << "  Current: Single tile execution for matrices <= 16×16\n";

    // Check if matrices fit in single tile
    if (config.M <= config.tile_size && config.K <= config.tile_size &&
        config.N <= config.tile_size) {

        std::cout << "  Executing single-tile matmul...\n";

        // Simple single-tile execution (similar to autonomous test)
        const size_t l3_tile_id = 0;
        const size_t l2_bank_id = 0;
        const size_t l1_buffer_id = 0;
        const size_t block_mover_id = 0;
        const size_t compute_tile_id = 0;

        const Address l3_a_addr = 0x0000;
        const Address l3_b_addr = 0x4000;
        const Address l2_a_addr = 0x0000;
        const Address l2_b_addr = 0x2000;
        const Address l1_a_addr = 0x0000;
        const Address l1_b_addr = 0x1000;
        const Address l1_c_addr = 0x2000;

        // Simple pipeline: Bank→L3→L2→L1→Compute→L1→L2→L3→Bank
        std::vector<uint8_t> temp_buffer(std::max(A.size(), B.size()) * sizeof(float));

        // Bank→L3
        kpu->read_memory_bank(bank_id, bank_a_addr, temp_buffer.data(), A.size() * sizeof(float));
        kpu->write_l3_tile(l3_tile_id, l3_a_addr, temp_buffer.data(), A.size() * sizeof(float));

        kpu->read_memory_bank(bank_id, bank_b_addr, temp_buffer.data(), B.size() * sizeof(float));
        kpu->write_l3_tile(l3_tile_id, l3_b_addr, temp_buffer.data(), B.size() * sizeof(float));

        // L3→L2→L1→Compute (synchronous for now)
        auto kpu_start = std::chrono::high_resolution_clock::now();

        // L3→L2
        kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_a_addr, l2_bank_id, l2_a_addr,
                                   config.M, config.K, sizeof(float),
                                   BlockMover::TransformType::IDENTITY, nullptr);
        kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_b_addr, l2_bank_id, l2_b_addr,
                                   config.K, config.N, sizeof(float),
                                   BlockMover::TransformType::IDENTITY, nullptr);
        kpu->run_until_idle();

        // L2→L1
        kpu->start_row_stream(0, l2_bank_id, l1_buffer_id, l2_a_addr, l1_a_addr,
                              config.M, config.K, sizeof(float), config.tile_size,
                              Streamer::StreamDirection::L2_TO_L1, nullptr);
        kpu->start_column_stream(1, l2_bank_id, l1_buffer_id, l2_b_addr, l1_b_addr,
                                 config.K, config.N, sizeof(float), config.tile_size,
                                 Streamer::StreamDirection::L2_TO_L1, nullptr);
        kpu->run_until_idle();

        // Compute
        kpu->start_matmul(compute_tile_id, l1_buffer_id, config.M, config.N, config.K,
                         l1_a_addr, l1_b_addr, l1_c_addr, nullptr);
        kpu->run_until_idle();

        auto kpu_end = std::chrono::high_resolution_clock::now();
        config.execution_time_ms = std::chrono::duration<double, std::milli>(kpu_end - kpu_start).count();
        config.total_cycles = kpu->get_current_cycle();

        // Readback: L1→L2→L3→Bank
        const Address l2_c_addr = 0x4000;
        const Address l3_c_addr = 0x8000;

        kpu->start_row_stream(0, l2_bank_id, l1_buffer_id, l2_c_addr, l1_c_addr,
                              config.M, config.N, sizeof(float), config.tile_size,
                              Streamer::StreamDirection::L1_TO_L2, nullptr);
        kpu->run_until_idle();

        // L2→L3 (manual)
        std::vector<uint8_t> c_buffer(C.size() * sizeof(float));
        kpu->read_l2_bank(l2_bank_id, l2_c_addr, c_buffer.data(), c_buffer.size());
        kpu->write_l3_tile(l3_tile_id, l3_c_addr, c_buffer.data(), c_buffer.size());

        // L3→Bank
        kpu->read_l3_tile(l3_tile_id, l3_c_addr, c_buffer.data(), c_buffer.size());
        kpu->write_memory_bank(bank_id, bank_c_addr, c_buffer.data(), c_buffer.size());

        // Bank→Host
        kpu->read_memory_bank(bank_id, bank_c_addr, C.data(), C.size() * sizeof(float));

        std::cout << "  Execution complete\n";
    } else {
        // Multi-tile execution
        std::cout << "  Executing multi-tile matmul...\n";

        // Calculate number of tiles in each dimension
        size_t m_tiles = (config.M + config.tile_size - 1) / config.tile_size;
        size_t k_tiles = (config.K + config.tile_size - 1) / config.tile_size;
        size_t n_tiles = (config.N + config.tile_size - 1) / config.tile_size;
        size_t total_tiles = m_tiles * k_tiles * n_tiles;

        std::cout << "  Tile grid: " << m_tiles << "×" << k_tiles << "×" << n_tiles
                  << " = " << total_tiles << " tiles\n";

        // Component IDs
        const size_t l3_tile_id = 0;
        const size_t l2_bank_id = 0;
        const size_t l1_buffer_id = 0;
        const size_t block_mover_id = 0;
        const size_t compute_tile_id = 0;

        // L3 addresses for tile storage
        const Address l3_a_base = 0x0000;
        const Address l3_b_base = 0x40000;  // 256KB offset
        const Address l3_c_base = 0x80000;  // 512KB offset

        // L2 addresses
        const Address l2_a_addr = 0x0000;
        const Address l2_b_addr = 0x2000;
        const Address l2_c_addr = 0x4000;

        // L1 (scratchpad) addresses
        const Address l1_a_addr = 0x0000;
        const Address l1_b_addr = 0x1000;
        const Address l1_c_addr = 0x2000;

        // Load all of A and B into L3 (tiled format)
        std::cout << "  Loading matrices to L3...\n";
        auto kpu_start = std::chrono::high_resolution_clock::now();
        Cycle start_cycle = kpu->get_current_cycle();

        // Load A tiles to L3
        for (size_t ti = 0; ti < m_tiles; ++ti) {
            for (size_t tk = 0; tk < k_tiles; ++tk) {
                size_t tile_m = std::min(config.tile_size, config.M - ti * config.tile_size);
                size_t tile_k = std::min(config.tile_size, config.K - tk * config.tile_size);
                size_t tile_bytes = tile_m * tile_k * sizeof(float);

                // Calculate L3 address for this A tile
                Address l3_addr = l3_a_base + (ti * k_tiles + tk) * config.tile_size * config.tile_size * sizeof(float);

                // Extract tile from host matrix A
                std::vector<float> tile_data(tile_m * tile_k);
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t k = 0; k < tile_k; ++k) {
                        size_t global_i = ti * config.tile_size + i;
                        size_t global_k = tk * config.tile_size + k;
                        tile_data[i * tile_k + k] = A[global_i * config.K + global_k];
                    }
                }

                // Write tile to memory bank, then to L3
                Address bank_tile_addr = bank_a_addr + (ti * k_tiles + tk) * config.tile_size * config.tile_size * sizeof(float);
                kpu->write_memory_bank(bank_id, bank_tile_addr, tile_data.data(), tile_bytes);
                kpu->read_memory_bank(bank_id, bank_tile_addr, tile_data.data(), tile_bytes);
                kpu->write_l3_tile(l3_tile_id, l3_addr, tile_data.data(), tile_bytes);

                if (config.profile) {
                    profiler.record_memory_transfer("Bank", "L3", tile_bytes, 1);
                }
            }
        }

        // Load B tiles to L3
        for (size_t tk = 0; tk < k_tiles; ++tk) {
            for (size_t tj = 0; tj < n_tiles; ++tj) {
                size_t tile_k = std::min(config.tile_size, config.K - tk * config.tile_size);
                size_t tile_n = std::min(config.tile_size, config.N - tj * config.tile_size);
                size_t tile_bytes = tile_k * tile_n * sizeof(float);

                // Calculate L3 address for this B tile
                Address l3_addr = l3_b_base + (tk * n_tiles + tj) * config.tile_size * config.tile_size * sizeof(float);

                // Extract tile from host matrix B
                std::vector<float> tile_data(tile_k * tile_n);
                for (size_t k = 0; k < tile_k; ++k) {
                    for (size_t j = 0; j < tile_n; ++j) {
                        size_t global_k = tk * config.tile_size + k;
                        size_t global_j = tj * config.tile_size + j;
                        tile_data[k * tile_n + j] = B[global_k * config.N + global_j];
                    }
                }

                // Write tile to memory bank, then to L3
                Address bank_tile_addr = bank_b_addr + (tk * n_tiles + tj) * config.tile_size * config.tile_size * sizeof(float);
                kpu->write_memory_bank(bank_id, bank_tile_addr, tile_data.data(), tile_bytes);
                kpu->read_memory_bank(bank_id, bank_tile_addr, tile_data.data(), tile_bytes);
                kpu->write_l3_tile(l3_tile_id, l3_addr, tile_data.data(), tile_bytes);

                if (config.profile) {
                    profiler.record_memory_transfer("Bank", "L3", tile_bytes, 1);
                }
            }
        }

        std::cout << "  Computing tiles...\n";

        // Triple nested loop: for each output tile C[ti,tj], accumulate across K
        size_t tile_count = 0;
        for (size_t ti = 0; ti < m_tiles; ++ti) {
            for (size_t tj = 0; tj < n_tiles; ++tj) {
                // Accumulator for C[ti,tj] tile
                size_t tile_m = std::min(config.tile_size, config.M - ti * config.tile_size);
                size_t tile_n = std::min(config.tile_size, config.N - tj * config.tile_size);
                std::vector<float> c_tile(tile_m * tile_n, 0.0f);

                for (size_t tk = 0; tk < k_tiles; ++tk) {
                    tile_count++;
                    Cycle tile_start = kpu->get_current_cycle();

                    if (config.verbose) {
                        std::cout << "    Tile [" << ti << "," << tj << "," << tk << "] ("
                                  << tile_count << "/" << total_tiles << ")\n";
                    }

                    if (config.profile) {
                        profiler.start_tile(tile_count, ti, tj, tk, tile_start);
                    }

                    size_t tile_k = std::min(config.tile_size, config.K - tk * config.tile_size);

                    // Load A[ti,tk] tile: L3→L2→L1
                    Cycle load_a_start = kpu->get_current_cycle();
                    Address l3_a_addr = l3_a_base + (ti * k_tiles + tk) * config.tile_size * config.tile_size * sizeof(float);

                    kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_a_addr, l2_bank_id, l2_a_addr,
                                               tile_m, tile_k, sizeof(float),
                                               BlockMover::TransformType::IDENTITY, nullptr);
                    kpu->run_until_idle();

                    kpu->start_row_stream(0, l2_bank_id, l1_buffer_id, l2_a_addr, l1_a_addr,
                                          tile_m, tile_k, sizeof(float), config.tile_size,
                                          Streamer::StreamDirection::L2_TO_L1, nullptr);
                    kpu->run_until_idle();
                    Cycle load_a_cycles = kpu->get_current_cycle() - load_a_start;

                    if (config.profile) {
                        profiler.record_component_usage("BlockMover", load_a_cycles);
                        profiler.record_memory_transfer("L3", "L2", tile_m * tile_k * sizeof(float), load_a_cycles);
                    }

                    // Load B[tk,tj] tile: L3→L2→L1
                    Cycle load_b_start = kpu->get_current_cycle();
                    Address l3_b_addr = l3_b_base + (tk * n_tiles + tj) * config.tile_size * config.tile_size * sizeof(float);

                    kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_b_addr, l2_bank_id, l2_b_addr,
                                               tile_k, tile_n, sizeof(float),
                                               BlockMover::TransformType::IDENTITY, nullptr);
                    kpu->run_until_idle();

                    kpu->start_column_stream(1, l2_bank_id, l1_buffer_id, l2_b_addr, l1_b_addr,
                                             tile_k, tile_n, sizeof(float), config.tile_size,
                                             Streamer::StreamDirection::L2_TO_L1, nullptr);
                    kpu->run_until_idle();
                    Cycle load_b_cycles = kpu->get_current_cycle() - load_b_start;

                    if (config.profile) {
                        profiler.record_component_usage("BlockMover", load_b_cycles);
                        profiler.record_memory_transfer("L3", "L2", tile_k * tile_n * sizeof(float), load_b_cycles);
                    }

                    // Compute: C_partial = A[ti,tk] × B[tk,tj]
                    Cycle compute_start = kpu->get_current_cycle();
                    kpu->start_matmul(compute_tile_id, l1_buffer_id, tile_m, tile_n, tile_k,
                                     l1_a_addr, l1_b_addr, l1_c_addr, nullptr);
                    kpu->run_until_idle();
                    Cycle compute_cycles = kpu->get_current_cycle() - compute_start;

                    if (config.profile) {
                        profiler.record_component_usage("SystolicArray", compute_cycles);
                    }

                    // Read partial result and accumulate into c_tile
                    std::vector<float> c_partial(tile_m * tile_n);
                    kpu->read_l1_buffer(l1_buffer_id, l1_c_addr, c_partial.data(), c_partial.size() * sizeof(float));

                    for (size_t i = 0; i < c_partial.size(); ++i) {
                        c_tile[i] += c_partial[i];
                    }

                    Cycle tile_end = kpu->get_current_cycle();
                    if (config.profile) {
                        profiler.end_tile(tile_end, load_a_cycles, load_b_cycles, compute_cycles, 0);
                    }
                }

                // Store completed C[ti,tj] tile back
                Cycle store_start = kpu->get_current_cycle();

                // Write c_tile to L1
                kpu->write_l1_buffer(l1_buffer_id, l1_c_addr, c_tile.data(), c_tile.size() * sizeof(float));

                // L1→L2→L3
                kpu->start_row_stream(0, l2_bank_id, l1_buffer_id, l2_c_addr, l1_c_addr,
                                      tile_m, tile_n, sizeof(float), config.tile_size,
                                      Streamer::StreamDirection::L1_TO_L2, nullptr);
                kpu->run_until_idle();

                // L2→L3 (manual)
                std::vector<uint8_t> c_tile_bytes(c_tile.size() * sizeof(float));
                kpu->read_l2_bank(l2_bank_id, l2_c_addr, c_tile_bytes.data(), c_tile_bytes.size());

                Address l3_c_addr = l3_c_base + (ti * n_tiles + tj) * config.tile_size * config.tile_size * sizeof(float);
                kpu->write_l3_tile(l3_tile_id, l3_c_addr, c_tile_bytes.data(), c_tile_bytes.size());

                Cycle store_cycles = kpu->get_current_cycle() - store_start;
                if (config.profile) {
                    profiler.record_memory_transfer("L1", "L3", c_tile.size() * sizeof(float), store_cycles);
                }
            }
        }

        auto kpu_end = std::chrono::high_resolution_clock::now();
        config.execution_time_ms = std::chrono::duration<double, std::milli>(kpu_end - kpu_start).count();
        config.total_cycles = kpu->get_current_cycle() - start_cycle;

        // Readback: Assemble C tiles from L3 back to host
        std::cout << "  Reading back result tiles...\n";
        for (size_t ti = 0; ti < m_tiles; ++ti) {
            for (size_t tj = 0; tj < n_tiles; ++tj) {
                size_t tile_m = std::min(config.tile_size, config.M - ti * config.tile_size);
                size_t tile_n = std::min(config.tile_size, config.N - tj * config.tile_size);

                Address l3_c_addr = l3_c_base + (ti * n_tiles + tj) * config.tile_size * config.tile_size * sizeof(float);

                std::vector<float> c_tile(tile_m * tile_n);
                kpu->read_l3_tile(l3_tile_id, l3_c_addr, c_tile.data(), c_tile.size() * sizeof(float));

                // Copy tile back to result matrix C
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t j = 0; j < tile_n; ++j) {
                        size_t global_i = ti * config.tile_size + i;
                        size_t global_j = tj * config.tile_size + j;
                        C[global_i * config.N + global_j] = c_tile[i * tile_n + j];
                    }
                }
            }
        }

        std::cout << "  ✓ Multi-tile execution complete (" << tile_count << " tiles processed)\n";
    }

    // Performance metrics
    config.print_performance();

    // Profiler output
    if (config.profile) {
        profiler.print_summary(config.total_cycles);
    }
    if (config.show_timeline) {
        profiler.print_detailed_timeline();
    }

    // Validation
    if (config.validate) {
        std::cout << "\n[5] Validation\n";
        return validate_results(C, C_ref);
    }

    return true;
}

/**
 * @brief Create minimal KPU configuration (1 L3 tile + 1 Compute tile)
 */
void create_minimal_kpu_config(SystemConfig& config) {
    std::cout << "\n========================================\n";
    std::cout << "  Creating Minimal KPU Configuration\n";
    std::cout << "========================================\n";

    config.clear();
    config.system.name = "Minimal KPU for Tiled MatMul";
    config.system.description = "Single L3 tile + Single Compute tile";

    // Host
    config.host.cpu.core_count = 8;
    config.host.cpu.frequency_mhz = 3000;

    MemoryModuleConfig mem;
    mem.id = "ddr5_dimm_0";
    mem.type = "DDR5";
    mem.capacity_gb = 16;
    mem.bandwidth_gbps = 51.2f;
    config.host.memory.modules.push_back(mem);

    // KPU accelerator
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "MinimalKPU";

    KPUConfig kpu;
    kpu.memory.type = "GDDR6";

    // 1 Memory bank (large enough for 256×256 matrices)
    KPUMemoryBankConfig bank;
    bank.id = "bank_0";
    bank.capacity_mb = 256;  // 256MB
    bank.bandwidth_gbps = 150.0f;
    kpu.memory.banks.push_back(bank);

    // 1 L3 tile (large capacity for hundreds of 16×16 tiles)
    KPUTileConfig l3;
    l3.id = "l3_0";
    l3.capacity_kb = 1024;  // 1MB - can hold many tiles
    kpu.memory.l3_tiles.push_back(l3);

    // 1 L2 bank
    KPUTileConfig l2;
    l2.id = "l2_0";
    l2.capacity_kb = 128;
    kpu.memory.l2_banks.push_back(l2);

    // 1 Scratchpad (L1)
    KPUScratchpadConfig scratch;
    scratch.id = "scratch_0";
    scratch.capacity_kb = 128;
    kpu.memory.scratchpads.push_back(scratch);

    // 1 Compute tile (16×16 systolic array)
    ComputeTileConfig tile;
    tile.id = "tile_0";
    tile.type = "systolic";
    tile.systolic_rows = 16;
    tile.systolic_cols = 16;
    tile.datatype = "fp32";
    kpu.compute_fabric.tiles.push_back(tile);

    // 1 DMA engine
    DMAEngineConfig dma;
    dma.id = "dma_0";
    dma.bandwidth_gbps = 75.0f;
    kpu.data_movement.dma_engines.push_back(dma);

    // 1 Block mover
    BlockMoverConfig mover;
    mover.id = "block_mover_0";
    kpu.data_movement.block_movers.push_back(mover);

    // 2 Streamers (for row and column streaming)
    for (int i = 0; i < 2; ++i) {
        StreamerConfig streamer;
        streamer.id = "streamer_" + std::to_string(i);
        kpu.data_movement.streamers.push_back(streamer);
    }

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Interconnect
    config.interconnect.host_to_accelerator.type = "PCIe";
    PCIeConfig pcie;
    pcie.generation = 4;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 32.0f;
    config.interconnect.host_to_accelerator.pcie_config = pcie;

    std::cout << "\nConfiguration created:\n";
    std::cout << config;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -m <M>          Rows of matrix A (default: 16)\n";
    std::cout << "  -k <K>          Cols of A / Rows of B (default: 16)\n";
    std::cout << "  -n <N>          Cols of matrix B (default: 16)\n";
    std::cout << "  -t <tile>       Tile size (default: 16)\n";
    std::cout << "  -v, --verbose   Verbose output\n";
    std::cout << "  --profile       Enable detailed profiling\n";
    std::cout << "  --timeline      Show detailed event timeline\n";
    std::cout << "  --no-validate   Skip validation\n";
    std::cout << "  -h, --help      Show this help\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog_name << " -m 256 -k 256 -n 256           # 256×256 square\n";
    std::cout << "  " << prog_name << " -m 128 -k 512 -n 256           # Rectangular\n";
    std::cout << "  " << prog_name << " -m 16 -k 16 -n 16 --profile   # With profiling\n";
    std::cout << "  " << prog_name << " -m 16 -k 16 -n 16 --timeline  # With timeline\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    size_t M = 16, K = 16, N = 16;
    size_t tile_size = 16;
    bool verbose = false;
    bool validate = true;
    bool profile = false;
    bool timeline = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            M = std::stoul(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            K = std::stoul(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            N = std::stoul(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            tile_size = std::stoul(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--timeline") {
            timeline = true;
        } else if (arg == "--no-validate") {
            validate = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "===========================================\n";
    std::cout << " Tiled Matrix Multiplication - Autonomous KPU\n";
    std::cout << "===========================================\n";

    try {
        // Create minimal KPU configuration
        SystemConfig sys_config;
        create_minimal_kpu_config(sys_config);

        // Initialize system
        SystemSimulator sim(sys_config);
        if (!sim.initialize()) {
            std::cerr << "ERROR: System initialization failed\n";
            return 1;
        }

        auto* kpu = sim.get_kpu(0);
        if (!kpu) {
            std::cerr << "ERROR: Could not get KPU instance\n";
            return 1;
        }

        // Execute tiled matmul
        MatMulConfig config(M, K, N, tile_size, verbose, validate, profile, timeline);
        bool success = execute_tiled_matmul(kpu, config);

        sim.shutdown();

        std::cout << "\n===========================================\n";
        if (success) {
            std::cout << " SUCCESS: Matrix multiplication completed!\n";
        } else {
            std::cout << " FAILED: Matrix multiplication failed!\n";
        }
        std::cout << "===========================================\n";

        return success ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}

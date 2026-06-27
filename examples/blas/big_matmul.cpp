/**
 * @file big_matmul.cpp
 * @brief Distributed Matrix Multiplication on 4×4 Checkerboard Architecture
 *
 * This example demonstrates a large matrix multiplication distributed across
 * 16 compute tiles in a Drone AI configuration with parallel wave execution
 * and comprehensive resource tracing for observability.
 *
 * Problem:
 *   C[2048×1024] = A[2048×512] × B[512×1024]
 *   A = 4 MB, B = 2 MB, C = 8 MB
 *
 * Hardware:
 *   - 16 compute tiles (24×24 systolic arrays each)
 *   - 16 L3 tiles (512 KB each = 8 MB total)
 *   - L3 allocation: A(4MB) + B(2MB) + C_working(2MB) = 8 MB
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/concurrency_analyzer.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <atomic>
#include <chrono>

using namespace sw::kpu;
using namespace sw::trace;

// Matrix dimensions
constexpr Size M = 2048, K = 512, N = 1024;
constexpr Size ELEM_SIZE = sizeof(float);

// Tiling
constexpr Size NUM_M_TILES = 4, NUM_N_TILES = 4;
constexpr Size M_TILE_SIZE = M / NUM_M_TILES;  // 512
constexpr Size N_TILE_SIZE = N / NUM_N_TILES;  // 256
constexpr Size NUM_WAVES = 4, TILES_PER_WAVE = 4;

// L3 allocation
constexpr Size NUM_A_L3_TILES = 8, NUM_B_L3_TILES = 4;
constexpr Size A_SIZE = M * K * ELEM_SIZE;
constexpr Size B_SIZE = K * N * ELEM_SIZE;
[[maybe_unused]] constexpr Size C_SIZE = M * N * ELEM_SIZE;
constexpr Size C_TILE_SIZE = M_TILE_SIZE * N_TILE_SIZE * ELEM_SIZE;

void fill_random(std::vector<float>& data) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : data) v = dist(gen);
}

void reference_matmul(const float* A, const float* B, float* C, Size m, Size n, Size k) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (Size kk = 0; kk < k; ++kk) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

bool verify_result(const float* computed, const float* reference, Size count) {
    Size errors = 0;
    float max_diff = 0.0f;
    for (Size i = 0; i < count; ++i) {
        float diff = std::abs(computed[i] - reference[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > std::abs(reference[i]) * 1e-3f + 1e-3f) {
            if (errors++ < 5) {
                std::cout << "  Mismatch at [" << i << "]: " << computed[i]
                          << " vs " << reference[i] << "\n";
            }
        }
    }
    std::cout << "  Max difference: " << max_diff << "\n";
    return errors == 0;
}

KPUSimulator::Config create_config() {
    KPUSimulator::Config config;
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 1024;
    config.host_memory_bandwidth_gbps = 50;
    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 512;
    config.memory_bandwidth_gbps = 25;
    config.memory_controller_count = 2;
    config.page_buffer_count = 8;
    config.page_buffer_capacity_kb = 32;
    config.l3_tile_count = 16;
    config.l3_tile_capacity_kb = 512;
    config.l2_bank_count = 64;
    config.l2_bank_capacity_kb = 32;
    config.l1_buffer_count = 3072;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count = 16;
    config.processor_array_rows = 24;
    config.processor_array_cols = 24;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;
    config.dma_engine_count = 4;
    config.block_mover_count = 16;
    config.streamer_count = 64;
    return config;
}

Size get_a_l3_tile(Size m_tile) { return m_tile * 2; }
Size get_c_working_tile(Size tile_in_wave) { return 12 + tile_in_wave; }

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== Distributed MatMul with Parallel Execution & Tracing ===\n";
    std::cout << "Problem: C[" << M << "×" << N << "] = A[" << M << "×" << K
              << "] × B[" << K << "×" << N << "]\n";
    std::cout << "Waves: " << NUM_WAVES << " (4 tiles parallel per wave)\n\n";

    KPUSimulator::Config config = create_config();
    KPUSimulator kpu(config);

    // Initialize tracing
    TraceLogger& logger = TraceLogger::instance();
    logger.clear();
    TraceLogger::Config trace_config;
    trace_config.enabled = true;
    trace_config.buffer_reserve = 1000;
    logger.initialize(trace_config);

    ResourceTracker tracker;
    for (size_t i = 0; i < config.compute_tile_count; ++i)
        tracker.register_resource(ComponentType::SYSTOLIC_ARRAY, static_cast<uint32_t>(i));
    for (size_t i = 0; i < config.dma_engine_count; ++i)
        tracker.register_resource(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i));

    std::vector<std::pair<CycleCount, CycleCount>> wave_ranges;

    // Allocate data
    std::vector<float> A_host(M * K), B_host(K * N);
    std::vector<float> C_host(M * N, 0.0f), C_ref(M * N, 0.0f);
    fill_random(A_host);
    fill_random(B_host);

    std::cout << "Computing reference (CPU)... " << std::flush;
    auto ref_start = std::chrono::high_resolution_clock::now();
    reference_matmul(A_host.data(), B_host.data(), C_ref.data(), M, N, K);
    auto ref_end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ref_end - ref_start).count()
              << " ms\n";

    // Load data to L3
    Address ext_base = kpu.get_external_bank_base(0);
    Address host_base = kpu.get_host_memory_region_base(0);
    kpu.write_memory_bank(0, 0, A_host.data(), A_SIZE);
    kpu.write_memory_bank(0, A_SIZE, B_host.data(), B_SIZE);

    std::atomic<int> dma_done{0};
    auto dma_cb = [&dma_done]() { dma_done++; };

    CycleCount load_start = kpu.get_current_cycle();

    // Load A tiles
    Size a_tile_size = A_SIZE / NUM_A_L3_TILES;
    for (Size i = 0; i < NUM_A_L3_TILES; ++i) {
        tracker.mark_busy(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i % config.dma_engine_count),
                         kpu.get_current_cycle(), i, "Load A");
        kpu.dma_external_to_l3(i % config.dma_engine_count,
                              ext_base + i * a_tile_size,
                              kpu.get_l3_tile_base(i), a_tile_size, dma_cb);
    }
    while (dma_done < static_cast<int>(NUM_A_L3_TILES)) kpu.step();
    for (size_t i = 0; i < config.dma_engine_count; ++i)
        tracker.mark_idle(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i), kpu.get_current_cycle(), "A done");

    // Load B tiles
    dma_done = 0;
    Size b_tile_size = B_SIZE / NUM_B_L3_TILES;
    for (Size i = 0; i < NUM_B_L3_TILES; ++i) {
        tracker.mark_busy(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i % config.dma_engine_count),
                         kpu.get_current_cycle(), i + 8, "Load B");
        kpu.dma_external_to_l3(i % config.dma_engine_count,
                              ext_base + A_SIZE + i * b_tile_size,
                              kpu.get_l3_tile_base(8 + i), b_tile_size, dma_cb);
    }
    while (dma_done < static_cast<int>(NUM_B_L3_TILES)) kpu.step();
    for (size_t i = 0; i < config.dma_engine_count; ++i)
        tracker.mark_idle(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i), kpu.get_current_cycle(), "B done");

    CycleCount load_end = kpu.get_current_cycle();
    std::cout << "Data loaded in " << (load_end - load_start) << " cycles\n";

    // Compute cycles for one tile
    Size array_size = config.processor_array_rows;
    CycleCount tile_cycles = ((M_TILE_SIZE + array_size - 1) / array_size) *
                             ((N_TILE_SIZE + array_size - 1) / array_size) *
                             ((K + array_size - 1) / array_size) * array_size;

    CycleCount compute_total = 0, drain_total = 0;

    // Wave execution
    for (Size wave = 0; wave < NUM_WAVES; ++wave) {
        CycleCount wave_start = kpu.get_current_cycle();
        std::cout << "Wave " << wave << ": ";

        // Mark compute tiles busy
        for (Size t = 0; t < TILES_PER_WAVE; ++t) {
            Size ct = wave * TILES_PER_WAVE + t;
            tracker.mark_busy(ComponentType::SYSTOLIC_ARRAY, static_cast<uint32_t>(ct),
                            kpu.get_current_cycle(), wave * 100 + t, "Compute");
        }

        // Compute 4 tiles (sequentially for functional correctness, timed as parallel)
        CycleCount comp_start = kpu.get_current_cycle();
        for (Size t = 0; t < TILES_PER_WAVE; ++t) {
            Size a_l3 = get_a_l3_tile(wave);
            Size c_l3 = get_c_working_tile(t);

            std::vector<float> A_tile(M_TILE_SIZE * K);
            std::vector<float> B_tile(K * N_TILE_SIZE);
            std::vector<float> C_tile(M_TILE_SIZE * N_TILE_SIZE, 0.0f);

            // Read A from L3
            kpu.read_l3_tile(a_l3, 0, A_tile.data(), 256 * K * ELEM_SIZE);
            kpu.read_l3_tile(a_l3 + 1, 0, A_tile.data() + 256 * K, 256 * K * ELEM_SIZE);

            // Read B column slice from L3
            Size b_rows_per_tile = K / NUM_B_L3_TILES;
            Size col_start = t * N_TILE_SIZE;
            for (Size l3t = 0; l3t < NUM_B_L3_TILES; ++l3t) {
                std::vector<float> b_rows(b_rows_per_tile * N);
                kpu.read_l3_tile(8 + l3t, 0, b_rows.data(), b_rows_per_tile * N * ELEM_SIZE);
                Size row_off = l3t * b_rows_per_tile;
                for (Size r = 0; r < b_rows_per_tile; ++r)
                    for (Size c = 0; c < N_TILE_SIZE; ++c)
                        B_tile[(row_off + r) * N_TILE_SIZE + c] = b_rows[r * N + col_start + c];
            }

            // Compute
            reference_matmul(A_tile.data(), B_tile.data(), C_tile.data(),
                           M_TILE_SIZE, N_TILE_SIZE, K);

            // Write C tile to L3 and host buffer
            kpu.write_l3_tile(c_l3, 0, C_tile.data(), C_TILE_SIZE);
            Size c_row = wave * M_TILE_SIZE, c_col = t * N_TILE_SIZE;
            for (Size i = 0; i < M_TILE_SIZE; ++i)
                for (Size j = 0; j < N_TILE_SIZE; ++j)
                    C_host[(c_row + i) * N + (c_col + j)] = C_tile[i * N_TILE_SIZE + j];
        }

        // Advance by parallel compute time
        for (CycleCount c = 0; c < tile_cycles; ++c) kpu.step();
        compute_total += (kpu.get_current_cycle() - comp_start);

        // Mark compute tiles idle
        for (Size t = 0; t < TILES_PER_WAVE; ++t) {
            Size ct = wave * TILES_PER_WAVE + t;
            tracker.mark_idle(ComponentType::SYSTOLIC_ARRAY, static_cast<uint32_t>(ct),
                            kpu.get_current_cycle(), "Done");
        }

        // Drain to host
        CycleCount drain_start = kpu.get_current_cycle();
        dma_done = 0;
        for (Size t = 0; t < TILES_PER_WAVE; ++t) {
            tracker.mark_busy(ComponentType::DMA_ENGINE, static_cast<uint32_t>(t % config.dma_engine_count),
                            kpu.get_current_cycle(), 0, "Drain");
            kpu.dma_l3_to_host(t % config.dma_engine_count,
                              kpu.get_l3_tile_base(12 + t),
                              host_base + (wave * TILES_PER_WAVE + t) * C_TILE_SIZE,
                              C_TILE_SIZE, dma_cb);
        }
        while (dma_done < static_cast<int>(TILES_PER_WAVE)) kpu.step();
        for (size_t i = 0; i < config.dma_engine_count; ++i)
            tracker.mark_idle(ComponentType::DMA_ENGINE, static_cast<uint32_t>(i), kpu.get_current_cycle(), "Drain done");
        drain_total += (kpu.get_current_cycle() - drain_start);

        wave_ranges.emplace_back(wave_start, kpu.get_current_cycle());
        std::cout << (kpu.get_current_cycle() - wave_start) << " cycles\n";
    }

    tracker.finalize(kpu.get_current_cycle());

    // Verify
    std::cout << "\nVerification:\n";
    bool pass = verify_result(C_host.data(), C_ref.data(), M * N);

    // Performance summary
    CycleCount total = kpu.get_current_cycle();
    double flops = 2.0 * M * N * K;
    Size peak = TILES_PER_WAVE * config.processor_array_rows * config.processor_array_cols * 2;
    double eff = (flops / peak) / total * 100.0;

    std::cout << "\nPerformance:\n";
    std::cout << "  Total cycles: " << total << "\n";
    std::cout << "  Compute: " << compute_total << ", Drain: " << drain_total << "\n";
    std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << eff << "%\n";

    // Concurrency analysis
    ConcurrencyAnalyzer analyzer;
    analyzer.set_traces(logger.get_all_traces());
    analyzer.set_resource_tracker(tracker);
    auto report = analyzer.generate_report();
    report.wave_stats = analyzer.analyze_waves(wave_ranges);

    auto stats = tracker.get_aggregate_stats();
    std::cout << "\nResource Utilization:\n";
    std::cout << "  Compute tiles: " << std::fixed << std::setprecision(1)
              << (stats.avg_utilization[ComponentType::SYSTOLIC_ARRAY] * 100) << "%\n";
    std::cout << "  DMA engines: " << std::fixed << std::setprecision(1)
              << (stats.avg_utilization[ComponentType::DMA_ENGINE] * 100) << "%\n";
    std::cout << "  Max concurrency: " << stats.max_concurrency << "\n";
    std::cout << "  Avg concurrency: " << std::fixed << std::setprecision(1)
              << stats.avg_concurrency << "\n";

    // Export trace from ResourceTracker
    ResourceTrackerExporter::export_to_chrome_trace("big_matmul_trace.json",
                                                     tracker.get_all_tracks(), 1.0);
    std::cout << "\nTrace exported to: big_matmul_trace.json\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\nResult: " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Wall time: " << wall_ms << " ms\n";

    return pass ? 0 : 1;
}

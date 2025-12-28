/**
 * @file matmul_schedule_analysis.cpp
 * @brief Analysis of tiled matrix multiplication scheduling strategies
 *
 * This program uses the analysis framework to compare different
 * scheduling strategies for the C[2048×1024] = A[2048×512] × B[512×1024]
 * matrix multiplication on the 16-tile KPU architecture.
 */

#include <sw/analysis/tiled_matmul_model.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace sw::analysis;

// Configuration record for CSV export
struct TilingConfig {
    // Tiling parameters
    size_t m_tiles, n_tiles, k_tiles;

    // Tile sizes (bytes)
    size_t a_tile_bytes, b_tile_bytes, c_tile_bytes;
    size_t total_per_l3;
    bool fits_in_l3;

    // Problem characteristics
    size_t total_c_tiles;
    size_t active_compute_tiles;  // min(total_c_tiles, hw.compute_tile_count)

    // Compute metrics
    Cycles subtile_compute_cycles;
    Cycles tile_compute_cycles;
    Cycles theoretical_min_cycles;

    // Data movement metrics
    Cycles load_a_cycles, load_b_cycles;
    Cycles drain_c_cycles;
    Cycles l3_transfer_a_cycles, l3_transfer_b_cycles;

    // Schedule metrics - Serialized
    Cycles serial_total_cycles;
    double serial_compute_util;
    double serial_efficiency;
    size_t serial_peak_concurrency;

    // Schedule metrics - Parallel
    Cycles parallel_total_cycles;
    double parallel_compute_util;
    double parallel_efficiency;
    size_t parallel_peak_concurrency;

    // Derived metrics
    double speedup;
    double arithmetic_intensity;  // FLOPs / bytes moved
    double memory_bandwidth_gbps;  // Required bandwidth at 1GHz

    // Pareto optimality flags
    bool pareto_efficiency;
    bool pareto_memory;
};

void analyze_current_implementation() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     MATMUL SCHEDULING ANALYSIS: C[2048×1024] = A[2048×512]×B[512×1024]  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    // Hardware configuration (matches big_matmul.cpp)
    HardwareConfig hw;
    hw.compute_tile_count = 16;
    hw.systolic_rows = 24;
    hw.systolic_cols = 24;
    hw.l3_tile_count = 16;
    hw.l3_tile_capacity = 512 * 1024;
    hw.dma_engine_count = 4;
    hw.block_mover_count = 16;

    // Current implementation: 4×4 tiling with K=1 (no K subtiling)
    TileGeometry geom;
    geom.M = 2048;
    geom.K = 512;
    geom.N = 1024;
    geom.M_tiles = 4;
    geom.N_tiles = 4;
    geom.K_tiles = 1;  // No K subtiling

    print_analysis(hw, geom);
}

void analyze_memory_constraints() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    L3 MEMORY CONSTRAINT ANALYSIS                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    // Problem: C[2048×1024] = A[2048×512] × B[512×1024]
    size_t M = 2048, K = 512, N = 1024;
    size_t elem_size = 4;

    std::cout << "\nFull matrix sizes:\n";
    std::cout << "  A[" << M << "×" << K << "] = " << (M * K * elem_size) / (1024*1024) << " MB\n";
    std::cout << "  B[" << K << "×" << N << "] = " << (K * N * elem_size) / (1024*1024) << " MB\n";
    std::cout << "  C[" << M << "×" << N << "] = " << (M * N * elem_size) / (1024*1024) << " MB\n";
    std::cout << "  Total: " << ((M*K + K*N + M*N) * elem_size) / (1024*1024) << " MB\n";

    std::cout << "\nL3 capacity: 16 tiles × 512KB = 8 MB\n";

    std::cout << "\n┌─ 4×4 Tiling Analysis ───────────────────────────────────────────────┐\n";
    size_t M_tile = M / 4;  // 512
    size_t N_tile = N / 4;  // 256

    std::cout << "│ C tile size: " << M_tile << "×" << N_tile << " = "
              << (M_tile * N_tile * elem_size) / 1024 << " KB\n";
    std::cout << "│ A strip (for row): " << M_tile << "×" << K << " = "
              << (M_tile * K * elem_size) / 1024 << " KB\n";
    std::cout << "│ B strip (for col): " << K << "×" << N_tile << " = "
              << (K * N_tile * elem_size) / 1024 << " KB\n";

    size_t per_tile_need = M_tile * N_tile + M_tile * K + K * N_tile;
    std::cout << "│\n";
    std::cout << "│ Per compute tile needs: C + A_strip + B_strip = "
              << (per_tile_need * elem_size) / 1024 << " KB\n";
    std::cout << "│ L3 capacity per tile: 512 KB\n";
    std::cout << "│ Status: " << (per_tile_need * elem_size <= 512*1024 ? "FITS" : "DOES NOT FIT") << "\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";

    std::cout << "\n┌─ K-Subtiling to Fit L3 ────────────────────────────────────────────┐\n";

    // Find K_tiles such that everything fits
    for (size_t K_tiles : {1, 2, 4, 8, 16}) {
        size_t K_tile = (K + K_tiles - 1) / K_tiles;
        size_t a_tile = M_tile * K_tile * elem_size;
        size_t b_tile = K_tile * N_tile * elem_size;
        size_t c_tile = M_tile * N_tile * elem_size;
        size_t total = a_tile + b_tile + c_tile;

        std::cout << "│ K_tiles=" << std::setw(2) << K_tiles
                  << ": A_tile=" << std::setw(4) << a_tile/1024 << "KB"
                  << " + B_tile=" << std::setw(4) << b_tile/1024 << "KB"
                  << " + C_tile=" << std::setw(4) << c_tile/1024 << "KB"
                  << " = " << std::setw(4) << total/1024 << "KB "
                  << (total <= 512*1024 ? "✓ FITS" : "✗") << "\n";
    }
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";
}

void analyze_data_sharing() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    DATA SHARING PATTERN ANALYSIS                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::cout << R"(
For C[i,j] = A[i,:] × B[:,j]:

    B columns →  j=0   j=1   j=2   j=3
                 ┌─────┬─────┬─────┬─────┐
A rows    i=0   │C0,0 │C0,1 │C0,2 │C0,3 │  ← All need A row 0
  ↓             ├─────┼─────┼─────┼─────┤
          i=1   │C1,0 │C1,1 │C1,2 │C1,3 │  ← All need A row 1
                ├─────┼─────┼─────┼─────┤
          i=2   │C2,0 │C2,1 │C2,2 │C2,3 │  ← All need A row 2
                ├─────┼─────┼─────┼─────┤
          i=3   │C3,0 │C3,1 │C3,2 │C3,3 │  ← All need A row 3
                └─────┴─────┴─────┴─────┘
                  ↑     ↑     ↑     ↑
                  All need same B column

Data Sharing Requirements for Full Parallelism:
  - Each A row strip must be accessible by 4 compute tiles (broadcast or replicate)
  - Each B column strip must be accessible by 4 compute tiles (broadcast or replicate)

)";

    std::cout << "┌─ Sharing Strategies ───────────────────────────────────────────────┐\n";
    std::cout << "│                                                                     │\n";
    std::cout << "│ 1. REPLICATION: Load 4 copies of each A/B strip to different L3s   │\n";
    std::cout << "│    - Pros: No L3↔L3 transfers during compute                       │\n";
    std::cout << "│    - Cons: 4× memory bandwidth for loading, 4× L3 usage            │\n";
    std::cout << "│                                                                     │\n";
    std::cout << "│ 2. BROADCAST: Single load, broadcast to 4 L3s via NoC              │\n";
    std::cout << "│    - Pros: 1× load bandwidth, data shared in flight                │\n";
    std::cout << "│    - Cons: Requires broadcast support in DMA                       │\n";
    std::cout << "│                                                                     │\n";
    std::cout << "│ 3. RING EXCHANGE: Load to one L3, rotate through ring              │\n";
    std::cout << "│    - Pros: 1× load bandwidth, streaming through compute            │\n";
    std::cout << "│    - Cons: Serializes K-dimension, requires careful timing         │\n";
    std::cout << "│                                                                     │\n";
    std::cout << "│ 4. OWNER-COMPUTE: Each L3 owns A/B subtiles, computes partial C    │\n";
    std::cout << "│    - Pros: No sharing overhead, maximizes locality                 │\n";
    std::cout << "│    - Cons: Requires reduction of partial C tiles                   │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";
}

void analyze_optimal_tiling() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    OPTIMAL TILING EXPLORATION                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    HardwareConfig hw;
    hw.compute_tile_count = 16;
    hw.systolic_rows = 24;
    hw.systolic_cols = 24;
    hw.l3_tile_count = 16;
    hw.l3_tile_capacity = 512 * 1024;
    hw.dma_engine_count = 4;

    size_t M = 2048, K = 512, N = 1024;
    size_t elem_size = 4;

    std::cout << "\nExploring tiling configurations that fit in L3:\n\n";
    std::cout << "┌───────┬───────┬───────┬────────┬────────┬────────┬──────────┬──────────┐\n";
    std::cout << "│M_tiles│N_tiles│K_tiles│A_tile  │B_tile  │C_tile  │Total/L3  │Efficiency│\n";
    std::cout << "├───────┼───────┼───────┼────────┼────────┼────────┼──────────┼──────────┤\n";

    struct Config {
        size_t m_tiles, n_tiles, k_tiles;
        double efficiency;
    };
    std::vector<Config> valid_configs;

    for (size_t m_tiles : {2, 4, 8}) {
        for (size_t n_tiles : {2, 4, 8}) {
            for (size_t k_tiles : {1, 2, 4, 8, 16}) {
                size_t m_tile = (M + m_tiles - 1) / m_tiles;
                size_t n_tile = (N + n_tiles - 1) / n_tiles;
                size_t k_tile = (K + k_tiles - 1) / k_tiles;

                size_t a_tile_bytes = m_tile * k_tile * elem_size;
                size_t b_tile_bytes = k_tile * n_tile * elem_size;
                size_t c_tile_bytes = m_tile * n_tile * elem_size;
                size_t total = a_tile_bytes + b_tile_bytes + c_tile_bytes;

                if (total > hw.l3_tile_capacity) continue;

                size_t total_c_tiles = m_tiles * n_tiles;
                if (total_c_tiles > hw.compute_tile_count * 4) continue;  // Too many tiles

                TileGeometry geom;
                geom.M = M; geom.K = K; geom.N = N;
                geom.M_tiles = m_tiles;
                geom.N_tiles = n_tiles;
                geom.K_tiles = k_tiles;

                ScheduleAnalyzer analyzer(hw, geom);
                auto sched = analyzer.generate_fully_parallel_schedule();
                auto metrics = analyzer.analyze(sched);

                std::cout << "│   " << std::setw(2) << m_tiles << "  "
                          << "│   " << std::setw(2) << n_tiles << "  "
                          << "│   " << std::setw(2) << k_tiles << "  "
                          << "│" << std::setw(6) << a_tile_bytes/1024 << "KB"
                          << "│" << std::setw(6) << b_tile_bytes/1024 << "KB"
                          << "│" << std::setw(6) << c_tile_bytes/1024 << "KB"
                          << "│" << std::setw(7) << total/1024 << "KB "
                          << "│  " << std::fixed << std::setprecision(1)
                          << std::setw(5) << metrics.overall_efficiency * 100 << "%  │\n";

                valid_configs.push_back({m_tiles, n_tiles, k_tiles, metrics.overall_efficiency});
            }
        }
    }

    std::cout << "└───────┴───────┴───────┴────────┴────────┴────────┴──────────┴──────────┘\n";

    // Find best
    if (!valid_configs.empty()) {
        auto best = std::max_element(valid_configs.begin(), valid_configs.end(),
            [](const Config& a, const Config& b) { return a.efficiency < b.efficiency; });

        std::cout << "\nBest configuration: " << best->m_tiles << "×" << best->n_tiles
                  << " C tiles, " << best->k_tiles << " K steps"
                  << " (efficiency: " << std::setprecision(1) << best->efficiency * 100 << "%)\n";
    }
}

void print_recommended_approach() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    RECOMMENDED IMPLEMENTATION                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::cout << R"(
Based on the analysis, the recommended approach for maximum utilization:

1. TILING CONFIGURATION
   - M_tiles=4, N_tiles=4 (16 C tiles = 16 compute tiles)
   - K_tiles=8 (subtile K to fit A+B+C in 512KB L3)

2. L3 ALLOCATION (per compute tile)
   - A subtile: 512×64×4 = 128KB
   - B subtile: 64×256×4 = 64KB
   - C tile:    512×256×4 = 512KB (partial, accumulated in-place)
   - Total: 704KB... still too big!

3. REVISED APPROACH: K-Streaming with A/B Double-Buffering
   - Use 2 L3 tiles per compute tile (32 L3 tiles needed... we only have 16)
   - OR: Stream A/B from external memory during compute (compute-bound)

4. PRACTICAL SOLUTION: Row-Major Streaming
   - Keep B entirely in L3 (fits in 8 L3 tiles: 2MB)
   - Stream A rows from external memory
   - Each compute tile in a row shares the same A stream
   - This gives 4× parallelism per row, 4 sequential rows

5. ADVANCED: Systolic Data Flow
   - Use Block Movers to shift A tiles across L3s during compute
   - Each A subtile visits 4 L3s as computation proceeds
   - Zero external memory traffic after initial load

)";
}

std::vector<TilingConfig> enumerate_all_configurations() {
    std::vector<TilingConfig> configs;

    // Hardware configuration
    HardwareConfig hw;
    hw.compute_tile_count = 16;
    hw.systolic_rows = 24;
    hw.systolic_cols = 24;
    hw.l3_tile_count = 16;
    hw.l3_tile_capacity = 512 * 1024;
    hw.dma_engine_count = 4;
    hw.block_mover_count = 16;

    // Problem dimensions
    size_t M = 2048, K = 512, N = 1024;
    size_t elem_size = 4;

    // Enumerate all reasonable tiling configurations
    std::vector<size_t> m_options = {1, 2, 4, 8, 16, 32};
    std::vector<size_t> n_options = {1, 2, 4, 8, 16, 32};
    std::vector<size_t> k_options = {1, 2, 4, 8, 16, 32, 64, 128};

    for (size_t m_tiles : m_options) {
        if (m_tiles > M) continue;  // Can't have more tiles than elements

        for (size_t n_tiles : n_options) {
            if (n_tiles > N) continue;

            for (size_t k_tiles : k_options) {
                if (k_tiles > K) continue;

                TilingConfig cfg;
                cfg.m_tiles = m_tiles;
                cfg.n_tiles = n_tiles;
                cfg.k_tiles = k_tiles;

                // Calculate tile sizes
                size_t m_tile = (M + m_tiles - 1) / m_tiles;
                size_t n_tile = (N + n_tiles - 1) / n_tiles;
                size_t k_tile = (K + k_tiles - 1) / k_tiles;

                cfg.a_tile_bytes = m_tile * k_tile * elem_size;
                cfg.b_tile_bytes = k_tile * n_tile * elem_size;
                cfg.c_tile_bytes = m_tile * n_tile * elem_size;
                cfg.total_per_l3 = cfg.a_tile_bytes + cfg.b_tile_bytes + cfg.c_tile_bytes;
                cfg.fits_in_l3 = cfg.total_per_l3 <= hw.l3_tile_capacity;

                cfg.total_c_tiles = m_tiles * n_tiles;
                cfg.active_compute_tiles = std::min(cfg.total_c_tiles, hw.compute_tile_count);

                // Skip configurations with too many tiles (impractical)
                if (cfg.total_c_tiles > 256) continue;

                // Set up geometry and models
                TileGeometry geom;
                geom.M = M; geom.K = K; geom.N = N;
                geom.M_tiles = m_tiles;
                geom.N_tiles = n_tiles;
                geom.K_tiles = k_tiles;
                geom.elem_size = elem_size;

                ComputeModel compute(hw, geom);
                DataMovementModel dm(hw, geom);

                cfg.subtile_compute_cycles = compute.subtile_compute_cycles();
                cfg.tile_compute_cycles = compute.tile_compute_cycles();
                cfg.theoretical_min_cycles = compute.theoretical_min_cycles();

                cfg.load_a_cycles = dm.load_A_tile_cycles();
                cfg.load_b_cycles = dm.load_B_tile_cycles();
                cfg.drain_c_cycles = dm.drain_C_tile_cycles();
                cfg.l3_transfer_a_cycles = dm.l3_to_l3_A_tile_cycles();
                cfg.l3_transfer_b_cycles = dm.l3_to_l3_B_tile_cycles();

                // Generate and analyze schedules
                ScheduleAnalyzer analyzer(hw, geom);

                auto serial_sched = analyzer.generate_serialized_wave_schedule();
                auto serial_metrics = analyzer.analyze(serial_sched);
                cfg.serial_total_cycles = serial_metrics.total_cycles;
                cfg.serial_compute_util = serial_metrics.compute_utilization;
                cfg.serial_efficiency = serial_metrics.overall_efficiency;
                cfg.serial_peak_concurrency = serial_metrics.peak_compute_concurrency;

                auto parallel_sched = analyzer.generate_fully_parallel_schedule();
                auto parallel_metrics = analyzer.analyze(parallel_sched);
                cfg.parallel_total_cycles = parallel_metrics.total_cycles;
                cfg.parallel_compute_util = parallel_metrics.compute_utilization;
                cfg.parallel_efficiency = parallel_metrics.overall_efficiency;
                cfg.parallel_peak_concurrency = parallel_metrics.peak_compute_concurrency;

                // Derived metrics
                cfg.speedup = static_cast<double>(cfg.serial_total_cycles) / cfg.parallel_total_cycles;

                // Arithmetic intensity: FLOPs / bytes moved
                uint64_t total_flops = 2ULL * M * N * K;
                size_t total_bytes = M * K * elem_size + K * N * elem_size + M * N * elem_size;
                cfg.arithmetic_intensity = static_cast<double>(total_flops) / total_bytes;

                // Memory bandwidth (GB/s at 1 GHz)
                double total_cycles = static_cast<double>(cfg.parallel_total_cycles);
                cfg.memory_bandwidth_gbps = total_bytes / total_cycles;

                cfg.pareto_efficiency = false;  // Will be set later
                cfg.pareto_memory = false;

                configs.push_back(cfg);
            }
        }
    }

    return configs;
}

void identify_pareto_optimal(std::vector<TilingConfig>& configs) {
    // Pareto frontier for efficiency vs memory
    // A config is Pareto-optimal if no other config has both:
    //   - higher parallel_efficiency AND
    //   - lower total_per_l3

    for (size_t i = 0; i < configs.size(); ++i) {
        bool dominated = false;
        for (size_t j = 0; j < configs.size(); ++j) {
            if (i == j) continue;

            // Check if j dominates i
            bool j_better_efficiency = configs[j].parallel_efficiency > configs[i].parallel_efficiency;
            bool j_better_memory = configs[j].total_per_l3 < configs[i].total_per_l3;
            bool j_equal_efficiency = std::abs(configs[j].parallel_efficiency - configs[i].parallel_efficiency) < 0.001;
            bool j_equal_memory = configs[j].total_per_l3 == configs[i].total_per_l3;

            if ((j_better_efficiency && (j_better_memory || j_equal_memory)) ||
                (j_better_memory && (j_better_efficiency || j_equal_efficiency))) {
                dominated = true;
                break;
            }
        }
        configs[i].pareto_efficiency = !dominated && configs[i].fits_in_l3;
    }
}

void export_csv(const std::vector<TilingConfig>& configs, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }

    // Header
    file << "m_tiles,n_tiles,k_tiles,"
         << "a_tile_kb,b_tile_kb,c_tile_kb,total_per_l3_kb,fits_in_l3,"
         << "total_c_tiles,active_compute_tiles,"
         << "subtile_compute_cycles,tile_compute_cycles,theoretical_min_cycles,"
         << "load_a_cycles,load_b_cycles,drain_c_cycles,"
         << "l3_transfer_a_cycles,l3_transfer_b_cycles,"
         << "serial_total_cycles,serial_compute_util,serial_efficiency,serial_peak_concurrency,"
         << "parallel_total_cycles,parallel_compute_util,parallel_efficiency,parallel_peak_concurrency,"
         << "speedup,arithmetic_intensity,memory_bandwidth_gbps,"
         << "pareto_optimal\n";

    // Data rows
    for (const auto& cfg : configs) {
        file << cfg.m_tiles << "," << cfg.n_tiles << "," << cfg.k_tiles << ","
             << cfg.a_tile_bytes/1024 << "," << cfg.b_tile_bytes/1024 << ","
             << cfg.c_tile_bytes/1024 << "," << cfg.total_per_l3/1024 << ","
             << (cfg.fits_in_l3 ? 1 : 0) << ","
             << cfg.total_c_tiles << "," << cfg.active_compute_tiles << ","
             << cfg.subtile_compute_cycles << "," << cfg.tile_compute_cycles << ","
             << cfg.theoretical_min_cycles << ","
             << cfg.load_a_cycles << "," << cfg.load_b_cycles << ","
             << cfg.drain_c_cycles << ","
             << cfg.l3_transfer_a_cycles << "," << cfg.l3_transfer_b_cycles << ","
             << cfg.serial_total_cycles << "," << std::fixed << std::setprecision(4)
             << cfg.serial_compute_util << "," << cfg.serial_efficiency << ","
             << cfg.serial_peak_concurrency << ","
             << cfg.parallel_total_cycles << "," << cfg.parallel_compute_util << ","
             << cfg.parallel_efficiency << "," << cfg.parallel_peak_concurrency << ","
             << std::setprecision(3) << cfg.speedup << ","
             << cfg.arithmetic_intensity << "," << cfg.memory_bandwidth_gbps << ","
             << (cfg.pareto_efficiency ? 1 : 0) << "\n";
    }

    file.close();
    std::cout << "Exported " << configs.size() << " configurations to " << filename << "\n";
}

void print_summary_table(const std::vector<TilingConfig>& configs) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              TILING CONFIGURATION SUMMARY                                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    // Count by category
    size_t total = configs.size();
    size_t fits_l3 = 0, pareto_count = 0;
    double max_efficiency = 0, min_memory = 1e9;
    const TilingConfig* best_efficiency = nullptr;
    const TilingConfig* best_memory = nullptr;

    for (const auto& cfg : configs) {
        if (cfg.fits_in_l3) fits_l3++;
        if (cfg.pareto_efficiency) pareto_count++;
        if (cfg.parallel_efficiency > max_efficiency && cfg.fits_in_l3) {
            max_efficiency = cfg.parallel_efficiency;
            best_efficiency = &cfg;
        }
        if (cfg.total_per_l3 < min_memory && cfg.fits_in_l3) {
            min_memory = cfg.total_per_l3;
            best_memory = &cfg;
        }
    }

    std::cout << "Total configurations explored: " << total << "\n";
    std::cout << "Configurations fitting in L3:  " << fits_l3 << "\n";
    std::cout << "Pareto-optimal configurations: " << pareto_count << "\n\n";

    if (best_efficiency) {
        std::cout << "┌─ Best Efficiency (fits in L3) ──────────────────────────────────────┐\n";
        std::cout << "│ Tiling: " << best_efficiency->m_tiles << "×" << best_efficiency->n_tiles
                  << " C tiles, " << best_efficiency->k_tiles << " K steps\n";
        std::cout << "│ Memory: A=" << best_efficiency->a_tile_bytes/1024 << "KB + B="
                  << best_efficiency->b_tile_bytes/1024 << "KB + C="
                  << best_efficiency->c_tile_bytes/1024 << "KB = "
                  << best_efficiency->total_per_l3/1024 << "KB\n";
        std::cout << "│ Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << best_efficiency->parallel_efficiency * 100 << "%\n";
        std::cout << "│ Speedup over serial: " << std::setprecision(2)
                  << best_efficiency->speedup << "×\n";
        std::cout << "│ Peak concurrency: " << best_efficiency->parallel_peak_concurrency << "\n";
        std::cout << "└──────────────────────────────────────────────────────────────────────┘\n\n";
    }

    if (best_memory && best_memory != best_efficiency) {
        std::cout << "┌─ Smallest Memory Footprint (fits in L3) ─────────────────────────────┐\n";
        std::cout << "│ Tiling: " << best_memory->m_tiles << "×" << best_memory->n_tiles
                  << " C tiles, " << best_memory->k_tiles << " K steps\n";
        std::cout << "│ Memory: A=" << best_memory->a_tile_bytes/1024 << "KB + B="
                  << best_memory->b_tile_bytes/1024 << "KB + C="
                  << best_memory->c_tile_bytes/1024 << "KB = "
                  << best_memory->total_per_l3/1024 << "KB\n";
        std::cout << "│ Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << best_memory->parallel_efficiency * 100 << "%\n";
        std::cout << "│ Speedup over serial: " << std::setprecision(2)
                  << best_memory->speedup << "×\n";
        std::cout << "└──────────────────────────────────────────────────────────────────────┘\n\n";
    }

    // Print Pareto frontier
    std::cout << "┌─ Pareto Frontier (Efficiency vs Memory) ───────────────────────────────────────────────┐\n";
    std::cout << "│ M×N×K    │ Total/L3 │ Par.Eff. │ Speedup │ Peak Conc. │ Cycles      │ Bottleneck │\n";
    std::cout << "├──────────┼──────────┼──────────┼─────────┼────────────┼─────────────┼────────────┤\n";

    for (const auto& cfg : configs) {
        if (!cfg.pareto_efficiency) continue;
        std::cout << "│ " << std::setw(2) << cfg.m_tiles << "×" << std::setw(2) << cfg.n_tiles
                  << "×" << std::setw(2) << cfg.k_tiles << " │ "
                  << std::setw(6) << cfg.total_per_l3/1024 << "KB │ "
                  << std::setw(7) << std::fixed << std::setprecision(1)
                  << cfg.parallel_efficiency * 100 << "% │ "
                  << std::setw(6) << std::setprecision(2) << cfg.speedup << "× │ "
                  << std::setw(10) << cfg.parallel_peak_concurrency << " │ "
                  << std::setw(11) << cfg.parallel_total_cycles << " │ "
                  << std::setw(10) << (cfg.parallel_compute_util > 0.5 ? "COMPUTE" : "MEMORY") << " │\n";
    }
    std::cout << "└──────────┴──────────┴──────────┴─────────┴────────────┴─────────────┴────────────┘\n";
}

void full_enumeration_analysis() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         FULL TILING ENUMERATION ANALYSIS                                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Enumerating all tiling configurations...\n";
    auto configs = enumerate_all_configurations();
    std::cout << "Generated " << configs.size() << " configurations\n";

    std::cout << "Identifying Pareto-optimal configurations...\n";
    identify_pareto_optimal(configs);

    // Export to CSV
    std::string csv_filename = "tiling_configurations.csv";
    export_csv(configs, csv_filename);

    // Print summary
    print_summary_table(configs);

    std::cout << "\n";
    std::cout << "CSV file '" << csv_filename << "' contains all data for plotting.\n";
    std::cout << "Suggested plots:\n";
    std::cout << "  1. Scatter: total_per_l3_kb vs parallel_efficiency (color by fits_in_l3)\n";
    std::cout << "  2. Scatter: parallel_total_cycles vs parallel_efficiency\n";
    std::cout << "  3. Heatmap: m_tiles × n_tiles, color by parallel_efficiency\n";
    std::cout << "  4. Line: k_tiles vs parallel_efficiency (for fixed m×n)\n";
    std::cout << "  5. Pareto frontier: highlight pareto_optimal=1 points\n";
}

int main(int argc, char* argv[]) {
    bool full_only = (argc > 1 && std::string(argv[1]) == "--full");

    if (!full_only) {
        analyze_memory_constraints();
        analyze_data_sharing();
        analyze_current_implementation();
        analyze_optimal_tiling();
        print_recommended_approach();
    }

    // Always run full enumeration and CSV export
    full_enumeration_analysis();

    return 0;
}

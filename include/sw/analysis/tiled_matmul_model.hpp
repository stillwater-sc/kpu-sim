/**
 * @file tiled_matmul_model.hpp
 * @brief Analysis Framework for Tiled Matrix Multiplication Scheduling
 *
 * This framework provides tools to reason about:
 * - Tile geometry and memory requirements
 * - L3 allocation strategies
 * - Compute time for subtiles
 * - Data movement time between L3s
 * - Schedule construction and analysis
 * - Utilization metrics and bottleneck identification
 *
 * The goal is to enable constructive design of resource allocation
 * and timing to maximize efficiency of the compute fabric.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <optional>
#include <functional>
#include <iostream>

namespace sw::analysis {

using Cycles = uint64_t;
using Bytes = uint64_t;

//=============================================================================
// Hardware Configuration
//=============================================================================

struct HardwareConfig {
    // Compute fabric
    size_t compute_tile_count = 16;
    size_t systolic_rows = 24;
    size_t systolic_cols = 24;

    // Memory hierarchy
    size_t l3_tile_count = 16;
    Bytes l3_tile_capacity = 512 * 1024;  // 512 KB

    // Data movement engines
    size_t dma_engine_count = 4;
    size_t block_mover_count = 16;

    // Bandwidth (bytes per cycle)
    double external_to_l3_bw = 32.0;   // ~32 GB/s at 1GHz
    double l3_to_l3_bw = 64.0;          // Internal NoC
    double l3_to_l2_bw = 128.0;         // Block mover

    // Latencies (cycles)
    Cycles dma_startup_latency = 100;
    Cycles block_mover_startup_latency = 20;

    // Derived
    size_t ops_per_cycle() const {
        return compute_tile_count * systolic_rows * systolic_cols * 2;  // MAC = 2 ops
    }

    Bytes total_l3_capacity() const {
        return l3_tile_count * l3_tile_capacity;
    }
};

//=============================================================================
// Matrix and Tile Geometry
//=============================================================================

struct MatrixDims {
    size_t rows;
    size_t cols;
    size_t elem_size = 4;  // bytes per element (float32)

    Bytes size_bytes() const { return rows * cols * elem_size; }
    size_t elements() const { return rows * cols; }
};

struct TileCoord {
    size_t row;
    size_t col;

    bool operator<(const TileCoord& o) const {
        return row < o.row || (row == o.row && col < o.col);
    }
    bool operator==(const TileCoord& o) const {
        return row == o.row && col == o.col;
    }
};

struct TileGeometry {
    // Problem dimensions: C[M×N] = A[M×K] × B[K×N]
    size_t M, K, N;
    size_t elem_size = 4;

    // Tiling parameters
    size_t M_tiles;  // Number of tiles along M dimension
    size_t N_tiles;  // Number of tiles along N dimension
    size_t K_tiles;  // Number of tiles along K dimension (for streaming)

    // Derived tile sizes
    size_t M_tile_size() const { return (M + M_tiles - 1) / M_tiles; }
    size_t N_tile_size() const { return (N + N_tiles - 1) / N_tiles; }
    size_t K_tile_size() const { return (K + K_tiles - 1) / K_tiles; }

    // Memory requirements per tile type
    Bytes A_tile_bytes() const { return M_tile_size() * K_tile_size() * elem_size; }
    Bytes B_tile_bytes() const { return K_tile_size() * N_tile_size() * elem_size; }
    Bytes C_tile_bytes() const { return M_tile_size() * N_tile_size() * elem_size; }

    // Full strip sizes (for sharing analysis)
    Bytes A_strip_bytes([[maybe_unused]] size_t m_tile) const {
        return M_tile_size() * K * elem_size;
    }
    Bytes B_strip_bytes([[maybe_unused]] size_t n_tile) const {
        return K * N_tile_size() * elem_size;
    }

    // Total tiles
    size_t total_C_tiles() const { return M_tiles * N_tiles; }
    size_t total_A_tiles() const { return M_tiles * K_tiles; }
    size_t total_B_tiles() const { return K_tiles * N_tiles; }

    // Total data sizes
    Bytes total_A_bytes() const { return M * K * elem_size; }
    Bytes total_B_bytes() const { return K * N * elem_size; }
    Bytes total_C_bytes() const { return M * N * elem_size; }

    // FLOPS for full matmul
    uint64_t total_flops() const { return 2ULL * M * N * K; }

    // FLOPS for one C tile (full K reduction)
    uint64_t tile_flops() const {
        return 2ULL * M_tile_size() * N_tile_size() * K;
    }

    // FLOPS for one C tile with K subtiling
    uint64_t subtile_flops() const {
        return 2ULL * M_tile_size() * N_tile_size() * K_tile_size();
    }

    std::string describe() const {
        std::ostringstream oss;
        oss << "C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]\n";
        oss << "Tiling: " << M_tiles << "×" << N_tiles << " C tiles, " << K_tiles << " K steps\n";
        oss << "Tile sizes: C=" << M_tile_size() << "×" << N_tile_size()
            << ", A=" << M_tile_size() << "×" << K_tile_size()
            << ", B=" << K_tile_size() << "×" << N_tile_size() << "\n";
        oss << "Memory: A_tile=" << A_tile_bytes()/1024 << "KB, B_tile=" << B_tile_bytes()/1024
            << "KB, C_tile=" << C_tile_bytes()/1024 << "KB\n";
        oss << "Total FLOPs: " << static_cast<double>(total_flops()) / 1e9 << " GFLOPs\n";
        return oss.str();
    }
};

//=============================================================================
// Compute Model
//=============================================================================

struct ComputeModel {
    const HardwareConfig& hw;
    const TileGeometry& geom;

    ComputeModel(const HardwareConfig& h, const TileGeometry& g) : hw(h), geom(g) {}

    // Cycles to compute one C subtile (M_tile × N_tile) with K_tile accumulation
    Cycles subtile_compute_cycles() const {
        size_t m_blocks = (geom.M_tile_size() + hw.systolic_rows - 1) / hw.systolic_rows;
        size_t n_blocks = (geom.N_tile_size() + hw.systolic_cols - 1) / hw.systolic_cols;
        size_t k_steps = (geom.K_tile_size() + hw.systolic_rows - 1) / hw.systolic_rows;

        // Systolic array: pipeline fill + steady state + drain
        // Simplified: blocks × k_steps × systolic_dim cycles
        return m_blocks * n_blocks * k_steps * hw.systolic_rows;
    }

    // Cycles to compute one full C tile (all K steps)
    Cycles tile_compute_cycles() const {
        return subtile_compute_cycles() * geom.K_tiles;
    }

    // Theoretical minimum cycles for full matmul (all tiles parallel)
    Cycles theoretical_min_cycles() const {
        Cycles per_tile = tile_compute_cycles();
        size_t tiles = geom.total_C_tiles();
        size_t parallelism = std::min(tiles, hw.compute_tile_count);
        return per_tile * ((tiles + parallelism - 1) / parallelism);
    }

    // Peak FLOPS per cycle
    double peak_flops_per_cycle() const {
        return static_cast<double>(hw.ops_per_cycle());
    }

    std::string describe() const {
        std::ostringstream oss;
        oss << "Compute Model:\n";
        oss << "  Subtile compute: " << subtile_compute_cycles() << " cycles\n";
        oss << "  Full tile compute: " << tile_compute_cycles() << " cycles\n";
        oss << "  Theoretical min (all parallel): " << theoretical_min_cycles() << " cycles\n";
        oss << "  Peak throughput: " << peak_flops_per_cycle() << " FLOPS/cycle\n";
        return oss.str();
    }
};

//=============================================================================
// Data Movement Model
//=============================================================================

struct DataMovementModel {
    const HardwareConfig& hw;
    const TileGeometry& geom;

    DataMovementModel(const HardwareConfig& h, const TileGeometry& g) : hw(h), geom(g) {}

    // External memory → L3 (DMA)
    Cycles load_A_tile_cycles() const {
        return hw.dma_startup_latency +
               static_cast<Cycles>(static_cast<double>(geom.A_tile_bytes()) / hw.external_to_l3_bw);
    }

    Cycles load_B_tile_cycles() const {
        return hw.dma_startup_latency +
               static_cast<Cycles>(static_cast<double>(geom.B_tile_bytes()) / hw.external_to_l3_bw);
    }

    // L3 → External (drain C)
    Cycles drain_C_tile_cycles() const {
        return hw.dma_startup_latency +
               static_cast<Cycles>(static_cast<double>(geom.C_tile_bytes()) / hw.external_to_l3_bw);
    }

    // L3 → L3 transfer (for A/B sharing)
    Cycles l3_to_l3_A_tile_cycles() const {
        return hw.block_mover_startup_latency +
               static_cast<Cycles>(static_cast<double>(geom.A_tile_bytes()) / hw.l3_to_l3_bw);
    }

    Cycles l3_to_l3_B_tile_cycles() const {
        return hw.block_mover_startup_latency +
               static_cast<Cycles>(static_cast<double>(geom.B_tile_bytes()) / hw.l3_to_l3_bw);
    }

    // Total load time for all A and B (sequential)
    Cycles total_load_cycles_sequential() const {
        return geom.total_A_tiles() * load_A_tile_cycles() +
               geom.total_B_tiles() * load_B_tile_cycles();
    }

    // Total load time with DMA parallelism
    Cycles total_load_cycles_parallel() const {
        Cycles a_cycles = geom.total_A_tiles() * load_A_tile_cycles();
        Cycles b_cycles = geom.total_B_tiles() * load_B_tile_cycles();
        return (a_cycles + b_cycles) / hw.dma_engine_count;
    }

    std::string describe() const {
        std::ostringstream oss;
        oss << "Data Movement Model:\n";
        oss << "  Load A tile: " << load_A_tile_cycles() << " cycles\n";
        oss << "  Load B tile: " << load_B_tile_cycles() << " cycles\n";
        oss << "  Drain C tile: " << drain_C_tile_cycles() << " cycles\n";
        oss << "  L3→L3 A tile: " << l3_to_l3_A_tile_cycles() << " cycles\n";
        oss << "  L3→L3 B tile: " << l3_to_l3_B_tile_cycles() << " cycles\n";
        oss << "  Total load (parallel): " << total_load_cycles_parallel() << " cycles\n";
        return oss.str();
    }
};

//=============================================================================
// L3 Allocation Strategy
//=============================================================================

enum class L3AllocationStrategy {
    DEDICATED,      // Each L3 dedicated to one compute tile
    SHARED_A,       // A tiles shared across row of L3s
    SHARED_B,       // B tiles shared across column of L3s
    BROADCAST,      // Data broadcast to multiple L3s
    RING_EXCHANGE   // Tiles rotate through L3s
};

struct L3Allocation {
    // Which data is stored in which L3
    // L3 index → list of (tile_type, tile_coord)
    enum class TileType { A, B, C };

    struct TileRef {
        TileType type;
        TileCoord coord;
        Bytes size;
        Cycles available_at;  // When this data becomes available
    };

    std::map<size_t, std::vector<TileRef>> allocations;

    // Check if allocation fits in L3 capacity
    bool validate(const HardwareConfig& hw) const {
        for (const auto& [l3_id, tiles] : allocations) {
            Bytes total = 0;
            for (const auto& t : tiles) total += t.size;
            if (total > hw.l3_tile_capacity) return false;
        }
        return true;
    }

    // Get total bytes allocated to an L3
    Bytes get_usage(size_t l3_id) const {
        auto it = allocations.find(l3_id);
        if (it == allocations.end()) return 0;
        Bytes total = 0;
        for (const auto& t : it->second) total += t.size;
        return total;
    }

    std::string describe(const HardwareConfig& hw) const {
        std::ostringstream oss;
        oss << "L3 Allocation:\n";
        for (const auto& [l3_id, tiles] : allocations) {
            Bytes usage = get_usage(l3_id);
            oss << "  L3[" << l3_id << "]: " << usage/1024 << "KB / "
                << hw.l3_tile_capacity/1024 << "KB (";
            for (size_t i = 0; i < tiles.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << (tiles[i].type == TileType::A ? "A" :
                       tiles[i].type == TileType::B ? "B" : "C")
                    << "[" << tiles[i].coord.row << "," << tiles[i].coord.col << "]";
            }
            oss << ")\n";
        }
        return oss.str();
    }
};

//=============================================================================
// Schedule Representation
//=============================================================================

struct ScheduleEvent {
    enum class Type {
        LOAD_A,         // DMA: External → L3
        LOAD_B,
        DRAIN_C,        // DMA: L3 → External
        TRANSFER_A,     // Block Mover: L3 → L3
        TRANSFER_B,
        COMPUTE         // Systolic array computation
    };

    Type type;
    size_t resource_id;     // Which DMA/BlockMover/ComputeTile
    TileCoord tile;         // Which tile
    size_t k_step = 0;      // For K-tiled computation

    Cycles start_cycle;
    Cycles duration;
    Cycles end_cycle() const { return start_cycle + duration; }

    std::string describe() const {
        static const char* type_names[] = {
            "LOAD_A", "LOAD_B", "DRAIN_C", "XFER_A", "XFER_B", "COMPUTE"
        };
        std::ostringstream oss;
        oss << type_names[static_cast<int>(type)]
            << "[" << tile.row << "," << tile.col << "]";
        if (type == Type::COMPUTE) oss << ".k" << k_step;
        oss << " @" << start_cycle << "+" << duration << " on R" << resource_id;
        return oss.str();
    }
};

struct Schedule {
    std::vector<ScheduleEvent> events;

    void add(ScheduleEvent e) {
        events.push_back(e);
    }

    Cycles total_cycles() const {
        Cycles max_end = 0;
        for (const auto& e : events) {
            max_end = std::max(max_end, e.end_cycle());
        }
        return max_end;
    }

    // Get all events for a specific resource type
    std::vector<ScheduleEvent> get_events(ScheduleEvent::Type type) const {
        std::vector<ScheduleEvent> result;
        for (const auto& e : events) {
            if (e.type == type) result.push_back(e);
        }
        return result;
    }

    // Sort events by start time
    void sort_by_time() {
        std::sort(events.begin(), events.end(),
            [](const ScheduleEvent& a, const ScheduleEvent& b) {
                return a.start_cycle < b.start_cycle;
            });
    }
};

//=============================================================================
// Schedule Analysis
//=============================================================================

struct UtilizationMetrics {
    // Time metrics
    Cycles total_cycles;
    Cycles compute_cycles;      // Sum of all compute time
    Cycles load_cycles;         // Sum of all load time
    Cycles drain_cycles;        // Sum of all drain time
    Cycles transfer_cycles;     // Sum of all L3↔L3 time

    // Utilization
    double compute_utilization;     // compute_cycles / (total × num_tiles)
    double memory_utilization;      // active data movement / total
    double overall_efficiency;      // theoretical_min / actual

    // Bottleneck analysis
    Cycles compute_critical_path;
    Cycles memory_critical_path;
    bool is_compute_bound;

    // Concurrency
    size_t peak_compute_concurrency;
    size_t peak_memory_concurrency;
    double avg_compute_concurrency;

    std::string describe() const {
        std::ostringstream oss;
        oss << "\n╔══════════════════════════════════════════════════════════════╗\n";
        oss << "║                    UTILIZATION ANALYSIS                       ║\n";
        oss << "╚══════════════════════════════════════════════════════════════╝\n";
        oss << "\n┌─ Timing ────────────────────────────────────────────────────┐\n";
        oss << "│ Total cycles:        " << std::setw(12) << total_cycles << "\n";
        oss << "│ Compute cycles:      " << std::setw(12) << compute_cycles << "\n";
        oss << "│ Load cycles:         " << std::setw(12) << load_cycles << "\n";
        oss << "│ Transfer cycles:     " << std::setw(12) << transfer_cycles << "\n";
        oss << "│ Drain cycles:        " << std::setw(12) << drain_cycles << "\n";
        oss << "└─────────────────────────────────────────────────────────────┘\n";
        oss << "\n┌─ Utilization ───────────────────────────────────────────────┐\n";
        oss << "│ Compute utilization: " << std::setw(11) << std::fixed << std::setprecision(1)
            << compute_utilization * 100 << "%\n";
        oss << "│ Memory utilization:  " << std::setw(11) << memory_utilization * 100 << "%\n";
        oss << "│ Overall efficiency:  " << std::setw(11) << overall_efficiency * 100 << "%\n";
        oss << "│ Bottleneck:          " << std::setw(12)
            << (is_compute_bound ? "COMPUTE" : "MEMORY") << "\n";
        oss << "└─────────────────────────────────────────────────────────────┘\n";
        oss << "\n┌─ Concurrency ───────────────────────────────────────────────┐\n";
        oss << "│ Peak compute:        " << std::setw(12) << peak_compute_concurrency << "\n";
        oss << "│ Peak memory:         " << std::setw(12) << peak_memory_concurrency << "\n";
        oss << "│ Avg compute:         " << std::setw(11) << std::setprecision(1)
            << avg_compute_concurrency << "\n";
        oss << "└─────────────────────────────────────────────────────────────┘\n";
        return oss.str();
    }
};

class ScheduleAnalyzer {
public:
    ScheduleAnalyzer(const HardwareConfig& hw, const TileGeometry& geom)
        : hw_(hw), geom_(geom),
          compute_(hw, geom),
          data_move_(hw, geom) {}

    UtilizationMetrics analyze(const Schedule& schedule) const {
        UtilizationMetrics m{};

        m.total_cycles = schedule.total_cycles();

        // Sum up cycles by type
        for (const auto& e : schedule.events) {
            switch (e.type) {
                case ScheduleEvent::Type::COMPUTE:
                    m.compute_cycles += e.duration;
                    break;
                case ScheduleEvent::Type::LOAD_A:
                case ScheduleEvent::Type::LOAD_B:
                    m.load_cycles += e.duration;
                    break;
                case ScheduleEvent::Type::DRAIN_C:
                    m.drain_cycles += e.duration;
                    break;
                case ScheduleEvent::Type::TRANSFER_A:
                case ScheduleEvent::Type::TRANSFER_B:
                    m.transfer_cycles += e.duration;
                    break;
            }
        }

        // Compute utilization
        Cycles max_compute = m.total_cycles * hw_.compute_tile_count;
        m.compute_utilization = max_compute > 0 ?
            static_cast<double>(m.compute_cycles) / static_cast<double>(max_compute) : 0;

        // Memory utilization (DMA + block movers)
        Cycles max_memory = m.total_cycles * (hw_.dma_engine_count + hw_.block_mover_count);
        Cycles total_memory = m.load_cycles + m.drain_cycles + m.transfer_cycles;
        m.memory_utilization = max_memory > 0 ?
            static_cast<double>(total_memory) / static_cast<double>(max_memory) : 0;

        // Efficiency vs theoretical (capped at 100%)
        Cycles theoretical = compute_.theoretical_min_cycles();
        m.overall_efficiency = theoretical > 0 ?
            std::min(1.0, static_cast<double>(theoretical) / static_cast<double>(m.total_cycles)) : 0;

        // Critical paths
        m.compute_critical_path = find_critical_path(schedule, true);
        m.memory_critical_path = find_critical_path(schedule, false);
        m.is_compute_bound = m.compute_critical_path >= m.memory_critical_path;

        // Concurrency analysis
        analyze_concurrency(schedule, m);

        return m;
    }

    // Generate the "current poor schedule" (serialized waves)
    Schedule generate_serialized_wave_schedule() const {
        Schedule sched;
        Cycles current_cycle = 0;

        // Load all data first
        Cycles load_a = data_move_.load_A_tile_cycles();
        Cycles load_b = data_move_.load_B_tile_cycles();

        for (size_t m = 0; m < geom_.M_tiles; ++m) {
            for (size_t k = 0; k < geom_.K_tiles; ++k) {
                sched.add({ScheduleEvent::Type::LOAD_A,
                          m % hw_.dma_engine_count,
                          {m, k}, 0, current_cycle, load_a});
                current_cycle += load_a / hw_.dma_engine_count;
            }
        }
        for (size_t k = 0; k < geom_.K_tiles; ++k) {
            for (size_t n = 0; n < geom_.N_tiles; ++n) {
                sched.add({ScheduleEvent::Type::LOAD_B,
                          (k * geom_.N_tiles + n) % hw_.dma_engine_count,
                          {k, n}, 0, current_cycle, load_b});
                current_cycle += load_b / hw_.dma_engine_count;
            }
        }

        [[maybe_unused]] Cycles load_end = current_cycle;

        // Compute in waves (serialized rows)
        Cycles tile_compute = compute_.tile_compute_cycles();
        Cycles drain = data_move_.drain_C_tile_cycles();

        for (size_t wave = 0; wave < geom_.M_tiles; ++wave) {
            Cycles wave_start = current_cycle;

            // All tiles in this wave compute in parallel
            for (size_t n = 0; n < geom_.N_tiles; ++n) {
                size_t compute_tile = wave * geom_.N_tiles + n;
                if (compute_tile < hw_.compute_tile_count) {
                    sched.add({ScheduleEvent::Type::COMPUTE,
                              compute_tile,
                              {wave, n}, 0, wave_start, tile_compute});
                }
            }

            current_cycle = wave_start + tile_compute;

            // Drain this wave's results
            for (size_t n = 0; n < geom_.N_tiles; ++n) {
                sched.add({ScheduleEvent::Type::DRAIN_C,
                          n % hw_.dma_engine_count,
                          {wave, n}, 0, current_cycle, drain});
            }
            current_cycle += drain;
        }

        return sched;
    }

    // Generate fully parallel schedule (all tiles concurrent)
    // This requires data replication/streaming
    Schedule generate_fully_parallel_schedule() const {
        Schedule sched;

        // For fully parallel, we need to stream A and B tiles
        // Each compute tile needs access to its A row and B column

        Cycles current_cycle = 0;
        Cycles load_a = data_move_.load_A_tile_cycles();
        Cycles load_b = data_move_.load_B_tile_cycles();
        Cycles subtile_compute = compute_.subtile_compute_cycles();
        [[maybe_unused]] Cycles transfer_a = data_move_.l3_to_l3_A_tile_cycles();
        [[maybe_unused]] Cycles transfer_b = data_move_.l3_to_l3_B_tile_cycles();

        // Phase 1: Initial data load
        // Load first K-tile of A and B
        for (size_t m = 0; m < geom_.M_tiles; ++m) {
            sched.add({ScheduleEvent::Type::LOAD_A,
                      m % hw_.dma_engine_count,
                      {m, 0}, 0, current_cycle, load_a});
        }
        for (size_t n = 0; n < geom_.N_tiles; ++n) {
            sched.add({ScheduleEvent::Type::LOAD_B,
                      n % hw_.dma_engine_count,
                      {0, n}, 0, current_cycle, load_b});
        }
        current_cycle += std::max(load_a, load_b);

        // Phase 2: Pipelined compute with data streaming
        for (size_t k = 0; k < geom_.K_tiles; ++k) {
            Cycles k_step_start = current_cycle;

            // All compute tiles work on this K-step
            for (size_t m = 0; m < geom_.M_tiles; ++m) {
                for (size_t n = 0; n < geom_.N_tiles; ++n) {
                    size_t ct = m * geom_.N_tiles + n;
                    if (ct < hw_.compute_tile_count) {
                        sched.add({ScheduleEvent::Type::COMPUTE,
                                  ct, {m, n}, k, k_step_start, subtile_compute});
                    }
                }
            }

            // Overlap: Load next K-step's data while computing
            if (k + 1 < geom_.K_tiles) {
                for (size_t m = 0; m < geom_.M_tiles; ++m) {
                    sched.add({ScheduleEvent::Type::LOAD_A,
                              m % hw_.dma_engine_count,
                              {m, k+1}, 0, k_step_start, load_a});
                }
                for (size_t n = 0; n < geom_.N_tiles; ++n) {
                    sched.add({ScheduleEvent::Type::LOAD_B,
                              n % hw_.dma_engine_count,
                              {k+1, n}, 0, k_step_start, load_b});
                }
            }

            current_cycle = k_step_start + subtile_compute;
        }

        // Phase 3: Drain all C tiles
        Cycles drain = data_move_.drain_C_tile_cycles();
        for (size_t m = 0; m < geom_.M_tiles; ++m) {
            for (size_t n = 0; n < geom_.N_tiles; ++n) {
                sched.add({ScheduleEvent::Type::DRAIN_C,
                          (m * geom_.N_tiles + n) % hw_.dma_engine_count,
                          {m, n}, 0, current_cycle, drain});
            }
        }
        current_cycle += drain * geom_.total_C_tiles() / hw_.dma_engine_count;

        return sched;
    }

private:
    const HardwareConfig& hw_;
    const TileGeometry& geom_;
    ComputeModel compute_;
    DataMovementModel data_move_;

    Cycles find_critical_path(const Schedule& schedule, bool compute_only) const {
        Cycles max_end = 0;
        for (const auto& e : schedule.events) {
            if (compute_only && e.type != ScheduleEvent::Type::COMPUTE) continue;
            if (!compute_only && e.type == ScheduleEvent::Type::COMPUTE) continue;
            max_end = std::max(max_end, e.end_cycle());
        }
        return max_end;
    }

    void analyze_concurrency(const Schedule& schedule, UtilizationMetrics& m) const {
        if (schedule.events.empty()) return;

        // Build timeline of active operations
        struct Event { Cycles cycle; int delta; bool is_compute; };
        std::vector<Event> timeline;

        for (const auto& e : schedule.events) {
            bool is_compute = (e.type == ScheduleEvent::Type::COMPUTE);
            timeline.push_back({e.start_cycle, +1, is_compute});
            timeline.push_back({e.end_cycle(), -1, is_compute});
        }

        std::sort(timeline.begin(), timeline.end(),
            [](const Event& a, const Event& b) { return a.cycle < b.cycle; });

        int compute_active = 0, memory_active = 0;
        m.peak_compute_concurrency = 0;
        m.peak_memory_concurrency = 0;

        Cycles last_cycle = 0;
        double weighted_compute = 0;

        for (const auto& ev : timeline) {
            if (ev.cycle > last_cycle) {
                weighted_compute += compute_active * static_cast<double>(ev.cycle - last_cycle);
                last_cycle = ev.cycle;
            }

            if (ev.is_compute) {
                compute_active += ev.delta;
                m.peak_compute_concurrency = std::max(m.peak_compute_concurrency,
                    static_cast<size_t>(compute_active));
            } else {
                memory_active += ev.delta;
                m.peak_memory_concurrency = std::max(m.peak_memory_concurrency,
                    static_cast<size_t>(memory_active));
            }
        }

        m.avg_compute_concurrency = m.total_cycles > 0 ?
            weighted_compute / static_cast<double>(m.total_cycles) : 0;
    }
};

//=============================================================================
// Convenience Functions
//=============================================================================

inline void print_analysis(const HardwareConfig& hw, const TileGeometry& geom) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TILED MATMUL ANALYSIS\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << geom.describe() << "\n";

    ComputeModel compute(hw, geom);
    std::cout << compute.describe() << "\n";

    DataMovementModel dm(hw, geom);
    std::cout << dm.describe() << "\n";

    ScheduleAnalyzer analyzer(hw, geom);

    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "SERIALIZED WAVE SCHEDULE (Current Implementation)\n";
    std::cout << std::string(70, '-') << "\n";

    auto serial_sched = analyzer.generate_serialized_wave_schedule();
    auto serial_metrics = analyzer.analyze(serial_sched);
    std::cout << serial_metrics.describe();

    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "FULLY PARALLEL SCHEDULE (Optimized)\n";
    std::cout << std::string(70, '-') << "\n";

    auto parallel_sched = analyzer.generate_fully_parallel_schedule();
    auto parallel_metrics = analyzer.analyze(parallel_sched);
    std::cout << parallel_metrics.describe();

    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "IMPROVEMENT\n";
    std::cout << std::string(70, '-') << "\n";
    double speedup = static_cast<double>(serial_metrics.total_cycles) /
                     static_cast<double>(parallel_metrics.total_cycles);
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "Efficiency improvement: " << std::setprecision(1)
              << (parallel_metrics.overall_efficiency - serial_metrics.overall_efficiency) * 100
              << " percentage points\n";
}

} // namespace sw::analysis

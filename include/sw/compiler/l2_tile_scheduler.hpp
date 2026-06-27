/**
 * @file l2_tile_scheduler.hpp
 * @brief L2 tile scheduler for minimizing DRAM accesses in KPU matrix multiplication
 *
 * The L2 tile scheduler manages the allocation and scheduling of matrix tiles in the
 * L2 cache banks to minimize L3 reloads and ultimately DRAM accesses. It tracks which
 * tiles are resident in L2 at any given time and generates an optimal sequence of
 * tile loads/reloads to complete a block matrix multiplication.
 *
 * Memory Hierarchy:
 *   DRAM → L3 Tiles (128KB × 4) → L2 Banks (64KB × 8) → L1 ASM → Systolic Array
 *
 * Key Objectives:
 *   1. Minimize the number of L2 tile reloads from L3
 *   2. Maximize data reuse in L2 for A and B tiles
 *   3. Track L2 capacity and manage tile evictions
 *   4. Generate the sequence of L2 loads for output-stationary execution
 */

#pragma once

#include <sw/compiler/tile_optimizer.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <optional>

namespace sw::kpu::compiler {

/**
 * @brief L2 tile scheduler for managing tile allocations and load sequences
 *
 * This scheduler operates at the L2 cache level and determines:
 * - Which tiles should be loaded into L2 at what time
 * - Where in L2 each tile should be allocated
 * - When tiles can be evicted to make room for new tiles
 * - The optimal ordering to minimize L3 reloads
 */
class L2TileScheduler {
public:
    /// Tile identifier for A, B, or C matrices
    struct TileID {
        char matrix;        ///< 'A', 'B', or 'C'
        Size row_idx;       ///< Row index in tile grid
        Size col_idx;       ///< Column index in tile grid

        TileID() : matrix('A'), row_idx(0), col_idx(0) {}
        TileID(char m, Size r, Size c) : matrix(m), row_idx(r), col_idx(c) {}

        bool operator<(const TileID& other) const {
            if (matrix != other.matrix) return matrix < other.matrix;
            if (row_idx != other.row_idx) return row_idx < other.row_idx;
            return col_idx < other.col_idx;
        }

        bool operator==(const TileID& other) const {
            return matrix == other.matrix &&
                   row_idx == other.row_idx &&
                   col_idx == other.col_idx;
        }

        std::string to_string() const {
            return std::string(1, matrix) + "[" +
                   std::to_string(row_idx) + "," +
                   std::to_string(col_idx) + "]";
        }
    };

    /// L2 bank allocation slot
    struct L2Slot {
        size_t bank_id;         ///< Which L2 bank (0-7 for default KPU)
        Address offset;          ///< Offset within bank
        Size size_bytes;         ///< Size of allocation
        std::optional<TileID> tile;  ///< Which tile is allocated (empty if free)

        L2Slot() : bank_id(0), offset(0), size_bytes(0), tile(std::nullopt) {}
        L2Slot(size_t bank, Address off, Size sz)
            : bank_id(bank), offset(off), size_bytes(sz), tile(std::nullopt) {}

        bool is_free() const { return !tile.has_value(); }

        std::string to_string() const {
            std::string s = "L2[" + std::to_string(bank_id) + "]@" +
                          std::to_string(offset);
            if (tile) {
                s += " <- " + tile->to_string();
            } else {
                s += " [FREE]";
            }
            return s;
        }
    };

    /// Tile load/reload event in the schedule
    struct TileLoad {
        enum class Type {
            INITIAL_LOAD,   ///< First load from L3
            RELOAD,         ///< Reload after eviction
            PREFETCH        ///< Speculative prefetch
        };

        Type type;
        TileID tile_id;
        size_t slot_index;      ///< Which L2 slot to load into
        Size time_step;         ///< When this load occurs
        bool from_dram;         ///< Load from DRAM (L3 miss) vs L3 hit

        // Context: which compute operation needs this tile
        Size compute_ti;        ///< Compute tile index in M dimension
        Size compute_tj;        ///< Compute tile index in N dimension
        Size compute_tk;        ///< Compute tile index in K dimension

        TileLoad() : type(Type::INITIAL_LOAD), tile_id(), slot_index(0),
                    time_step(0), from_dram(false),
                    compute_ti(0), compute_tj(0), compute_tk(0) {}

        std::string to_string() const {
            std::string type_str;
            switch (type) {
                case Type::INITIAL_LOAD: type_str = "LOAD"; break;
                case Type::RELOAD: type_str = "RELOAD"; break;
                case Type::PREFETCH: type_str = "PREFETCH"; break;
            }

            std::string source = from_dram ? "DRAM" : "L3";

            return "t=" + std::to_string(time_step) + ": " +
                   type_str + " " + tile_id.to_string() +
                   " -> L2Slot[" + std::to_string(slot_index) + "]" +
                   " from " + source +
                   " for C[" + std::to_string(compute_ti) + "," +
                   std::to_string(compute_tj) + "]";
        }
    };

    /// Cache replacement policy
    enum class ReplacementPolicy {
        LRU,            ///< Least Recently Used
        FIFO,           ///< First In First Out
        OPTIMAL,        ///< Belady's optimal (requires future knowledge)
        MANUAL          ///< User-specified eviction order
    };

    /// Scheduling strategy for tile loads
    enum class SchedulingStrategy {
        WEIGHT_STATIONARY,      ///< B (weights) stay resident, stream A and C
        INPUT_STATIONARY,       ///< A (inputs) stay resident, stream B and C
        OUTPUT_STATIONARY,      ///< C tiles stay in systolic array, A/B streamed
        ROW_MAJOR_C,           ///< Process C tiles in row-major order
        COLUMN_MAJOR_C,        ///< Process C tiles in column-major order
        MORTON_ORDER,          ///< Z-order/Morton curve for spatial locality
        CUSTOM                 ///< User-defined tile ordering
    };

    /// Complete L2 schedule for a matrix multiplication
    struct L2Schedule {
        // Matrix dimensions
        Size M, N, K;

        // Tile configuration
        TileOptimizer::TileConfig config;

        // Scheduling strategy used to generate this schedule
        SchedulingStrategy strategy;

        // Tile grid dimensions
        Size num_tile_rows_A;   ///< Number of A tile rows (M / Ti)
        Size num_tile_cols_A;   ///< Number of A tile cols (K / Tk)
        Size num_tile_rows_B;   ///< Number of B tile rows (K / Tk)
        Size num_tile_cols_B;   ///< Number of B tile cols (N / Tj)
        Size num_tile_rows_C;   ///< Number of C tile rows (M / Ti)
        Size num_tile_cols_C;   ///< Number of C tile cols (N / Tj)

        // L2 configuration
        size_t num_l2_banks;
        Size l2_bank_size;
        Size l2_total_capacity;
        size_t max_l2_slots;    ///< Maximum number of tiles that fit in L2

        // Allocation state
        std::vector<L2Slot> slots;  ///< L2 slot allocations

        // Load sequence
        std::vector<TileLoad> load_sequence;

        // Statistics
        Size total_loads;       ///< Total number of tile loads
        Size initial_loads;     ///< Number of initial loads
        Size reloads;           ///< Number of reloads (cache misses)
        Size l3_hits;           ///< Loads satisfied from L3
        Size l3_misses;         ///< Loads requiring DRAM access

        Size total_bytes_loaded;    ///< Total data movement L3→L2
        Size wasted_loads;          ///< Tiles loaded but evicted before use

        double l2_hit_rate;     ///< Percentage of tile accesses satisfied by L2
        double l3_hit_rate;     ///< Percentage of L2 loads satisfied by L3

        // Reuse tracking
        std::map<TileID, Size> tile_access_count;   ///< How many times each tile is accessed
        std::map<TileID, Size> tile_load_count;     ///< How many times each tile is loaded

        L2Schedule() : M(0), N(0), K(0),
                      num_tile_rows_A(0), num_tile_cols_A(0),
                      num_tile_rows_B(0), num_tile_cols_B(0),
                      num_tile_rows_C(0), num_tile_cols_C(0),
                      num_l2_banks(0), l2_bank_size(0), l2_total_capacity(0),
                      max_l2_slots(0),
                      total_loads(0), initial_loads(0), reloads(0),
                      l3_hits(0), l3_misses(0),
                      total_bytes_loaded(0), wasted_loads(0),
                      l2_hit_rate(0.0), l3_hit_rate(0.0) {}
    };

public:
    /**
     * @brief Constructor
     * @param mem Memory hierarchy specification (from TileOptimizer)
     * @param systolic_size Systolic array dimension (default 16×16)
     */
    explicit L2TileScheduler(
        const TileOptimizer::MemoryHierarchy& mem = TileOptimizer::MemoryHierarchy(),
        Size systolic_size = 16);

    /**
     * @brief Generate L2 tile schedule for matrix multiplication
     *
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param config Tile configuration from TileOptimizer
     * @param policy Cache replacement policy
     * @param strategy Tile scheduling strategy
     * @return Complete L2 schedule with load sequence
     */
    L2Schedule generate_schedule(
        Size M, Size N, Size K,
        const TileOptimizer::TileConfig& config,
        ReplacementPolicy policy = ReplacementPolicy::LRU,
        SchedulingStrategy strategy = SchedulingStrategy::OUTPUT_STATIONARY);

    /**
     * @brief Allocate L2 slots for initial tile set
     *
     * Determines which tiles should be resident in L2 initially and
     * allocates L2 slots for them.
     */
    void allocate_initial_tiles(L2Schedule& schedule);

    /**
     * @brief Generate the complete tile load sequence
     *
     * Walks through the computation order (output-stationary) and generates
     * tile loads/reloads as needed.
     */
    void generate_load_sequence(L2Schedule& schedule);

    /**
     * @brief Find or allocate an L2 slot for a tile
     *
     * @param schedule Current schedule
     * @param tile_id Tile to allocate
     * @param time_step Current time step
     * @return Slot index, or nullopt if no space (eviction needed)
     */
    std::optional<size_t> find_or_allocate_slot(
        L2Schedule& schedule,
        const TileID& tile_id,
        Size time_step);

    /**
     * @brief Evict a tile from L2 to make room
     *
     * @param schedule Current schedule
     * @param time_step Current time step
     * @return Index of evicted slot, or nullopt if eviction failed
     */
    std::optional<size_t> evict_tile(
        L2Schedule& schedule,
        Size time_step);

    /**
     * @brief Check if a tile is currently resident in L2
     */
    bool is_resident(const L2Schedule& schedule, const TileID& tile_id) const;

    /**
     * @brief Get the slot index for a resident tile
     */
    std::optional<size_t> get_slot_index(
        const L2Schedule& schedule,
        const TileID& tile_id) const;

    /**
     * @brief Calculate reuse statistics for all tiles
     */
    void calculate_reuse_stats(L2Schedule& schedule) const;

    /**
     * @brief Print L2 schedule in human-readable format
     */
    void print_schedule(const L2Schedule& schedule, bool verbose = false) const;

    /**
     * @brief Print current L2 allocation state
     */
    void print_l2_state(const L2Schedule& schedule) const;

    /**
     * @brief Print tile load sequence
     */
    void print_load_sequence(const L2Schedule& schedule, Size max_entries = 20) const;

    /**
     * @brief Print reuse statistics
     */
    void print_reuse_stats(const L2Schedule& schedule) const;

    /**
     * @brief Export schedule to JSON format
     */
    std::string export_json(const L2Schedule& schedule) const;

    /**
     * @brief Visualize L2 state over time (ASCII art)
     */
    void visualize_l2_timeline(const L2Schedule& schedule) const;

    // Accessors
    const TileOptimizer::MemoryHierarchy& memory_hierarchy() const { return memory_; }
    void set_memory_hierarchy(const TileOptimizer::MemoryHierarchy& mem) { memory_ = mem; }

    ReplacementPolicy replacement_policy() const { return policy_; }
    void set_replacement_policy(ReplacementPolicy policy) { policy_ = policy; }

    SchedulingStrategy scheduling_strategy() const { return strategy_; }
    void set_scheduling_strategy(SchedulingStrategy strategy) { strategy_ = strategy; }

private:
    TileOptimizer::MemoryHierarchy memory_;
    [[maybe_unused]] Size systolic_size_;
    ReplacementPolicy policy_;
    SchedulingStrategy strategy_;

    // LRU tracking
    std::map<TileID, Size> tile_last_access_;  ///< Last access time for each tile

    // Helper methods

    /**
     * @brief Calculate tile dimensions for a specific tile index
     */
    Size get_tile_dimension(Size global_dim, Size tile_idx, Size tile_size) const {
        return std::min(tile_size, global_dim - tile_idx * tile_size);
    }

    /**
     * @brief Calculate size in bytes for a tile
     */
    Size calculate_tile_bytes(const TileID& tile_id,
                             const L2Schedule& schedule) const;

    /**
     * @brief Generate compute tile iteration order
     *
     * Returns a vector of (ti, tj, tk) tuples in the order they should be computed
     */
    std::vector<std::tuple<Size, Size, Size>> generate_compute_order(
        const L2Schedule& schedule) const;

    /**
     * @brief Determine which tiles are needed for a compute operation
     */
    struct TileSet {
        TileID a_tile;
        TileID b_tile;
        TileID c_tile;
    };

    TileSet get_required_tiles(Size ti, Size tj, Size tk) const {
        return TileSet{
            TileID{'A', ti, tk},
            TileID{'B', tk, tj},
            TileID{'C', ti, tj}
        };
    }

    /**
     * @brief Select victim tile for eviction based on policy
     */
    std::optional<size_t> select_victim(
        const L2Schedule& schedule,
        Size time_step) const;

    /**
     * @brief LRU victim selection
     */
    std::optional<size_t> select_lru_victim(
        const L2Schedule& schedule) const;

    /**
     * @brief Optimal (Belady) victim selection
     * Requires future knowledge of tile accesses
     */
    std::optional<size_t> select_optimal_victim(
        const L2Schedule& schedule,
        Size time_step) const;

    /**
     * @brief Check if L3 contains the tile (assumes simple L3 capacity model)
     */
    bool is_in_l3(const TileID& tile_id, const L2Schedule& schedule) const;

    /**
     * @brief Update L3 state after a load
     */
    void update_l3_state(const TileID& tile_id, L2Schedule& schedule);
};

} // namespace sw::kpu::compiler

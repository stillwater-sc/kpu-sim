// ============================================================================
// include/sw/kpu/dataflow/block_mover_compiler.hpp
// Compiler from TileDataFlowGraph to BlockMover programs
// ============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sw/kpu/components/block_mover_isa.hpp"
#include "sw/kpu/components/l3_interconnect.hpp"
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"

namespace sw::kpu::dataflow {

// Import Direction from sw::kpu namespace
using sw::kpu::Direction;

// ============================================================================
// Compiled Schedule
// ============================================================================

/// A complete compiled schedule ready for execution
struct CompiledSchedule {
    /// BlockMover programs for each L3 tile
    std::array<BlockMoverProgram, 16> blockmover_programs;

    /// Estimated total cycles
    int64_t estimated_cycles = 0;

    /// Estimated compute cycles (actual compute time)
    int64_t compute_cycles = 0;

    /// Estimated data movement cycles
    int64_t data_movement_cycles = 0;

    /// Statistics
    struct Stats {
        size_t total_commands = 0;
        size_t total_l3_transfers = 0;
        size_t total_l2_transfers = 0;
        size_t total_dma_ops = 0;
        size_t total_compute_ops = 0;
        size_t total_triggers = 0;
        size_t total_barriers = 0;
    };
    Stats stats;

    /// Get program for a specific L3 tile
    const BlockMoverProgram& program(uint8_t l3_id) const {
        return blockmover_programs[l3_id];
    }

    BlockMoverProgram& program(uint8_t l3_id) {
        return blockmover_programs[l3_id];
    }

    /// Check if any programs are non-empty
    bool has_work() const {
        for (const auto& prog : blockmover_programs) {
            if (!prog.empty()) return true;
        }
        return false;
    }

    /// Get all non-empty programs
    std::vector<std::pair<uint8_t, const BlockMoverProgram*>> active_programs() const {
        std::vector<std::pair<uint8_t, const BlockMoverProgram*>> result;
        for (uint8_t i = 0; i < 16; ++i) {
            if (!blockmover_programs[i].empty()) {
                result.push_back({i, &blockmover_programs[i]});
            }
        }
        return result;
    }

    /// Debug output
    std::string to_string() const;
    void print_summary() const;
};

// ============================================================================
// BlockMover Compiler
// ============================================================================

/// Compiles TileDataFlowGraph to BlockMover programs
class BlockMoverCompiler {
public:
    struct Config {
        uint8_t num_l3_tiles = 16;
        uint8_t mesh_rows = 4;
        uint8_t mesh_cols = 4;

        // Optimization options
        bool enable_pipelining = true;          // Overlap loads with compute
        bool enable_double_buffering = false;   // Use double-buffered L3 regions
        bool emit_barriers = true;              // Insert barriers between phases

        // Trigger allocation strategy
        bool use_explicit_triggers = true;      // vs implicit ordering
    };

    /// Default constructor
    BlockMoverCompiler();

    /// Constructor with custom configuration
    explicit BlockMoverCompiler(const Config& config);

    /// Compile a TileDataFlowGraph to BlockMover programs
    CompiledSchedule compile(const TileDataFlowGraph& dfg);

    /// Compile a TileDataFlowGraph with an existing schedule
    CompiledSchedule compile(const TileDataFlowGraph& dfg, const DFGSchedule& schedule);

    /// Get the last compilation statistics
    const CompiledSchedule::Stats& last_stats() const { return last_stats_; }

private:
    Config config_;
    CompiledSchedule::Stats last_stats_;

    // Trigger allocation
    struct TriggerAllocation {
        std::map<size_t, uint8_t> node_to_trigger;  // node_id → trigger channel
        uint8_t next_trigger = 0;
    };
    TriggerAllocation trigger_alloc_;

    // ========== Compilation Phases ==========

    /// Phase 1: Assign triggers for synchronization
    void assign_triggers(const TileDataFlowGraph& dfg);

    /// Phase 2: Generate commands for each node
    void emit_node(const DFNode& node, const TileDataFlowGraph& dfg,
                   CompiledSchedule& schedule);

    /// Phase 3: Insert barriers and synchronization
    void insert_barriers(CompiledSchedule& schedule);

    /// Phase 4: Assemble programs (resolve loop offsets)
    void assemble_programs(CompiledSchedule& schedule);

    // ========== Node-specific Emission ==========

    void emit_dma_load(const DFNode& node, BlockMoverProgram& prog);
    void emit_dma_store(const DFNode& node, BlockMoverProgram& prog);
    void emit_l3_transfer(const DFNode& node, BlockMoverProgram& prog);
    void emit_l3_to_l2(const DFNode& node, BlockMoverProgram& prog);
    void emit_l2_to_l3(const DFNode& node, BlockMoverProgram& prog);
    void emit_matmul(const DFNode& node, const TileDataFlowGraph& dfg,
                     BlockMoverProgram& prog);
    void emit_barrier(const DFNode& node, BlockMoverProgram& prog);

    // ========== Helpers ==========

    /// Get direction from source L3 to destination L3
    std::optional<Direction> get_direction(uint8_t src_l3, uint8_t dst_l3) const;

    /// Allocate a trigger channel
    uint8_t allocate_trigger();

    /// Get trigger for a node (allocates if not already assigned)
    uint8_t get_trigger_for_node(size_t node_id);

    /// Get mesh position from L3 ID
    std::pair<uint8_t, uint8_t> get_mesh_position(uint8_t l3_id) const;

    /// Get L3 ID from mesh position
    uint8_t get_l3_id(uint8_t row, uint8_t col) const;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Compile a matmul to BlockMover programs
CompiledSchedule compile_matmul(
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles,
    const BlockMoverCompiler::Config& config = {});

/// Print a compiled schedule for debugging
void print_compiled_schedule(const CompiledSchedule& schedule);

} // namespace sw::kpu::dataflow

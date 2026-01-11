// ============================================================================
// include/sw/kpu/dataflow/tile_dependency_analyzer.hpp
// Tile Dependency Analyzer - Value-use chain analysis for tile data flow
// ============================================================================

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include "sw/kpu/dataflow/tile_flow_tracer.hpp"

namespace sw::kpu::dataflow {

// ============================================================================
// Tile Dependency Chain
// ============================================================================

/// Represents the complete dependency chain for an output tile
struct TileDependencyChain {
    TileDescriptor output_tile;               // The C[m,n] tile
    std::vector<TileDescriptor> a_tiles;      // All A[m,k] tiles needed
    std::vector<TileDescriptor> b_tiles;      // All B[k,n] tiles needed

    /// Movement path for a single tile through L3 mesh
    struct TileMovement {
        TileDescriptor tile;
        uint8_t start_l3;                     // Where tile enters mesh
        uint8_t end_l3;                       // Where tile is consumed
        std::vector<uint8_t> path;            // L3 IDs traversed (hop-and-forward)
    };

    std::vector<TileMovement> a_movements;    // Paths for A tiles
    std::vector<TileMovement> b_movements;    // Paths for B tiles

    /// Get all tiles involved in this dependency chain
    std::set<std::tuple<uint8_t, uint16_t, uint16_t, uint16_t>> get_all_tile_keys() const {
        std::set<std::tuple<uint8_t, uint16_t, uint16_t, uint16_t>> keys;

        // Output tile (C)
        keys.insert({output_tile.tensor, output_tile.m_tile,
                     output_tile.n_tile, output_tile.k_tile});

        // Input tiles (A)
        for (const auto& t : a_tiles) {
            keys.insert({t.tensor, t.m_tile, t.n_tile, t.k_tile});
        }

        // Input tiles (B)
        for (const auto& t : b_tiles) {
            keys.insert({t.tensor, t.m_tile, t.n_tile, t.k_tile});
        }

        return keys;
    }

    /// Get count of all tiles
    size_t total_tiles() const {
        return 1 + a_tiles.size() + b_tiles.size();  // C + A tiles + B tiles
    }
};

// ============================================================================
// Tile Dependency Analyzer
// ============================================================================

/// Analyzes tile dependencies for matmul operations
class TileDependencyAnalyzer {
public:
    /// Configuration for the matmul tiling
    struct Config {
        size_t m_tiles = 4;      // Number of M-dimension tiles
        size_t n_tiles = 4;      // Number of N-dimension tiles
        size_t k_tiles = 4;      // Number of K-dimension tiles

        size_t tile_m = 64;      // Tile size in M dimension
        size_t tile_n = 64;      // Tile size in N dimension
        size_t tile_k = 64;      // Tile size in K dimension

        size_t mesh_rows = 4;    // L3 mesh rows
        size_t mesh_cols = 4;    // L3 mesh columns
    };

    explicit TileDependencyAnalyzer(const Config& config)
        : config_(config) {}

    /// Get all dependencies for output tile C[m,n]
    TileDependencyChain get_dependencies_for_output(
        uint16_t m_tile, uint16_t n_tile) const {

        TileDependencyChain chain;

        // Output tile: C[m,n]
        chain.output_tile.tensor = 2;  // C tensor
        chain.output_tile.m_tile = m_tile;
        chain.output_tile.n_tile = n_tile;
        chain.output_tile.k_tile = 0;
        chain.output_tile.size = config_.tile_m * config_.tile_n * sizeof(float);

        // For matmul: C[m,n] = sum_k(A[m,k] * B[k,n])
        // We need A[m,k] for all k, and B[k,n] for all k
        for (uint16_t k = 0; k < config_.k_tiles; k++) {
            // A tile: A[m,k]
            TileDescriptor a_tile;
            a_tile.tensor = 0;  // A tensor
            a_tile.m_tile = m_tile;
            a_tile.n_tile = 0;  // A doesn't have n dimension in tiling
            a_tile.k_tile = k;
            a_tile.size = config_.tile_m * config_.tile_k * sizeof(float);
            chain.a_tiles.push_back(a_tile);

            // B tile: B[k,n]
            TileDescriptor b_tile;
            b_tile.tensor = 1;  // B tensor
            b_tile.m_tile = 0;  // B doesn't have m dimension in tiling
            b_tile.n_tile = n_tile;
            b_tile.k_tile = k;
            b_tile.size = config_.tile_k * config_.tile_n * sizeof(float);
            chain.b_tiles.push_back(b_tile);
        }

        // Compute movement paths through the L3 mesh
        compute_movement_paths(chain);

        return chain;
    }

    /// Filter events to only those related to a specific output tile
    std::vector<TileFlowEvent> filter_events(
        const std::vector<TileFlowEvent>& all_events,
        const TileDependencyChain& chain) const {

        auto tile_keys = chain.get_all_tile_keys();
        std::vector<TileFlowEvent> filtered;
        filtered.reserve(all_events.size() / 4);  // Estimate

        for (const auto& event : all_events) {
            auto key = std::make_tuple(
                event.tile.tensor,
                event.tile.m_tile,
                event.tile.n_tile,
                event.tile.k_tile
            );

            if (tile_keys.count(key) > 0) {
                filtered.push_back(event);
            }
        }

        return filtered;
    }

    /// Filter events for a specific output tile by coordinates
    std::vector<TileFlowEvent> filter_events_for_output(
        const std::vector<TileFlowEvent>& all_events,
        uint16_t m_tile, uint16_t n_tile) const {

        auto chain = get_dependencies_for_output(m_tile, n_tile);
        return filter_events(all_events, chain);
    }

    /// Get L3 tiles involved in computing C[m,n]
    std::set<uint8_t> get_involved_l3_tiles(uint16_t m_tile, uint16_t n_tile) const {
        std::set<uint8_t> l3_tiles;

        // The output L3 tile (where C[m,n] is computed)
        uint8_t output_l3 = static_cast<uint8_t>(m_tile * config_.mesh_cols + n_tile);
        l3_tiles.insert(output_l3);

        // For hop-and-forward:
        // A tiles enter from left column (col 0) and hop right
        // B tiles enter from top row (row 0) and hop down

        // A tiles path: from (m_tile, 0) to (m_tile, n_tile)
        for (uint16_t col = 0; col <= n_tile; col++) {
            l3_tiles.insert(static_cast<uint8_t>(m_tile * config_.mesh_cols + col));
        }

        // B tiles path: from (0, n_tile) to (m_tile, n_tile)
        for (uint16_t row = 0; row <= m_tile; row++) {
            l3_tiles.insert(static_cast<uint8_t>(row * config_.mesh_cols + n_tile));
        }

        return l3_tiles;
    }

    /// Get configuration
    const Config& config() const { return config_; }

    /// Get summary string for a dependency chain
    static std::string summarize(const TileDependencyChain& chain) {
        std::ostringstream ss;
        ss << "C[" << chain.output_tile.m_tile << ","
           << chain.output_tile.n_tile << "] requires "
           << chain.a_tiles.size() << " A tiles + "
           << chain.b_tiles.size() << " B tiles";
        return ss.str();
    }

private:
    Config config_;

    /// Compute hop-and-forward movement paths for tiles
    void compute_movement_paths(TileDependencyChain& chain) const {
        uint16_t m = chain.output_tile.m_tile;
        uint16_t n = chain.output_tile.n_tile;

        // A tiles hop from left (col 0) to destination column
        for (size_t i = 0; i < chain.a_tiles.size(); i++) {
            TileDependencyChain::TileMovement movement;
            movement.tile = chain.a_tiles[i];
            movement.start_l3 = static_cast<uint8_t>(m * config_.mesh_cols);  // (m, 0)
            movement.end_l3 = static_cast<uint8_t>(m * config_.mesh_cols + n);  // (m, n)

            // Path: hop right from col 0 to col n
            for (uint16_t col = 0; col <= n; col++) {
                movement.path.push_back(static_cast<uint8_t>(m * config_.mesh_cols + col));
            }
            chain.a_movements.push_back(movement);
        }

        // B tiles hop from top (row 0) to destination row
        for (size_t i = 0; i < chain.b_tiles.size(); i++) {
            TileDependencyChain::TileMovement movement;
            movement.tile = chain.b_tiles[i];
            movement.start_l3 = static_cast<uint8_t>(n);  // (0, n)
            movement.end_l3 = static_cast<uint8_t>(m * config_.mesh_cols + n);  // (m, n)

            // Path: hop down from row 0 to row m
            for (uint16_t row = 0; row <= m; row++) {
                movement.path.push_back(static_cast<uint8_t>(row * config_.mesh_cols + n));
            }
            chain.b_movements.push_back(movement);
        }
    }
};

} // namespace sw::kpu::dataflow

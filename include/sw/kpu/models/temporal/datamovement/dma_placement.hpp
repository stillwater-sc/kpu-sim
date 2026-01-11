// ============================================================================
// include/sw/kpu/models/temporal/datamovement/dma_placement.hpp
// DMA Placement Configuration for NoC Edge Integration
//
// This header defines the placement of DMA engines on NoC edges:
// - West edge DMAs for row ingress (A tiles flowing East)
// - North edge DMAs for column ingress (B tiles flowing South)
// ============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/models/interfaces/memory_controller_interface.hpp>
#include <sw/kpu/models/interfaces/noc_interface.hpp>

namespace sw::kpu {

// ============================================================================
// DMA Placement Configuration
// ============================================================================

/// Single DMA engine placement on a NoC edge
struct DMAPlacement {
    /// Edge of the mesh where DMA is placed
    enum class Edge : uint8_t {
        WEST,   // Column 0 routers (row ingress)
        NORTH,  // Row 0 routers (column ingress)
        EAST,   // Last column routers
        SOUTH   // Last row routers
    };

    Edge edge;                      // Which edge
    uint8_t position;               // Position along edge (row for W/E, col for N/S)
    uint8_t router_id;              // Computed router ID
    uint8_t memory_controller_id;   // Which memory controller to use

    /// Compute router ID from edge and position
    static uint8_t compute_router_id(Edge edge, uint8_t pos,
                                      uint8_t mesh_rows, uint8_t mesh_cols) {
        switch (edge) {
            case Edge::WEST:
                return pos * mesh_cols;                     // Column 0
            case Edge::EAST:
                return pos * mesh_cols + (mesh_cols - 1);   // Last column
            case Edge::NORTH:
                return pos;                                  // Row 0
            case Edge::SOUTH:
                return (mesh_rows - 1) * mesh_cols + pos;   // Last row
            default:
                return 0;
        }
    }

    /// Get string name for edge
    static const char* edge_name(Edge edge) {
        switch (edge) {
            case Edge::WEST:  return "WEST";
            case Edge::NORTH: return "NORTH";
            case Edge::EAST:  return "EAST";
            case Edge::SOUTH: return "SOUTH";
            default: return "UNKNOWN";
        }
    }
};

/// System-wide DMA placement configuration
struct DMASystemConfig {
    /// West edge DMAs (one per row, for A tile loading)
    std::vector<DMAPlacement> west_edge_dmas;

    /// North edge DMAs (one per column, for B tile loading)
    std::vector<DMAPlacement> north_edge_dmas;

    /// East edge DMAs (optional, for output draining)
    std::vector<DMAPlacement> east_edge_dmas;

    /// South edge DMAs (optional, for output draining)
    std::vector<DMAPlacement> south_edge_dmas;

    /// Create standard configuration for a mesh
    ///
    /// Creates one DMA per west edge row and one per north edge column.
    /// Memory controllers are distributed round-robin.
    ///
    /// @param mesh_rows Number of rows in the NoC mesh
    /// @param mesh_cols Number of columns in the NoC mesh
    /// @param num_memory_controllers Number of memory controllers available
    /// @return DMASystemConfig with placements
    static DMASystemConfig create_standard(uint8_t mesh_rows,
                                            uint8_t mesh_cols,
                                            uint8_t num_memory_controllers = 1) {
        DMASystemConfig config;

        // West edge: one DMA per row
        for (uint8_t row = 0; row < mesh_rows; ++row) {
            DMAPlacement p;
            p.edge = DMAPlacement::Edge::WEST;
            p.position = row;
            p.router_id = DMAPlacement::compute_router_id(
                p.edge, row, mesh_rows, mesh_cols);
            p.memory_controller_id = row % num_memory_controllers;
            config.west_edge_dmas.push_back(p);
        }

        // North edge: one DMA per column
        for (uint8_t col = 0; col < mesh_cols; ++col) {
            DMAPlacement p;
            p.edge = DMAPlacement::Edge::NORTH;
            p.position = col;
            p.router_id = DMAPlacement::compute_router_id(
                p.edge, col, mesh_rows, mesh_cols);
            p.memory_controller_id = col % num_memory_controllers;
            config.north_edge_dmas.push_back(p);
        }

        return config;
    }

    /// Create minimal configuration (one DMA per edge direction)
    static DMASystemConfig create_minimal(uint8_t mesh_rows,
                                           uint8_t mesh_cols) {
        DMASystemConfig config;

        // Just one west DMA (position 0)
        DMAPlacement west;
        west.edge = DMAPlacement::Edge::WEST;
        west.position = 0;
        west.router_id = 0;
        west.memory_controller_id = 0;
        config.west_edge_dmas.push_back(west);

        // Just one north DMA (position 0)
        DMAPlacement north;
        north.edge = DMAPlacement::Edge::NORTH;
        north.position = 0;
        north.router_id = 0;
        north.memory_controller_id = 0;
        config.north_edge_dmas.push_back(north);

        return config;
    }

    /// Get total number of DMAs
    size_t total_dmas() const {
        return west_edge_dmas.size() + north_edge_dmas.size() +
               east_edge_dmas.size() + south_edge_dmas.size();
    }

    /// Get all placements as a flat vector
    std::vector<DMAPlacement> all_placements() const {
        std::vector<DMAPlacement> all;
        all.insert(all.end(), west_edge_dmas.begin(), west_edge_dmas.end());
        all.insert(all.end(), north_edge_dmas.begin(), north_edge_dmas.end());
        all.insert(all.end(), east_edge_dmas.begin(), east_edge_dmas.end());
        all.insert(all.end(), south_edge_dmas.begin(), south_edge_dmas.end());
        return all;
    }
};

// ============================================================================
// DMA System (Collection of DMA Engines)
// ============================================================================

/// Container for all DMA engines in the system
struct DMASystem {
    /// West edge DMA engines (for A tile loading)
    std::vector<std::unique_ptr<CycleAccurateDMAEngine>> west_dmas;

    /// North edge DMA engines (for B tile loading)
    std::vector<std::unique_ptr<CycleAccurateDMAEngine>> north_dmas;

    /// East edge DMA engines (optional)
    std::vector<std::unique_ptr<CycleAccurateDMAEngine>> east_dmas;

    /// South edge DMA engines (optional)
    std::vector<std::unique_ptr<CycleAccurateDMAEngine>> south_dmas;

    /// Tick all DMA engines
    void tick() {
        for (auto& dma : west_dmas) dma->tick();
        for (auto& dma : north_dmas) dma->tick();
        for (auto& dma : east_dmas) dma->tick();
        for (auto& dma : south_dmas) dma->tick();
    }

    /// Drain all DMA engines
    void drain() {
        for (auto& dma : west_dmas) dma->drain();
        for (auto& dma : north_dmas) dma->drain();
        for (auto& dma : east_dmas) dma->drain();
        for (auto& dma : south_dmas) dma->drain();
    }

    /// Reset all DMA engines
    void reset() {
        for (auto& dma : west_dmas) dma->reset();
        for (auto& dma : north_dmas) dma->reset();
        for (auto& dma : east_dmas) dma->reset();
        for (auto& dma : south_dmas) dma->reset();
    }

    /// Set cycle for all engines
    void set_cycle(uint64_t cycle) {
        for (auto& dma : west_dmas) dma->set_cycle(cycle);
        for (auto& dma : north_dmas) dma->set_cycle(cycle);
        for (auto& dma : east_dmas) dma->set_cycle(cycle);
        for (auto& dma : south_dmas) dma->set_cycle(cycle);
    }

    /// Check if any DMA has pending transfers
    bool has_pending() const {
        for (const auto& dma : west_dmas) if (dma->has_pending()) return true;
        for (const auto& dma : north_dmas) if (dma->has_pending()) return true;
        for (const auto& dma : east_dmas) if (dma->has_pending()) return true;
        for (const auto& dma : south_dmas) if (dma->has_pending()) return true;
        return false;
    }

    /// Get total number of engines
    size_t size() const {
        return west_dmas.size() + north_dmas.size() +
               east_dmas.size() + south_dmas.size();
    }
};

// ============================================================================
// DMA System Factory
// ============================================================================

/// Create a DMA system based on placement configuration
///
/// @param placement DMA placement configuration
/// @param engine_config Configuration for each DMA engine
/// @param memory_controllers Vector of memory controller pointers
/// @param noc NoC pointer for tile injection
/// @return DMASystem with all engines created and bound
inline DMASystem create_dma_system(
    const DMASystemConfig& placement,
    const CycleAccurateDMAConfig& engine_config,
    const std::vector<IMemoryController*>& memory_controllers,
    noc::INoC* noc)
{
    DMASystem system;

    // Create west edge DMAs
    for (const auto& p : placement.west_edge_dmas) {
        auto dma = std::make_unique<CycleAccurateDMAEngine>(engine_config);

        // Bind to memory controller
        if (p.memory_controller_id < memory_controllers.size()) {
            dma->bind_memory_controller(memory_controllers[p.memory_controller_id]);
        }

        // Bind to NoC
        if (noc) {
            dma->bind_noc(noc, p.router_id);
        }

        system.west_dmas.push_back(std::move(dma));
    }

    // Create north edge DMAs
    for (const auto& p : placement.north_edge_dmas) {
        auto dma = std::make_unique<CycleAccurateDMAEngine>(engine_config);

        if (p.memory_controller_id < memory_controllers.size()) {
            dma->bind_memory_controller(memory_controllers[p.memory_controller_id]);
        }

        if (noc) {
            dma->bind_noc(noc, p.router_id);
        }

        system.north_dmas.push_back(std::move(dma));
    }

    // Create east edge DMAs (if configured)
    for (const auto& p : placement.east_edge_dmas) {
        auto dma = std::make_unique<CycleAccurateDMAEngine>(engine_config);

        if (p.memory_controller_id < memory_controllers.size()) {
            dma->bind_memory_controller(memory_controllers[p.memory_controller_id]);
        }

        if (noc) {
            dma->bind_noc(noc, p.router_id);
        }

        system.east_dmas.push_back(std::move(dma));
    }

    // Create south edge DMAs (if configured)
    for (const auto& p : placement.south_edge_dmas) {
        auto dma = std::make_unique<CycleAccurateDMAEngine>(engine_config);

        if (p.memory_controller_id < memory_controllers.size()) {
            dma->bind_memory_controller(memory_controllers[p.memory_controller_id]);
        }

        if (noc) {
            dma->bind_noc(noc, p.router_id);
        }

        system.south_dmas.push_back(std::move(dma));
    }

    return system;
}

} // namespace sw::kpu

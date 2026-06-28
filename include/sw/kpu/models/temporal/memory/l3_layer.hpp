#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>
#include <sw/kpu/models/temporal/memory/l3_tile.hpp>
#include <sw/kpu/models/temporal/datamovement/block_mover.hpp>
#include <sw/kpu/models/temporal/datamovement/l3_interconnect.hpp>

namespace sw::kpu {

// Forward declaration: L2Bank is only referenced by reference (process_block_movers).
class L2Bank;

// ============================================================================
// L3 Layer configuration
//
// The L3 layer is the whole SoC-wide L3 resource: a (possibly non-uniform)
// collection of L3Tile elements, the per-tile BlockMovers that push tiles down
// into the L2 layer, and the L3 interconnect/NoC. Element micro-architecture
// (capacity, banks, ports) belongs to the element; the *layer* describes how
// many elements there are, their grouping, and the layer-level resources.
//
// Non-uniformity (heterogeneous compute fabrics) is expressed with the
// `group -> element-config -> multiplicity` shape: each named group binds one
// element spec to a count of identical tiles, and a layer holds many groups.
// ============================================================================

/// Per-element (L3Tile) micro-architecture spec at the temporal-model level.
struct L3TileSpec {
    /// Capacity per tile in KB.
    Size capacity_kb = 128;
};

/// A named group of identical L3Tiles: element-config bound to a multiplicity.
struct L3TileGroup {
    std::string name;          ///< Symbolic group name (e.g. "default", "hbw").
    L3TileSpec tile;           ///< Element configuration shared by the group.
    size_t multiplicity = 1;   ///< Number of identical tiles in the group.
};

/// Configuration for an L3Layer aggregate.
struct L3LayerConfig {
    /// Tile groups: group -> element-config -> multiplicity (supports non-uniform layers).
    std::vector<L3TileGroup> tile_groups;

    /// Number of BlockMovers the layer owns (round-robin associated to tiles).
    /// Target micro-architecture is one BlockMover per L3Tile.
    size_t block_mover_count = 0;

    /// Construct and own an L3Interconnect instance (ownership seam; routing
    /// integration is a separate concern from ownership).
    bool enable_interconnect = false;

    /// BlockMover timing parameters.
    double block_mover_clock_ghz = 1.0;
    double block_mover_bandwidth_gb_s = 100.0;

    /// Total number of tiles across all groups.
    size_t total_tiles() const {
        size_t n = 0;
        for (const auto& g : tile_groups) {
            n += g.multiplicity;
        }
        return n;
    }

    /// Convenience constructor for a uniform layer: `num_tiles` identical tiles
    /// of `capacity_kb`, with `block_mover_count` movers. Used to bridge the
    /// legacy flat configuration until per-group config is wired through.
    static L3LayerConfig uniform(size_t num_tiles, Size capacity_kb,
                                 size_t block_mover_count = 0) {
        L3LayerConfig cfg;
        if (num_tiles > 0) {
            cfg.tile_groups.push_back(L3TileGroup{"default", L3TileSpec{capacity_kb}, num_tiles});
        }
        cfg.block_mover_count = block_mover_count;
        return cfg;
    }
};

// ============================================================================
// L3 Layer aggregate
// ============================================================================

/// The whole SoC-wide L3 resource. Owns the L3Tile elements, the BlockMovers
/// attached to them, and the (optional) L3 interconnect.
class KPU_API L3Layer {
public:
    explicit L3Layer(const L3LayerConfig& config = {});
    ~L3Layer();

    // Non-copyable (owns a unique_ptr); movable.
    L3Layer(const L3Layer&) = delete;
    L3Layer& operator=(const L3Layer&) = delete;
    L3Layer(L3Layer&&) noexcept;
    L3Layer& operator=(L3Layer&&) noexcept;

    // --- L3Tile elements -----------------------------------------------------
    size_t tile_count() const { return tiles_.size(); }
    L3Tile& tile(size_t index);
    const L3Tile& tile(size_t index) const;

    /// Direct access to the tile collection, for components whose APIs take a
    /// `std::vector<L3Tile>&` (e.g. the DMA engine and BlockMover).
    std::vector<L3Tile>& tiles() { return tiles_; }
    const std::vector<L3Tile>& tiles() const { return tiles_; }

    // --- BlockMovers (owned by the layer) ------------------------------------
    size_t block_mover_count() const { return block_movers_.size(); }
    BlockMover& block_mover(size_t index);
    const BlockMover& block_mover(size_t index) const;
    std::vector<BlockMover>& block_movers() { return block_movers_; }

    /// Drive all owned BlockMovers one step, pushing tiles down into the L2 layer.
    void process_block_movers(std::vector<L2Bank>& l2_banks);
    /// Set the simulation cycle on all owned BlockMovers.
    void set_block_mover_cycle(Cycle cycle);
    /// True if any owned BlockMover still has work in flight.
    bool any_block_mover_busy() const;

    // --- Interconnect (ownership seam; may be absent) ------------------------
    bool has_interconnect() const { return interconnect_ != nullptr; }
    L3Interconnect& interconnect();
    const L3Interconnect& interconnect() const;

    // --- Lifecycle -----------------------------------------------------------
    void reset();

    const L3LayerConfig& config() const { return config_; }

private:
    L3LayerConfig config_;
    std::vector<L3Tile> tiles_;
    std::vector<BlockMover> block_movers_;
    std::unique_ptr<L3Interconnect> interconnect_;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

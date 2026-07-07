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
#include <sw/kpu/models/temporal/compute/cu_layer.hpp>
#include <sw/kpu/models/temporal/compute/vu_layer.hpp>
#include <sw/kpu/models/temporal/datamovement/streamer.hpp>

namespace sw::kpu {

// Forward declaration: 
class ComputeFabric;

// ============================================================================
// CU Layer configuration
//
// The CU layer is the whole SoC-wide Compute Unit (CU) resource: a (possibly non-uniform)
// collection of ComputTile elements, the per-tile Streamers that create streams
// that are pushed into the ComputeFabric, or receive streams out of the ComputeFabric.
//
// Non-uniformity (heterogeneous compute tiles) is expressed with the
// `group -> element-config -> multiplicity` shape: each named group binds one
// element spec to a count of identical tiles, and a layer holds many groups.
// ============================================================================

/// Processing element spec
struct PESpec {
    /// how do we specify the capabilities of a PE?
};

/// Per-element (CUTile) micro-architecture spec at the temporal-model level.
struct CUTileSpec {
    Size rows = 16;
    Size cols = 16;
    Size vus = 1;
    bool systolic_array = false;  // is the ComputeFabric a pure systolic array or not
    PESpec pe;
};

/// A named group of identical L3Tiles: element-config bound to a multiplicity.
struct CUTileGroup {
    std::string name;          ///< Symbolic group name (e.g. "default", "hbw").
    CUTileSpec tile;           ///< Element configuration shared by the group.
    size_t multiplicity = 1;   ///< Number of identical tiles in the group.
};

/// Configuration for an CULayer aggregate.
///
/// Canonical way to describe the compute tiles:
///  - populate `tile_groups` with named
///    `group -> element-config -> multiplicity` entries.
struct CULayerConfig {
    /// Tile groups: group -> element-config -> multiplicity (supports non-uniform layers).
    std::vector<CUTileGroup> tile_groups;

    /// Number of Streamers the compute layer owns 
    /// Target micro-architecture is four Streamers per CUTile, one for each NEWS direction.
    /// Calculated automatically in CULayer constructor; can be overridden for testing.
    size_t streamer_count = 0;

    /// Streamer attributes.
    float streamer_clock_ghz = 1.0;
    int   streamer_buswidth_bits = 64;
    // Stream bandwidth in GB/s (per streamer, not aggregate) = streamer_buswidth_bits * streamer_clock_ghz / 8

    /// Total number of tiles: sum of group multiplicities, or the uniform count.
    size_t total_tiles() const {
        size_t n = 0;
        for (const auto& g : tile_groups) {
            n += g.multiplicity;
        }
        return n;
    }
};

// ============================================================================
// CU Layer aggregate
// ============================================================================

/// The whole SoC-wide CU resource. Owns the CUTile elements, the Streamers
/// attached to them, and any SFU vector units for fused operations.
class KPU_API CULayer {
public:
    explicit CULayer(const CULayerConfig& config = {});
    ~CULayer();

    // Non-copyable (owns a unique_ptr); movable.
    CULayer(const CULayer&) = delete;
    CULayer& operator=(const CULayer&) = delete;
    CULayer(CULayer&&) noexcept;
    CULayer& operator=(CULayer&&) noexcept;

    // --- L3Tile elements -----------------------------------------------------
    size_t tile_count() const { return tiles_.size(); }
    L3Tile& tile(size_t index);
    const L3Tile& tile(size_t index) const;

    /// Direct access to the tile collection, for components whose APIs take a
    /// `std::vector<CUTileGroup>&` (e.g. the ResourceManager).
    std::vector<CUTileGroup>& tiles() { return tiles_; }
    const std::vector<CUTileGroup>& tiles() const { return tiles_; }

    // --- Streamers (owned by the layer) ------------------------------------
    size_t streamer_count() const { return streamers_.size(); }
    Streamer& streamer(size_t index);
    const Streamer& streamer(size_t index) const;
    std::vector<Streamer>& streamers() { return streamers_; }
    const std::vector<Streamer>& streamers() const { return streamers_; }

    /// Drive all owned Streamers one step, extracting rows and columns from the 
    /// L2 layer, and creating a data stream for the ComputeFabric.
    void process_streamers(std::vector<Streamer>& streamers);
    /// Set the simulation cycle on all owned Streamers.
    void set_streamer_cycle(Cycle cycle);
    /// True if any owned Streamers still has work in flight.
    bool any_streamer_busy() const;

    // --- Vector Units (ownership seam; may be absent) ------------------------
    //bool has_vecture_unit() const { return vector_unit_ != nullptr; }
    //VectorUnit& vector_unit();
    //const VectorUnit& vector_unit() const;

    // --- Lifecycle -----------------------------------------------------------
    void reset();

    const CULayerConfig& config() const { return config_; }

private:
    CULayerConfig config_;
    std::vector<CUTileGroup> tiles_;
    std::vector<Streamer> streamers_;
//    std::vector<VUTile> vus_;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

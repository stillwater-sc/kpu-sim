#pragma once

#include <vector>
#include <string>
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

namespace sw::kpu {

// Forward declaration: 

enum class SFUType {
	NONE,
	TANH,
	SIGMOID,
	RELU,
	LEAKY_RELU,
	SOFTMAX
};

/// Vector Unit Tile specification
struct VUTileSpec {
    /// Vector Width
	Size vector_width = 16; // Number of elements processed in parallel
	std::vector<SFUType> supported_sfu; // List of supported SFU types for this tile
};

/// A named group of identical VUTiles: element-config bound to a multiplicity.
struct VUTileGroup {
    std::string name;          ///< Symbolic group name (e.g. "default", "sfu", "bias", "fusion").
    VUTileSpec tile;           ///< Element configuration shared by the group.
    size_t multiplicity = 1;   ///< Number of identical tiles in the group.
};

/// Vector Unit tiles in the configuration for an VULayer aggregate.
///
/// **Canonical Way** to describe the tiles:
///  -  populate `tile_groups` with named
///    `group -> element-config -> multiplicity` entries.
struct VULayerConfig {
    /// Tile groups: group -> element-config -> multiplicity (supports non-uniform layers).
    std::vector<VUTileGroup> tile_groups;

    /// Total number of tiles: sum of group multiplicities
    size_t total_tiles() const {
        size_t n = 0;
        for (const auto& g : tile_groups) {
            n += g.multiplicity;
        }
        return n;
    }
};

// ============================================================================
// Vector Unit Layer aggregate
// ============================================================================

/// The whole SoC-wide VU resource. Owns just the Vector Units in all the  
/// Compute Tiles.
class KPU_API VULayer {
public:
    explicit VULayer(const VULayerConfig& config = {});
    ~VULayer();

    // Non-copyable (owns a unique_ptr); movable.
    VULayer(const VULayer&) = delete;
    VULayer& operator=(const VULayer&) = delete;
    VULayer(VULayer&&) noexcept;
    VULayer& operator=(VULayer&&) noexcept;

private:

};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

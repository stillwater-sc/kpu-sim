#pragma once

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
#include <sw/kpu/models/temporal/memory/l1_buffer.hpp>

namespace sw::kpu {

// ============================================================================
// L1 Layer configuration
//
// The L1 layer is the whole SoC-wide L1 stream-buffer resource that feeds the
// compute fabric: a (possibly non-uniform) collection of L1Buffer elements.
//
// IMPORTANT: L1Layer (like L2Layer / L3Layer) is a *conceptual resource owner*
// for monitoring/debug — it reflects the physical distribution of storage and
// lets us interrogate the L1 resources (count, capacity, per-element state). It
// is NOT a ResourceManager / dataflow API: the actual data movement and credit
// flow through the hierarchy is driven by the distributed CSP engines (DMA,
// BlockMover, Streamer). The Layer observes; the Streamer drives L2->L1.
//
// Non-uniformity (heterogeneous compute fabrics) is expressed with the
// `group -> element-config -> multiplicity` shape.
// ============================================================================

/// Per-element (L1Buffer) micro-architecture spec at the temporal-model level.
struct L1BufferSpec {
    /// Capacity per buffer in KB.
    Size capacity_kb = 32;
};

/// A named group of identical L1Buffers: element-config bound to a multiplicity.
struct L1BufferGroup {
    std::string name;          ///< Symbolic group name (e.g. "default").
    L1BufferSpec buffer;       ///< Element configuration shared by the group.
    size_t multiplicity = 1;   ///< Number of identical buffers in the group.
};

/// Configuration for an L1Layer aggregate.
///
/// Two ways to describe the buffers:
///  - **Canonical (non-uniform):** populate `buffer_groups` with named
///    `group -> element-config -> multiplicity` entries.
///  - **Uniform convenience:** leave `buffer_groups` empty and set the scalar
///    `num_buffers` / `capacity_kb` fields; the layer is then built as
///    `num_buffers` identical buffers of `capacity_kb`.
/// When `buffer_groups` is non-empty it takes precedence over the scalar fields.
///
/// Note: the L1 buffer count is typically *derived* from the processor-array
/// topology (see processor_array_topology.hpp). That derivation is a caller
/// responsibility — set `num_buffers` (or `buffer_groups`) accordingly.
struct L1LayerConfig {
    /// Buffer groups: group -> element-config -> multiplicity (non-uniform layers).
    std::vector<L1BufferGroup> buffer_groups;

    /// Uniform convenience: number of identical buffers (used iff `buffer_groups` empty).
    size_t num_buffers = 0;
    /// Uniform convenience: per-buffer capacity in KB (used iff `buffer_groups` empty).
    Size capacity_kb = 32;

    /// Total number of buffers: sum of group multiplicities, or the uniform count.
    size_t total_buffers() const {
        if (!buffer_groups.empty()) {
            size_t n = 0;
            for (const auto& g : buffer_groups) {
                n += g.multiplicity;
            }
            return n;
        }
        return num_buffers;
    }

    /// Convenience factory for a uniform layer.
    static L1LayerConfig uniform(size_t num_buffers, Size capacity_kb) {
        L1LayerConfig cfg;
        cfg.num_buffers = num_buffers;
        cfg.capacity_kb = capacity_kb;
        return cfg;
    }
};

// ============================================================================
// L1 Layer aggregate
// ============================================================================

/// The whole SoC-wide L1 stream-buffer resource. Owns the L1Buffer elements.
/// Ownership / monitoring structure only — not a dataflow API (see header note).
class KPU_API L1Layer {
public:
    explicit L1Layer(const L1LayerConfig& config = {});

    // --- L1Buffer elements ---------------------------------------------------
    size_t buffer_count() const { return buffers_.size(); }
    L1Buffer& buffer(size_t index);
    const L1Buffer& buffer(size_t index) const;

    /// Direct access to the buffer collection, for components whose APIs take a
    /// `std::vector<L1Buffer>&` (e.g. the Streamer and ComputeFabric).
    std::vector<L1Buffer>& buffers() { return buffers_; }
    const std::vector<L1Buffer>& buffers() const { return buffers_; }

    // --- Monitoring / inspection (not a dataflow API) ------------------------
    /// Aggregate capacity of the layer in bytes.
    Size total_capacity_bytes() const;

    // --- Lifecycle -----------------------------------------------------------
    void reset();

    const L1LayerConfig& config() const { return config_; }

private:
    L1LayerConfig config_;
    std::vector<L1Buffer> buffers_;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

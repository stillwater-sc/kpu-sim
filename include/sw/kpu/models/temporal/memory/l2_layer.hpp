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
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>

namespace sw::kpu {

// ============================================================================
// L2 Layer configuration
//
// The L2 layer is the whole SoC-wide L2 resource that sits between the L3 layer
// and the stream buffers feeding the compute fabric: a (possibly non-uniform)
// collection of L2Bank elements plus the layer's access ports. Element
// micro-architecture (capacity, ports) belongs to the element; the *layer*
// describes how many elements there are and their grouping.
//
// Non-uniformity (heterogeneous compute fabrics) is expressed with the
// `group -> element-config -> multiplicity` shape: each named group binds one
// element spec to a count of identical banks, and a layer holds many groups.
//
// (An L2 interconnect/topology is intentionally out of scope for now; see the
//  memory-layer restructuring plan.)
// ============================================================================

/// Per-element (L2Bank) micro-architecture spec at the temporal-model level.
struct L2BankSpec {
    /// Capacity per bank in KB.
    Size capacity_kb = 64;
};

/// A named group of identical L2Banks: element-config bound to a multiplicity.
struct L2BankGroup {
    std::string name;          ///< Symbolic group name (e.g. "default").
    L2BankSpec bank;           ///< Element configuration shared by the group.
    size_t multiplicity = 1;   ///< Number of identical banks in the group.
};

/// Configuration for an L2Layer aggregate.
///
/// Two ways to describe the banks:
///  - **Canonical (non-uniform):** populate `bank_groups` with named
///    `group -> element-config -> multiplicity` entries.
///  - **Uniform convenience:** leave `bank_groups` empty and set the scalar
///    `num_banks` / `capacity_kb` fields; the layer is then built as `num_banks`
///    identical banks of `capacity_kb`.
/// When `bank_groups` is non-empty it takes precedence over the scalar fields.
struct L2LayerConfig {
    /// Bank groups: group -> element-config -> multiplicity (supports non-uniform layers).
    std::vector<L2BankGroup> bank_groups;

    /// Uniform convenience: number of identical banks (used iff `bank_groups` empty).
    size_t num_banks = 0;
    /// Uniform convenience: per-bank capacity in KB (used iff `bank_groups` empty).
    Size capacity_kb = 64;

    /// Total number of banks: sum of group multiplicities, or the uniform count.
    size_t total_banks() const {
        if (!bank_groups.empty()) {
            size_t n = 0;
            for (const auto& g : bank_groups) {
                n += g.multiplicity;
            }
            return n;
        }
        return num_banks;
    }

    /// Convenience factory for a uniform layer.
    static L2LayerConfig uniform(size_t num_banks, Size capacity_kb) {
        L2LayerConfig cfg;
        cfg.num_banks = num_banks;
        cfg.capacity_kb = capacity_kb;
        return cfg;
    }
};

// ============================================================================
// L2 Layer aggregate
// ============================================================================

/// The whole SoC-wide L2 resource. Owns the L2Bank elements.
class KPU_API L2Layer {
public:
    explicit L2Layer(const L2LayerConfig& config = {});

    // --- L2Bank elements -----------------------------------------------------
    size_t bank_count() const { return banks_.size(); }
    L2Bank& bank(size_t index);
    const L2Bank& bank(size_t index) const;

    /// Direct access to the bank collection, for components whose APIs take a
    /// `std::vector<L2Bank>&` (e.g. the BlockMover and Streamer).
    std::vector<L2Bank>& banks() { return banks_; }
    const std::vector<L2Bank>& banks() const { return banks_; }

    // --- Lifecycle -----------------------------------------------------------
    void reset();

    const L2LayerConfig& config() const { return config_; }

private:
    L2LayerConfig config_;
    std::vector<L2Bank> banks_;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

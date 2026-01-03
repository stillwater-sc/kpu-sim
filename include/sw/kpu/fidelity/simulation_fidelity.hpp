// ============================================================================
// include/sw/kpu/fidelity/simulation_fidelity.hpp
// Core types for multi-fidelity simulation framework
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <string_view>

namespace sw::kpu {

// ============================================================================
// Simulation Fidelity Levels
// ============================================================================

/// Determines the accuracy/speed tradeoff for component simulation
enum class SimulationFidelity : uint8_t {
    /// Functional correctness only, instant or fixed latency
    /// - No state machine, no queuing, no contention
    /// - ~100-1000x faster than cycle-accurate
    /// - Use for: software bring-up, functional verification, CI/CD
    BEHAVIORAL,

    /// Transaction-level timing with statistical models
    /// - Queue-based contention, aggregate latencies
    /// - ~10-100x faster than cycle-accurate
    /// - Use for: architecture exploration, bottleneck identification
    TRANSACTIONAL,

    /// Full protocol state machine with per-cycle timing
    /// - Complete timing model, all protocol constraints
    /// - Baseline speed (1x)
    /// - Use for: performance analysis, timing validation
    CYCLE_ACCURATE
};

/// Convert fidelity level to string
constexpr std::string_view to_string(SimulationFidelity fidelity) {
    switch (fidelity) {
        case SimulationFidelity::BEHAVIORAL:    return "BEHAVIORAL";
        case SimulationFidelity::TRANSACTIONAL: return "TRANSACTIONAL";
        case SimulationFidelity::CYCLE_ACCURATE: return "CYCLE_ACCURATE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Verification Levels
// ============================================================================

/// Runtime verification/checking level for components
enum class VerificationLevel : uint8_t {
    /// No runtime checks (maximum speed)
    NONE,

    /// Basic sanity checks via assert()
    ASSERTIONS,

    /// Full formal invariant checking with violation reporting
    INVARIANTS,

    /// Protocol compliance checking (e.g., JEDEC timing rules)
    PROTOCOL
};

constexpr std::string_view to_string(VerificationLevel level) {
    switch (level) {
        case VerificationLevel::NONE:       return "NONE";
        case VerificationLevel::ASSERTIONS: return "ASSERTIONS";
        case VerificationLevel::INVARIANTS: return "INVARIANTS";
        case VerificationLevel::PROTOCOL:   return "PROTOCOL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Memory Technology Standards
// ============================================================================

/// JEDEC memory technology standards
enum class MemoryTechnology : uint8_t {
    /// Ideal/configurable memory (user-provided timing)
    IDEAL,

    /// JEDEC LPDDR5 (mobile, edge AI)
    LPDDR5,

    /// JEDEC LPDDR5X (high-performance mobile)
    LPDDR5X,

    /// JEDEC DDR5 (servers, workstations)
    DDR5,

    /// JEDEC HBM3 (datacenter AI accelerators)
    HBM3,

    /// JEDEC HBM3E (next-generation AI)
    HBM3E,

    /// JEDEC GDDR6 (discrete graphics)
    GDDR6,

    /// JEDEC GDDR7 (next-generation graphics)
    GDDR7
};

constexpr std::string_view to_string(MemoryTechnology tech) {
    switch (tech) {
        case MemoryTechnology::IDEAL:   return "IDEAL";
        case MemoryTechnology::LPDDR5:  return "LPDDR5";
        case MemoryTechnology::LPDDR5X: return "LPDDR5X";
        case MemoryTechnology::DDR5:    return "DDR5";
        case MemoryTechnology::HBM3:    return "HBM3";
        case MemoryTechnology::HBM3E:   return "HBM3E";
        case MemoryTechnology::GDDR6:   return "GDDR6";
        case MemoryTechnology::GDDR7:   return "GDDR7";
        default: return "UNKNOWN";
    }
}

/// Check if technology is a low-power variant
constexpr bool is_low_power(MemoryTechnology tech) {
    return tech == MemoryTechnology::LPDDR5 || tech == MemoryTechnology::LPDDR5X;
}

/// Check if technology is high-bandwidth memory
constexpr bool is_hbm(MemoryTechnology tech) {
    return tech == MemoryTechnology::HBM3 || tech == MemoryTechnology::HBM3E;
}

/// Check if technology is graphics memory
constexpr bool is_gddr(MemoryTechnology tech) {
    return tech == MemoryTechnology::GDDR6 || tech == MemoryTechnology::GDDR7;
}

// ============================================================================
// Compute Technology
// ============================================================================

/// Compute unit technology/datatype configuration
enum class ComputeTechnology : uint8_t {
    /// Ideal compute (configurable throughput)
    IDEAL,

    /// INT8 systolic array (inference)
    INT8_SYSTOLIC,

    /// FP16 systolic array (training/inference)
    FP16_SYSTOLIC,

    /// BF16 systolic array (training)
    BF16_SYSTOLIC,

    /// FP32 SIMD vector unit
    FP32_SIMD,

    /// Mixed precision with dynamic selection
    MIXED_PRECISION
};

constexpr std::string_view to_string(ComputeTechnology tech) {
    switch (tech) {
        case ComputeTechnology::IDEAL:           return "IDEAL";
        case ComputeTechnology::INT8_SYSTOLIC:   return "INT8_SYSTOLIC";
        case ComputeTechnology::FP16_SYSTOLIC:   return "FP16_SYSTOLIC";
        case ComputeTechnology::BF16_SYSTOLIC:   return "BF16_SYSTOLIC";
        case ComputeTechnology::FP32_SIMD:       return "FP32_SIMD";
        case ComputeTechnology::MIXED_PRECISION: return "MIXED_PRECISION";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Interconnect Technology
// ============================================================================

/// Network-on-Chip topology and technology
enum class InterconnectTechnology : uint8_t {
    /// Ideal zero-latency crossbar
    IDEAL,

    /// 2D mesh network
    MESH_2D,

    /// 2D torus network
    TORUS_2D,

    /// Hierarchical (ring of meshes, etc.)
    HIERARCHICAL,

    /// Custom/configurable
    CUSTOM
};

constexpr std::string_view to_string(InterconnectTechnology tech) {
    switch (tech) {
        case InterconnectTechnology::IDEAL:        return "IDEAL";
        case InterconnectTechnology::MESH_2D:      return "MESH_2D";
        case InterconnectTechnology::TORUS_2D:     return "TORUS_2D";
        case InterconnectTechnology::HIERARCHICAL: return "HIERARCHICAL";
        case InterconnectTechnology::CUSTOM:       return "CUSTOM";
        default: return "UNKNOWN";
    }
}

} // namespace sw::kpu

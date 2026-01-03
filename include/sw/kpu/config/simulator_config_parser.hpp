// ============================================================================
// include/sw/kpu/config/simulator_config_parser.hpp
// Configuration parser for multi-fidelity simulation
//
// Parses JSON configuration files to configure simulator components
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <stdexcept>

namespace sw::kpu::config {

// ============================================================================
// Configuration Parse Errors
// ============================================================================

class ConfigParseError : public std::runtime_error {
public:
    explicit ConfigParseError(const std::string& msg)
        : std::runtime_error(msg) {}
};

// ============================================================================
// String to Enum Conversion
// ============================================================================

/// Parse simulation fidelity from string
inline SimulationFidelity parse_fidelity(const std::string& str) {
    if (str == "BEHAVIORAL" || str == "behavioral") {
        return SimulationFidelity::BEHAVIORAL;
    } else if (str == "TRANSACTIONAL" || str == "transactional") {
        return SimulationFidelity::TRANSACTIONAL;
    } else if (str == "CYCLE_ACCURATE" || str == "cycle_accurate") {
        return SimulationFidelity::CYCLE_ACCURATE;
    }
    throw ConfigParseError("Unknown fidelity level: " + str);
}

/// Parse memory technology from string
inline MemoryTechnology parse_memory_technology(const std::string& str) {
    if (str == "IDEAL" || str == "ideal") return MemoryTechnology::IDEAL;
    if (str == "LPDDR5" || str == "lpddr5") return MemoryTechnology::LPDDR5;
    if (str == "LPDDR5X" || str == "lpddr5x") return MemoryTechnology::LPDDR5X;
    if (str == "DDR5" || str == "ddr5") return MemoryTechnology::DDR5;
    if (str == "HBM3" || str == "hbm3") return MemoryTechnology::HBM3;
    if (str == "HBM3E" || str == "hbm3e") return MemoryTechnology::HBM3E;
    if (str == "GDDR6" || str == "gddr6") return MemoryTechnology::GDDR6;
    if (str == "GDDR7" || str == "gddr7") return MemoryTechnology::GDDR7;
    throw ConfigParseError("Unknown memory technology: " + str);
}

/// Parse compute technology from string
inline ComputeTechnology parse_compute_technology(const std::string& str) {
    if (str == "IDEAL" || str == "ideal") return ComputeTechnology::IDEAL;
    if (str == "INT8_SYSTOLIC" || str == "int8_systolic") return ComputeTechnology::INT8_SYSTOLIC;
    if (str == "FP16_SYSTOLIC" || str == "fp16_systolic") return ComputeTechnology::FP16_SYSTOLIC;
    if (str == "BF16_SYSTOLIC" || str == "bf16_systolic") return ComputeTechnology::BF16_SYSTOLIC;
    if (str == "FP32_SIMD" || str == "fp32_simd") return ComputeTechnology::FP32_SIMD;
    if (str == "MIXED_PRECISION" || str == "mixed_precision") return ComputeTechnology::MIXED_PRECISION;
    throw ConfigParseError("Unknown compute technology: " + str);
}

/// Parse interconnect technology from string
inline InterconnectTechnology parse_interconnect_technology(const std::string& str) {
    if (str == "IDEAL" || str == "ideal") return InterconnectTechnology::IDEAL;
    if (str == "MESH_2D" || str == "mesh_2d") return InterconnectTechnology::MESH_2D;
    if (str == "TORUS_2D" || str == "torus_2d") return InterconnectTechnology::TORUS_2D;
    if (str == "HIERARCHICAL" || str == "hierarchical") return InterconnectTechnology::HIERARCHICAL;
    throw ConfigParseError("Unknown interconnect technology: " + str);
}

/// Parse verification level from string
inline VerificationLevel parse_verification(const std::string& str) {
    if (str == "NONE" || str == "none") return VerificationLevel::NONE;
    if (str == "ASSERTIONS" || str == "assertions") return VerificationLevel::ASSERTIONS;
    if (str == "INVARIANTS" || str == "invariants") return VerificationLevel::INVARIANTS;
    if (str == "PROTOCOL" || str == "protocol") return VerificationLevel::PROTOCOL;
    throw ConfigParseError("Unknown verification level: " + str);
}

// ============================================================================
// JSON Parsing Helpers
// ============================================================================

/// Get optional value from JSON
template<typename T>
std::optional<T> get_optional(const nlohmann::json& j, const std::string& key) {
    if (j.contains(key) && !j[key].is_null()) {
        return j[key].get<T>();
    }
    return std::nullopt;
}

/// Get value with default from JSON
template<typename T>
T get_with_default(const nlohmann::json& j, const std::string& key, const T& default_val) {
    if (j.contains(key) && !j[key].is_null()) {
        return j[key].get<T>();
    }
    return default_val;
}

// ============================================================================
// Configuration Parser
// ============================================================================

class SimulatorConfigParser {
public:
    /// Parse configuration from JSON file
    static SimulatorConfig parse_file(const std::filesystem::path& path);

    /// Parse configuration from JSON string
    static SimulatorConfig parse_string(const std::string& json_str);

    /// Parse configuration from JSON object
    static SimulatorConfig parse_json(const nlohmann::json& j);

    /// Serialize configuration to JSON
    static nlohmann::json to_json(const SimulatorConfig& config);

    /// Write configuration to file
    static void write_file(const SimulatorConfig& config, const std::filesystem::path& path);

private:
    /// Parse simulation section
    static void parse_simulation(const nlohmann::json& j, SimulatorConfig& config);

    /// Parse memory controller section
    static MemoryControllerConfig parse_memory(const nlohmann::json& j,
                                                const SimulatorConfig& defaults);

    /// Parse DMA engine section
    static DMAEngineConfig parse_dma(const nlohmann::json& j,
                                     const SimulatorConfig& defaults);

    /// Parse L3 tile section
    static L3TileConfig parse_l3(const nlohmann::json& j,
                                 const SimulatorConfig& defaults);

    /// Parse compute fabric section
    static ComputeFabricConfig parse_compute(const nlohmann::json& j,
                                             const SimulatorConfig& defaults);

    /// Parse NoC section
    static NoCConfig parse_noc(const nlohmann::json& j,
                               const SimulatorConfig& defaults);
};

// ============================================================================
// CLI Argument Parser
// ============================================================================

/// Command-line options for fidelity configuration
struct CLIFidelityOptions {
    std::optional<SimulationFidelity> global_fidelity;
    std::optional<SimulationFidelity> memory_fidelity;
    std::optional<SimulationFidelity> compute_fidelity;
    std::optional<SimulationFidelity> noc_fidelity;
    std::optional<std::string> config_file;
    bool enable_tracing = false;
    std::optional<VerificationLevel> verification;
};

/// Parse command-line arguments for fidelity settings
///
/// Supported arguments:
///   --fidelity=<BEHAVIORAL|TRANSACTIONAL|CYCLE_ACCURATE>
///   --memory-fidelity=<...>
///   --compute-fidelity=<...>
///   --noc-fidelity=<...>
///   --config=<path>
///   --tracing
///   --verification=<NONE|ASSERTIONS|INVARIANTS|PROTOCOL>
CLIFidelityOptions parse_cli_args(int argc, char* argv[]);

/// Apply CLI options to simulator config
void apply_cli_options(SimulatorConfig& config, const CLIFidelityOptions& opts);

} // namespace sw::kpu::config

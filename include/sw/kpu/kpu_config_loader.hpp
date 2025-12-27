#pragma once

/**
 * @file kpu_config_loader.hpp
 * @brief Loader for KPU simulator configuration files (YAML and JSON)
 *
 * Supports loading KPUSimulator::Config from:
 * - YAML files (.yaml, .yml)
 * - JSON files (.json)
 *
 * The configuration format is simpler and more focused than the full
 * SystemConfig, targeting direct use with KPUSimulator.
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <optional>
#include <vector>

namespace sw::kpu {

/**
 * @brief Validation result for configuration files
 */
struct ConfigValidationResult {
    bool valid = false;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    explicit operator bool() const { return valid; }
};

/**
 * @brief Loader for KPU simulator configuration files
 *
 * Supports YAML and JSON formats with automatic detection based on
 * file extension. Provides validation and factory methods for common
 * configurations.
 */
class KPUConfigLoader {
public:
    /**
     * @brief Load configuration from file (auto-detect format)
     * @param file_path Path to configuration file (.yaml, .yml, .json)
     * @return KPUSimulator::Config
     * @throws std::runtime_error on file read or parse errors
     */
    static KPUSimulator::Config load(const std::filesystem::path& file_path);

    /**
     * @brief Load configuration from JSON file
     * @param file_path Path to JSON configuration file
     * @return KPUSimulator::Config
     * @throws std::runtime_error on file read or parse errors
     */
    static KPUSimulator::Config load_json(const std::filesystem::path& file_path);

    /**
     * @brief Load configuration from YAML file
     * @param file_path Path to YAML configuration file
     * @return KPUSimulator::Config
     * @throws std::runtime_error on file read or parse errors
     */
    static KPUSimulator::Config load_yaml(const std::filesystem::path& file_path);

    /**
     * @brief Load configuration from JSON string
     * @param json_string JSON configuration as string
     * @return KPUSimulator::Config
     */
    static KPUSimulator::Config from_json_string(const std::string& json_string);

    /**
     * @brief Load configuration from YAML string
     * @param yaml_string YAML configuration as string
     * @return KPUSimulator::Config
     */
    static KPUSimulator::Config from_yaml_string(const std::string& yaml_string);

    /**
     * @brief Save configuration to JSON file
     * @param config Configuration to save
     * @param file_path Output file path
     * @param pretty Pretty-print with indentation
     */
    static void save_json(const KPUSimulator::Config& config,
                          const std::filesystem::path& file_path,
                          bool pretty = true);

    /**
     * @brief Save configuration to YAML file
     * @param config Configuration to save
     * @param file_path Output file path
     */
    static void save_yaml(const KPUSimulator::Config& config,
                          const std::filesystem::path& file_path);

    /**
     * @brief Convert configuration to JSON string
     * @param config Configuration to serialize
     * @param pretty Pretty-print with indentation
     * @return JSON string
     */
    static std::string to_json_string(const KPUSimulator::Config& config,
                                       bool pretty = true);

    /**
     * @brief Convert configuration to YAML string
     * @param config Configuration to serialize
     * @return YAML string
     */
    static std::string to_yaml_string(const KPUSimulator::Config& config);

    /**
     * @brief Validate configuration file without loading
     * @param file_path Path to configuration file
     * @return Validation result with errors and warnings
     */
    static ConfigValidationResult validate(const std::filesystem::path& file_path);

    /**
     * @brief Validate configuration object
     * @param config Configuration to validate
     * @return Validation result with errors and warnings
     */
    static ConfigValidationResult validate(const KPUSimulator::Config& config);

    // =========================================
    // Factory Methods for Common Configurations
    // =========================================

    /**
     * @brief Create minimal configuration for testing
     * @return Minimal KPUSimulator::Config
     */
    static KPUSimulator::Config create_minimal();

    /**
     * @brief Create edge AI configuration (power-efficient)
     * @return Edge AI KPUSimulator::Config
     */
    static KPUSimulator::Config create_edge_ai();

    /**
     * @brief Create embodied AI configuration (robotics/autonomous systems)
     * @return Embodied AI KPUSimulator::Config
     */
    static KPUSimulator::Config create_embodied_ai();

    /**
     * @brief Create datacenter configuration (high-performance)
     * @return Datacenter KPUSimulator::Config
     */
    static KPUSimulator::Config create_datacenter();

    /**
     * @brief Create custom configuration with specified dimensions
     * @param m Matrix dimension M
     * @param n Matrix dimension N
     * @param k Matrix dimension K
     * @param array_size Systolic array size (rows = cols = array_size)
     * @return Configured KPUSimulator::Config
     */
    static KPUSimulator::Config create_for_matmul(Size m, Size n, Size k,
                                                   Size array_size = 16);

private:
    // JSON parsing
    static KPUSimulator::Config parse_json(const nlohmann::json& j);
    static nlohmann::json to_json(const KPUSimulator::Config& config);

    // YAML to JSON conversion (YAML is parsed via JSON intermediate)
    static nlohmann::json yaml_to_json(const std::string& yaml_string);
    static std::string json_to_yaml(const nlohmann::json& j, int indent = 0);

    // File format detection
    static bool is_yaml_file(const std::filesystem::path& file_path);
    static bool is_json_file(const std::filesystem::path& file_path);

    // Validation helpers
    static void validate_config(const KPUSimulator::Config& config,
                                 ConfigValidationResult& result);

    // Helper to safely get optional fields with defaults
    template<typename T>
    static T get_or_default(const nlohmann::json& j, const std::string& key,
                            const T& default_value) {
        if (j.contains(key)) {
            return j[key].get<T>();
        }
        return default_value;
    }

    template<typename T>
    static T get_nested_or_default(const nlohmann::json& j,
                                    const std::string& key1,
                                    const std::string& key2,
                                    const T& default_value) {
        if (j.contains(key1) && j[key1].contains(key2)) {
            return j[key1][key2].get<T>();
        }
        return default_value;
    }

    template<typename T>
    static T get_nested_or_default(const nlohmann::json& j,
                                    const std::string& key1,
                                    const std::string& key2,
                                    const std::string& key3,
                                    const T& default_value) {
        if (j.contains(key1) && j[key1].contains(key2) && j[key1][key2].contains(key3)) {
            return j[key1][key2][key3].get<T>();
        }
        return default_value;
    }
};

} // namespace sw::kpu

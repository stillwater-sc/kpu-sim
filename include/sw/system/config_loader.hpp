#pragma once

#include "system_config.hpp"
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace sw::sim {

/**
 * @brief Loads and validates system configurations from JSON files
 */
class ConfigLoader {
public:
    /**
     * @brief Load system configuration from JSON file
     * @param file_path Path to JSON configuration file
     * @return SystemConfig object
     * @throws std::runtime_error if file cannot be read or JSON is invalid
     */
    static SystemConfig load_from_file(const std::filesystem::path& file_path);

    /**
     * @brief Load system configuration from JSON string
     * @param json_string JSON configuration as string
     * @return SystemConfig object
     * @throws std::runtime_error if JSON is invalid
     */
    static SystemConfig load_from_string(const std::string& json_string);

    /**
     * @brief Save system configuration to JSON file
     * @param config SystemConfig to save
     * @param file_path Path where to save JSON file
     * @throws std::runtime_error if file cannot be written
     */
    static void save_to_file(const SystemConfig& config, const std::filesystem::path& file_path);

    /**
     * @brief Convert system configuration to JSON string
     * @param config SystemConfig to serialize
     * @param pretty Pretty-print with indentation
     * @return JSON string
     */
    static std::string to_json_string(const SystemConfig& config, bool pretty = true);

    /**
     * @brief Validate JSON schema without creating config object
     * @param file_path Path to JSON file
     * @return true if valid, false otherwise
     */
    static bool validate_file(const std::filesystem::path& file_path);

    /**
     * @brief Get validation errors for a JSON file
     * @param file_path Path to JSON file
     * @return Vector of error messages (empty if valid)
     */
    static std::vector<std::string> get_validation_errors(const std::filesystem::path& file_path);

private:
    // JSON parsing helpers
    static SystemConfig parse_json(const nlohmann::json& j);
    static SystemInfo parse_system_info(const nlohmann::json& j);
    static HostConfig parse_host_config(const nlohmann::json& j);
    static CPUConfig parse_cpu_config(const nlohmann::json& j);
    static HostMemoryConfig parse_host_memory_config(const nlohmann::json& j);
    static DRAMControllerConfig parse_dram_controller_config(const nlohmann::json& j);
    static MemoryModuleConfig parse_memory_module_config(const nlohmann::json& j);
    static StorageConfig parse_storage_config(const nlohmann::json& j);
    static std::vector<AcceleratorConfig> parse_accelerators(const nlohmann::json& j);
    static AcceleratorConfig parse_accelerator_config(const nlohmann::json& j);
    static KPUConfig parse_kpu_config(const nlohmann::json& j);
    static KPUMemoryConfig parse_kpu_memory_config(const nlohmann::json& j);
    static KPUMemoryBankConfig parse_kpu_memory_bank_config(const nlohmann::json& j);
    static KPUTileConfig parse_kpu_tile_config(const nlohmann::json& j);
    static KPUScratchpadConfig parse_kpu_scratchpad_config(const nlohmann::json& j);
    static KPUComputeConfig parse_kpu_compute_config(const nlohmann::json& j);
    static ComputeTileConfig parse_compute_tile_config(const nlohmann::json& j);
    static KPUDataMovementConfig parse_kpu_data_movement_config(const nlohmann::json& j);
    static DMAEngineConfig parse_dma_engine_config(const nlohmann::json& j);
    static BlockMoverConfig parse_block_mover_config(const nlohmann::json& j);
    static StreamerConfig parse_streamer_config(const nlohmann::json& j);
    static GPUConfig parse_gpu_config(const nlohmann::json& j);
    static GPUMemoryConfig parse_gpu_memory_config(const nlohmann::json& j);
    static NPUConfig parse_npu_config(const nlohmann::json& j);
    static NPUMemoryConfig parse_npu_memory_config(const nlohmann::json& j);
    static InterconnectConfig parse_interconnect_config(const nlohmann::json& j);
    static HostToAcceleratorConfig parse_host_to_accelerator_config(const nlohmann::json& j);
    static PCIeConfig parse_pcie_config(const nlohmann::json& j);
    static CXLConfig parse_cxl_config(const nlohmann::json& j);
    static AcceleratorToAcceleratorConfig parse_accelerator_to_accelerator_config(const nlohmann::json& j);
    static NoCConfig parse_noc_config(const nlohmann::json& j);
    static OnChipConfig parse_on_chip_config(const nlohmann::json& j);
    static AMBAConfig parse_amba_config(const nlohmann::json& j);
    static NetworkConfig parse_network_config(const nlohmann::json& j);
    static SystemServicesConfig parse_system_services_config(const nlohmann::json& j);
    static MemoryManagerConfig parse_memory_manager_config(const nlohmann::json& j);
    static InterruptControllerConfig parse_interrupt_controller_config(const nlohmann::json& j);
    static PowerManagementConfig parse_power_management_config(const nlohmann::json& j);

    // JSON serialization helpers
    static nlohmann::json to_json(const SystemConfig& config);
    static nlohmann::json to_json(const SystemInfo& info);
    static nlohmann::json to_json(const HostConfig& config);
    static nlohmann::json to_json(const CPUConfig& config);
    static nlohmann::json to_json(const HostMemoryConfig& config);
    static nlohmann::json to_json(const DRAMControllerConfig& config);
    static nlohmann::json to_json(const MemoryModuleConfig& config);
    static nlohmann::json to_json(const StorageConfig& config);
    static nlohmann::json to_json(const AcceleratorConfig& config);
    static nlohmann::json to_json(const KPUConfig& config);
    static nlohmann::json to_json(const KPUMemoryConfig& config);
    static nlohmann::json to_json(const KPUMemoryBankConfig& config);
    static nlohmann::json to_json(const KPUTileConfig& config);
    static nlohmann::json to_json(const KPUScratchpadConfig& config);
    static nlohmann::json to_json(const KPUComputeConfig& config);
    static nlohmann::json to_json(const ComputeTileConfig& config);
    static nlohmann::json to_json(const KPUDataMovementConfig& config);
    static nlohmann::json to_json(const DMAEngineConfig& config);
    static nlohmann::json to_json(const BlockMoverConfig& config);
    static nlohmann::json to_json(const StreamerConfig& config);
    static nlohmann::json to_json(const GPUConfig& config);
    static nlohmann::json to_json(const GPUMemoryConfig& config);
    static nlohmann::json to_json(const NPUConfig& config);
    static nlohmann::json to_json(const NPUMemoryConfig& config);
    static nlohmann::json to_json(const InterconnectConfig& config);
    static nlohmann::json to_json(const HostToAcceleratorConfig& config);
    static nlohmann::json to_json(const PCIeConfig& config);
    static nlohmann::json to_json(const CXLConfig& config);
    static nlohmann::json to_json(const AcceleratorToAcceleratorConfig& config);
    static nlohmann::json to_json(const NoCConfig& config);
    static nlohmann::json to_json(const OnChipConfig& config);
    static nlohmann::json to_json(const AMBAConfig& config);
    static nlohmann::json to_json(const NetworkConfig& config);
    static nlohmann::json to_json(const SystemServicesConfig& config);
    static nlohmann::json to_json(const MemoryManagerConfig& config);
    static nlohmann::json to_json(const InterruptControllerConfig& config);
    static nlohmann::json to_json(const PowerManagementConfig& config);

    // Validation helpers
    static void validate_json_schema(const nlohmann::json& j);
    static AcceleratorType string_to_accelerator_type(const std::string& type_str);
    static std::string accelerator_type_to_string(AcceleratorType type);

    // Helper to safely get optional fields with defaults
    template<typename T>
    static T get_or_default(const nlohmann::json& j, const std::string& key, const T& default_value) {
        if (j.contains(key)) {
            return j[key].get<T>();
        }
        return default_value;
    }

    // Helper to resolve $ref references to external JSON files
    static nlohmann::json resolve_ref(const nlohmann::json& j, const std::filesystem::path& base_path);
};

} // namespace sw::sim

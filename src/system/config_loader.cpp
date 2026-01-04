#include "sw/system/config_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace sw::sim {

using json = nlohmann::json;

//=============================================================================
// Public API - File I/O
//=============================================================================

SystemConfig ConfigLoader::load_from_file(const std::filesystem::path& file_path) {
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Configuration file does not exist: " + file_path.string());
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + file_path.string());
    }

    json j;
    try {
        file >> j;
    }
    catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + file_path.string() + ": " + e.what());
    }

    // Resolve any $ref references to external config files
    json resolved = resolve_ref(j, file_path);

    return parse_json(resolved);
}

SystemConfig ConfigLoader::load_from_string(const std::string& json_string) {
    json j;
    try {
        j = json::parse(json_string);
    }
    catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    return parse_json(j);
}

void ConfigLoader::save_to_file(const SystemConfig& config, const std::filesystem::path& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create configuration file: " + file_path.string());
    }

    json j = to_json(config);
    file << j.dump(2);  // Pretty print with 2-space indentation
}

std::string ConfigLoader::to_json_string(const SystemConfig& config, bool pretty) {
    json j = to_json(config);
    return pretty ? j.dump(2) : j.dump();
}

bool ConfigLoader::validate_file(const std::filesystem::path& file_path) {
    try {
        load_from_file(file_path);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> ConfigLoader::get_validation_errors(const std::filesystem::path& file_path) {
    std::vector<std::string> errors;

    try {
        auto config = load_from_file(file_path);
        std::string validation_errors = config.get_validation_errors();
        if (!validation_errors.empty()) {
            std::istringstream iss(validation_errors);
            std::string line;
            while (std::getline(iss, line)) {
                if (!line.empty()) {
                    errors.push_back(line);
                }
            }
        }
    }
    catch (const std::exception& e) {
        errors.push_back(std::string("Parse error: ") + e.what());
    }

    return errors;
}

//=============================================================================
// JSON Parsing - Top Level
//=============================================================================

SystemConfig ConfigLoader::parse_json(const json& j) {
    SystemConfig config;

    if (j.contains("system")) {
        config.system = parse_system_info(j["system"]);
    }

    if (j.contains("host")) {
        config.host = parse_host_config(j["host"]);
    }

    if (j.contains("accelerators")) {
        config.accelerators = parse_accelerators(j["accelerators"]);
    }

    if (j.contains("interconnect")) {
        config.interconnect = parse_interconnect_config(j["interconnect"]);
    }

    if (j.contains("system_services")) {
        config.system_services = parse_system_services_config(j["system_services"]);
    }

    return config;
}

SystemInfo ConfigLoader::parse_system_info(const json& j) {
    SystemInfo info;
    info.name = get_or_default(j, "name", std::string("Unnamed System"));
    info.description = get_or_default(j, "description", std::string(""));

    if (j.contains("clock_frequency_mhz")) {
        info.clock_frequency_mhz = j["clock_frequency_mhz"].get<uint32_t>();
    }

    return info;
}

//=============================================================================
// JSON Parsing - Host Configuration
//=============================================================================

HostConfig ConfigLoader::parse_host_config(const json& j) {
    HostConfig config;

    if (j.contains("cpu")) {
        config.cpu = parse_cpu_config(j["cpu"]);
    }

    if (j.contains("memory")) {
        config.memory = parse_host_memory_config(j["memory"]);
    }

    if (j.contains("storage") && j["storage"].is_array()) {
        for (const auto& storage_json : j["storage"]) {
            config.storage.push_back(parse_storage_config(storage_json));
        }
    }

    return config;
}

CPUConfig ConfigLoader::parse_cpu_config(const json& j) {
    CPUConfig config;
    config.core_count = get_or_default(j, "core_count", 4u);
    config.frequency_mhz = get_or_default(j, "frequency_mhz", 2400u);
    config.cache_l1_kb = get_or_default(j, "cache_l1_kb", 32u);
    config.cache_l2_kb = get_or_default(j, "cache_l2_kb", 256u);
    config.cache_l3_kb = get_or_default(j, "cache_l3_kb", 8192u);
    return config;
}

HostMemoryConfig ConfigLoader::parse_host_memory_config(const json& j) {
    HostMemoryConfig config;

    if (j.contains("dram_controller")) {
        config.dram_controller = parse_dram_controller_config(j["dram_controller"]);
    }

    if (j.contains("modules") && j["modules"].is_array()) {
        for (const auto& module_json : j["modules"]) {
            config.modules.push_back(parse_memory_module_config(module_json));
        }
    }

    return config;
}

DRAMControllerConfig ConfigLoader::parse_dram_controller_config(const json& j) {
    DRAMControllerConfig config;
    config.channel_count = get_or_default(j, "channel_count", 2u);
    config.data_width_bits = get_or_default(j, "data_width_bits", 64u);
    return config;
}

MemoryModuleConfig ConfigLoader::parse_memory_module_config(const json& j) {
    MemoryModuleConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.type = get_or_default(j, "type", std::string("DDR4"));
    config.form_factor = get_or_default(j, "form_factor", std::string("DIMM"));
    config.capacity_gb = get_or_default(j, "capacity_gb", 16u);
    config.frequency_mhz = get_or_default(j, "frequency_mhz", 3200u);
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 25.6f);
    config.latency_ns = get_or_default(j, "latency_ns", 80u);
    config.channels = get_or_default(j, "channels", 2u);
    return config;
}

StorageConfig ConfigLoader::parse_storage_config(const json& j) {
    StorageConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.type = get_or_default(j, "type", std::string("SSD"));
    config.capacity_gb = get_or_default(j, "capacity_gb", 256u);
    config.read_bandwidth_mbps = get_or_default(j, "read_bandwidth_mbps", 3500u);
    config.write_bandwidth_mbps = get_or_default(j, "write_bandwidth_mbps", 3000u);
    config.latency_us = get_or_default(j, "latency_us", 100u);
    return config;
}

//=============================================================================
// JSON Parsing - Accelerators
//=============================================================================

std::vector<AcceleratorConfig> ConfigLoader::parse_accelerators(const json& j) {
    std::vector<AcceleratorConfig> accelerators;

    if (j.is_array()) {
        for (const auto& accel_json : j) {
            accelerators.push_back(parse_accelerator_config(accel_json));
        }
    }

    return accelerators;
}

AcceleratorConfig ConfigLoader::parse_accelerator_config(const json& j) {
    AcceleratorConfig config;

    std::string type_str = get_or_default(j, "type", std::string("KPU"));
    config.type = string_to_accelerator_type(type_str);
    config.id = get_or_default(j, "id", std::string(""));
    config.description = get_or_default(j, "description", std::string(""));

    // Parse type-specific configuration
    // Support both new naming ("kpu") and legacy naming ("kpu_config")
    if (config.type == AcceleratorType::KPU) {
        if (j.contains("kpu")) {
            config.kpu_config = parse_kpu_config(j["kpu"]);
        } else if (j.contains("kpu_config")) {
            config.kpu_config = parse_kpu_config(j["kpu_config"]);
        }
    }
    else if (config.type == AcceleratorType::GPU && j.contains("gpu_config")) {
        config.gpu_config = parse_gpu_config(j["gpu_config"]);
    }
    else if (config.type == AcceleratorType::NPU && j.contains("npu_config")) {
        config.npu_config = parse_npu_config(j["npu_config"]);
    }

    return config;
}

//=============================================================================
// JSON Parsing - KPU Configuration
//=============================================================================

KPUConfig ConfigLoader::parse_kpu_config(const json& j) {
    KPUConfig config;

    if (j.contains("memory")) {
        config.memory = parse_kpu_memory_config(j["memory"]);
    }

    if (j.contains("compute_fabric")) {
        config.compute_fabric = parse_kpu_compute_config(j["compute_fabric"]);
    }

    if (j.contains("data_movement")) {
        config.data_movement = parse_kpu_data_movement_config(j["data_movement"]);
    }

    return config;
}

KPUMemoryConfig ConfigLoader::parse_kpu_memory_config(const json& j) {
    KPUMemoryConfig config;
    config.type = get_or_default(j, "type", std::string("GDDR6"));
    config.form_factor = get_or_default(j, "form_factor", std::string("PCB"));

    if (j.contains("banks") && j["banks"].is_array()) {
        for (const auto& bank_json : j["banks"]) {
            config.banks.push_back(parse_kpu_memory_bank_config(bank_json));
        }
    }

    if (j.contains("l3_tiles") && j["l3_tiles"].is_array()) {
        for (const auto& tile_json : j["l3_tiles"]) {
            config.l3_tiles.push_back(parse_kpu_tile_config(tile_json));
        }
    }

    if (j.contains("l2_banks") && j["l2_banks"].is_array()) {
        for (const auto& bank_json : j["l2_banks"]) {
            config.l2_banks.push_back(parse_kpu_tile_config(bank_json));
        }
    }

    if (j.contains("scratchpads") && j["scratchpads"].is_array()) {
        for (const auto& scratch_json : j["scratchpads"]) {
            config.scratchpads.push_back(parse_kpu_scratchpad_config(scratch_json));
        }
    }

    return config;
}

KPUMemoryBankConfig ConfigLoader::parse_kpu_memory_bank_config(const json& j) {
    KPUMemoryBankConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.capacity_mb = get_or_default(j, "capacity_mb", 1024u);
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 100.0f);
    config.latency_ns = get_or_default(j, "latency_ns", 20u);
    return config;
}

KPUTileConfig ConfigLoader::parse_kpu_tile_config(const json& j) {
    KPUTileConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.capacity_kb = get_or_default(j, "capacity_kb", 128u);
    return config;
}

KPUScratchpadConfig ConfigLoader::parse_kpu_scratchpad_config(const json& j) {
    KPUScratchpadConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.capacity_kb = get_or_default(j, "capacity_kb", 64u);
    return config;
}

KPUComputeConfig ConfigLoader::parse_kpu_compute_config(const json& j) {
    KPUComputeConfig config;

    if (j.contains("tiles") && j["tiles"].is_array()) {
        for (const auto& tile_json : j["tiles"]) {
            config.tiles.push_back(parse_compute_tile_config(tile_json));
        }
    }

    return config;
}

ComputeTileConfig ConfigLoader::parse_compute_tile_config(const json& j) {
    ComputeTileConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.type = get_or_default(j, "type", std::string("systolic"));
    config.systolic_rows = get_or_default(j, "systolic_rows", 16u);
    config.systolic_cols = get_or_default(j, "systolic_cols", 16u);
    config.datatype = get_or_default(j, "datatype", std::string("fp32"));
    return config;
}

KPUDataMovementConfig ConfigLoader::parse_kpu_data_movement_config(const json& j) {
    KPUDataMovementConfig config;

    if (j.contains("dma_engines") && j["dma_engines"].is_array()) {
        for (const auto& dma_json : j["dma_engines"]) {
            config.dma_engines.push_back(parse_dma_engine_config(dma_json));
        }
    }

    if (j.contains("block_movers") && j["block_movers"].is_array()) {
        for (const auto& mover_json : j["block_movers"]) {
            config.block_movers.push_back(parse_block_mover_config(mover_json));
        }
    }

    if (j.contains("streamers") && j["streamers"].is_array()) {
        for (const auto& streamer_json : j["streamers"]) {
            config.streamers.push_back(parse_streamer_config(streamer_json));
        }
    }

    return config;
}

DMAEngineConfig ConfigLoader::parse_dma_engine_config(const json& j) {
    DMAEngineConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 50.0f);
    config.channels = get_or_default(j, "channels", 1u);
    return config;
}

BlockMoverConfig ConfigLoader::parse_block_mover_config(const json& j) {
    BlockMoverConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    return config;
}

StreamerConfig ConfigLoader::parse_streamer_config(const json& j) {
    StreamerConfig config;
    config.id = get_or_default(j, "id", std::string(""));
    return config;
}

//=============================================================================
// JSON Parsing - GPU Configuration
//=============================================================================

GPUConfig ConfigLoader::parse_gpu_config(const json& j) {
    GPUConfig config;
    config.compute_units = get_or_default(j, "compute_units", 64u);
    config.clock_mhz = get_or_default(j, "clock_mhz", 1800u);

    if (j.contains("memory")) {
        config.memory = parse_gpu_memory_config(j["memory"]);
    }

    return config;
}

GPUMemoryConfig ConfigLoader::parse_gpu_memory_config(const json& j) {
    GPUMemoryConfig config;
    config.type = get_or_default(j, "type", std::string("GDDR6"));
    config.form_factor = get_or_default(j, "form_factor", std::string("PCB"));
    config.capacity_gb = get_or_default(j, "capacity_gb", 8u);
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 448.0f);
    config.bus_width_bits = get_or_default(j, "bus_width_bits", 256u);
    return config;
}

//=============================================================================
// JSON Parsing - NPU Configuration
//=============================================================================

NPUConfig ConfigLoader::parse_npu_config(const json& j) {
    NPUConfig config;
    config.tops_int8 = get_or_default(j, "tops_int8", 40u);
    config.tops_fp16 = get_or_default(j, "tops_fp16", 20u);

    if (j.contains("memory")) {
        config.memory = parse_npu_memory_config(j["memory"]);
    }

    return config;
}

NPUMemoryConfig ConfigLoader::parse_npu_memory_config(const json& j) {
    NPUMemoryConfig config;
    config.type = get_or_default(j, "type", std::string("OnChip"));
    config.capacity_mb = get_or_default(j, "capacity_mb", 16u);
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 200.0f);
    return config;
}

//=============================================================================
// JSON Parsing - Interconnect Configuration
//=============================================================================

InterconnectConfig ConfigLoader::parse_interconnect_config(const json& j) {
    InterconnectConfig config;

    if (j.contains("host_to_accelerator")) {
        config.host_to_accelerator = parse_host_to_accelerator_config(j["host_to_accelerator"]);
    }

    if (j.contains("accelerator_to_accelerator")) {
        config.accelerator_to_accelerator = parse_accelerator_to_accelerator_config(j["accelerator_to_accelerator"]);
    }

    if (j.contains("on_chip")) {
        config.on_chip = parse_on_chip_config(j["on_chip"]);
    }

    if (j.contains("network")) {
        config.network = parse_network_config(j["network"]);
    }

    return config;
}

HostToAcceleratorConfig ConfigLoader::parse_host_to_accelerator_config(const json& j) {
    HostToAcceleratorConfig config;
    config.type = get_or_default(j, "type", std::string("PCIe"));

    if (j.contains("pcie_config")) {
        config.pcie_config = parse_pcie_config(j["pcie_config"]);
    }

    if (j.contains("cxl_config")) {
        config.cxl_config = parse_cxl_config(j["cxl_config"]);
    }

    return config;
}

PCIeConfig ConfigLoader::parse_pcie_config(const json& j) {
    PCIeConfig config;
    config.generation = get_or_default(j, "generation", 4u);
    config.lanes = get_or_default(j, "lanes", 16u);
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 32.0f);
    return config;
}

CXLConfig ConfigLoader::parse_cxl_config(const json& j) {
    CXLConfig config;
    config.version = get_or_default(j, "version", std::string("2.0"));
    config.bandwidth_gbps = get_or_default(j, "bandwidth_gbps", 64.0f);
    return config;
}

AcceleratorToAcceleratorConfig ConfigLoader::parse_accelerator_to_accelerator_config(const json& j) {
    AcceleratorToAcceleratorConfig config;
    config.type = get_or_default(j, "type", std::string("None"));

    if (j.contains("noc_config")) {
        config.noc_config = parse_noc_config(j["noc_config"]);
    }

    return config;
}

NoCConfig ConfigLoader::parse_noc_config(const json& j) {
    NoCConfig config;
    config.topology = get_or_default(j, "topology", std::string("mesh"));
    config.router_count = get_or_default(j, "router_count", 4u);
    config.link_bandwidth_gbps = get_or_default(j, "link_bandwidth_gbps", 16.0f);
    return config;
}

OnChipConfig ConfigLoader::parse_on_chip_config(const json& j) {
    OnChipConfig config;
    config.type = get_or_default(j, "type", std::string("AMBA"));

    if (j.contains("amba_config")) {
        config.amba_config = parse_amba_config(j["amba_config"]);
    }

    return config;
}

AMBAConfig ConfigLoader::parse_amba_config(const json& j) {
    AMBAConfig config;
    config.protocol = get_or_default(j, "protocol", std::string("AXI4"));
    return config;
}

NetworkConfig ConfigLoader::parse_network_config(const json& j) {
    NetworkConfig config;
    config.enabled = get_or_default(j, "enabled", false);
    config.type = get_or_default(j, "type", std::string("Ethernet"));
    config.speed_gbps = get_or_default(j, "speed_gbps", 100u);
    return config;
}

//=============================================================================
// JSON Parsing - System Services
//=============================================================================

SystemServicesConfig ConfigLoader::parse_system_services_config(const json& j) {
    SystemServicesConfig config;

    if (j.contains("memory_manager")) {
        config.memory_manager = parse_memory_manager_config(j["memory_manager"]);
    }

    if (j.contains("interrupt_controller")) {
        config.interrupt_controller = parse_interrupt_controller_config(j["interrupt_controller"]);
    }

    if (j.contains("power_management")) {
        config.power_management = parse_power_management_config(j["power_management"]);
    }

    return config;
}

MemoryManagerConfig ConfigLoader::parse_memory_manager_config(const json& j) {
    MemoryManagerConfig config;
    config.enabled = get_or_default(j, "enabled", true);
    config.pool_size_mb = get_or_default(j, "pool_size_mb", 512u);
    config.alignment_bytes = get_or_default(j, "alignment_bytes", 64u);
    return config;
}

InterruptControllerConfig ConfigLoader::parse_interrupt_controller_config(const json& j) {
    InterruptControllerConfig config;
    config.enabled = get_or_default(j, "enabled", true);
    return config;
}

PowerManagementConfig ConfigLoader::parse_power_management_config(const json& j) {
    PowerManagementConfig config;
    config.enabled = get_or_default(j, "enabled", false);
    return config;
}

//=============================================================================
// Helpers
//=============================================================================

json ConfigLoader::resolve_ref(const json& j, const std::filesystem::path& base_path) {
    // If this is a $ref object, load the referenced file
    if (j.is_object() && j.contains("$ref")) {
        std::string ref_path = j["$ref"].get<std::string>();
        std::filesystem::path resolved_path;

        // Handle relative paths
        if (ref_path[0] != '/') {
            resolved_path = base_path.parent_path() / ref_path;
        } else {
            resolved_path = ref_path;
        }

        if (!std::filesystem::exists(resolved_path)) {
            throw std::runtime_error("Referenced config file not found: " + resolved_path.string());
        }

        std::ifstream file(resolved_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open referenced file: " + resolved_path.string());
        }

        json ref_json;
        file >> ref_json;

        // Recursively resolve any nested $ref
        return resolve_ref(ref_json, resolved_path);
    }

    // For arrays, resolve each element
    if (j.is_array()) {
        json result = json::array();
        for (const auto& elem : j) {
            result.push_back(resolve_ref(elem, base_path));
        }
        return result;
    }

    // For objects, resolve each value
    if (j.is_object()) {
        json result = json::object();
        for (auto& [key, value] : j.items()) {
            result[key] = resolve_ref(value, base_path);
        }
        return result;
    }

    // Primitives are returned as-is
    return j;
}

AcceleratorType ConfigLoader::string_to_accelerator_type(const std::string& type_str) {
    if (type_str == "KPU") return AcceleratorType::KPU;
    if (type_str == "GPU") return AcceleratorType::GPU;
    if (type_str == "NPU") return AcceleratorType::NPU;
    if (type_str == "DSP") return AcceleratorType::DSP;
    if (type_str == "FPGA") return AcceleratorType::FPGA;
    throw std::runtime_error("Unknown accelerator type: " + type_str);
}

std::string ConfigLoader::accelerator_type_to_string(AcceleratorType type) {
    switch (type) {
        case AcceleratorType::KPU: return "KPU";
        case AcceleratorType::GPU: return "GPU";
        case AcceleratorType::NPU: return "NPU";
        case AcceleratorType::DSP: return "DSP";
        case AcceleratorType::FPGA: return "FPGA";
        default: return "Unknown";
    }
}

//=============================================================================
// JSON Serialization (to_json methods)
// Implementation continues in next part...
//=============================================================================

json ConfigLoader::to_json(const SystemConfig& config) {
    json j;
    j["system"] = to_json(config.system);
    j["host"] = to_json(config.host);

    json accelerators_array = json::array();
    for (const auto& accel : config.accelerators) {
        accelerators_array.push_back(to_json(accel));
    }
    j["accelerators"] = accelerators_array;

    j["interconnect"] = to_json(config.interconnect);
    j["system_services"] = to_json(config.system_services);

    return j;
}

json ConfigLoader::to_json(const SystemInfo& info) {
    json j;
    j["name"] = info.name;
    if (!info.description.empty()) {
        j["description"] = info.description;
    }
    if (info.clock_frequency_mhz.has_value()) {
        j["clock_frequency_mhz"] = info.clock_frequency_mhz.value();
    }
    return j;
}

json ConfigLoader::to_json(const HostConfig& config) {
    json j;
    j["cpu"] = to_json(config.cpu);
    j["memory"] = to_json(config.memory);

    if (!config.storage.empty()) {
        json storage_array = json::array();
        for (const auto& storage : config.storage) {
            storage_array.push_back(to_json(storage));
        }
        j["storage"] = storage_array;
    }

    return j;
}

json ConfigLoader::to_json(const CPUConfig& config) {
    json j;
    j["core_count"] = config.core_count;
    j["frequency_mhz"] = config.frequency_mhz;
    j["cache_l1_kb"] = config.cache_l1_kb;
    j["cache_l2_kb"] = config.cache_l2_kb;
    j["cache_l3_kb"] = config.cache_l3_kb;
    return j;
}

json ConfigLoader::to_json(const HostMemoryConfig& config) {
    json j;
    j["dram_controller"] = to_json(config.dram_controller);

    json modules_array = json::array();
    for (const auto& module : config.modules) {
        modules_array.push_back(to_json(module));
    }
    j["modules"] = modules_array;

    return j;
}

json ConfigLoader::to_json(const DRAMControllerConfig& config) {
    json j;
    j["channel_count"] = config.channel_count;
    j["data_width_bits"] = config.data_width_bits;
    return j;
}

json ConfigLoader::to_json(const MemoryModuleConfig& config) {
    json j;
    j["id"] = config.id;
    j["type"] = config.type;
    j["form_factor"] = config.form_factor;
    j["capacity_gb"] = config.capacity_gb;
    j["frequency_mhz"] = config.frequency_mhz;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    j["latency_ns"] = config.latency_ns;
    j["channels"] = config.channels;
    return j;
}

json ConfigLoader::to_json(const StorageConfig& config) {
    json j;
    j["id"] = config.id;
    j["type"] = config.type;
    j["capacity_gb"] = config.capacity_gb;
    j["read_bandwidth_mbps"] = config.read_bandwidth_mbps;
    j["write_bandwidth_mbps"] = config.write_bandwidth_mbps;
    j["latency_us"] = config.latency_us;
    return j;
}

json ConfigLoader::to_json(const AcceleratorConfig& config) {
    json j;
    j["type"] = accelerator_type_to_string(config.type);
    j["id"] = config.id;
    if (!config.description.empty()) {
        j["description"] = config.description;
    }

    // Use new naming convention ("kpu" instead of "kpu_config")
    if (config.kpu_config.has_value()) {
        j["kpu"] = to_json(config.kpu_config.value());
    }
    else if (config.gpu_config.has_value()) {
        j["gpu_config"] = to_json(config.gpu_config.value());
    }
    else if (config.npu_config.has_value()) {
        j["npu_config"] = to_json(config.npu_config.value());
    }

    return j;
}

json ConfigLoader::to_json(const KPUConfig& config) {
    json j;
    j["memory"] = to_json(config.memory);
    j["compute_fabric"] = to_json(config.compute_fabric);
    j["data_movement"] = to_json(config.data_movement);
    return j;
}

json ConfigLoader::to_json(const KPUMemoryConfig& config) {
    json j;
    j["type"] = config.type;
    j["form_factor"] = config.form_factor;

    json banks_array = json::array();
    for (const auto& bank : config.banks) {
        banks_array.push_back(to_json(bank));
    }
    j["banks"] = banks_array;

    json l3_array = json::array();
    for (const auto& tile : config.l3_tiles) {
        l3_array.push_back(to_json(tile));
    }
    j["l3_tiles"] = l3_array;

    json l2_array = json::array();
    for (const auto& bank : config.l2_banks) {
        l2_array.push_back(to_json(bank));
    }
    j["l2_banks"] = l2_array;

    json scratch_array = json::array();
    for (const auto& scratch : config.scratchpads) {
        scratch_array.push_back(to_json(scratch));
    }
    j["scratchpads"] = scratch_array;

    return j;
}

json ConfigLoader::to_json(const KPUMemoryBankConfig& config) {
    json j;
    j["id"] = config.id;
    j["capacity_mb"] = config.capacity_mb;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    j["latency_ns"] = config.latency_ns;
    return j;
}

json ConfigLoader::to_json(const KPUTileConfig& config) {
    json j;
    j["id"] = config.id;
    j["capacity_kb"] = config.capacity_kb;
    return j;
}

json ConfigLoader::to_json(const KPUScratchpadConfig& config) {
    json j;
    j["id"] = config.id;
    j["capacity_kb"] = config.capacity_kb;
    return j;
}

json ConfigLoader::to_json(const KPUComputeConfig& config) {
    json j;
    json tiles_array = json::array();
    for (const auto& tile : config.tiles) {
        tiles_array.push_back(to_json(tile));
    }
    j["tiles"] = tiles_array;
    return j;
}

json ConfigLoader::to_json(const ComputeTileConfig& config) {
    json j;
    j["id"] = config.id;
    j["type"] = config.type;
    j["systolic_rows"] = config.systolic_rows;
    j["systolic_cols"] = config.systolic_cols;
    j["datatype"] = config.datatype;
    return j;
}

json ConfigLoader::to_json(const KPUDataMovementConfig& config) {
    json j;

    json dma_array = json::array();
    for (const auto& dma : config.dma_engines) {
        dma_array.push_back(to_json(dma));
    }
    j["dma_engines"] = dma_array;

    json mover_array = json::array();
    for (const auto& mover : config.block_movers) {
        mover_array.push_back(to_json(mover));
    }
    j["block_movers"] = mover_array;

    json streamer_array = json::array();
    for (const auto& streamer : config.streamers) {
        streamer_array.push_back(to_json(streamer));
    }
    j["streamers"] = streamer_array;

    return j;
}

json ConfigLoader::to_json(const DMAEngineConfig& config) {
    json j;
    j["id"] = config.id;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    j["channels"] = config.channels;
    return j;
}

json ConfigLoader::to_json(const BlockMoverConfig& config) {
    json j;
    j["id"] = config.id;
    return j;
}

json ConfigLoader::to_json(const StreamerConfig& config) {
    json j;
    j["id"] = config.id;
    return j;
}

json ConfigLoader::to_json(const GPUConfig& config) {
    json j;
    j["compute_units"] = config.compute_units;
    j["clock_mhz"] = config.clock_mhz;
    j["memory"] = to_json(config.memory);
    return j;
}

json ConfigLoader::to_json(const GPUMemoryConfig& config) {
    json j;
    j["type"] = config.type;
    j["form_factor"] = config.form_factor;
    j["capacity_gb"] = config.capacity_gb;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    j["bus_width_bits"] = config.bus_width_bits;
    return j;
}

json ConfigLoader::to_json(const NPUConfig& config) {
    json j;
    j["tops_int8"] = config.tops_int8;
    j["tops_fp16"] = config.tops_fp16;
    j["memory"] = to_json(config.memory);
    return j;
}

json ConfigLoader::to_json(const NPUMemoryConfig& config) {
    json j;
    j["type"] = config.type;
    j["capacity_mb"] = config.capacity_mb;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    return j;
}

json ConfigLoader::to_json(const InterconnectConfig& config) {
    json j;
    j["host_to_accelerator"] = to_json(config.host_to_accelerator);
    j["accelerator_to_accelerator"] = to_json(config.accelerator_to_accelerator);
    j["on_chip"] = to_json(config.on_chip);
    if (config.network.has_value()) {
        j["network"] = to_json(config.network.value());
    }
    return j;
}

json ConfigLoader::to_json(const HostToAcceleratorConfig& config) {
    json j;
    j["type"] = config.type;
    if (config.pcie_config.has_value()) {
        j["pcie_config"] = to_json(config.pcie_config.value());
    }
    if (config.cxl_config.has_value()) {
        j["cxl_config"] = to_json(config.cxl_config.value());
    }
    return j;
}

json ConfigLoader::to_json(const PCIeConfig& config) {
    json j;
    j["generation"] = config.generation;
    j["lanes"] = config.lanes;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    return j;
}

json ConfigLoader::to_json(const CXLConfig& config) {
    json j;
    j["version"] = config.version;
    j["bandwidth_gbps"] = config.bandwidth_gbps;
    return j;
}

json ConfigLoader::to_json(const AcceleratorToAcceleratorConfig& config) {
    json j;
    j["type"] = config.type;
    if (config.noc_config.has_value()) {
        j["noc_config"] = to_json(config.noc_config.value());
    }
    return j;
}

json ConfigLoader::to_json(const NoCConfig& config) {
    json j;
    j["topology"] = config.topology;
    j["router_count"] = config.router_count;
    j["link_bandwidth_gbps"] = config.link_bandwidth_gbps;
    return j;
}

json ConfigLoader::to_json(const OnChipConfig& config) {
    json j;
    j["type"] = config.type;
    if (config.amba_config.has_value()) {
        j["amba_config"] = to_json(config.amba_config.value());
    }
    return j;
}

json ConfigLoader::to_json(const AMBAConfig& config) {
    json j;
    j["protocol"] = config.protocol;
    return j;
}

json ConfigLoader::to_json(const NetworkConfig& config) {
    json j;
    j["enabled"] = config.enabled;
    j["type"] = config.type;
    j["speed_gbps"] = config.speed_gbps;
    return j;
}

json ConfigLoader::to_json(const SystemServicesConfig& config) {
    json j;
    j["memory_manager"] = to_json(config.memory_manager);
    j["interrupt_controller"] = to_json(config.interrupt_controller);
    j["power_management"] = to_json(config.power_management);
    return j;
}

json ConfigLoader::to_json(const MemoryManagerConfig& config) {
    json j;
    j["enabled"] = config.enabled;
    j["pool_size_mb"] = config.pool_size_mb;
    j["alignment_bytes"] = config.alignment_bytes;
    return j;
}

json ConfigLoader::to_json(const InterruptControllerConfig& config) {
    json j;
    j["enabled"] = config.enabled;
    return j;
}

json ConfigLoader::to_json(const PowerManagementConfig& config) {
    json j;
    j["enabled"] = config.enabled;
    return j;
}

} // namespace sw::sim

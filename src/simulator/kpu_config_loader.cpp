/**
 * @file kpu_config_loader.cpp
 * @brief Implementation of KPU configuration file loader
 */

#include <sw/kpu/kpu_config_loader.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>

namespace sw::kpu {

// =========================================
// File Loading
// =========================================

KPUSimulator::Config KPUConfigLoader::load(const std::filesystem::path& file_path) {
    if (is_yaml_file(file_path)) {
        return load_yaml(file_path);
    } else if (is_json_file(file_path)) {
        return load_json(file_path);
    } else {
        throw std::runtime_error("Unsupported file format: " + file_path.string() +
                                 " (expected .yaml, .yml, or .json)");
    }
}

KPUSimulator::Config KPUConfigLoader::load_json(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    try {
        nlohmann::json j = nlohmann::json::parse(file);
        return parse_json(j);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parse error in " + file_path.string() + ": " + e.what());
    }
}

KPUSimulator::Config KPUConfigLoader::load_yaml(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_yaml_string(buffer.str());
}

KPUSimulator::Config KPUConfigLoader::from_json_string(const std::string& json_string) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_string);
        return parse_json(j);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }
}

KPUSimulator::Config KPUConfigLoader::from_yaml_string(const std::string& yaml_string) {
    nlohmann::json j = yaml_to_json(yaml_string);
    return parse_json(j);
}

// =========================================
// File Saving
// =========================================

void KPUConfigLoader::save_json(const KPUSimulator::Config& config,
                                 const std::filesystem::path& file_path,
                                 bool pretty) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + file_path.string());
    }

    nlohmann::json j = to_json(config);
    file << (pretty ? j.dump(2) : j.dump());
}

void KPUConfigLoader::save_yaml(const KPUSimulator::Config& config,
                                 const std::filesystem::path& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + file_path.string());
    }

    file << to_yaml_string(config);
}

std::string KPUConfigLoader::to_json_string(const KPUSimulator::Config& config, bool pretty) {
    nlohmann::json j = to_json(config);
    return pretty ? j.dump(2) : j.dump();
}

std::string KPUConfigLoader::to_yaml_string(const KPUSimulator::Config& config) {
    nlohmann::json j = to_json(config);
    return json_to_yaml(j);
}

// =========================================
// JSON Parsing
// =========================================

KPUSimulator::Config KPUConfigLoader::parse_json(const nlohmann::json& j) {
    KPUSimulator::Config config;

    // Host memory
    config.host_memory_region_count = get_nested_or_default<Size>(j, "host_memory", "region_count", 0);
    config.host_memory_region_capacity_mb = get_nested_or_default<Size>(j, "host_memory", "region_capacity_mb", 0);
    config.host_memory_bandwidth_gbps = get_nested_or_default<Size>(j, "host_memory", "bandwidth_gbps", 0);

    // External memory
    config.memory_bank_count = get_nested_or_default<Size>(j, "external_memory", "bank_count", 0);
    config.memory_bank_capacity_mb = get_nested_or_default<Size>(j, "external_memory", "bank_capacity_mb", 0);
    config.memory_bandwidth_gbps = get_nested_or_default<Size>(j, "external_memory", "bandwidth_gbps", 0);

    // Memory controller
    config.memory_controller_count = get_nested_or_default<Size>(j, "memory_controller", "controller_count", 0);
    config.page_buffer_count = get_nested_or_default<Size>(j, "memory_controller", "page_buffer_count", 0);
    config.page_buffer_capacity_kb = get_nested_or_default<Size>(j, "memory_controller", "page_buffer_capacity_kb", 0);

    // On-chip memory hierarchy
    config.l3_tile_count = get_nested_or_default<Size>(j, "on_chip_memory", "l3", "tile_count", 0);
    config.l3_tile_capacity_kb = get_nested_or_default<Size>(j, "on_chip_memory", "l3", "tile_capacity_kb", 0);
    config.l2_bank_count = get_nested_or_default<Size>(j, "on_chip_memory", "l2", "bank_count", 0);
    config.l2_bank_capacity_kb = get_nested_or_default<Size>(j, "on_chip_memory", "l2", "bank_capacity_kb", 0);
    config.l1_buffer_count = get_nested_or_default<Size>(j, "on_chip_memory", "l1", "buffer_count", 0);
    config.l1_buffer_capacity_kb = get_nested_or_default<Size>(j, "on_chip_memory", "l1", "buffer_capacity_kb", 0);

    // Data movement
    config.dma_engine_count = get_nested_or_default<Size>(j, "data_movement", "dma_engine_count", 0);
    config.block_mover_count = get_nested_or_default<Size>(j, "data_movement", "block_mover_count", 0);
    config.streamer_count = get_nested_or_default<Size>(j, "data_movement", "streamer_count", 0);

    // Compute
    config.compute_tile_count = get_nested_or_default<Size>(j, "compute", "tile_count", 0);
    config.processor_array_rows = get_nested_or_default<Size>(j, "compute", "processor_array", "rows", 0);
    config.processor_array_cols = get_nested_or_default<Size>(j, "compute", "processor_array", "cols", 0);
    config.use_systolic_array_mode = get_nested_or_default<bool>(j, "compute", "systolic_mode", false);

    // Parse topology (default: rectangular)
    std::string topology_str = get_nested_or_default<std::string>(j, "compute", "processor_array", "topology", "rectangular");
    try {
        config.processor_array_topology = topology_from_string(topology_str);
    } catch (...) {
        config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    }

    // Auto-compute L1 buffer count if not explicitly set
    // L1 buffers are derived from processor array configuration
    if (config.l1_buffer_count == 0 && config.compute_tile_count > 0 &&
        config.processor_array_rows > 0) {
        config.l1_buffer_count = compute_l1_buffer_count(
            config.processor_array_topology,
            config.processor_array_rows,
            config.processor_array_cols,
            config.compute_tile_count
        );
    }

    // Address map (optional)
    if (j.contains("address_map")) {
        const auto& am = j["address_map"];
        config.host_memory_base = get_or_default<Address>(am, "host_memory_base", 0);
        config.external_memory_base = get_or_default<Address>(am, "external_memory_base", 0);
        config.l3_tile_base = get_or_default<Address>(am, "l3_tile_base", 0);
        config.l2_bank_base = get_or_default<Address>(am, "l2_bank_base", 0);
        config.l1_buffer_base = get_or_default<Address>(am, "l1_buffer_base", 0);
        config.page_buffer_base = get_or_default<Address>(am, "page_buffer_base", 0);
    }

    return config;
}

nlohmann::json KPUConfigLoader::to_json(const KPUSimulator::Config& config) {
    nlohmann::json j;

    // Host memory
    j["host_memory"]["region_count"] = config.host_memory_region_count;
    j["host_memory"]["region_capacity_mb"] = config.host_memory_region_capacity_mb;
    j["host_memory"]["bandwidth_gbps"] = config.host_memory_bandwidth_gbps;

    // External memory
    j["external_memory"]["bank_count"] = config.memory_bank_count;
    j["external_memory"]["bank_capacity_mb"] = config.memory_bank_capacity_mb;
    j["external_memory"]["bandwidth_gbps"] = config.memory_bandwidth_gbps;

    // Memory controller
    j["memory_controller"]["controller_count"] = config.memory_controller_count;
    j["memory_controller"]["page_buffer_count"] = config.page_buffer_count;
    j["memory_controller"]["page_buffer_capacity_kb"] = config.page_buffer_capacity_kb;

    // On-chip memory hierarchy
    j["on_chip_memory"]["l3"]["tile_count"] = config.l3_tile_count;
    j["on_chip_memory"]["l3"]["tile_capacity_kb"] = config.l3_tile_capacity_kb;
    j["on_chip_memory"]["l2"]["bank_count"] = config.l2_bank_count;
    j["on_chip_memory"]["l2"]["bank_capacity_kb"] = config.l2_bank_capacity_kb;
    j["on_chip_memory"]["l1"]["buffer_count"] = config.l1_buffer_count;
    j["on_chip_memory"]["l1"]["buffer_capacity_kb"] = config.l1_buffer_capacity_kb;

    // Data movement
    j["data_movement"]["dma_engine_count"] = config.dma_engine_count;
    j["data_movement"]["block_mover_count"] = config.block_mover_count;
    j["data_movement"]["streamer_count"] = config.streamer_count;

    // Compute
    j["compute"]["tile_count"] = config.compute_tile_count;
    j["compute"]["processor_array"]["rows"] = config.processor_array_rows;
    j["compute"]["processor_array"]["cols"] = config.processor_array_cols;
    j["compute"]["processor_array"]["topology"] = topology_to_string(config.processor_array_topology);
    j["compute"]["systolic_mode"] = config.use_systolic_array_mode;

    // Address map (only if non-default)
    if (config.host_memory_base != 0 || config.external_memory_base != 0 ||
        config.l3_tile_base != 0 || config.l2_bank_base != 0 ||
        config.l1_buffer_base != 0 || config.page_buffer_base != 0) {
        j["address_map"]["host_memory_base"] = config.host_memory_base;
        j["address_map"]["external_memory_base"] = config.external_memory_base;
        j["address_map"]["l3_tile_base"] = config.l3_tile_base;
        j["address_map"]["l2_bank_base"] = config.l2_bank_base;
        j["address_map"]["l1_buffer_base"] = config.l1_buffer_base;
        j["address_map"]["page_buffer_base"] = config.page_buffer_base;
    }

    return j;
}

// =========================================
// Simple YAML Parser (subset for config files)
// =========================================

// A simple YAML to JSON converter that handles our config file format
// This is a minimal implementation that doesn't require external YAML library
nlohmann::json KPUConfigLoader::yaml_to_json(const std::string& yaml_string) {
    nlohmann::json result;
    std::vector<std::pair<int, nlohmann::json*>> stack;
    stack.push_back({-1, &result});

    std::istringstream stream(yaml_string);
    std::string line;

    while (std::getline(stream, line)) {
        // Skip empty lines and comments
        size_t first_non_space = line.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) continue;
        if (line[first_non_space] == '#') continue;

        // Calculate indentation
        int indent = static_cast<int>(first_non_space);

        // Trim the line
        std::string trimmed = line.substr(first_non_space);
        size_t last_non_space = trimmed.find_last_not_of(" \t\r\n");
        if (last_non_space != std::string::npos) {
            trimmed = trimmed.substr(0, last_non_space + 1);
        }

        // Skip if empty after trim
        if (trimmed.empty()) continue;

        // Pop stack until we find the right parent
        while (stack.size() > 1 && stack.back().first >= indent) {
            stack.pop_back();
        }

        // Find key-value separator
        size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = trimmed.substr(0, colon_pos);
        std::string value = (colon_pos + 1 < trimmed.size()) ?
                            trimmed.substr(colon_pos + 1) : "";

        // Trim key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        if (!value.empty()) {
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
        }

        // Remove quotes from value
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        nlohmann::json* parent = stack.back().second;

        if (value.empty()) {
            // This is a nested object
            (*parent)[key] = nlohmann::json::object();
            stack.push_back({indent, &(*parent)[key]});
        } else {
            // This is a value
            // Try to parse as different types
            if (value == "true") {
                (*parent)[key] = true;
            } else if (value == "false") {
                (*parent)[key] = false;
            } else {
                // Try to parse as number
                try {
                    // Check for hex number
                    if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
                        (*parent)[key] = std::stoull(value, nullptr, 16);
                    } else if (value.find('.') != std::string::npos) {
                        (*parent)[key] = std::stod(value);
                    } else {
                        (*parent)[key] = std::stoll(value);
                    }
                } catch (...) {
                    // Store as string
                    (*parent)[key] = value;
                }
            }
        }
    }

    return result;
}

std::string KPUConfigLoader::json_to_yaml(const nlohmann::json& j, int indent) {
    std::ostringstream ss;
    std::string indent_str(indent * 2, ' ');

    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            ss << indent_str << it.key() << ":";
            if (it.value().is_object()) {
                ss << "\n" << json_to_yaml(it.value(), indent + 1);
            } else if (it.value().is_boolean()) {
                ss << " " << (it.value().get<bool>() ? "true" : "false") << "\n";
            } else if (it.value().is_number_integer()) {
                ss << " " << it.value().get<int64_t>() << "\n";
            } else if (it.value().is_number_float()) {
                ss << " " << it.value().get<double>() << "\n";
            } else if (it.value().is_string()) {
                std::string str = it.value().get<std::string>();
                if (str.find(' ') != std::string::npos ||
                    str.find(':') != std::string::npos) {
                    ss << " \"" << str << "\"\n";
                } else {
                    ss << " " << str << "\n";
                }
            } else {
                ss << " " << it.value().dump() << "\n";
            }
        }
    }

    return ss.str();
}

// =========================================
// Validation
// =========================================

ConfigValidationResult KPUConfigLoader::validate(const std::filesystem::path& file_path) {
    ConfigValidationResult result;

    try {
        KPUSimulator::Config config = load(file_path);
        validate_config(config, result);
    } catch (const std::exception& e) {
        result.valid = false;
        result.errors.push_back(e.what());
    }

    return result;
}

ConfigValidationResult KPUConfigLoader::validate(const KPUSimulator::Config& config) {
    ConfigValidationResult result;
    validate_config(config, result);
    return result;
}

void KPUConfigLoader::validate_config(const KPUSimulator::Config& config,
                                       ConfigValidationResult& result) {
    result.valid = true;

    // Check for required resources
    if (config.compute_tile_count == 0) {
        result.warnings.push_back("No compute tiles configured");
    }

    if (config.memory_bank_count == 0 && config.host_memory_region_count == 0) {
        result.errors.push_back("No memory configured (external or host)");
        result.valid = false;
    }

    // Check processor array consistency
    if (config.processor_array_rows == 0 || config.processor_array_cols == 0) {
        if (config.compute_tile_count > 0) {
            result.warnings.push_back("Compute tiles configured but processor array dimensions are 0");
        }
    }

    // Check data movement resources
    if (config.dma_engine_count == 0 && config.memory_bank_count > 0) {
        result.warnings.push_back("External memory configured but no DMA engines");
    }

    // Check on-chip memory hierarchy
    if (config.l1_buffer_count == 0 && config.compute_tile_count > 0) {
        result.warnings.push_back("Compute tiles configured but no L1 buffers");
    }

    // Validate L1 buffer count matches processor array configuration
    if (config.l1_buffer_count > 0 && config.compute_tile_count > 0 &&
        config.processor_array_rows > 0) {
        Size expected_l1_count = compute_l1_buffer_count(
            config.processor_array_topology,
            config.processor_array_rows,
            config.processor_array_cols,
            config.compute_tile_count
        );
        if (config.l1_buffer_count != expected_l1_count) {
            result.warnings.push_back(
                "L1 buffer count mismatch: configured " +
                std::to_string(config.l1_buffer_count) +
                " but processor array requires " +
                std::to_string(expected_l1_count) +
                " (4 × (rows + cols) × compute_tiles for rectangular arrays)"
            );
        }
    }

    if (config.l2_bank_count == 0 && config.l1_buffer_count > 0) {
        result.warnings.push_back("L1 buffers configured but no L2 banks");
    }

    if (config.streamer_count == 0 && config.l1_buffer_count > 0) {
        result.warnings.push_back("L1 buffers configured but no streamers");
    }

    if (config.block_mover_count == 0 && config.l3_tile_count > 0) {
        result.warnings.push_back("L3 tiles configured but no block movers");
    }
}

// =========================================
// File Format Detection
// =========================================

bool KPUConfigLoader::is_yaml_file(const std::filesystem::path& file_path) {
    std::string ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".yaml" || ext == ".yml";
}

bool KPUConfigLoader::is_json_file(const std::filesystem::path& file_path) {
    std::string ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".json";
}

// =========================================
// Factory Methods
// =========================================

KPUSimulator::Config KPUConfigLoader::create_minimal() {
    // Minimal configuration: smallest viable KPU for testing
    // 1 compute tile, 8×8 array, 1 L3 tile, 4 L2 banks
    KPUSimulator::Config config;

    // Host memory (minimal for testing)
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 128;
    config.host_memory_bandwidth_gbps = 25;

    // External memory (single LPDDR4x channel - low power)
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 256;
    config.memory_bandwidth_gbps = 25;

    // Memory controller
    config.memory_controller_count = 1;
    config.page_buffer_count = 2;
    config.page_buffer_capacity_kb = 16;

    // Compute: 1 tile, 8×8 array
    config.compute_tile_count = 1;
    config.processor_array_rows = 8;
    config.processor_array_cols = 8;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // On-chip memory: L3 = global working set, L2 = local tile buffers
    // L3 should be significantly larger than L2 total
    config.l3_tile_count = 1;
    config.l3_tile_capacity_kb = 256;    // 256 KB L3 total
    config.l2_bank_count = 4;
    config.l2_bank_capacity_kb = 16;     // 64 KB L2 total (4:1 ratio)
    // L1 buffers: 4 × (8 + 8) × 1 tile = 64
    config.l1_buffer_count = compute_l1_buffer_count(
        config.processor_array_topology,
        config.processor_array_rows,
        config.processor_array_cols,
        config.compute_tile_count
    );
    config.l1_buffer_capacity_kb = 32;

    // Data movement (minimal)
    config.dma_engine_count = 1;
    config.block_mover_count = 1;
    config.streamer_count = 4;

    return config;
}

KPUSimulator::Config KPUConfigLoader::create_edge_ai() {
    // Edge AI configuration: 2 compute tiles for power-efficient inference
    // 2 tiles, 16×16 arrays, 2 L3 tiles, 8 L2 banks per L3 (16 total)
    KPUSimulator::Config config;

    // Host memory (embedded system)
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 512;
    config.host_memory_bandwidth_gbps = 50;

    // External memory (LPDDR5 quad channel, 64-bit total)
    // 4 channels × 16-bit × 6400 MT/s ≈ 50 GB/s total
    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 256;
    config.memory_bandwidth_gbps = 12;  // ~12.8 GB/s per 16-bit channel

    // Memory controller
    config.memory_controller_count = 2;
    config.page_buffer_count = 4;
    config.page_buffer_capacity_kb = 32;

    // Compute: 2 tiles, 16×16 arrays
    config.compute_tile_count = 2;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // On-chip memory: L3 = global working set, L2 = local tile buffers
    // L3 should be significantly larger than L2 total
    config.l3_tile_count = 2;
    config.l3_tile_capacity_kb = 512;    // 1 MB L3 total
    config.l2_bank_count = 8;            // 4 per compute tile
    config.l2_bank_capacity_kb = 32;     // 256 KB L2 total (4:1 ratio)
    // L1 buffers: 4 × (16 + 16) × 2 tiles = 256
    config.l1_buffer_count = compute_l1_buffer_count(
        config.processor_array_topology,
        config.processor_array_rows,
        config.processor_array_cols,
        config.compute_tile_count
    );
    config.l1_buffer_capacity_kb = 64;

    // Data movement
    config.dma_engine_count = 2;
    config.block_mover_count = 4;
    config.streamer_count = 16;

    return config;
}

KPUSimulator::Config KPUConfigLoader::create_embodied_ai() {
    // Embodied AI configuration: robotics and autonomous systems
    // Modeled after Jetson Orin: 256-bit LPDDR5 @ 204 GB/s, power-efficient
    // 64 tiles (8×8 layout), 24×24 arrays, 64 L3 tiles, 16 L2 banks per L3
    KPUSimulator::Config config;

    // Host memory (embedded high-performance)
    config.host_memory_region_count = 2;
    config.host_memory_region_capacity_mb = 2048;  // 2GB per region = 4GB total
    config.host_memory_bandwidth_gbps = 100;

    // External memory (8-channel LPDDR5, 256-bit total, Jetson Orin style)
    // 8 channels × 32-bit × 25 GB/s = 200 GB/s total, very power efficient
    config.memory_bank_count = 8;
    config.memory_bank_capacity_mb = 512;   // 512MB per channel = 4GB total
    config.memory_bandwidth_gbps = 25;      // 25 GB/s per channel

    // Memory controller (one per 2 channels)
    config.memory_controller_count = 4;
    config.page_buffer_count = 16;
    config.page_buffer_capacity_kb = 32;

    // Compute: 64 tiles (8×8 layout), 24×24 arrays each
    config.compute_tile_count = 64;
    config.processor_array_rows = 24;
    config.processor_array_cols = 24;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // On-chip memory: L3 = global working set, L2 = local tile buffers
    // L3 should be significantly larger than L2 total
    config.l3_tile_count = 16;
    config.l3_tile_capacity_kb = 2048;   // 32 MB L3 total
    config.l2_bank_count = 256;          // 4 per compute tile
    config.l2_bank_capacity_kb = 32;     // 8 MB L2 total (4:1 ratio)
    // L1 buffers: 4 × (24 + 24) × 64 tiles = 12,288
    config.l1_buffer_count = compute_l1_buffer_count(
        config.processor_array_topology,
        config.processor_array_rows,
        config.processor_array_cols,
        config.compute_tile_count
    );
    config.l1_buffer_capacity_kb = 64;

    // Data movement (scaled for 64 tiles)
    config.dma_engine_count = 8;
    config.block_mover_count = 64;   // 1 per L3 tile
    config.streamer_count = 256;     // 4 per L3 tile

    return config;
}

KPUSimulator::Config KPUConfigLoader::create_datacenter() {
    // Datacenter configuration: massive parallel compute
    // 256 tiles (16×16 checkerboard), 32×32 arrays, 256 L3 tiles, 16 L2 banks per L3
    KPUSimulator::Config config;

    // Host memory (NUMA, large capacity)
    config.host_memory_region_count = 4;
    config.host_memory_region_capacity_mb = 16384;  // 16GB per region = 64GB total
    config.host_memory_bandwidth_gbps = 200;

    // External memory (6 HBM3 channels @ 800 GB/s each = 4.8 TB/s)
    config.memory_bank_count = 6;
    config.memory_bank_capacity_mb = 4096;  // 4GB per channel = 24GB total
    config.memory_bandwidth_gbps = 800;

    // Memory controller (one per HBM channel)
    config.memory_controller_count = 6;
    config.page_buffer_count = 32;
    config.page_buffer_capacity_kb = 128;

    // Compute: 256 tiles (16×16 checkerboard), 32×32 arrays each
    config.compute_tile_count = 256;
    config.processor_array_rows = 32;
    config.processor_array_cols = 32;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // On-chip memory: L3 = global working set, L2 = local tile buffers
    // L3 should be significantly larger than L2 total
    config.l3_tile_count = 64;
    config.l3_tile_capacity_kb = 4096;   // 256 MB L3 total
    config.l2_bank_count = 1024;         // 4 per compute tile
    config.l2_bank_capacity_kb = 64;     // 64 MB L2 total (4:1 ratio)
    // L1 buffers: 4 × (32 + 32) × 256 tiles = 65,536
    config.l1_buffer_count = compute_l1_buffer_count(
        config.processor_array_topology,
        config.processor_array_rows,
        config.processor_array_cols,
        config.compute_tile_count
    );
    config.l1_buffer_capacity_kb = 128;

    // Data movement (scaled for 256 tiles)
    config.dma_engine_count = 32;
    config.block_mover_count = 256;  // 1 per L3 tile
    config.streamer_count = 1024;    // 4 per L3 tile

    return config;
}

KPUSimulator::Config KPUConfigLoader::create_for_matmul(Size m, Size n, Size k, Size array_size) {
    KPUSimulator::Config config;

    // Calculate memory requirements
    Size matrix_a_bytes = m * k * sizeof(float);
    Size matrix_b_bytes = k * n * sizeof(float);
    Size matrix_c_bytes = m * n * sizeof(float);
    Size total_bytes = matrix_a_bytes + matrix_b_bytes + matrix_c_bytes;

    // Round up to MB with 50% overhead
    Size required_mb = (total_bytes * 3 / 2) / (1024 * 1024) + 1;

    // Host memory
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = required_mb;
    config.host_memory_bandwidth_gbps = 50;

    // External memory (2x requirement for double buffering)
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = required_mb;
    config.memory_bandwidth_gbps = 100;

    // Memory controller
    config.memory_controller_count = 1;
    config.page_buffer_count = 2;
    config.page_buffer_capacity_kb = 32;

    // Compute
    config.compute_tile_count = 1;
    config.processor_array_rows = array_size;
    config.processor_array_cols = array_size;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // On-chip memory (sized for tiling)
    Size tile_bytes = array_size * array_size * sizeof(float);
    Size l1_capacity_kb = (tile_bytes / 1024) + 1;
    Size l2_capacity_kb = l1_capacity_kb * 4;
    Size l3_capacity_kb = l2_capacity_kb * 4;

    config.l3_tile_count = 4;
    config.l3_tile_capacity_kb = l3_capacity_kb;
    config.l2_bank_count = 8;
    config.l2_bank_capacity_kb = l2_capacity_kb;
    // L1 buffers derived from array size
    config.l1_buffer_count = compute_l1_buffer_count(
        config.processor_array_topology,
        config.processor_array_rows,
        config.processor_array_cols,
        config.compute_tile_count
    );
    config.l1_buffer_capacity_kb = l1_capacity_kb;

    // Data movement
    config.dma_engine_count = 2;
    config.block_mover_count = 4;
    config.streamer_count = 8;

    return config;
}

} // namespace sw::kpu

// ============================================================================
// src/config/simulator_config_parser.cpp
// Configuration parser implementation
// ============================================================================

#include <sw/kpu/config/simulator_config_parser.hpp>

#include <fstream>
#include <sstream>

namespace sw::kpu::config {

// ============================================================================
// File I/O
// ============================================================================

SimulatorConfig SimulatorConfigParser::parse_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw ConfigParseError("Configuration file not found: " + path.string());
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        throw ConfigParseError("Failed to open configuration file: " + path.string());
    }

    try {
        nlohmann::json j = nlohmann::json::parse(file);
        return parse_json(j);
    } catch (const nlohmann::json::parse_error& e) {
        throw ConfigParseError("JSON parse error in " + path.string() + ": " + e.what());
    }
}

SimulatorConfig SimulatorConfigParser::parse_string(const std::string& json_str) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);
        return parse_json(j);
    } catch (const nlohmann::json::parse_error& e) {
        throw ConfigParseError(std::string("JSON parse error: ") + e.what());
    }
}

void SimulatorConfigParser::write_file(const SimulatorConfig& config,
                                        const std::filesystem::path& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw ConfigParseError("Failed to open file for writing: " + path.string());
    }

    nlohmann::json j = to_json(config);
    file << j.dump(2);  // Pretty print with 2-space indent
}

// ============================================================================
// JSON Parsing
// ============================================================================

SimulatorConfig SimulatorConfigParser::parse_json(const nlohmann::json& j) {
    SimulatorConfig config;

    // Parse global simulation settings
    if (j.contains("simulation")) {
        parse_simulation(j["simulation"], config);
    }

    // Parse per-component configurations
    if (j.contains("memory")) {
        config.memory_controller = parse_memory(j["memory"], config);
    }

    if (j.contains("dma")) {
        config.dma_engine = parse_dma(j["dma"], config);
    }

    if (j.contains("l3")) {
        config.l3_tile = parse_l3(j["l3"], config);
    }

    if (j.contains("compute")) {
        config.compute_fabric = parse_compute(j["compute"], config);
    }

    if (j.contains("noc")) {
        config.noc = parse_noc(j["noc"], config);
    }

    return config;
}

void SimulatorConfigParser::parse_simulation(const nlohmann::json& j,
                                              SimulatorConfig& config) {
    if (j.contains("default_fidelity")) {
        config.default_fidelity = parse_fidelity(j["default_fidelity"].get<std::string>());
    }

    if (j.contains("default_verification")) {
        config.default_verification = parse_verification(
            j["default_verification"].get<std::string>());
    }

    config.default_tracing = get_with_default(j, "enable_tracing", false);

    // Component counts
    config.num_memory_controllers = get_with_default<uint32_t>(j, "num_memory_controllers", 1);
    config.num_dma_engines = get_with_default<uint32_t>(j, "num_dma_engines", 4);
    config.num_l3_tiles = get_with_default<uint32_t>(j, "num_l3_tiles", 4);
    config.num_l2_banks = get_with_default<uint32_t>(j, "num_l2_banks", 16);
    config.num_compute_tiles = get_with_default<uint32_t>(j, "num_compute_tiles", 16);
}

MemoryControllerConfig SimulatorConfigParser::parse_memory(const nlohmann::json& j,
                                                            const SimulatorConfig& defaults) {
    MemoryControllerConfig config;

    // Apply defaults
    config.fidelity = defaults.default_fidelity;
    config.verification = defaults.default_verification;
    config.enable_tracing = defaults.default_tracing;

    // Override with specific values
    if (j.contains("fidelity")) {
        config.fidelity = parse_fidelity(j["fidelity"].get<std::string>());
    }

    if (j.contains("technology")) {
        config.technology = parse_memory_technology(j["technology"].get<std::string>());
    }

    if (j.contains("verification")) {
        config.verification = parse_verification(j["verification"].get<std::string>());
    }

    config.speed_mt_s = get_with_default<uint32_t>(j, "speed_mt_s", 6400);
    config.num_channels = get_with_default<uint8_t>(j, "channels", 1);
    config.banks_per_channel = get_with_default<uint8_t>(j, "banks_per_channel", 16);
    config.bank_groups = get_with_default<uint8_t>(j, "bank_groups", 4);
    config.queue_depth = get_with_default<uint32_t>(j, "queue_depth", 32);

    // Apply default timing based on technology
    config.timing = MemoryControllerConfig::get_default_timing(
        config.technology, config.speed_mt_s);

    return config;
}

DMAEngineConfig SimulatorConfigParser::parse_dma(const nlohmann::json& j,
                                                  const SimulatorConfig& defaults) {
    DMAEngineConfig config;

    config.fidelity = defaults.default_fidelity;
    config.verification = defaults.default_verification;
    config.enable_tracing = defaults.default_tracing;

    if (j.contains("fidelity")) {
        config.fidelity = parse_fidelity(j["fidelity"].get<std::string>());
    }

    config.num_channels = get_with_default<uint32_t>(j, "channels", 8);
    config.max_burst_size = get_with_default<uint32_t>(j, "max_burst_size", 256);
    config.bandwidth_gbps = get_with_default<uint32_t>(j, "bandwidth_gbps", 100);
    config.queue_depth_per_channel = get_with_default<uint32_t>(j, "queue_depth", 16);

    return config;
}

L3TileConfig SimulatorConfigParser::parse_l3(const nlohmann::json& j,
                                              const SimulatorConfig& defaults) {
    L3TileConfig config;

    config.fidelity = defaults.default_fidelity;
    config.verification = defaults.default_verification;
    config.enable_tracing = defaults.default_tracing;

    if (j.contains("fidelity")) {
        config.fidelity = parse_fidelity(j["fidelity"].get<std::string>());
    }

    config.capacity_kb = get_with_default<uint32_t>(j, "capacity_kb", 256);
    config.num_banks = get_with_default<uint8_t>(j, "num_banks", 8);
    config.num_ports = get_with_default<uint8_t>(j, "num_ports", 4);
    config.bank_width_bytes = get_with_default<uint32_t>(j, "bank_width_bytes", 64);
    config.access_latency_cycles = get_with_default<uint32_t>(j, "access_latency", 4);

    return config;
}

ComputeFabricConfig SimulatorConfigParser::parse_compute(const nlohmann::json& j,
                                                          const SimulatorConfig& defaults) {
    ComputeFabricConfig config;

    config.fidelity = defaults.default_fidelity;
    config.verification = defaults.default_verification;
    config.enable_tracing = defaults.default_tracing;

    if (j.contains("fidelity")) {
        config.fidelity = parse_fidelity(j["fidelity"].get<std::string>());
    }

    if (j.contains("technology")) {
        config.technology = parse_compute_technology(j["technology"].get<std::string>());
    }

    config.array_rows = get_with_default<uint32_t>(j, "array_rows", 16);
    config.array_cols = get_with_default<uint32_t>(j, "array_cols", 16);
    config.macs_per_cycle = get_with_default<uint32_t>(j, "macs_per_cycle", 256);
    config.pipeline_depth = get_with_default<uint32_t>(j, "pipeline_depth", 4);

    // Handle array_size as [rows, cols]
    if (j.contains("array_size") && j["array_size"].is_array() && j["array_size"].size() == 2) {
        config.array_rows = j["array_size"][0].get<uint32_t>();
        config.array_cols = j["array_size"][1].get<uint32_t>();
    }

    return config;
}

NoCConfig SimulatorConfigParser::parse_noc(const nlohmann::json& j,
                                            const SimulatorConfig& defaults) {
    NoCConfig config;

    config.fidelity = defaults.default_fidelity;
    config.verification = defaults.default_verification;
    config.enable_tracing = defaults.default_tracing;

    if (j.contains("fidelity")) {
        config.fidelity = parse_fidelity(j["fidelity"].get<std::string>());
    }

    if (j.contains("topology")) {
        config.technology = parse_interconnect_technology(j["topology"].get<std::string>());
    }

    config.mesh_rows = get_with_default<uint32_t>(j, "mesh_rows", 4);
    config.mesh_cols = get_with_default<uint32_t>(j, "mesh_cols", 4);
    config.link_bandwidth = get_with_default<uint32_t>(j, "link_bandwidth", 64);
    config.router_latency = get_with_default<uint32_t>(j, "router_latency", 1);

    // Handle dimensions as [rows, cols]
    if (j.contains("dimensions") && j["dimensions"].is_array() && j["dimensions"].size() == 2) {
        config.mesh_rows = j["dimensions"][0].get<uint32_t>();
        config.mesh_cols = j["dimensions"][1].get<uint32_t>();
    }

    return config;
}

// ============================================================================
// JSON Serialization
// ============================================================================

nlohmann::json SimulatorConfigParser::to_json(const SimulatorConfig& config) {
    nlohmann::json j;

    // Simulation section
    j["simulation"] = {
        {"default_fidelity", to_string(config.default_fidelity)},
        {"default_verification", to_string(config.default_verification)},
        {"enable_tracing", config.default_tracing},
        {"num_memory_controllers", config.num_memory_controllers},
        {"num_dma_engines", config.num_dma_engines},
        {"num_l3_tiles", config.num_l3_tiles},
        {"num_l2_banks", config.num_l2_banks},
        {"num_compute_tiles", config.num_compute_tiles}
    };

    // Memory section
    if (config.memory_controller) {
        const auto& mc = *config.memory_controller;
        j["memory"] = {
            {"fidelity", to_string(mc.fidelity)},
            {"technology", to_string(mc.technology)},
            {"speed_mt_s", mc.speed_mt_s},
            {"channels", mc.num_channels},
            {"banks_per_channel", mc.banks_per_channel},
            {"verification", to_string(mc.verification)}
        };
    }

    // DMA section
    if (config.dma_engine) {
        const auto& dma = *config.dma_engine;
        j["dma"] = {
            {"fidelity", to_string(dma.fidelity)},
            {"channels", dma.num_channels},
            {"bandwidth_gbps", dma.bandwidth_gbps}
        };
    }

    // L3 section
    if (config.l3_tile) {
        const auto& l3 = *config.l3_tile;
        j["l3"] = {
            {"fidelity", to_string(l3.fidelity)},
            {"capacity_kb", l3.capacity_kb},
            {"num_banks", l3.num_banks},
            {"num_ports", l3.num_ports}
        };
    }

    // Compute section
    if (config.compute_fabric) {
        const auto& cf = *config.compute_fabric;
        j["compute"] = {
            {"fidelity", to_string(cf.fidelity)},
            {"technology", to_string(cf.technology)},
            {"array_size", {cf.array_rows, cf.array_cols}},
            {"macs_per_cycle", cf.macs_per_cycle}
        };
    }

    // NoC section
    if (config.noc) {
        const auto& noc = *config.noc;
        j["noc"] = {
            {"fidelity", to_string(noc.fidelity)},
            {"topology", to_string(noc.technology)},
            {"dimensions", {noc.mesh_rows, noc.mesh_cols}}
        };
    }

    return j;
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

CLIFidelityOptions parse_cli_args(int argc, char* argv[]) {
    CLIFidelityOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // --fidelity=VALUE
        if (arg.find("--fidelity=") == 0) {
            std::string value = arg.substr(11);
            opts.global_fidelity = parse_fidelity(value);
        }
        // --memory-fidelity=VALUE
        else if (arg.find("--memory-fidelity=") == 0) {
            std::string value = arg.substr(18);
            opts.memory_fidelity = parse_fidelity(value);
        }
        // --compute-fidelity=VALUE
        else if (arg.find("--compute-fidelity=") == 0) {
            std::string value = arg.substr(19);
            opts.compute_fidelity = parse_fidelity(value);
        }
        // --noc-fidelity=VALUE
        else if (arg.find("--noc-fidelity=") == 0) {
            std::string value = arg.substr(15);
            opts.noc_fidelity = parse_fidelity(value);
        }
        // --config=PATH
        else if (arg.find("--config=") == 0) {
            opts.config_file = arg.substr(9);
        }
        // --tracing
        else if (arg == "--tracing") {
            opts.enable_tracing = true;
        }
        // --verification=VALUE
        else if (arg.find("--verification=") == 0) {
            std::string value = arg.substr(15);
            opts.verification = parse_verification(value);
        }
    }

    return opts;
}

void apply_cli_options(SimulatorConfig& config, const CLIFidelityOptions& opts) {
    // Apply global fidelity
    if (opts.global_fidelity) {
        config.set_fidelity(*opts.global_fidelity);
    }

    // Apply component-specific fidelity
    if (opts.memory_fidelity) {
        config.set_memory_fidelity(*opts.memory_fidelity);
    }

    if (opts.compute_fidelity) {
        config.set_compute_fidelity(*opts.compute_fidelity);
    }

    if (opts.noc_fidelity) {
        config.set_interconnect_fidelity(*opts.noc_fidelity);
    }

    // Apply tracing
    if (opts.enable_tracing) {
        config.enable_all_tracing(true);
    }

    // Apply verification
    if (opts.verification) {
        config.default_verification = *opts.verification;
    }
}

} // namespace sw::kpu::config

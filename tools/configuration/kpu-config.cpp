// ============================================================================
// tools/configuration/kpu-config.cpp
// Configuration management tool for KPU Simulator
//
// Commands:
//   generate - Generate a default configuration file
//   validate - Validate a configuration file
//   show     - Display configuration in human-readable format
//   convert  - Convert between formats (future: JSON <-> YAML)
// ============================================================================

#include <sw/kpu/config/simulator_config_parser.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

using namespace sw::kpu;
using namespace sw::kpu::config;

// ============================================================================
// Help Text
// ============================================================================

void print_usage() {
    std::cout << R"(
KPU Simulator Configuration Tool

Usage: kpu-config <command> [options]

Commands:
  generate [options]      Generate a configuration file
  validate <file>         Validate a configuration file
  show <file>             Display configuration in human-readable format
  presets                 List available configuration presets

Options for 'generate':
  --output=<file>         Output file path (default: stdout)
  --preset=<name>         Use a preset configuration:
                            fast       - All components BEHAVIORAL
                            balanced   - Memory TRANSACTIONAL, others BEHAVIORAL
                            accurate   - All components CYCLE_ACCURATE
                            mixed      - Memory/NoC CYCLE_ACCURATE, compute BEHAVIORAL
  --fidelity=<level>      Set global fidelity (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE)
  --memory-tech=<tech>    Set memory technology (LPDDR5, HBM3, DDR5, etc.)
  --mesh=<rows>x<cols>    Set mesh dimensions (e.g., 4x4)
  --tracing               Enable tracing by default
  --pretty                Pretty-print JSON output (default: true)

Examples:
  kpu-config generate --preset=fast --output=config.json
  kpu-config generate --fidelity=TRANSACTIONAL --memory-tech=LPDDR5
  kpu-config validate config.json
  kpu-config show config.json
)";
}

// ============================================================================
// Preset Configurations
// ============================================================================

SimulatorConfig get_preset(const std::string& name) {
    SimulatorConfig config;

    if (name == "fast" || name == "behavioral") {
        // All behavioral - maximum simulation speed
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.set_fidelity(SimulationFidelity::BEHAVIORAL);
    }
    else if (name == "balanced" || name == "transactional") {
        // Memory transactional, compute behavioral
        config.default_fidelity = SimulationFidelity::TRANSACTIONAL;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::TRANSACTIONAL;
        mc.technology = MemoryTechnology::LPDDR5;
        config.memory_controller = mc;

        DMAEngineConfig dma;
        dma.fidelity = SimulationFidelity::TRANSACTIONAL;
        config.dma_engine = dma;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        config.compute_fabric = cf;
    }
    else if (name == "accurate" || name == "cycle_accurate") {
        // All cycle-accurate - maximum fidelity
        config.default_fidelity = SimulationFidelity::CYCLE_ACCURATE;
        config.default_verification = VerificationLevel::INVARIANTS;
        config.set_fidelity(SimulationFidelity::CYCLE_ACCURATE);

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        mc.technology = MemoryTechnology::LPDDR5;
        mc.verification = VerificationLevel::INVARIANTS;
        config.memory_controller = mc;
    }
    else if (name == "mixed") {
        // Mixed fidelity - accurate memory/NoC, fast compute
        config.default_fidelity = SimulationFidelity::TRANSACTIONAL;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        mc.technology = MemoryTechnology::LPDDR5;
        config.memory_controller = mc;

        DMAEngineConfig dma;
        dma.fidelity = SimulationFidelity::TRANSACTIONAL;
        config.dma_engine = dma;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        config.compute_fabric = cf;

        NoCConfig noc;
        noc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        noc.technology = InterconnectTechnology::MESH_2D;
        config.noc = noc;
    }
    else {
        throw ConfigParseError("Unknown preset: " + name);
    }

    return config;
}

void list_presets() {
    std::cout << R"(
Available Configuration Presets:

  fast        All components use BEHAVIORAL fidelity
              - Fastest simulation speed
              - Functional correctness only
              - Use for: CI/CD, unit tests, software bring-up

  balanced    Memory TRANSACTIONAL, compute BEHAVIORAL
              - Good balance of speed and accuracy
              - Queue-based memory timing
              - Use for: Architecture exploration, bottleneck analysis

  accurate    All components use CYCLE_ACCURATE fidelity
              - Maximum timing accuracy
              - Full protocol modeling
              - Use for: Performance validation, hardware correlation

  mixed       Memory/NoC CYCLE_ACCURATE, compute BEHAVIORAL
              - Focus on data movement timing
              - Fast compute for algorithm iteration
              - Use for: Memory subsystem analysis
)";
}

// ============================================================================
// Commands
// ============================================================================

int cmd_generate(int argc, char* argv[]) {
    SimulatorConfig config;
    std::string output_file;
    bool pretty = true;

    // Parse options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.find("--output=") == 0) {
            output_file = arg.substr(9);
        }
        else if (arg.find("--preset=") == 0) {
            std::string preset = arg.substr(9);
            config = get_preset(preset);
        }
        else if (arg.find("--fidelity=") == 0) {
            std::string value = arg.substr(11);
            config.set_fidelity(parse_fidelity(value));
        }
        else if (arg.find("--memory-tech=") == 0) {
            std::string value = arg.substr(14);
            if (!config.memory_controller) {
                config.memory_controller = MemoryControllerConfig{};
            }
            config.memory_controller->technology = parse_memory_technology(value);
        }
        else if (arg.find("--mesh=") == 0) {
            std::string value = arg.substr(7);
            size_t x_pos = value.find('x');
            if (x_pos != std::string::npos) {
                uint32_t rows = std::stoul(value.substr(0, x_pos));
                uint32_t cols = std::stoul(value.substr(x_pos + 1));
                if (!config.noc) {
                    config.noc = NoCConfig{};
                }
                config.noc->mesh_rows = rows;
                config.noc->mesh_cols = cols;
            }
        }
        else if (arg == "--tracing") {
            config.enable_all_tracing(true);
        }
        else if (arg == "--no-pretty") {
            pretty = false;
        }
    }

    // Generate JSON
    nlohmann::json j = SimulatorConfigParser::to_json(config);

    // Output
    if (output_file.empty()) {
        if (pretty) {
            std::cout << j.dump(2) << std::endl;
        } else {
            std::cout << j.dump() << std::endl;
        }
    } else {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open output file: " << output_file << std::endl;
            return 1;
        }
        if (pretty) {
            file << j.dump(2) << std::endl;
        } else {
            file << j.dump() << std::endl;
        }
        std::cout << "Configuration written to: " << output_file << std::endl;
    }

    return 0;
}

int cmd_validate(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: No configuration file specified" << std::endl;
        std::cerr << "Usage: kpu-config validate <file>" << std::endl;
        return 1;
    }

    std::string file = argv[2];

    try {
        SimulatorConfig config = SimulatorConfigParser::parse_file(file);
        std::cout << "Configuration file is valid: " << file << std::endl;

        // Print summary
        std::cout << "\nConfiguration Summary:" << std::endl;
        std::cout << "  Default fidelity: " << to_string(config.default_fidelity) << std::endl;
        std::cout << "  Verification: " << to_string(config.default_verification) << std::endl;
        std::cout << "  Tracing: " << (config.default_tracing ? "enabled" : "disabled") << std::endl;

        if (config.memory_controller) {
            std::cout << "  Memory: " << to_string(config.memory_controller->fidelity)
                      << " (" << to_string(config.memory_controller->technology) << ")" << std::endl;
        }
        if (config.dma_engine) {
            std::cout << "  DMA: " << to_string(config.dma_engine->fidelity) << std::endl;
        }
        if (config.compute_fabric) {
            std::cout << "  Compute: " << to_string(config.compute_fabric->fidelity) << std::endl;
        }
        if (config.noc) {
            std::cout << "  NoC: " << to_string(config.noc->fidelity)
                      << " (" << config.noc->mesh_rows << "x" << config.noc->mesh_cols << ")" << std::endl;
        }

        return 0;
    } catch (const ConfigParseError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    }
}

int cmd_show(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: No configuration file specified" << std::endl;
        std::cerr << "Usage: kpu-config show <file>" << std::endl;
        return 1;
    }

    std::string file = argv[2];

    try {
        SimulatorConfig config = SimulatorConfigParser::parse_file(file);

        std::cout << "\n=== KPU Simulator Configuration ===" << std::endl;
        std::cout << "Source: " << file << std::endl;
        std::cout << std::endl;

        std::cout << "Global Settings:" << std::endl;
        std::cout << "  Default Fidelity:    " << to_string(config.default_fidelity) << std::endl;
        std::cout << "  Verification Level:  " << to_string(config.default_verification) << std::endl;
        std::cout << "  Tracing Enabled:     " << (config.default_tracing ? "yes" : "no") << std::endl;
        std::cout << std::endl;

        std::cout << "Component Counts:" << std::endl;
        std::cout << "  Memory Controllers:  " << config.num_memory_controllers << std::endl;
        std::cout << "  DMA Engines:         " << config.num_dma_engines << std::endl;
        std::cout << "  L3 Tiles:            " << config.num_l3_tiles << std::endl;
        std::cout << "  L2 Banks:            " << config.num_l2_banks << std::endl;
        std::cout << "  Compute Tiles:       " << config.num_compute_tiles << std::endl;
        std::cout << std::endl;

        if (config.memory_controller) {
            const auto& mc = *config.memory_controller;
            std::cout << "Memory Controller:" << std::endl;
            std::cout << "  Fidelity:            " << to_string(mc.fidelity) << std::endl;
            std::cout << "  Technology:          " << to_string(mc.technology) << std::endl;
            std::cout << "  Speed:               " << mc.speed_mt_s << " MT/s" << std::endl;
            std::cout << "  Channels:            " << static_cast<int>(mc.num_channels) << std::endl;
            std::cout << "  Banks per Channel:   " << static_cast<int>(mc.banks_per_channel) << std::endl;
            std::cout << std::endl;
        }

        if (config.dma_engine) {
            const auto& dma = *config.dma_engine;
            std::cout << "DMA Engine:" << std::endl;
            std::cout << "  Fidelity:            " << to_string(dma.fidelity) << std::endl;
            std::cout << "  Channels:            " << dma.num_channels << std::endl;
            std::cout << "  Bandwidth:           " << dma.bandwidth_gbps << " GB/s" << std::endl;
            std::cout << std::endl;
        }

        if (config.l3_tile) {
            const auto& l3 = *config.l3_tile;
            std::cout << "L3 Tile:" << std::endl;
            std::cout << "  Fidelity:            " << to_string(l3.fidelity) << std::endl;
            std::cout << "  Capacity:            " << l3.capacity_kb << " KB" << std::endl;
            std::cout << "  Banks:               " << static_cast<int>(l3.num_banks) << std::endl;
            std::cout << "  Ports:               " << static_cast<int>(l3.num_ports) << std::endl;
            std::cout << std::endl;
        }

        if (config.compute_fabric) {
            const auto& cf = *config.compute_fabric;
            std::cout << "Compute Fabric:" << std::endl;
            std::cout << "  Fidelity:            " << to_string(cf.fidelity) << std::endl;
            std::cout << "  Technology:          " << to_string(cf.technology) << std::endl;
            std::cout << "  Array Size:          " << cf.array_rows << " x " << cf.array_cols << std::endl;
            std::cout << "  MACs/cycle:          " << cf.macs_per_cycle << std::endl;
            std::cout << std::endl;
        }

        if (config.noc) {
            const auto& noc = *config.noc;
            std::cout << "Network-on-Chip:" << std::endl;
            std::cout << "  Fidelity:            " << to_string(noc.fidelity) << std::endl;
            std::cout << "  Topology:            " << to_string(noc.technology) << std::endl;
            std::cout << "  Dimensions:          " << noc.mesh_rows << " x " << noc.mesh_cols << std::endl;
            std::cout << "  Link Bandwidth:      " << noc.link_bandwidth << " bytes/cycle" << std::endl;
            std::cout << std::endl;
        }

        return 0;
    } catch (const ConfigParseError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    try {
        if (command == "generate") {
            return cmd_generate(argc, argv);
        }
        else if (command == "validate") {
            return cmd_validate(argc, argv);
        }
        else if (command == "show") {
            return cmd_show(argc, argv);
        }
        else if (command == "presets") {
            list_presets();
            return 0;
        }
        else if (command == "-h" || command == "--help" || command == "help") {
            print_usage();
            return 0;
        }
        else {
            std::cerr << "Unknown command: " << command << std::endl;
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

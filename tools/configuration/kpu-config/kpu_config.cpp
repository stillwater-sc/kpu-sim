/**
 * @file kpu_config.cpp
 * @brief KPU Configuration Tool - Manage KPU configuration files
 *
 * Commands:
 *   validate <file>           Validate a configuration file
 *   convert <file> -o <out>   Convert between YAML and JSON formats
 *   show <file>               Display configuration in formatted output
 *   generate <type> -o <out>  Generate template configuration
 *   get <file> <path>         Query a specific configuration value
 *   diff <file1> <file2>      Compare two configuration files
 */

#include <sw/kpu/kpu_config_loader.hpp>
#include <sw/kpu/kpu_simulator.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace sw::kpu;

// =========================================
// Utility Functions
// =========================================

void print_usage(const char* program) {
    std::cout << "KPU Configuration Tool - Manage KPU configuration files\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  validate <file>              Validate a configuration file\n";
    std::cout << "  convert <file> -o <output>   Convert between YAML and JSON\n";
    std::cout << "  show <file>                  Display formatted configuration\n";
    std::cout << "  generate <type> [-o <file>]  Generate template (minimal|edge_ai|embodied_ai|datacenter)\n";
    std::cout << "  get <file> <path>            Query config value (e.g., external_memory.bank_count)\n";
    std::cout << "  diff <file1> <file2>         Compare two configurations\n";
    std::cout << "  list-templates               List available template types\n";
    std::cout << "  stats-table                  Display statistics table for all configurations\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --output <file>   Output file\n";
    std::cout << "  -f, --format <fmt>    Output format: yaml, json (default: auto from extension)\n";
    std::cout << "  -q, --quiet           Quiet mode (minimal output)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " validate configs/kpu/my_config.yaml\n";
    std::cout << "  " << program << " convert config.yaml -o config.json\n";
    std::cout << "  " << program << " generate embodied_ai -o robot.yaml\n";
    std::cout << "  " << program << " get config.yaml external_memory.bank_count\n";
    std::cout << "  " << program << " stats-table\n";
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string get_format_from_extension(const std::string& filename) {
    if (ends_with(filename, ".yaml") || ends_with(filename, ".yml")) {
        return "yaml";
    } else if (ends_with(filename, ".json")) {
        return "json";
    }
    return "unknown";
}

// =========================================
// Command: validate
// =========================================

int cmd_validate(const std::string& filename, bool quiet) {
    if (!quiet) {
        std::cout << "Validating: " << filename << "\n";
    }

    try {
        std::string format = get_format_from_extension(filename);
        KPUSimulator::Config config;

        if (format == "yaml") {
            config = KPUConfigLoader::load_yaml(filename);
        } else if (format == "json") {
            config = KPUConfigLoader::load_json(filename);
        } else {
            std::cerr << "Error: Unknown file format. Use .yaml, .yml, or .json extension.\n";
            return 1;
        }

        // Validate the configuration
        auto result = KPUConfigLoader::validate(config);
        if (!result.valid) {
            std::cerr << "Validation FAILED:\n";
            for (const auto& err : result.errors) {
                std::cerr << "  - " << err << "\n";
            }
            return 1;
        }
        if (!result.warnings.empty()) {
            std::cerr << "Warnings:\n";
            for (const auto& warn : result.warnings) {
                std::cerr << "  - " << warn << "\n";
            }
        }

        if (!quiet) {
            std::cout << "Validation PASSED\n";
            std::cout << "\nConfiguration summary:\n";
            std::cout << "  External memory: " << config.memory_bank_count << " banks x "
                      << config.memory_bank_capacity_mb << " MB\n";
            std::cout << "  L3 tiles:        " << config.l3_tile_count << " x "
                      << config.l3_tile_capacity_kb << " KB\n";
            std::cout << "  L2 banks:        " << config.l2_bank_count << " x "
                      << config.l2_bank_capacity_kb << " KB\n";
            std::cout << "  L1 buffers:      " << config.l1_buffer_count << " x "
                      << config.l1_buffer_capacity_kb << " KB\n";
            std::cout << "  Compute tiles:   " << config.compute_tile_count << "\n";
        } else {
            std::cout << "OK\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: convert
// =========================================

int cmd_convert(const std::string& input_file, const std::string& output_file,
                const std::string& output_format, bool quiet) {
    if (output_file.empty()) {
        std::cerr << "Error: Output file required. Use -o <file>\n";
        return 1;
    }

    try {
        // Load input
        std::string input_fmt = get_format_from_extension(input_file);
        KPUSimulator::Config config;

        if (input_fmt == "yaml") {
            config = KPUConfigLoader::load_yaml(input_file);
        } else if (input_fmt == "json") {
            config = KPUConfigLoader::load_json(input_file);
        } else {
            std::cerr << "Error: Unknown input format. Use .yaml, .yml, or .json extension.\n";
            return 1;
        }

        // Determine output format
        std::string out_fmt = output_format;
        if (out_fmt.empty() || out_fmt == "auto") {
            out_fmt = get_format_from_extension(output_file);
        }

        if (out_fmt == "unknown") {
            std::cerr << "Error: Cannot determine output format. Use -f yaml|json or proper extension.\n";
            return 1;
        }

        // Save output
        if (out_fmt == "yaml") {
            KPUConfigLoader::save_yaml(config, output_file);
        } else if (out_fmt == "json") {
            KPUConfigLoader::save_json(config, output_file);
        }

        if (!quiet) {
            std::cout << "Converted: " << input_file << " (" << input_fmt << ") -> "
                      << output_file << " (" << out_fmt << ")\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: show
// =========================================

void print_separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int cmd_show(const std::string& filename, bool quiet) {
    try {
        std::string format = get_format_from_extension(filename);
        KPUSimulator::Config config;

        if (format == "yaml") {
            config = KPUConfigLoader::load_yaml(filename);
        } else if (format == "json") {
            config = KPUConfigLoader::load_json(filename);
        } else {
            std::cerr << "Error: Unknown file format.\n";
            return 1;
        }

        std::cout << "Configuration: " << filename << "\n";
        std::cout << std::string(60, '=') << "\n";

        print_separator("Host Memory");
        std::cout << "  region_count:        " << config.host_memory_region_count << "\n";
        std::cout << "  region_capacity_mb:  " << config.host_memory_region_capacity_mb << " MB\n";
        std::cout << "  bandwidth_gbps:      " << config.host_memory_bandwidth_gbps << " GB/s\n";

        print_separator("External Memory (KPU Local)");
        std::cout << "  bank_count:          " << config.memory_bank_count << "\n";
        std::cout << "  bank_capacity_mb:    " << config.memory_bank_capacity_mb << " MB\n";
        std::cout << "  bandwidth_gbps:      " << config.memory_bandwidth_gbps << " GB/s\n";
        std::cout << "  total_capacity:      " << (config.memory_bank_count * config.memory_bank_capacity_mb) << " MB\n";

        print_separator("Memory Controller");
        std::cout << "  controller_count:    " << config.memory_controller_count << "\n";
        std::cout << "  page_buffer_count:   " << config.page_buffer_count << "\n";
        std::cout << "  page_buffer_kb:      " << config.page_buffer_capacity_kb << " KB\n";

        print_separator("L3 Global Buffer");
        std::cout << "  tile_count:          " << config.l3_tile_count << "\n";
        std::cout << "  tile_capacity_kb:    " << config.l3_tile_capacity_kb << " KB\n";
        std::cout << "  total_capacity:      " << (config.l3_tile_count * config.l3_tile_capacity_kb) << " KB\n";

        print_separator("L2 Tile Buffer");
        std::cout << "  bank_count:          " << config.l2_bank_count << "\n";
        std::cout << "  bank_capacity_kb:    " << config.l2_bank_capacity_kb << " KB\n";
        std::cout << "  total_capacity:      " << (config.l2_bank_count * config.l2_bank_capacity_kb) << " KB\n";

        print_separator("L1 Streaming Buffer");
        std::cout << "  buffer_count:        " << config.l1_buffer_count << "\n";
        std::cout << "  buffer_capacity_kb:  " << config.l1_buffer_capacity_kb << " KB\n";
        std::cout << "  total_capacity:      " << (config.l1_buffer_count * config.l1_buffer_capacity_kb) << " KB\n";

        print_separator("Data Movement");
        std::cout << "  dma_engine_count:    " << config.dma_engine_count << "\n";
        std::cout << "  block_mover_count:   " << config.block_mover_count << "\n";
        std::cout << "  streamer_count:      " << config.streamer_count << "\n";

        print_separator("Compute Fabric");
        std::cout << "  compute_tile_count:  " << config.compute_tile_count << "\n";
        std::cout << "  processor_rows:      " << config.processor_array_rows << "\n";
        std::cout << "  processor_cols:      " << config.processor_array_cols << "\n";
        std::cout << "  processor_topology:  " << topology_to_string(config.processor_array_topology) << "\n";
        std::cout << "  total_MACs:          " << (config.compute_tile_count *
                     config.processor_array_rows * config.processor_array_cols) << "\n";

        // Show L1 buffer derivation
        Size expected_l1 = compute_l1_buffer_count(
            config.processor_array_topology,
            config.processor_array_rows,
            config.processor_array_cols,
            config.compute_tile_count
        );
        std::cout << "  derived_l1_buffers:  " << expected_l1;
        if (config.l1_buffer_count != expected_l1) {
            std::cout << " (configured: " << config.l1_buffer_count << " - MISMATCH!)";
        }
        std::cout << "\n";

        print_separator("Memory Map (Computed)");
        // Create a temporary simulator to get address info
        KPUSimulator sim(config);
        std::cout << std::hex << std::setfill('0');
        std::cout << "  host_memory_base:    0x" << std::setw(12) << sim.get_host_memory_region_base(0) << "\n";
        std::cout << "  external_mem_base:   0x" << std::setw(12) << sim.get_external_bank_base(0) << "\n";
        std::cout << "  l3_tile_base:        0x" << std::setw(12) << sim.get_l3_tile_base(0) << "\n";
        std::cout << "  l2_bank_base:        0x" << std::setw(12) << sim.get_l2_bank_base(0) << "\n";
        std::cout << "  l1_buffer_base:      0x" << std::setw(12) << sim.get_l1_buffer_base(0) << "\n";
        std::cout << std::dec;

        std::cout << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: generate
// =========================================

int cmd_generate(const std::string& template_type, const std::string& output_file,
                 const std::string& output_format, bool quiet) {
    try {
        KPUSimulator::Config config;

        if (template_type == "minimal") {
            config = KPUConfigLoader::create_minimal();
        } else if (template_type == "edge_ai" || template_type == "edge") {
            config = KPUConfigLoader::create_edge_ai();
        } else if (template_type == "embodied_ai" || template_type == "embodied") {
            config = KPUConfigLoader::create_embodied_ai();
        } else if (template_type == "datacenter" || template_type == "dc") {
            config = KPUConfigLoader::create_datacenter();
        } else {
            std::cerr << "Error: Unknown template type: " << template_type << "\n";
            std::cerr << "Available templates: minimal, edge_ai, embodied_ai, datacenter\n";
            return 1;
        }

        // Determine output format and destination
        std::string out_fmt = output_format;
        std::string out_file = output_file;

        if (out_file.empty()) {
            // Output to stdout
            out_fmt = out_fmt.empty() ? "yaml" : out_fmt;

            if (out_fmt == "yaml") {
                // Generate YAML to stdout
                std::cout << "# KPU Configuration - " << template_type << "\n";
                std::cout << "# Generated by kpu-config\n\n";
                std::cout << "name: \"" << template_type << "\"\n";
                std::cout << "description: \"Generated " << template_type << " configuration\"\n\n";

                std::cout << "host_memory:\n";
                std::cout << "  region_count: " << config.host_memory_region_count << "\n";
                std::cout << "  region_capacity_mb: " << config.host_memory_region_capacity_mb << "\n";
                std::cout << "  bandwidth_gbps: " << config.host_memory_bandwidth_gbps << "\n\n";

                std::cout << "external_memory:\n";
                std::cout << "  bank_count: " << config.memory_bank_count << "\n";
                std::cout << "  bank_capacity_mb: " << config.memory_bank_capacity_mb << "\n";
                std::cout << "  bandwidth_gbps: " << config.memory_bandwidth_gbps << "\n\n";

                std::cout << "memory_controller:\n";
                std::cout << "  controller_count: " << config.memory_controller_count << "\n";
                std::cout << "  page_buffer_count: " << config.page_buffer_count << "\n";
                std::cout << "  page_buffer_capacity_kb: " << config.page_buffer_capacity_kb << "\n\n";

                std::cout << "on_chip_memory:\n";
                std::cout << "  l3:\n";
                std::cout << "    tile_count: " << config.l3_tile_count << "\n";
                std::cout << "    tile_capacity_kb: " << config.l3_tile_capacity_kb << "\n";
                std::cout << "  l2:\n";
                std::cout << "    bank_count: " << config.l2_bank_count << "\n";
                std::cout << "    bank_capacity_kb: " << config.l2_bank_capacity_kb << "\n";
                std::cout << "  l1:\n";
                std::cout << "    buffer_count: " << config.l1_buffer_count << "\n";
                std::cout << "    buffer_capacity_kb: " << config.l1_buffer_capacity_kb << "\n\n";

                std::cout << "data_movement:\n";
                std::cout << "  dma_engine_count: " << config.dma_engine_count << "\n";
                std::cout << "  block_mover_count: " << config.block_mover_count << "\n";
                std::cout << "  streamer_count: " << config.streamer_count << "\n\n";

                std::cout << "compute:\n";
                std::cout << "  tile_count: " << config.compute_tile_count << "\n";
                std::cout << "  processor_array:\n";
                std::cout << "    rows: " << config.processor_array_rows << "\n";
                std::cout << "    cols: " << config.processor_array_cols << "\n";
                std::cout << "  systolic_mode: true\n";
            } else {
                // JSON to stdout - use the save function to a temp then cat
                KPUConfigLoader::save_json(config, "/tmp/kpu_config_temp.json");
                std::ifstream f("/tmp/kpu_config_temp.json");
                std::cout << f.rdbuf();
            }
        } else {
            // Output to file
            if (out_fmt.empty() || out_fmt == "auto") {
                out_fmt = get_format_from_extension(out_file);
            }

            if (out_fmt == "yaml") {
                KPUConfigLoader::save_yaml(config, out_file);
            } else if (out_fmt == "json") {
                KPUConfigLoader::save_json(config, out_file);
            } else {
                std::cerr << "Error: Unknown output format. Use .yaml or .json extension.\n";
                return 1;
            }

            if (!quiet) {
                std::cout << "Generated " << template_type << " configuration: " << out_file << "\n";
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: get
// =========================================

int cmd_get(const std::string& filename, const std::string& path, bool quiet) {
    try {
        std::string format = get_format_from_extension(filename);
        KPUSimulator::Config config;

        if (format == "yaml") {
            config = KPUConfigLoader::load_yaml(filename);
        } else if (format == "json") {
            config = KPUConfigLoader::load_json(filename);
        } else {
            std::cerr << "Error: Unknown file format.\n";
            return 1;
        }

        // Parse the path and get the value
        // Supported paths:
        //   host_memory.region_count, host_memory.region_capacity_mb, host_memory.bandwidth_gbps
        //   external_memory.bank_count, external_memory.bank_capacity_mb, external_memory.bandwidth_gbps
        //   l3.tile_count, l3.tile_capacity_kb
        //   l2.bank_count, l2.bank_capacity_kb
        //   l1.buffer_count, l1.buffer_capacity_kb
        //   data_movement.dma_engine_count, data_movement.block_mover_count, data_movement.streamer_count
        //   compute.tile_count, compute.systolic_rows, compute.systolic_cols

        std::string value;
        bool found = true;

        if (path == "host_memory.region_count") {
            value = std::to_string(config.host_memory_region_count);
        } else if (path == "host_memory.region_capacity_mb") {
            value = std::to_string(config.host_memory_region_capacity_mb);
        } else if (path == "host_memory.bandwidth_gbps") {
            value = std::to_string(config.host_memory_bandwidth_gbps);
        } else if (path == "external_memory.bank_count") {
            value = std::to_string(config.memory_bank_count);
        } else if (path == "external_memory.bank_capacity_mb") {
            value = std::to_string(config.memory_bank_capacity_mb);
        } else if (path == "external_memory.bandwidth_gbps") {
            value = std::to_string(config.memory_bandwidth_gbps);
        } else if (path == "l3.tile_count") {
            value = std::to_string(config.l3_tile_count);
        } else if (path == "l3.tile_capacity_kb") {
            value = std::to_string(config.l3_tile_capacity_kb);
        } else if (path == "l2.bank_count") {
            value = std::to_string(config.l2_bank_count);
        } else if (path == "l2.bank_capacity_kb") {
            value = std::to_string(config.l2_bank_capacity_kb);
        } else if (path == "l1.buffer_count") {
            value = std::to_string(config.l1_buffer_count);
        } else if (path == "l1.buffer_capacity_kb") {
            value = std::to_string(config.l1_buffer_capacity_kb);
        } else if (path == "data_movement.dma_engine_count") {
            value = std::to_string(config.dma_engine_count);
        } else if (path == "data_movement.block_mover_count") {
            value = std::to_string(config.block_mover_count);
        } else if (path == "data_movement.streamer_count") {
            value = std::to_string(config.streamer_count);
        } else if (path == "compute.tile_count") {
            value = std::to_string(config.compute_tile_count);
        } else if (path == "compute.processor_rows") {
            value = std::to_string(config.processor_array_rows);
        } else if (path == "compute.processor_cols") {
            value = std::to_string(config.processor_array_cols);
        } else {
            found = false;
        }

        if (!found) {
            std::cerr << "Error: Unknown config path: " << path << "\n";
            std::cerr << "\nAvailable paths:\n";
            std::cerr << "  host_memory.{region_count,region_capacity_mb,bandwidth_gbps}\n";
            std::cerr << "  external_memory.{bank_count,bank_capacity_mb,bandwidth_gbps}\n";
            std::cerr << "  l3.{tile_count,tile_capacity_kb}\n";
            std::cerr << "  l2.{bank_count,bank_capacity_kb}\n";
            std::cerr << "  l1.{buffer_count,buffer_capacity_kb}\n";
            std::cerr << "  data_movement.{dma_engine_count,block_mover_count,streamer_count}\n";
            std::cerr << "  compute.{tile_count,processor_rows,processor_cols}\n";
            return 1;
        }

        std::cout << value << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: diff
// =========================================

int cmd_diff(const std::string& file1, const std::string& file2, bool quiet) {
    try {
        // Load both configs
        auto load_config = [](const std::string& file) {
            std::string fmt = get_format_from_extension(file);
            if (fmt == "yaml") {
                return KPUConfigLoader::load_yaml(file);
            } else if (fmt == "json") {
                return KPUConfigLoader::load_json(file);
            } else {
                throw std::runtime_error("Unknown format for: " + file);
            }
        };

        auto c1 = load_config(file1);
        auto c2 = load_config(file2);

        std::cout << "Comparing:\n";
        std::cout << "  [1] " << file1 << "\n";
        std::cout << "  [2] " << file2 << "\n\n";

        bool identical = true;
        auto compare = [&](const std::string& name, auto v1, auto v2) {
            if (v1 != v2) {
                std::cout << "  " << std::left << std::setw(35) << name
                          << std::right << std::setw(12) << v1
                          << " -> " << std::setw(12) << v2 << "\n";
                identical = false;
            }
        };

        std::cout << "Differences (config1 -> config2):\n";
        std::cout << std::string(65, '-') << "\n";

        compare("host_memory.region_count", c1.host_memory_region_count, c2.host_memory_region_count);
        compare("host_memory.region_capacity_mb", c1.host_memory_region_capacity_mb, c2.host_memory_region_capacity_mb);
        compare("host_memory.bandwidth_gbps", c1.host_memory_bandwidth_gbps, c2.host_memory_bandwidth_gbps);

        compare("external_memory.bank_count", c1.memory_bank_count, c2.memory_bank_count);
        compare("external_memory.bank_capacity_mb", c1.memory_bank_capacity_mb, c2.memory_bank_capacity_mb);
        compare("external_memory.bandwidth_gbps", c1.memory_bandwidth_gbps, c2.memory_bandwidth_gbps);

        compare("l3.tile_count", c1.l3_tile_count, c2.l3_tile_count);
        compare("l3.tile_capacity_kb", c1.l3_tile_capacity_kb, c2.l3_tile_capacity_kb);

        compare("l2.bank_count", c1.l2_bank_count, c2.l2_bank_count);
        compare("l2.bank_capacity_kb", c1.l2_bank_capacity_kb, c2.l2_bank_capacity_kb);

        compare("l1.buffer_count", c1.l1_buffer_count, c2.l1_buffer_count);
        compare("l1.buffer_capacity_kb", c1.l1_buffer_capacity_kb, c2.l1_buffer_capacity_kb);

        compare("data_movement.dma_engine_count", c1.dma_engine_count, c2.dma_engine_count);
        compare("data_movement.block_mover_count", c1.block_mover_count, c2.block_mover_count);
        compare("data_movement.streamer_count", c1.streamer_count, c2.streamer_count);

        compare("compute.tile_count", c1.compute_tile_count, c2.compute_tile_count);
        compare("compute.processor_rows", c1.processor_array_rows, c2.processor_array_rows);
        compare("compute.processor_cols", c1.processor_array_cols, c2.processor_array_cols);

        if (identical) {
            std::cout << "  (configurations are identical)\n";
        }

        std::cout << "\n";
        return identical ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// =========================================
// Command: list-templates
// =========================================

int cmd_list_templates() {
    std::cout << "Available configuration templates:\n\n";

    std::cout << "  minimal      Smallest viable KPU for testing and development\n";
    std::cout << "               - 1 compute tile (8x8 rectangular systolic)\n";
    std::cout << "               - 1 external channel (256 MB, LPDDR4x)\n";
    std::cout << "               - 1 L3, 4 L2, 64 L1 buffers (derived: 4*(8+8)*1)\n\n";

    std::cout << "  edge_ai      Dual-tile configuration for edge AI inference\n";
    std::cout << "               - 2 compute tiles (16x16 rectangular systolic each)\n";
    std::cout << "               - 4 external channels (256 MB each, LPDDR5, 64-bit)\n";
    std::cout << "               - 2 L3, 16 L2, 256 L1 buffers (derived: 4*(16+16)*2)\n";
    std::cout << "               - Power-efficient 48 GB/s bandwidth\n\n";

    std::cout << "  embodied_ai  64-tile configuration for robotics/autonomous systems\n";
    std::cout << "               - 64 compute tiles (24x24 rectangular systolic each)\n";
    std::cout << "               - 8 external channels (512 MB each, LPDDR5)\n";
    std::cout << "               - Jetson Orin style: 256-bit, 200 GB/s, power-efficient\n";
    std::cout << "               - 64 L3, 1024 L2, 12288 L1 buffers (derived: 4*(24+24)*64)\n\n";

    std::cout << "  datacenter   256-tile configuration for datacenter-scale AI\n";
    std::cout << "               - 256 compute tiles (32x32 rectangular systolic each)\n";
    std::cout << "               - 6 external banks (4 GB each, HBM3)\n";
    std::cout << "               - 256 L3, 4096 L2, 65536 L1 buffers (derived: 4*(32+32)*256)\n";
    std::cout << "               - 4.8 TB/s memory bandwidth\n\n";

    std::cout << "L1 Buffer Derivation:\n";
    std::cout << "  L1 streaming buffers are derived from the processor array configuration.\n";
    std::cout << "  For rectangular arrays: L1_count = 4 * (rows + cols) * compute_tiles\n";
    std::cout << "  Each edge (TOP/BOTTOM/LEFT/RIGHT) has ingress + egress buffers.\n\n";

    std::cout << "Generate a template:\n";
    std::cout << "  kpu-config generate minimal -o my_config.yaml\n";
    std::cout << "  kpu-config generate embodied_ai -o robot.yaml\n";
    std::cout << "  kpu-config generate datacenter -o hpc_config.json\n";

    return 0;
}

// =========================================
// Command: stats-table
// =========================================

struct ConfigStats {
    std::string name;
    Size tiles;
    Size array_rows;
    Size array_cols;
    Size l3_count;
    Size l2_count;
    Size l1_count;
    Size l3_kb;
    Size l2_kb;
    Size l1_kb;
    Size clock_mhz;
    double ops_per_cycle_per_tile;
    double ops_per_cycle_total;
    double peak_tops;
    double l3_total_kb;
    double l2_total_kb;
    double l1_total_kb;
    double tdp_watts;
    double mem_bw_gbps;
};

ConfigStats compute_stats(const std::string& name, const KPUSimulator::Config& config, Size clock_mhz) {
    ConfigStats s;
    s.name = name;
    s.tiles = config.compute_tile_count;
    s.array_rows = config.processor_array_rows;
    s.array_cols = config.processor_array_cols;
    s.l3_count = config.l3_tile_count;
    s.l2_count = config.l2_bank_count;
    s.l1_count = config.l1_buffer_count;
    s.l3_kb = config.l3_tile_capacity_kb;
    s.l2_kb = config.l2_bank_capacity_kb;
    s.l1_kb = config.l1_buffer_capacity_kb;
    s.clock_mhz = clock_mhz;

    // Ops per cycle: 2 ops per PE per cycle (FMA = multiply + add)
    s.ops_per_cycle_per_tile = 2.0 * s.array_rows * s.array_cols;
    s.ops_per_cycle_total = s.ops_per_cycle_per_tile * s.tiles;

    // Peak throughput in TOPS: ops/cycle * clock_freq_GHz
    double clock_ghz = clock_mhz / 1000.0;
    s.peak_tops = (s.ops_per_cycle_total * clock_ghz) / 1000.0;  // Tera = 10^12

    // Total storage in KB
    s.l3_total_kb = static_cast<double>(s.l3_count) * s.l3_kb;
    s.l2_total_kb = static_cast<double>(s.l2_count) * s.l2_kb;

    // L1 buffers are FIFOs with depth = array dimension, not the config capacity
    // Each L1 buffer = FIFO_depth × element_size = array_dim × 4 bytes
    Size fifo_depth = std::max(s.array_rows, s.array_cols);
    Size element_size = 4;  // 4 bytes for float32
    Size l1_buffer_bytes = fifo_depth * element_size;
    s.l1_total_kb = static_cast<double>(s.l1_count) * l1_buffer_bytes / 1024.0;

    // Memory bandwidth
    s.mem_bw_gbps = static_cast<double>(config.memory_bank_count) * config.memory_bandwidth_gbps;

    // TDP estimation (rough model based on typical accelerators)
    // ~0.5W per TOPS for compute + memory overhead
    double compute_power = s.peak_tops * 0.5;
    double memory_power = s.mem_bw_gbps * 0.015;  // ~15mW per GB/s
    double on_chip_power = (s.l3_total_kb + s.l2_total_kb + s.l1_total_kb) * 0.0001;  // ~0.1mW per KB
    s.tdp_watts = compute_power + memory_power + on_chip_power;

    return s;
}

std::string format_size_kb(double kb) {
    if (kb >= 1024 * 1024) {
        return std::to_string(static_cast<int>(kb / (1024 * 1024))) + " GB";
    } else if (kb >= 1024) {
        return std::to_string(static_cast<int>(kb / 1024)) + " MB";
    } else {
        return std::to_string(static_cast<int>(kb)) + " KB";
    }
}

int cmd_stats_table() {
    // Define clock frequencies for each config (typical values)
    std::vector<std::pair<std::string, std::pair<KPUSimulator::Config, Size>>> configs = {
        {"Minimal", {KPUConfigLoader::create_minimal(), 500}},
        {"Edge AI", {KPUConfigLoader::create_edge_ai(), 750}},
        {"Embodied AI", {KPUConfigLoader::create_embodied_ai(), 1000}},
        {"Datacenter", {KPUConfigLoader::create_datacenter(), 1500}}
    };

    std::vector<ConfigStats> stats;
    for (const auto& [name, cfg_clock] : configs) {
        stats.push_back(compute_stats(name, cfg_clock.first, cfg_clock.second));
    }

    // Print header
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                     KPU Configuration Statistics                                                          ║\n";
    std::cout << "╠══════════════╦═══════╦═══════╦═══════╦═══════╦═══════════╦═══════════╦═══════════╦════════════╦════════════╦══════════════╣\n";
    std::cout << "║ Config       ║ Tiles ║ Array ║ Clock ║ Ops/  ║ Ops/Cycle ║ Peak      ║ L3        ║ L2         ║ L1         ║ Est. TDP     ║\n";
    std::cout << "║              ║       ║       ║ (MHz) ║ Cycle ║ Total     ║ (TOPS)    ║ Storage   ║ Storage    ║ Storage    ║ (Watts)      ║\n";
    std::cout << "║              ║       ║       ║       ║ /Tile ║           ║           ║           ║            ║            ║              ║\n";
    std::cout << "╠══════════════╬═══════╬═══════╬═══════╬═══════╬═══════════╬═══════════╬═══════════╬════════════╬════════════╬══════════════╣\n";

    for (const auto& s : stats) {
        std::cout << "║ " << std::left << std::setw(12) << s.name << " ║";
        std::cout << std::right << std::setw(5) << s.tiles << "  ║";
        std::cout << std::setw(3) << s.array_rows << "x" << std::left << std::setw(3) << s.array_cols << "║";
        std::cout << std::right << std::setw(5) << s.clock_mhz << "  ║";
        std::cout << std::setw(5) << static_cast<int>(s.ops_per_cycle_per_tile) << "  ║";
        std::cout << std::setw(9) << static_cast<int>(s.ops_per_cycle_total) << "  ║";
        std::cout << std::setw(9) << std::fixed << std::setprecision(1) << s.peak_tops << "  ║";
        std::cout << std::setw(9) << format_size_kb(s.l3_total_kb) << "  ║";
        std::cout << std::setw(10) << format_size_kb(s.l2_total_kb) << "  ║";
        std::cout << std::setw(10) << format_size_kb(s.l1_total_kb) << "  ║";
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << s.tdp_watts << " W  ║\n";
    }

    std::cout << "╠══════════════╩═══════╩═══════╩═══════╩═══════╩═══════════╩═══════════╩═══════════╩════════════╩════════════╩══════════════╣\n";

    // Print additional details
    std::cout << "║                                                                                                                           ║\n";
    std::cout << "║  Notes:                                                                                                                   ║\n";
    std::cout << "║  • Ops/Cycle/Tile = 2 × rows × cols (FMA operations in systolic array)                                                    ║\n";
    std::cout << "║  • Peak TOPS = Ops/Cycle Total × Clock (GHz) / 1000                                                                       ║\n";
    std::cout << "║  • L1 buffers derived: 4 × (rows + cols) × tiles                                                                          ║\n";
    std::cout << "║  • TDP is estimated based on typical accelerator power profiles                                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n";

    // Print memory interface table
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                           Memory Interface Summary                                                 ║\n";
    std::cout << "╠══════════════╦══════════╦══════════╦═════════╦═════════╦══════════╦═════════╦══════════╦═══════════╦═══════════════╣\n";
    std::cout << "║ Config       ║ Mem Type ║ Channels ║ Ch.Width║ Total   ║ BW/Ch    ║ Total   ║ Power/Ch ║ Total     ║ Power/Pin     ║\n";
    std::cout << "║              ║          ║          ║ (bits)  ║ Pins    ║ (GB/s)   ║ BW(GB/s)║ (W)      ║ Power (W) ║ (mW)          ║\n";
    std::cout << "╠══════════════╬══════════╬══════════╬═════════╬═════════╬══════════╬═════════╬══════════╬═══════════╬═══════════════╣\n";

    // Memory interface specifications:
    // GDDR6:  32-bit channel, ~5W/channel
    // LPDDR5: 16-bit channel, ~0.8W/channel (low power)
    // GDDR6X: 32-bit channel, ~8W/channel
    // HBM3:   1024-bit per stack, ~10W/stack
    struct MemInfo {
        std::string name;
        std::string mem_type;
        int channels;
        int ch_width_bits;
        int bw_per_ch;      // GB/s
        double power_per_ch; // Watts
    };

    // LPDDR5 @ 6400 MT/s: ~12.8 GB/s per 16-bit, ~25.6 GB/s per 32-bit
    std::vector<MemInfo> mem_info = {
        {"Minimal",     "LPDDR4x", 1, 16,   25,  0.4},   // 25 GB/s total
        {"Edge AI",     "LPDDR5",  4, 16,   12,  0.8},   // 48 GB/s total (64-bit)
        {"Embodied AI", "LPDDR5",  8, 32,   25,  1.0},   // 200 GB/s total (256-bit, Jetson Orin)
        {"Datacenter",  "HBM3",    6, 1024, 800, 10.0}   // 4800 GB/s total
    };

    for (const auto& m : mem_info) {
        int total_pins = m.channels * m.ch_width_bits;
        int total_bw = m.channels * m.bw_per_ch;
        double total_power = m.channels * m.power_per_ch;
        double power_per_pin = (total_power * 1000.0) / total_pins;  // mW per pin

        // Column widths must match header: 14,10,10,9,9,10,9,10,11,15
        std::cout << "║ " << std::left << std::setw(12) << m.name << " ║";
        std::cout << " " << std::left << std::setw(8) << m.mem_type << " ║";
        std::cout << std::right << std::setw(9) << m.channels << " ║";
        std::cout << std::setw(8) << m.ch_width_bits << " ║";
        std::cout << std::setw(8) << total_pins << " ║";
        std::cout << std::setw(9) << m.bw_per_ch << " ║";
        std::cout << std::setw(8) << total_bw << " ║";
        std::cout << std::setw(9) << std::fixed << std::setprecision(1) << m.power_per_ch << " ║";
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << total_power << " ║";
        std::cout << std::setw(14) << std::fixed << std::setprecision(2) << power_per_pin << " ║\n";
    }

    std::cout << "╚══════════════╩══════════╩══════════╩═════════╩═════════╩══════════╩═════════╩══════════╩═══════════╩═══════════════╝\n\n";

    return 0;
}

// =========================================
// Main
// =========================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "-h" || command == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    // Parse global options
    bool quiet = false;
    std::string output_file;
    std::string output_format;
    std::vector<std::string> positional_args;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-f" || arg == "--format") && i + 1 < argc) {
            output_format = argv[++i];
        } else if (arg[0] != '-') {
            positional_args.push_back(arg);
        }
    }

    // Dispatch commands
    if (command == "validate") {
        if (positional_args.empty()) {
            std::cerr << "Error: validate requires a file argument\n";
            return 1;
        }
        return cmd_validate(positional_args[0], quiet);

    } else if (command == "convert") {
        if (positional_args.empty()) {
            std::cerr << "Error: convert requires an input file\n";
            return 1;
        }
        return cmd_convert(positional_args[0], output_file, output_format, quiet);

    } else if (command == "show") {
        if (positional_args.empty()) {
            std::cerr << "Error: show requires a file argument\n";
            return 1;
        }
        return cmd_show(positional_args[0], quiet);

    } else if (command == "generate") {
        if (positional_args.empty()) {
            std::cerr << "Error: generate requires a template type (minimal, edge_ai, datacenter)\n";
            return 1;
        }
        return cmd_generate(positional_args[0], output_file, output_format, quiet);

    } else if (command == "get") {
        if (positional_args.size() < 2) {
            std::cerr << "Error: get requires <file> <path>\n";
            return 1;
        }
        return cmd_get(positional_args[0], positional_args[1], quiet);

    } else if (command == "diff") {
        if (positional_args.size() < 2) {
            std::cerr << "Error: diff requires two files\n";
            return 1;
        }
        return cmd_diff(positional_args[0], positional_args[1], quiet);

    } else if (command == "list-templates") {
        return cmd_list_templates();

    } else if (command == "stats-table") {
        return cmd_stats_table();

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage(argv[0]);
        return 1;
    }
}

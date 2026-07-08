/**
 * @file main.cpp
 * @brief KPU Loader - Load and execute KPU binary programs
 *
 * Loads .kpubin binary files and executes them on the behavioral or
 * transactional KPU simulator.
 *
 * Usage:
 *   kpu-loader program.kpubin [options]
 *
 * This tool:
 * 1. Reads the .kpubin binary file
 * 2. Loads input tensors from files
 * 3. Executes the program on the specified simulator fidelity
 * 4. Writes output tensors to files
 * 5. Optionally exports trace for visualization
 */

#include <sw/kpu/isa/program_serializer.hpp>
#include <sw/kpu/isa/program_executor_interface.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/models/temporal/memory/l3_tile.hpp>
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>
#include <sw/kpu/models/temporal/memory/l1_buffer.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <iomanip>

void print_usage(const char* program_name) {
    std::cout << "KPU Loader v1.0 - Load and execute KPU binary programs\n\n";
    std::cout << "Usage: " << program_name << " PROGRAM.kpubin [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --fidelity <level>  Simulation fidelity: 'behavioral' (default) or 'transactional'\n";
    std::cout << "  --input-a FILE      Input tensor A (binary float32)\n";
    std::cout << "  --input-b FILE      Input tensor B (binary float32)\n";
    std::cout << "  --output-c FILE     Output tensor C (binary float32)\n";
    std::cout << "  --trace FILE        Export Chrome Trace to FILE (transactional only)\n";
    std::cout << "  --dry-run           Load and validate but don't execute\n";
    std::cout << "  --stats             Print execution statistics\n";
    std::cout << "  -v, --verbose       Verbose output\n";
    std::cout << "  -h, --help          Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " matmul.kpubin --dry-run\n";
    std::cout << "  " << program_name << " matmul.kpubin --input-a A.bin --input-b B.bin --output-c C.bin\n";
    std::cout << "  " << program_name << " matmul.kpubin --fidelity transactional --trace trace.json\n";
}

bool read_tensor_file(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return file.good();
}

bool write_tensor_file(const std::string& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return file.good();
}

void print_program_info(const sw::kpu::isa::DMProgram& prog, bool verbose) {
    std::cout << "Program: " << prog.name << "\n";
    std::cout << "Dimensions: M=" << prog.M << " N=" << prog.N << " K=" << prog.K << "\n";
    std::cout << "Tiling: Ti=" << prog.Ti << " Tj=" << prog.Tj << " Tk=" << prog.Tk << "\n";
    std::cout << "Instructions: " << prog.instructions.size() << "\n";
    std::cout << "  DMA: " << prog.num_dma_ops() << "\n";
    std::cout << "  BM:  " << prog.num_bm_ops() << "\n";
    std::cout << "  STR: " << prog.num_str_ops() << "\n";
    std::cout << "  Sync:" << prog.num_sync_ops() << "\n";

    if (verbose) {
        std::cout << "\nMemory layout:\n";
        std::cout << "  A base: 0x" << std::hex << prog.memory_map.a_base << std::dec << "\n";
        std::cout << "  B base: 0x" << std::hex << prog.memory_map.b_base << std::dec << "\n";
        std::cout << "  C base: 0x" << std::hex << prog.memory_map.c_base << std::dec << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string program_file;
    std::string input_a_file;
    std::string input_b_file;
    std::string output_c_file;
    std::string trace_file;
    std::string fidelity_str = "behavioral";
    bool dry_run = false;
    bool show_stats = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--fidelity") == 0 && i + 1 < argc) {
            fidelity_str = argv[++i];
        } else if (std::strcmp(argv[i], "--input-a") == 0 && i + 1 < argc) {
            input_a_file = argv[++i];
        } else if (std::strcmp(argv[i], "--input-b") == 0 && i + 1 < argc) {
            input_b_file = argv[++i];
        } else if (std::strcmp(argv[i], "--output-c") == 0 && i + 1 < argc) {
            output_c_file = argv[++i];
        } else if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--dry-run") == 0) {
            dry_run = true;
        } else if (std::strcmp(argv[i], "--stats") == 0) {
            show_stats = true;
        } else if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (argv[i][0] != '-') {
            program_file = argv[i];
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            return 1;
        }
    }

    if (program_file.empty()) {
        std::cerr << "Error: No program file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    // Parse fidelity
    sw::kpu::SimulationFidelity fidelity;
    if (fidelity_str == "behavioral") {
        fidelity = sw::kpu::SimulationFidelity::BEHAVIORAL;
    } else if (fidelity_str == "transactional") {
        fidelity = sw::kpu::SimulationFidelity::TRANSACTIONAL;
    } else {
        std::cerr << "Error: Unknown fidelity level '" << fidelity_str << "'\n";
        std::cerr << "Valid options: 'behavioral', 'transactional'\n";
        return 1;
    }

    try {
        // Load program
        if (verbose) {
            std::cout << "Loading program: " << program_file << "\n";
        }

        sw::kpu::isa::ProgramSerializer serializer;
        sw::kpu::isa::DMProgram program;

        // Detect format from extension
        std::string fmt = sw::kpu::isa::ProgramSerializer::detect_format(program_file);
        if (fmt == "json") {
            program = serializer.load_json(program_file);
        } else {
            program = serializer.load(program_file);
        }

        print_program_info(program, verbose);

        if (dry_run) {
            std::cout << "\nDry run - not executing\n";
            return 0;
        }

        // Calculate memory sizes
        using sw::kpu::Size;
        using sw::kpu::Address;
        Size a_size = program.M * program.K;
        Size b_size = program.K * program.N;
        Size c_size = program.M * program.N;
        Size total_size = (a_size + b_size + c_size) * sizeof(float);

        // Set up memory bases if not specified in program
        Address a_base = program.memory_map.a_base;
        Address b_base = program.memory_map.b_base;
        Address c_base = program.memory_map.c_base;

        // If bases are 0, use default layout
        if (a_base == 0 && b_base == 0 && c_base == 0) {
            a_base = 0;
            b_base = a_size * sizeof(float);
            c_base = b_base + b_size * sizeof(float);
        }

        // Create hardware context
        Size ext_mem_kb = (total_size + 1023) / 1024 + 1;  // Round up to KB + buffer
        sw::kpu::ExternalMemory ext_mem(static_cast<uint32_t>(ext_mem_kb));
        std::vector<sw::kpu::L3Tile> l3_tiles;
        std::vector<sw::kpu::L2Bank> l2_banks;
        std::vector<sw::kpu::L1Buffer> l1_buffers;

        // Create appropriate number of components based on program needs
        for (int i = 0; i < 4; ++i) l3_tiles.emplace_back(i, 256);  // 256KB L3 tiles
        for (int i = 0; i < 8; ++i) l2_banks.emplace_back(i, 128);  // 128KB L2 banks
        for (int i = 0; i < 2; ++i) l1_buffers.emplace_back(i, 64 * 1024); // 64KB L1 buffers

        sw::kpu::isa::HardwareContext hw{ext_mem, l3_tiles, l2_banks, l1_buffers};

        // Load input tensors
        std::vector<float> a_data(a_size);
        std::vector<float> b_data(b_size);
        std::vector<float> c_data(c_size, 0.0f);

        if (!input_a_file.empty()) {
            if (verbose) std::cout << "Loading A from: " << input_a_file << "\n";
            if (!read_tensor_file(input_a_file, a_data)) {
                std::cerr << "Error: Failed to read input A from " << input_a_file << "\n";
                return 1;
            }
            if (a_data.size() != a_size) {
                std::cerr << "Error: Input A size mismatch (expected " << a_size
                          << ", got " << a_data.size() << ")\n";
                return 1;
            }
        } else {
            // Initialize with 1s for testing
            std::fill(a_data.begin(), a_data.end(), 1.0f);
        }

        if (!input_b_file.empty()) {
            if (verbose) std::cout << "Loading B from: " << input_b_file << "\n";
            if (!read_tensor_file(input_b_file, b_data)) {
                std::cerr << "Error: Failed to read input B from " << input_b_file << "\n";
                return 1;
            }
            if (b_data.size() != b_size) {
                std::cerr << "Error: Input B size mismatch (expected " << b_size
                          << ", got " << b_data.size() << ")\n";
                return 1;
            }
        } else {
            std::fill(b_data.begin(), b_data.end(), 1.0f);
        }

        // Write to external memory
        ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
        ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
        ext_mem.write(c_base, c_data.data(), c_data.size() * sizeof(float));

        // Create executor
        auto executor = sw::kpu::isa::create_program_executor(fidelity, hw);
        if (!executor) {
            std::cerr << "Error: Failed to create executor for fidelity " << fidelity_str << "\n";
            return 1;
        }

        if (verbose) {
            std::cout << "Executing with " << fidelity_str << " fidelity...\n";
        }

        // Execute
        auto start = std::chrono::high_resolution_clock::now();

        executor->load_program(program, a_base, b_base, c_base);
        bool halted = executor->run();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (!halted) {
            std::cerr << "Warning: Program did not halt normally\n";
        }

        // Read result
        ext_mem.read(c_base, c_data.data(), c_data.size() * sizeof(float));

        // Write output
        if (!output_c_file.empty()) {
            if (verbose) std::cout << "Writing C to: " << output_c_file << "\n";
            if (!write_tensor_file(output_c_file, c_data)) {
                std::cerr << "Error: Failed to write output C to " << output_c_file << "\n";
                return 1;
            }
        }

        // Export trace
        if (!trace_file.empty()) {
            if (executor->fidelity() == sw::kpu::SimulationFidelity::TRANSACTIONAL) {
                if (verbose) std::cout << "Exporting trace to: " << trace_file << "\n";
                executor->export_trace(trace_file);
            } else {
                std::cerr << "Warning: Trace export only available for transactional fidelity\n";
            }
        }

        // Print statistics
        std::cout << "\nExecution complete\n";
        std::cout << "Wall time: " << duration.count() << " us\n";
        if (executor->fidelity() == sw::kpu::SimulationFidelity::TRANSACTIONAL) {
            std::cout << "Simulated cycles: " << executor->total_cycles() << "\n";
        }

        if (show_stats) {
            std::cout << "\n" << executor->statistics_string();
        }

        // Quick result check
        if (verbose && c_size <= 256) {
            std::cout << "\nOutput C (first few elements):\n";
            for (size_t i = 0; i < std::min(c_size, Size(16)); ++i) {
                std::cout << "  C[" << i << "] = " << c_data[i] << "\n";
            }
        }

    } catch (const sw::kpu::isa::SerializationError& e) {
        std::cerr << "Serialization error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

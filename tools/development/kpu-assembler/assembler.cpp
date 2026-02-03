/**
 * @file assembler.cpp
 * @brief KPU Assembler - Assembly language assembler tool
 *
 * Assembles .kpuasm files into .kpubin binary format.
 *
 * Usage:
 *   kpu-assembler input.kpuasm -o output.kpubin
 *   kpu-assembler input.kpuasm -o output.kpujson --format json
 *   kpu-assembler input.kpuasm --print
 */

#include <sw/kpu/isa/assembler.hpp>
#include <sw/kpu/isa/program_serializer.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

void print_usage(const char* prog) {
    std::cerr << "KPU Assembler v1.0\n\n";
    std::cerr << "Usage: " << prog << " [options] <input.kpuasm>\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -o <file>       Output file (default: input with .kpubin extension)\n";
    std::cerr << "  --format <fmt>  Output format: 'binary' (default) or 'json'\n";
    std::cerr << "  --print         Print parsed program (no output file)\n";
    std::cerr << "  --stats         Print program statistics\n";
    std::cerr << "  -h, --help      Show this help message\n";
}

std::string get_opcode_name(sw::kpu::isa::DMOpcode op) {
    using namespace sw::kpu::isa;
    switch (op) {
        case DMOpcode::DMA_LOAD_TILE: return "DMA_LOAD_TILE";
        case DMOpcode::DMA_STORE_TILE: return "DMA_STORE_TILE";
        case DMOpcode::DMA_PREFETCH_TILE: return "DMA_PREFETCH_TILE";
        case DMOpcode::DMA_LOAD_GATHER: return "DMA_LOAD_GATHER";
        case DMOpcode::DMA_STORE_SCATTER: return "DMA_STORE_SCATTER";
        case DMOpcode::BM_MOVE_TILE: return "BM_MOVE_TILE";
        case DMOpcode::BM_TRANSPOSE_TILE: return "BM_TRANSPOSE_TILE";
        case DMOpcode::BM_WRITEBACK_TILE: return "BM_WRITEBACK_TILE";
        case DMOpcode::BM_RESHAPE_TILE: return "BM_RESHAPE_TILE";
        case DMOpcode::STR_FEED_ROWS: return "STR_FEED_ROWS";
        case DMOpcode::STR_FEED_COLS: return "STR_FEED_COLS";
        case DMOpcode::STR_DRAIN_OUTPUT: return "STR_DRAIN_OUTPUT";
        case DMOpcode::STR_BROADCAST_ROW: return "STR_BROADCAST_ROW";
        case DMOpcode::STR_BROADCAST_COL: return "STR_BROADCAST_COL";
        case DMOpcode::BARRIER: return "BARRIER";
        case DMOpcode::WAIT_DMA: return "WAIT_DMA";
        case DMOpcode::WAIT_BM: return "WAIT_BM";
        case DMOpcode::WAIT_STR: return "WAIT_STR";
        case DMOpcode::SIGNAL: return "SIGNAL";
        case DMOpcode::SET_TILE_SIZE: return "SET_TILE_SIZE";
        case DMOpcode::SET_BUFFER: return "SET_BUFFER";
        case DMOpcode::SET_STRIDE: return "SET_STRIDE";
        case DMOpcode::LOOP_BEGIN: return "LOOP_BEGIN";
        case DMOpcode::LOOP_END: return "LOOP_END";
        case DMOpcode::VE_ELEMENTWISE: return "VE_ELEMENTWISE";
        case DMOpcode::VE_REDUCE: return "VE_REDUCE";
        case DMOpcode::L2_SCRATCH_WRITE: return "L2_SCRATCH_WRITE";
        case DMOpcode::L2_SCRATCH_READ: return "L2_SCRATCH_READ";
        case DMOpcode::NOP: return "NOP";
        case DMOpcode::HALT: return "HALT";
        default: return "UNKNOWN";
    }
}

void print_program(const sw::kpu::isa::DMProgram& prog) {
    using namespace sw::kpu::isa;

    std::cout << "=== Program: " << prog.name << " ===\n";
    std::cout << "Version: " << prog.version << "\n";
    std::cout << "Dimensions: M=" << prog.M << " N=" << prog.N << " K=" << prog.K << "\n";
    std::cout << "Tiling: Ti=" << prog.Ti << " Tj=" << prog.Tj << " Tk=" << prog.Tk << "\n";
    std::cout << "L1_Ki: " << prog.L1_Ki << "\n";
    std::cout << "Dataflow: ";
    switch (prog.dataflow) {
        case DMProgram::Dataflow::OUTPUT_STATIONARY: std::cout << "output_stationary"; break;
        case DMProgram::Dataflow::WEIGHT_STATIONARY: std::cout << "weight_stationary"; break;
        case DMProgram::Dataflow::INPUT_STATIONARY: std::cout << "input_stationary"; break;
    }
    std::cout << "\n";
    std::cout << "Memory map:\n";
    std::cout << "  A base: 0x" << std::hex << prog.memory_map.a_base << std::dec << "\n";
    std::cout << "  B base: 0x" << std::hex << prog.memory_map.b_base << std::dec << "\n";
    std::cout << "  C base: 0x" << std::hex << prog.memory_map.c_base << std::dec << "\n";
    std::cout << "\nInstructions (" << prog.instructions.size() << "):\n";

    for (size_t i = 0; i < prog.instructions.size(); ++i) {
        const auto& instr = prog.instructions[i];
        std::cout << "  [" << i << "] " << get_opcode_name(instr.opcode);
        if (!instr.label.empty()) {
            std::cout << "  ; " << instr.label;
        }
        std::cout << "\n";
    }
}

void print_stats(const sw::kpu::isa::DMProgram& prog) {
    std::cout << "\n=== Program Statistics ===\n";
    std::cout << "Total instructions: " << prog.instructions.size() << "\n";
    std::cout << "DMA operations:     " << prog.num_dma_ops() << "\n";
    std::cout << "BM operations:      " << prog.num_bm_ops() << "\n";
    std::cout << "STR operations:     " << prog.num_str_ops() << "\n";
    std::cout << "Sync operations:    " << prog.num_sync_ops() << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string input_file;
    std::string output_file;
    std::string format = "binary";
    bool print_only = false;
    bool show_stats = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        } else if (std::strcmp(argv[i], "--print") == 0) {
            print_only = true;
        } else if (std::strcmp(argv[i], "--stats") == 0) {
            show_stats = true;
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            return 1;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    // Default output filename
    if (output_file.empty() && !print_only) {
        size_t dot = input_file.rfind('.');
        if (dot != std::string::npos) {
            output_file = input_file.substr(0, dot);
        } else {
            output_file = input_file;
        }
        output_file += (format == "json") ? ".kpujson" : ".kpubin";
    }

    // Assemble
    try {
        sw::kpu::isa::Assembler assembler;
        sw::kpu::isa::DMProgram program = assembler.assemble_file(input_file);

        // Print warnings
        for (const auto& warn : assembler.warnings()) {
            std::cerr << warn.to_string() << "\n";
        }

        // Print if requested
        if (print_only) {
            print_program(program);
            if (show_stats) {
                print_stats(program);
            }
            return 0;
        }

        // Show stats if requested
        if (show_stats) {
            print_stats(program);
        }

        // Save output
        sw::kpu::isa::ProgramSerializer serializer;

        if (format == "json") {
            serializer.save_json(program, output_file);
        } else {
            serializer.save(program, output_file);
        }

        std::cout << "Assembled " << program.instructions.size()
                  << " instructions to " << output_file << "\n";

    } catch (const sw::kpu::isa::AssemblyError& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

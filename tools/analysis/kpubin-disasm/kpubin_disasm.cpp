/**
 * @file kpubin_disasm.cpp
 * @brief KPU Binary Disassembler
 *
 * Reads .kpubin (DMProgram) and .kpukernel (Kernel) files and displays
 * their contents in a human-readable format.
 *
 * Usage:
 *   kpubin-disasm program.kpubin [options]
 *   kpubin-disasm kernel.kpukernel [options]
 *   kpubin-disasm program.json [options]
 *
 * Options:
 *   -h, --help          Show help
 *   -v, --verbose       Show all instruction details
 *   -s, --summary       Show summary only (no instructions)
 *   -j, --json          Output as JSON
 *   -i, --instructions  Show only instructions
 *   -m, --memory-map    Show only memory map
 */

#include <sw/kpu/isa/program_serializer.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/kernel_serializer.hpp>
#include <sw/kpu/kernel.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <algorithm>

using namespace sw::kpu;
using namespace sw::kpu::isa;

// ============================================================================
// Formatting Helpers
// ============================================================================

std::string format_bytes(Size bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

std::string format_number(uint64_t n) {
    std::string s = std::to_string(n);
    std::string result;
    int count = 0;
    for (int i = s.length() - 1; i >= 0; --i) {
        if (count > 0 && count % 3 == 0) {
            result = "," + result;
        }
        result = s[i] + result;
        count++;
    }
    return result;
}

std::string format_address(Address addr) {
    std::ostringstream oss;
    oss << "0x" << std::hex << addr;
    return oss.str();
}

const char* opcode_name(DMOpcode op) {
    switch (op) {
        case DMOpcode::DMA_LOAD_TILE: return "DMA_LOAD_TILE";
        case DMOpcode::DMA_STORE_TILE: return "DMA_STORE_TILE";
        case DMOpcode::DMA_PREFETCH_TILE: return "DMA_PREFETCH_TILE";
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
        case DMOpcode::NOP: return "NOP";
        case DMOpcode::HALT: return "HALT";
        default: return "UNKNOWN";
    }
}

const char* matrix_name(MatrixID m) {
    switch (m) {
        case MatrixID::A: return "A";
        case MatrixID::B: return "B";
        case MatrixID::C: return "C";
        default: return "?";
    }
}

const char* transform_name(Transform t) {
    switch (t) {
        case Transform::IDENTITY: return "identity";
        case Transform::TRANSPOSE: return "transpose";
        case Transform::RESHAPE: return "reshape";
        case Transform::SHUFFLE: return "shuffle";
        default: return "unknown";
    }
}

const char* buffer_name(BufferSlot b) {
    switch (b) {
        case BufferSlot::BUF_0: return "buf0";
        case BufferSlot::BUF_1: return "buf1";
        case BufferSlot::AUTO: return "auto";
        default: return "?";
    }
}

const char* dataflow_name(DMProgram::Dataflow df) {
    switch (df) {
        case DMProgram::Dataflow::OUTPUT_STATIONARY: return "OUTPUT_STATIONARY";
        case DMProgram::Dataflow::WEIGHT_STATIONARY: return "WEIGHT_STATIONARY";
        case DMProgram::Dataflow::INPUT_STATIONARY: return "INPUT_STATIONARY";
        default: return "UNKNOWN";
    }
}

std::string tile_coord_str(const TileCoord& t) {
    return "[" + std::to_string(t.ti) + "," + std::to_string(t.tj) + "," + std::to_string(t.tk) + "]";
}

// ============================================================================
// Instruction Disassembly
// ============================================================================

void disassemble_instruction(const DMInstruction& instr, size_t index, bool verbose) {
    std::cout << "  [" << std::setw(4) << index << "] ";
    std::cout << std::left << std::setw(18) << opcode_name(instr.opcode);

    std::visit([verbose](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, std::monostate>) {
            // No operands
        } else if constexpr (std::is_same_v<T, DMAOperands>) {
            std::cout << matrix_name(arg.matrix) << tile_coord_str(arg.tile)
                      << " ext:" << format_address(arg.ext_mem_addr)
                      << " -> L3[" << (int)arg.l3_tile_id << "]:" << format_address(arg.l3_offset)
                      << " (" << format_bytes(arg.size_bytes) << ")";
            if (verbose) {
                std::cout << " [" << buffer_name(arg.buffer) << "]";
            }
        } else if constexpr (std::is_same_v<T, BlockMoverOperands>) {
            std::cout << matrix_name(arg.matrix) << tile_coord_str(arg.tile)
                      << " L3[" << (int)arg.src_l3_tile_id << "]:" << format_address(arg.src_offset)
                      << " -> L2[" << (int)arg.dst_l2_bank_id << "]:" << format_address(arg.dst_offset)
                      << " (" << arg.height << "x" << arg.width << ", " << transform_name(arg.transform) << ")";
            if (verbose) {
                std::cout << " [elem=" << arg.element_size << "B, " << buffer_name(arg.buffer) << "]";
            }
        } else if constexpr (std::is_same_v<T, StreamerOperands>) {
            std::cout << matrix_name(arg.matrix) << tile_coord_str(arg.tile)
                      << " L2[" << (int)arg.l2_bank_id << "]:" << format_address(arg.l2_addr)
                      << " <-> L1[" << (int)arg.l1_buffer_id << "]:" << format_address(arg.l1_addr)
                      << " (" << arg.height << "x" << arg.width << ")";
            if (arg.ve_enabled) {
                std::cout << " [VE: " << activation_type_name(arg.ve_activation);
                if (arg.ve_bias_enabled) {
                    std::cout << "+bias@" << format_address(arg.ve_bias_addr);
                }
                std::cout << "]";
            }
        } else if constexpr (std::is_same_v<T, SyncOperands>) {
            if (arg.wait_mask != 0) {
                std::cout << "mask=0x" << std::hex << arg.wait_mask << std::dec;
            }
            if (arg.signal_id != 0) {
                std::cout << " signal=" << arg.signal_id;
            }
        } else if constexpr (std::is_same_v<T, LoopOperands>) {
            std::cout << "id=" << (int)arg.loop_id
                      << " count=" << arg.loop_count
                      << " stride=" << arg.loop_stride;
        } else if constexpr (std::is_same_v<T, ConfigOperands>) {
            std::cout << "Ti=" << arg.Ti << " Tj=" << arg.Tj << " Tk=" << arg.Tk
                      << " L1_Ki=" << arg.L1_Ki;
            if (verbose) {
                std::cout << " strides=(" << arg.stride_m << "," << arg.stride_n << "," << arg.stride_k << ")";
            }
        }
    }, instr.operands);

    if (verbose && !instr.label.empty()) {
        std::cout << "  ; " << instr.label;
    }

    if (verbose && !instr.dependencies.empty()) {
        std::cout << " (deps: ";
        for (size_t i = 0; i < instr.dependencies.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << instr.dependencies[i];
        }
        std::cout << ")";
    }

    std::cout << "\n";
}

// ============================================================================
// Program Disassembly
// ============================================================================

void disassemble_program(const DMProgram& program, bool verbose, bool summary_only,
                         bool instructions_only, bool memory_map_only) {
    if (!instructions_only && !memory_map_only) {
        std::cout << "=== KPU Program: " << program.name << " ===\n";
        std::cout << "Version: " << program.version << "\n";
        std::cout << "Dimensions: M=" << program.M << ", N=" << program.N << ", K=" << program.K << "\n";
        std::cout << "Tiles: Ti=" << program.Ti << ", Tj=" << program.Tj
                  << ", Tk=" << program.Tk << ", L1_Ki=" << program.L1_Ki << "\n";
        std::cout << "Dataflow: " << dataflow_name(program.dataflow) << "\n";
        std::cout << "\n";
    }

    // Operation counts
    if (!instructions_only && !memory_map_only) {
        size_t dma_count = 0, bm_count = 0, str_count = 0, sync_count = 0, other_count = 0;
        for (const auto& instr : program.instructions) {
            switch (instr.opcode) {
                case DMOpcode::DMA_LOAD_TILE:
                case DMOpcode::DMA_STORE_TILE:
                case DMOpcode::DMA_PREFETCH_TILE:
                    dma_count++;
                    break;
                case DMOpcode::BM_MOVE_TILE:
                case DMOpcode::BM_TRANSPOSE_TILE:
                case DMOpcode::BM_WRITEBACK_TILE:
                case DMOpcode::BM_RESHAPE_TILE:
                    bm_count++;
                    break;
                case DMOpcode::STR_FEED_ROWS:
                case DMOpcode::STR_FEED_COLS:
                case DMOpcode::STR_DRAIN_OUTPUT:
                case DMOpcode::STR_BROADCAST_ROW:
                case DMOpcode::STR_BROADCAST_COL:
                    str_count++;
                    break;
                case DMOpcode::BARRIER:
                case DMOpcode::WAIT_DMA:
                case DMOpcode::WAIT_BM:
                case DMOpcode::WAIT_STR:
                case DMOpcode::SIGNAL:
                    sync_count++;
                    break;
                default:
                    other_count++;
            }
        }

        std::cout << "Operations Summary:\n";
        std::cout << "  Total: " << program.instructions.size() << "\n";
        std::cout << "  DMA:      " << std::setw(6) << dma_count << " (External <-> L3)\n";
        std::cout << "  BM:       " << std::setw(6) << bm_count << " (L3 <-> L2)\n";
        std::cout << "  Streamer: " << std::setw(6) << str_count << " (L2 <-> L1)\n";
        std::cout << "  Sync:     " << std::setw(6) << sync_count << "\n";
        if (other_count > 0) {
            std::cout << "  Other:    " << std::setw(6) << other_count << "\n";
        }
        std::cout << "\n";
    }

    // Instructions
    if (!summary_only && !memory_map_only) {
        std::cout << "Instructions (" << program.instructions.size() << "):\n";
        for (size_t i = 0; i < program.instructions.size(); ++i) {
            disassemble_instruction(program.instructions[i], i, verbose);
        }
        std::cout << "\n";
    }

    // Memory Map
    if (!instructions_only && !summary_only) {
        std::cout << "Memory Map:\n";
        std::cout << "  A base: " << format_address(program.memory_map.a_base) << "\n";
        std::cout << "  B base: " << format_address(program.memory_map.b_base) << "\n";
        std::cout << "  C base: " << format_address(program.memory_map.c_base) << "\n";

        if (!program.memory_map.l3_allocations.empty()) {
            std::cout << "\n  L3 Allocations (" << program.memory_map.l3_allocations.size() << "):\n";
            for (const auto& alloc : program.memory_map.l3_allocations) {
                std::cout << "    Tile[" << (int)alloc.tile_id << "] "
                          << matrix_name(alloc.matrix)
                          << " offset=" << format_address(alloc.offset)
                          << " size=" << format_bytes(alloc.size)
                          << " [" << buffer_name(alloc.buffer) << "]\n";
            }
        }

        if (!program.memory_map.l2_allocations.empty()) {
            std::cout << "\n  L2 Allocations (" << program.memory_map.l2_allocations.size() << "):\n";
            for (const auto& alloc : program.memory_map.l2_allocations) {
                std::cout << "    Bank[" << (int)alloc.bank_id << "] "
                          << matrix_name(alloc.matrix)
                          << " offset=" << format_address(alloc.offset)
                          << " size=" << format_bytes(alloc.size)
                          << " [" << buffer_name(alloc.buffer) << "]\n";
            }
        }
        std::cout << "\n";
    }

    // Estimates
    if (!instructions_only && !memory_map_only) {
        std::cout << "Performance Estimates:\n";
        std::cout << "  Total cycles:         " << format_number(program.estimates.total_cycles) << "\n";
        std::cout << "  External memory:      " << format_bytes(program.estimates.external_mem_bytes) << "\n";
        std::cout << "  L3 traffic:           " << format_bytes(program.estimates.l3_bytes) << "\n";
        std::cout << "  L2 traffic:           " << format_bytes(program.estimates.l2_bytes) << "\n";
        std::cout << "  Arithmetic intensity: " << std::fixed << std::setprecision(2)
                  << program.estimates.arithmetic_intensity << " FLOP/byte\n";
        std::cout << "  Estimated GFLOPS:     " << std::fixed << std::setprecision(1)
                  << program.estimates.estimated_gflops << "\n";
    }
}

// ============================================================================
// Kernel Disassembly
// ============================================================================

void disassemble_kernel(const Kernel& kernel, bool verbose, bool summary_only,
                        bool instructions_only, bool memory_map_only) {
    if (!instructions_only && !memory_map_only) {
        std::cout << "=== KPU Kernel: " << kernel.name() << " ===\n";
        std::cout << "Operation: " << kernel_op_type_name(kernel.op_type()) << "\n";
        std::cout << "Data Type: " << dtype_name(kernel.dtype()) << "\n";
        std::cout << "Dimensions: M=" << kernel.M() << ", N=" << kernel.N() << ", K=" << kernel.K() << "\n";
        std::cout << "Tiles: Ti=" << kernel.Ti() << ", Tj=" << kernel.Tj()
                  << ", Tk=" << kernel.Tk() << "\n";

        if (kernel.op_type() == KernelOpType::MLP) {
            std::cout << "Activation: " << activation_type_name(kernel.activation()) << "\n";
            std::cout << "Has Bias: " << (kernel.has_bias() ? "yes" : "no") << "\n";
        }

        std::cout << "\nArguments (" << kernel.arguments().size() << "):\n";
        for (const auto& arg : kernel.arguments()) {
            std::cout << "  " << std::left << std::setw(8) << arg.name << " ";
            std::cout << std::setw(10) << dtype_name(arg.dtype) << " [";
            for (size_t i = 0; i < arg.shape.size(); ++i) {
                if (i > 0) std::cout << " x ";
                std::cout << arg.shape[i];
            }
            std::cout << "] " << format_bytes(arg.size_bytes);
            std::cout << (arg.is_output ? " (output)" : " (input)") << "\n";
        }

        std::cout << "\nMemory Footprint:\n";
        std::cout << "  Input:  " << format_bytes(kernel.total_input_bytes()) << "\n";
        std::cout << "  Output: " << format_bytes(kernel.total_output_bytes()) << "\n";
        std::cout << "  FLOPs:  " << format_number(kernel.total_flops()) << "\n";
        std::cout << "  Arithmetic Intensity: " << std::fixed << std::setprecision(2)
                  << kernel.arithmetic_intensity() << " FLOP/byte\n";
        std::cout << "\n";
    }

    // Disassemble the embedded program
    disassemble_program(kernel.program(), verbose, summary_only, instructions_only, memory_map_only);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program_name) {
    std::cout << "KPU Binary Disassembler\n\n";
    std::cout << "Reads .kpubin (DMProgram) and .kpukernel (Kernel) files and displays\n";
    std::cout << "their contents in a human-readable format.\n\n";
    std::cout << "Usage: " << program_name << " <file> [options]\n\n";
    std::cout << "Supported formats:\n";
    std::cout << "  .kpubin      - DMProgram binary format\n";
    std::cout << "  .kpukernel   - Kernel binary format\n";
    std::cout << "  .kpujson     - DMProgram JSON format\n";
    std::cout << "  .json        - Kernel or Program JSON format\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help          Show this help\n";
    std::cout << "  -v, --verbose       Show all instruction details (deps, labels, buffers)\n";
    std::cout << "  -s, --summary       Show summary only (no instructions)\n";
    std::cout << "  -i, --instructions  Show only instructions\n";
    std::cout << "  -m, --memory-map    Show only memory map\n";
    std::cout << "  -j, --json          Output as JSON\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filename;
    bool verbose = false;
    bool summary_only = false;
    bool instructions_only = false;
    bool memory_map_only = false;
    bool output_json = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-s" || arg == "--summary") {
            summary_only = true;
        } else if (arg == "-i" || arg == "--instructions") {
            instructions_only = true;
        } else if (arg == "-m" || arg == "--memory-map") {
            memory_map_only = true;
        } else if (arg == "-j" || arg == "--json") {
            output_json = true;
        } else if (arg[0] != '-') {
            filename = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    if (filename.empty()) {
        std::cerr << "Error: No input file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: File not found: " << filename << "\n";
        return 1;
    }

    try {
        // Detect file type by extension
        std::string ext = std::filesystem::path(filename).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        bool is_kernel = (ext == ".kpukernel");
        bool is_json = (ext == ".json" || ext == ".kpujson");

        if (is_kernel || ext == ".json") {
            // Try loading as kernel first
            KernelSerializer kernel_serializer;

            try {
                Kernel kernel;
                if (is_json) {
                    kernel = kernel_serializer.load_json(filename);
                } else {
                    kernel = kernel_serializer.load(filename);
                }

                if (output_json) {
                    std::cout << kernel_serializer.to_json(kernel, true) << "\n";
                } else {
                    disassemble_kernel(kernel, verbose, summary_only, instructions_only, memory_map_only);
                }
                return 0;
            } catch (const SerializationError&) {
                // If it's a .json file and kernel loading failed, try as program
                if (is_json) {
                    // Fall through to try program
                } else {
                    throw;
                }
            }
        }

        // Load as program
        ProgramSerializer program_serializer;
        DMProgram program;

        if (is_json || ext == ".kpujson") {
            program = program_serializer.load_json(filename);
        } else {
            program = program_serializer.load(filename);
        }

        if (output_json) {
            std::cout << program_serializer.to_json(program, true) << "\n";
        } else {
            disassemble_program(program, verbose, summary_only, instructions_only, memory_map_only);
        }

        return 0;

    } catch (const SerializationError& e) {
        std::cerr << "Serialization error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

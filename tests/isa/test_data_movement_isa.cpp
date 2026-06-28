/**
 * @file test_data_movement_isa.cpp
 * @brief Tests for Data Movement ISA and program execution
 */

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/program_executor.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <sstream>

using namespace sw::kpu::isa;
using sw::kpu::Size;

// ============================================================================
// Test Helpers
// ============================================================================

void print_test_header(const std::string& name) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_pass(const std::string& msg) {
    std::cout << "[PASS] " << msg << "\n";
}

void print_fail(const std::string& msg) {
    std::cout << "[FAIL] " << msg << "\n";
}

// ============================================================================
// Test: Program Builder Creates Valid Output-Stationary Program
// ============================================================================

bool test_output_stationary_program_builder() {
    print_test_header("Output-Stationary Program Builder");

    // Configure for a small matmul: C[64,64] = A[64,64] × B[64,64]
    OutputStationaryProgramBuilder::Config config;
    config.M = 64;
    config.N = 64;
    config.K = 64;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;  // float32

    config.l3_tile_capacity = 128 * 1024;   // 128 KB
    config.l2_bank_capacity = 64 * 1024;    // 64 KB
    config.l1_buffer_capacity = 32 * 1024;  // 32 KB

    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = true;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    // Validate program
    std::string error;
    bool valid = validate_program(program, error);

    if (!valid) {
        print_fail("Program validation failed: " + error);
        return false;
    }
    print_pass("Program validated successfully");

    // Check program metadata
    if (program.name.find("matmul") == std::string::npos) {
        print_fail("Program name should contain 'matmul'");
        return false;
    }
    print_pass("Program name: " + program.name);

    if (program.dataflow != DMProgram::Dataflow::OUTPUT_STATIONARY) {
        print_fail("Dataflow should be OUTPUT_STATIONARY");
        return false;
    }
    print_pass("Dataflow: Output-Stationary");

    // Check dimensions
    if (program.M != 64 || program.N != 64 || program.K != 64) {
        print_fail("Matrix dimensions incorrect");
        return false;
    }
    print_pass("Matrix dimensions: " + std::to_string(program.M) + "x" +
               std::to_string(program.N) + "x" + std::to_string(program.K));

    // Check tile sizes
    if (program.Ti != 16 || program.Tj != 16 || program.Tk != 16) {
        print_fail("Tile sizes incorrect");
        return false;
    }
    print_pass("Tile sizes: Ti=" + std::to_string(program.Ti) +
               " Tj=" + std::to_string(program.Tj) +
               " Tk=" + std::to_string(program.Tk));

    // Check instruction counts
    size_t num_dma = program.num_dma_ops();
    size_t num_bm = program.num_bm_ops();
    size_t num_str = program.num_str_ops();
    size_t num_sync = program.num_sync_ops();

    std::cout << "\nInstruction Statistics:\n";
    std::cout << "  Total:      " << program.instructions.size() << "\n";
    std::cout << "  DMA ops:    " << num_dma << "\n";
    std::cout << "  BM ops:     " << num_bm << "\n";
    std::cout << "  STR ops:    " << num_str << "\n";
    std::cout << "  SYNC ops:   " << num_sync << "\n";

    // For 64x64x64 with Ti=Tj=Tk=16:
    // - 4 output tiles in M (64/16)
    // - 4 output tiles in N (64/16)
    // - 4 reduction tiles in K (64/16)
    // - Each C tile: 4 K iterations of (load A, load B, BM A, BM B, stream A, stream B)
    //   then drain C and store C
    // Expected: 4*4 output tiles = 16
    // Per output tile: 4 K-iterations * (2 DMA + 2 BM + 2 STR) + 1 drain + 1 store + barriers
    //                  = 4 * (2+2+2) + 2 = 26 data ops per C tile
    //                  Plus barriers between phases

    if (num_dma == 0) {
        print_fail("No DMA operations generated");
        return false;
    }
    print_pass("DMA operations generated: " + std::to_string(num_dma));

    if (num_bm == 0) {
        print_fail("No BlockMover operations generated");
        return false;
    }
    print_pass("BlockMover operations generated: " + std::to_string(num_bm));

    if (num_str == 0) {
        print_fail("No Streamer operations generated");
        return false;
    }
    print_pass("Streamer operations generated: " + std::to_string(num_str));

    // Check last instruction is HALT
    if (program.instructions.back().opcode != DMOpcode::HALT) {
        print_fail("Last instruction should be HALT");
        return false;
    }
    print_pass("Program ends with HALT");

    // Check estimates
    std::cout << "\nPerformance Estimates:\n";
    std::cout << "  External memory bytes: " << program.estimates.external_mem_bytes << "\n";
    std::cout << "  L3 bytes: " << program.estimates.l3_bytes << "\n";
    std::cout << "  L2 bytes: " << program.estimates.l2_bytes << "\n";
    std::cout << "  Arithmetic intensity: " << std::fixed << std::setprecision(2)
              << program.estimates.arithmetic_intensity << " FLOPs/byte\n";

    print_pass("Program builder test completed");
    return true;
}

// ============================================================================
// Test: DMInstruction Static Constructors
// ============================================================================

bool test_instruction_constructors() {
    print_test_header("Instruction Constructors");

    // Test DMA load instruction
    TileCoord tile{0, 0, 0};
    auto dma_instr = DMInstruction::dma_load(MatrixID::A, tile, 0x1000, 0, 0x0, 4096);

    if (dma_instr.opcode != DMOpcode::DMA_LOAD_TILE) {
        print_fail("DMA load opcode incorrect");
        return false;
    }
    print_pass("DMA load instruction created");

    // Test BlockMover instruction
    auto bm_instr = DMInstruction::bm_move(MatrixID::A, tile, 0, 0, 0, 0, 16, 16, 4);

    if (bm_instr.opcode != DMOpcode::BM_MOVE_TILE) {
        print_fail("BM move opcode incorrect");
        return false;
    }
    print_pass("BlockMover instruction created");

    // Test Streamer instruction
    auto str_instr = DMInstruction::str_feed_rows(MatrixID::A, tile, 0, 0, 0, 0, 16, 16, 16);

    if (str_instr.opcode != DMOpcode::STR_FEED_ROWS) {
        print_fail("Streamer feed rows opcode incorrect");
        return false;
    }
    print_pass("Streamer instruction created");

    // Test barrier
    auto barrier_instr = DMInstruction::barrier();

    if (barrier_instr.opcode != DMOpcode::BARRIER) {
        print_fail("Barrier opcode incorrect");
        return false;
    }
    print_pass("Barrier instruction created");

    // Test halt
    auto halt_instr = DMInstruction::halt();

    if (halt_instr.opcode != DMOpcode::HALT) {
        print_fail("Halt opcode incorrect");
        return false;
    }
    print_pass("Halt instruction created");

    print_pass("All instruction constructors working");
    return true;
}

// ============================================================================
// Test: Program Disassembly
// ============================================================================

bool test_disassembly() {
    print_test_header("Program Disassembly");

    // Create a small program
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;

    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;  // Simpler for testing

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    // Disassemble to string stream
    std::ostringstream oss;
    disassemble_program(program, oss);

    std::string disasm = oss.str();

    // Check that disassembly contains key elements
    if (disasm.find("Output-Stationary") == std::string::npos) {
        print_fail("Disassembly missing dataflow type");
        return false;
    }
    print_pass("Disassembly shows dataflow type");

    if (disasm.find("DMA_LOAD") == std::string::npos) {
        print_fail("Disassembly missing DMA_LOAD instructions");
        return false;
    }
    print_pass("Disassembly shows DMA_LOAD instructions");

    if (disasm.find("BM_MOVE") == std::string::npos) {
        print_fail("Disassembly missing BM_MOVE instructions");
        return false;
    }
    print_pass("Disassembly shows BM_MOVE instructions");

    if (disasm.find("STR_") == std::string::npos) {
        print_fail("Disassembly missing STR_ instructions");
        return false;
    }
    print_pass("Disassembly shows STR_ instructions");

    if (disasm.find("HALT") == std::string::npos) {
        print_fail("Disassembly missing HALT instruction");
        return false;
    }
    print_pass("Disassembly shows HALT instruction");

    // Print a sample of the disassembly
    std::cout << "\nSample disassembly:\n";
    std::cout << disasm.substr(0, 1500) << "\n...\n";

    print_pass("Disassembly test completed");
    return true;
}

// ============================================================================
// Test: Larger MatMul Program Generation
// ============================================================================

bool test_large_matmul_program() {
    print_test_header("Large MatMul Program (1024x1024x1024)");

    OutputStationaryProgramBuilder::Config config;
    config.M = 1024;
    config.N = 1024;
    config.K = 1024;
    config.Ti = 64;
    config.Tj = 64;
    config.Tk = 64;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;

    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = true;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "Program: " << program.name << "\n";
    std::cout << "Instructions: " << program.instructions.size() << "\n";

    // Calculate expected tile counts
    Size m_tiles = (config.M + config.Ti - 1) / config.Ti;  // 16
    Size n_tiles = (config.N + config.Tj - 1) / config.Tj;  // 16
    Size k_tiles = (config.K + config.Tk - 1) / config.Tk;  // 16

    std::cout << "Tile counts: M=" << m_tiles << " N=" << n_tiles << " K=" << k_tiles << "\n";
    std::cout << "Output tiles: " << (m_tiles * n_tiles) << "\n";
    std::cout << "Total tile iterations: " << (m_tiles * n_tiles * k_tiles) << "\n";

    // Check traffic estimates
    std::cout << "\nTraffic Estimates:\n";
    std::cout << "  External memory: " << (static_cast<double>(program.estimates.external_mem_bytes) / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  L3 traffic: " << (static_cast<double>(program.estimates.l3_bytes) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  L2 traffic: " << (static_cast<double>(program.estimates.l2_bytes) / (1024.0 * 1024.0)) << " MB\n";

    // For 1024x1024x1024 with Ti=Tj=Tk=64:
    // Minimum DRAM read = A (4MB) + B (4MB) = 8MB
    // Minimum DRAM write = C (4MB)
    // Total minimum = 12MB
    // With output-stationary, each A row and B column is loaded multiple times
    // But reuse should keep it much lower than naive

    Size min_external = (config.M * config.K + config.K * config.N + config.M * config.N) * 4;
    double reuse_factor = static_cast<double>(program.estimates.external_mem_bytes) / static_cast<double>(min_external);

    std::cout << "  Minimum external: " << (static_cast<double>(min_external) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Reuse factor: " << std::fixed << std::setprecision(2) << reuse_factor << "x\n";
    std::cout << "  Arithmetic intensity: " << program.estimates.arithmetic_intensity << " FLOPs/byte\n";

    // 2*M*N*K FLOPs = 2*1024^3 = 2.147B FLOPs
    Size total_flops = 2ULL * config.M * config.N * config.K;
    std::cout << "  Total FLOPs: " << (static_cast<double>(total_flops) / 1e9) << " GFLOPs\n";

    std::string error;
    if (!validate_program(program, error)) {
        print_fail("Program validation failed: " + error);
        return false;
    }
    print_pass("Large program validated");

    print_pass("Large MatMul program test completed");
    return true;
}

// ============================================================================
// Test: TileCoord Equality
// ============================================================================

bool test_tile_coord() {
    print_test_header("TileCoord Operations");

    TileCoord a{1, 2, 3};
    TileCoord b{1, 2, 3};
    TileCoord c{1, 2, 4};

    if (!(a == b)) {
        print_fail("Equal TileCoords should compare equal");
        return false;
    }
    print_pass("Equal TileCoords compare equal");

    if (a == c) {
        print_fail("Different TileCoords should not compare equal");
        return false;
    }
    print_pass("Different TileCoords compare not equal");

    print_pass("TileCoord test completed");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Data Movement ISA Test Suite\n";
    std::cout << std::string(60, '*') << "\n";

    int passed = 0;
    int failed = 0;

    if (test_tile_coord()) passed++; else failed++;
    if (test_instruction_constructors()) passed++; else failed++;
    if (test_output_stationary_program_builder()) passed++; else failed++;
    if (test_disassembly()) passed++; else failed++;
    if (test_large_matmul_program()) passed++; else failed++;

    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Test Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << std::string(60, '*') << "\n\n";

    return failed > 0 ? 1 : 0;
}

/**
 * @file test_schedule_dsl.cpp
 * @brief Tests for Schedule DSL and three prototypical kernel schedules
 *
 * Verifies that:
 * 1. MatMul schedule compiles to a valid DMProgram
 * 2. Conv2D schedule compiles with im2col gather ops
 * 3. Softmax schedule compiles with VE reduction/elementwise ops
 */

#include <sw/kpu/dsl/schedule.hpp>
#include <sw/kpu/dsl/schedule_compiler.hpp>
#include <sw/kpu/schedules/matmul_schedule.hpp>
#include <sw/kpu/schedules/conv2d_schedule.hpp>
#include <sw/kpu/schedules/softmax_schedule.hpp>
#include <iostream>
#include <cassert>
#include <sstream>
#include <set>

using namespace sw::kpu;
using namespace sw::kpu::dsl;
using namespace sw::kpu::schedules;
using namespace sw::kpu::isa;

// ============================================================================
// Test Helpers
// ============================================================================

static int passed = 0;
static int failed = 0;

void check(bool condition, const std::string& msg) {
    if (condition) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

void print_program_stats(const DMProgram& prog) {
    std::cout << "  Program: " << prog.name << "\n";
    std::cout << "  Instructions: " << prog.instructions.size() << "\n";
    std::cout << "  DMA ops:  " << prog.num_dma_ops() << "\n";
    std::cout << "  BM ops:   " << prog.num_bm_ops() << "\n";
    std::cout << "  STR ops:  " << prog.num_str_ops() << "\n";
    std::cout << "  SYNC ops: " << prog.num_sync_ops() << "\n";
    std::cout << "  External memory: " << prog.estimates.external_mem_bytes << " bytes\n";
}

// ============================================================================
// Test 1: MatMul Schedule DSL
// ============================================================================

void test_matmul_schedule() {
    std::cout << "\n=== Test: MatMul Output-Stationary Schedule ===\n";

    auto sched = matmul_output_stationary(256, 256, 256, 64, 64, 64);

    check(sched.name().find("matmul_os") != std::string::npos,
          "Schedule name contains 'matmul_os'");
    check(sched.tensors().size() == 3, "Three tensors declared (A, B, C)");
    check(sched.get_tile("ti") == 64, "Ti = 64");
    check(sched.get_tile("tj") == 64, "Tj = 64");
    check(sched.get_tile("tk") == 64, "Tk = 64");
    check(sched.get_dataflow() == DMProgram::Dataflow::OUTPUT_STATIONARY,
          "Dataflow is output-stationary");

    // Compile
    DMProgram prog = compile_schedule(sched);

    print_program_stats(prog);

    check(prog.name.find("matmul_os") != std::string::npos,
          "DMProgram name contains 'matmul_os'");
    check(prog.dataflow == DMProgram::Dataflow::OUTPUT_STATIONARY,
          "DMProgram dataflow is output-stationary");
    check(prog.instructions.size() > 0, "DMProgram has instructions");
    check(prog.instructions.back().opcode == DMOpcode::HALT,
          "DMProgram ends with HALT");
    check(prog.num_dma_ops() > 0, "Has DMA operations");
    check(prog.num_bm_ops() > 0, "Has BlockMover operations");
    check(prog.num_str_ops() > 0, "Has Streamer operations");
    check(prog.num_sync_ops() > 0, "Has sync operations");

    // For 256x256x256 with Ti=Tj=Tk=64:
    //   4 M tiles x 4 N tiles = 16 output tiles
    //   4 K tiles per output tile
    //   Per K iteration: 2 DMA loads + 1 barrier + 2 BM moves + 2 STR feeds
    //   Per output tile: drain + writeback + store
    Size m_tiles = 256 / 64;  // 4
    Size n_tiles = 256 / 64;  // 4
    Size k_tiles = 256 / 64;  // 4
    // 2 loads per K iteration + 1 store per output tile
    Size expected_dma = m_tiles * n_tiles * k_tiles * 2 + m_tiles * n_tiles;
    check(prog.num_dma_ops() == expected_dma,
          "Expected " + std::to_string(expected_dma) + " DMA ops, got " +
          std::to_string(prog.num_dma_ops()));

    check(prog.estimates.external_mem_bytes > 0, "External memory traffic > 0");
}

// ============================================================================
// Test 2: Conv2D Schedule DSL
// ============================================================================

void test_conv2d_schedule() {
    std::cout << "\n=== Test: Conv2D Im2Col Schedule ===\n";

    // Small conv: batch=1, Ci=3, 8x8 input, Co=16, 3x3 kernel
    auto sched = conv2d_im2col(1, 3, 8, 8, 16, 3, 3, 1, 1, 64, 16);

    check(sched.name().find("conv2d_im2col") != std::string::npos,
          "Schedule name contains 'conv2d_im2col'");
    check(sched.tensors().size() == 3, "Three tensors (A_col, B_w, C_out)");
    check(sched.get_dataflow() == DMProgram::Dataflow::OUTPUT_STATIONARY,
          "Dataflow is output-stationary");

    // Compile
    DMProgram prog = compile_schedule(sched);

    print_program_stats(prog);

    check(prog.instructions.size() > 0, "DMProgram has instructions");
    check(prog.instructions.back().opcode == DMOpcode::HALT,
          "DMProgram ends with HALT");
    check(prog.num_dma_ops() > 0, "Has DMA operations (including gather)");
    check(prog.num_str_ops() > 0, "Has Streamer operations");

    // Check for im2col gather annotation
    bool has_gather = false;
    bool has_fused_drain = false;
    for (const auto& instr : prog.instructions) {
        if (instr.label.find("GATHER") != std::string::npos) has_gather = true;
        if (instr.label.find("+VE") != std::string::npos) has_fused_drain = true;
    }
    check(has_gather, "Has im2col gather DMA operation");
    check(has_fused_drain, "Has fused drain with VE (bias+activation)");
}

// ============================================================================
// Test 3: Softmax Schedule DSL
// ============================================================================

void test_softmax_schedule() {
    std::cout << "\n=== Test: Softmax Multi-Pass Schedule ===\n";

    auto sched = softmax(128, 512, 32);

    check(sched.name().find("softmax") != std::string::npos,
          "Schedule name contains 'softmax'");
    check(sched.tensors().size() == 2, "Two tensors (X, Y)");
    check(sched.get_tile("tb") == 32, "Tb = 32");

    // Compile
    DMProgram prog = compile_schedule(sched);

    print_program_stats(prog);

    check(prog.instructions.size() > 0, "DMProgram has instructions");
    check(prog.instructions.back().opcode == DMOpcode::HALT,
          "DMProgram ends with HALT");

    // Check for VE operations (emitted as annotated NOPs)
    bool has_reduce_max = false;
    bool has_reduce_sum = false;
    bool has_ewise_sub = false;
    bool has_ewise_exp = false;
    bool has_ewise_div = false;
    bool has_broadcast = false;
    bool has_scratch = false;

    for (const auto& instr : prog.instructions) {
        if (instr.label.find("VE_REDUCE(MAX)") != std::string::npos) has_reduce_max = true;
        if (instr.label.find("VE_REDUCE(SUM)") != std::string::npos) has_reduce_sum = true;
        if (instr.label.find("VE_ELEMENTWISE(SUB)") != std::string::npos) has_ewise_sub = true;
        if (instr.label.find("VE_ELEMENTWISE(EXP)") != std::string::npos) has_ewise_exp = true;
        if (instr.label.find("VE_ELEMENTWISE(DIV)") != std::string::npos) has_ewise_div = true;
        if (instr.opcode == DMOpcode::STR_BROADCAST_ROW) has_broadcast = true;
        if (instr.label.find("SCRATCH") != std::string::npos) has_scratch = true;
    }

    check(has_reduce_max, "Has VE_REDUCE(MAX) for pass 1");
    check(has_reduce_sum, "Has VE_REDUCE(SUM) for pass 3");
    check(has_ewise_sub, "Has VE_ELEMENTWISE(SUB) for x - max");
    check(has_ewise_exp, "Has VE_ELEMENTWISE(EXP) for exp(x - max)");
    check(has_ewise_div, "Has VE_ELEMENTWISE(DIV) for exp / sum");
    check(has_broadcast, "Has broadcast operations");
    check(has_scratch, "Has drain-to-scratch operations");

    // Softmax has 3 passes per batch tile, so instruction count should reflect that
    Size batch_tiles = 128 / 32;  // 4
    check(prog.num_str_ops() > batch_tiles,
          "More streamer ops than batch tiles (multiple passes)");
}

// ============================================================================
// Test 4: Schedule DSL Fluent API
// ============================================================================

void test_fluent_api() {
    std::cout << "\n=== Test: DSL Fluent API ===\n";

    Schedule sched("test_api");
    sched.tensor(Tensor{"X", MatrixID::A, {64, 64}, 0x1000, DataType::FP32})
         .tensor(Tensor{"Y", MatrixID::C, {64, 64}, 0x2000, DataType::FP32})
         .tile("ti", 16)
         .tile("tj", 16)
         .dataflow(DMProgram::Dataflow::OUTPUT_STATIONARY)
         .systolic_size(16)
         .l3_buffers(4)
         .l2_banks(8);

    check(sched.name() == "test_api", "Schedule name set");
    check(sched.tensors().size() == 2, "Two tensors declared");
    check(sched.get_tile("ti") == 16, "Ti = 16");
    check(sched.get_systolic_size() == 16, "Systolic size = 16");
    check(sched.get_l3_buffers() == 4, "L3 buffers = 4");
    check(sched.get_l2_banks() == 8, "L2 banks = 8");

    auto* t = sched.find_tensor(MatrixID::A);
    check(t != nullptr, "Found tensor A");
    check(t != nullptr && t->name == "X", "Tensor A name is 'X'");
    check(t != nullptr && t->base_addr == 0x1000, "Tensor A base addr = 0x1000");
}

// ============================================================================
// Test 5: Large MatMul Schedule
// ============================================================================

void test_large_matmul() {
    std::cout << "\n=== Test: Large MatMul Schedule (1024x1024x1024) ===\n";

    auto sched = matmul_output_stationary(1024, 1024, 1024, 128, 128, 64);
    DMProgram prog = compile_schedule(sched);

    print_program_stats(prog);

    Size m_tiles = 1024 / 128;  // 8
    Size n_tiles = 1024 / 128;  // 8
    Size k_tiles = 1024 / 64;   // 16

    Size expected_dma = m_tiles * n_tiles * k_tiles * 2 + m_tiles * n_tiles;
    check(prog.num_dma_ops() == expected_dma,
          "Correct DMA op count for large matmul (" + std::to_string(expected_dma) + ")");
    check(prog.instructions.back().opcode == DMOpcode::HALT, "Ends with HALT");
    check(prog.estimates.external_mem_bytes > 0, "Has traffic estimates");

    // Arithmetic intensity for 1024^3 matmul
    Size total_flops = 2ULL * 1024 * 1024 * 1024;
    double ai = static_cast<double>(total_flops) / prog.estimates.external_mem_bytes;
    std::cout << "  Arithmetic intensity: " << ai << " FLOPs/byte\n";
    check(ai > 0, "Arithmetic intensity > 0");
}

// ============================================================================
// Test 6: TileLayout Resource Distribution
// ============================================================================

void test_tile_layout_integration() {
    std::cout << "\n=== Test: TileLayout Resource Distribution ===\n";

    // Use iteration-aware layout (default) with 4 channels, 4 L3 tiles, 8 L2 banks
    auto sched = matmul_output_stationary(256, 256, 256, 64, 64, 64);
    DMProgram prog = compile_schedule(sched);

    // Collect unique L3 tile IDs and L2 bank IDs from DMA and BM instructions
    std::set<uint8_t> dma_l3_tiles;
    std::set<uint8_t> bm_src_l3;
    std::set<uint8_t> bm_dst_l2;
    std::set<uint8_t> str_l2_banks;

    for (const auto& instr : prog.instructions) {
        if (instr.opcode == DMOpcode::DMA_LOAD_TILE) {
            if (auto* d = std::get_if<DMAOperands>(&instr.operands)) {
                dma_l3_tiles.insert(d->l3_tile_id);
            }
        }
        if (instr.opcode == DMOpcode::BM_MOVE_TILE) {
            if (auto* b = std::get_if<BlockMoverOperands>(&instr.operands)) {
                bm_src_l3.insert(b->src_l3_tile_id);
                bm_dst_l2.insert(b->dst_l2_bank_id);
            }
        }
        if (instr.opcode == DMOpcode::STR_FEED_ROWS ||
            instr.opcode == DMOpcode::STR_FEED_COLS) {
            if (auto* s = std::get_if<StreamerOperands>(&instr.operands)) {
                str_l2_banks.insert(s->l2_bank_id);
            }
        }
    }

    std::cout << "  DMA L3 tile IDs used: " << dma_l3_tiles.size() << " {";
    for (auto id : dma_l3_tiles) std::cout << " " << (int)id;
    std::cout << " }\n";

    std::cout << "  BM src L3 IDs used:   " << bm_src_l3.size() << " {";
    for (auto id : bm_src_l3) std::cout << " " << (int)id;
    std::cout << " }\n";

    std::cout << "  BM dst L2 IDs used:   " << bm_dst_l2.size() << " {";
    for (auto id : bm_dst_l2) std::cout << " " << (int)id;
    std::cout << " }\n";

    std::cout << "  STR L2 bank IDs used: " << str_l2_banks.size() << " {";
    for (auto id : str_l2_banks) std::cout << " " << (int)id;
    std::cout << " }\n";

    // With iteration-aware layout and 4 channels, A tiles go to even channels (0,2)
    // and B tiles go to odd channels (1,3). L3 tile IDs = channel % num_l3_tiles.
    check(dma_l3_tiles.size() > 1,
          "DMA uses multiple L3 tile slots (got " + std::to_string(dma_l3_tiles.size()) + ")");
    check(bm_dst_l2.size() > 1,
          "BM distributes across multiple L2 banks (got " + std::to_string(bm_dst_l2.size()) + ")");
    check(str_l2_banks.size() > 1,
          "Streamer reads from multiple L2 banks (got " + std::to_string(str_l2_banks.size()) + ")");

    // Verify A and B tiles don't always land on the same L3 tile
    // (iteration-aware should separate them)
    std::set<uint8_t> a_l3, b_l3;
    for (const auto& instr : prog.instructions) {
        if (instr.opcode == DMOpcode::DMA_LOAD_TILE) {
            if (auto* d = std::get_if<DMAOperands>(&instr.operands)) {
                if (d->matrix == MatrixID::A) a_l3.insert(d->l3_tile_id);
                if (d->matrix == MatrixID::B) b_l3.insert(d->l3_tile_id);
            }
        }
    }

    std::cout << "  A tile L3 slots: {";
    for (auto id : a_l3) std::cout << " " << (int)id;
    std::cout << " }\n";
    std::cout << "  B tile L3 slots: {";
    for (auto id : b_l3) std::cout << " " << (int)id;
    std::cout << " }\n";

    // With iteration-aware: A on even channels, B on odd channels
    // So A L3 tiles and B L3 tiles should be different sets
    bool separated = true;
    for (auto a_id : a_l3) {
        if (b_l3.count(a_id)) { separated = false; break; }
    }
    check(separated, "A and B tiles use separate L3 slots (iteration-aware layout)");
}

// ============================================================================
// Test 7: Layout Policy Selection
// ============================================================================

void test_layout_policy_selection() {
    std::cout << "\n=== Test: Layout Policy Selection ===\n";

    // Test with round-robin layout
    Schedule sched("test_rr");
    sched.tensor(Tensor{"A", MatrixID::A, {128, 128}, 0x0000, DataType::FP32})
         .tensor(Tensor{"B", MatrixID::B, {128, 128}, 0x4000, DataType::FP32})
         .tensor(Tensor{"C", MatrixID::C, {128, 128}, 0x8000, DataType::FP32})
         .tile("ti", 64).tile("tj", 64).tile("tk", 64)
         .dataflow(DMProgram::Dataflow::OUTPUT_STATIONARY)
         .layout_policy(isa::LayoutPolicy::ROUND_ROBIN)
         .num_channels(4);

    sched.for_tiles("ti")
        .for_tiles("tj")
            .for_tiles("tk")
                .load(MatrixID::A)
                .load(MatrixID::B)
                .barrier()
                .move(MatrixID::A)
                .move(MatrixID::B)
                .stream_rows(MatrixID::A)
                .stream_cols(MatrixID::B)
            .end()
            .drain()
            .writeback(MatrixID::C)
            .store(MatrixID::C)
        .end()
    .end();

    DMProgram prog = compile_schedule(sched);

    check(prog.instructions.size() > 0, "Round-robin schedule compiles");
    check(prog.instructions.back().opcode == DMOpcode::HALT, "Ends with HALT");

    // Collect L3 tile IDs
    std::set<uint8_t> l3_tiles;
    for (const auto& instr : prog.instructions) {
        if (instr.opcode == DMOpcode::DMA_LOAD_TILE) {
            if (auto* d = std::get_if<DMAOperands>(&instr.operands)) {
                l3_tiles.insert(d->l3_tile_id);
            }
        }
    }
    check(l3_tiles.size() > 1,
          "Round-robin distributes across L3 tiles (got " + std::to_string(l3_tiles.size()) + ")");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Schedule DSL Test Suite\n";
    std::cout << std::string(60, '*') << "\n";

    test_fluent_api();
    test_matmul_schedule();
    test_conv2d_schedule();
    test_softmax_schedule();
    test_large_matmul();
    test_tile_layout_integration();
    test_layout_policy_selection();

    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << std::string(60, '*') << "\n\n";

    return failed > 0 ? 1 : 0;
}

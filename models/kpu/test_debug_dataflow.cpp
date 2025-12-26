#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>

using namespace sw::kpu;

int main() {
    // Create minimal config
    KPUSimulator::Config config;
    config.memory_bank_count = 1;
    config.l3_tile_count = 1;
    config.l2_bank_count = 1;
    config.l1_buffer_count = 1;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count = 1;
    config.block_mover_count = 1;
    config.streamer_count = 2;
    config.processor_array_rows = 16;
    config.processor_array_cols = 16;
    config.use_systolic_array_mode = true;

    KPUSimulator kpu(config);

    // Simple 2x2 x 2x2 = 2x2 matmul test
    std::vector<float> A = {1.0f, 2.0f,   // First row
                            3.0f, 4.0f};  // Second row
    std::vector<float> B = {5.0f, 6.0f,   // First row
                            7.0f, 8.0f};  // Second row
    // Expected C = [[19, 22], [43, 50]]

    std::cout << "=== Step 1: Write to Memory Bank ===\n";
    kpu.write_memory_bank(0, 0x0000, A.data(), A.size() * sizeof(float));
    kpu.write_memory_bank(0, 0x1000, B.data(), B.size() * sizeof(float));

    // Verify memory bank
    std::vector<float> verify_A(4), verify_B(4);
    kpu.read_memory_bank(0, 0x0000, verify_A.data(), verify_A.size() * sizeof(float));
    kpu.read_memory_bank(0, 0x1000, verify_B.data(), verify_B.size() * sizeof(float));
    std::cout << "Memory Bank A: " << verify_A[0] << ", " << verify_A[1] << ", " << verify_A[2] << ", " << verify_A[3] << "\n";
    std::cout << "Memory Bank B: " << verify_B[0] << ", " << verify_B[1] << ", " << verify_B[2] << ", " << verify_B[3] << "\n";

    std::cout << "\n=== Step 2: Write to L3 ===\n";
    kpu.write_l3_tile(0, 0x0000, A.data(), A.size() * sizeof(float));
    kpu.write_l3_tile(0, 0x1000, B.data(), B.size() * sizeof(float));

    // Verify L3
    kpu.read_l3_tile(0, 0x0000, verify_A.data(), verify_A.size() * sizeof(float));
    kpu.read_l3_tile(0, 0x1000, verify_B.data(), verify_B.size() * sizeof(float));
    std::cout << "L3 A: " << verify_A[0] << ", " << verify_A[1] << ", " << verify_A[2] << ", " << verify_A[3] << "\n";
    std::cout << "L3 B: " << verify_B[0] << ", " << verify_B[1] << ", " << verify_B[2] << ", " << verify_B[3] << "\n";

    std::cout << "\n=== Step 3: BlockMover L3->L2 ===\n";
    bool block_a_done = false, block_b_done = false;
    kpu.start_block_transfer(0, 0, 0x0000, 0, 0x0000, 2, 2, sizeof(float),
                             BlockMover::TransformType::IDENTITY,
                             [&](){ block_a_done = true; });
    kpu.start_block_transfer(0, 0, 0x1000, 0, 0x1000, 2, 2, sizeof(float),
                             BlockMover::TransformType::IDENTITY,
                             [&](){ block_b_done = true; });
    kpu.run_until_idle();
    std::cout << "BlockMover callbacks: A=" << block_a_done << ", B=" << block_b_done << "\n";

    // Verify L2
    kpu.read_l2_bank(0, 0x0000, verify_A.data(), verify_A.size() * sizeof(float));
    kpu.read_l2_bank(0, 0x1000, verify_B.data(), verify_B.size() * sizeof(float));
    std::cout << "L2 A: " << verify_A[0] << ", " << verify_A[1] << ", " << verify_A[2] << ", " << verify_A[3] << "\n";
    std::cout << "L2 B: " << verify_B[0] << ", " << verify_B[1] << ", " << verify_B[2] << ", " << verify_B[3] << "\n";

    std::cout << "\n=== Step 4: Streamer L2->L1 ===\n";
    bool stream_a_done = false, stream_b_done = false;
    kpu.start_row_stream(0, 0, 0, 0x0000, 0x0000, 2, 2, sizeof(float), 16,
                         Streamer::StreamDirection::L2_TO_L1,
                         [&](){ stream_a_done = true; });
    kpu.start_column_stream(1, 0, 0, 0x1000, 0x1000, 2, 2, sizeof(float), 16,
                            Streamer::StreamDirection::L2_TO_L1,
                            [&](){ stream_b_done = true; });
    kpu.run_until_idle();
    std::cout << "Streamer callbacks: A=" << stream_a_done << ", B=" << stream_b_done << "\n";

    // Verify L1
    kpu.read_l1_buffer(0, 0x0000, verify_A.data(), verify_A.size() * sizeof(float));
    kpu.read_l1_buffer(0, 0x1000, verify_B.data(), verify_B.size() * sizeof(float));
    std::cout << "L1 A: " << verify_A[0] << ", " << verify_A[1] << ", " << verify_A[2] << ", " << verify_A[3] << "\n";
    std::cout << "L1 B: " << verify_B[0] << ", " << verify_B[1] << ", " << verify_B[2] << ", " << verify_B[3] << "\n";

    std::cout << "\n=== Step 5: Compute ===\n";
    bool compute_done = false;
    kpu.start_matmul(0, 0, 2, 2, 2, 0x0000, 0x1000, 0x2000,
                     [&](){ compute_done = true; });
    kpu.run_until_idle();
    std::cout << "Compute callback: " << compute_done << "\n";

    // Verify result
    std::vector<float> C(4);
    kpu.read_l1_buffer(0, 0x2000, C.data(), C.size() * sizeof(float));
    std::cout << "Result C:\n";
    std::cout << "  [" << C[0] << ", " << C[1] << "]\n";
    std::cout << "  [" << C[2] << ", " << C[3] << "]\n";
    std::cout << "Expected: [[19, 22], [43, 50]]\n";

    bool success = (std::abs(C[0] - 19.0f) < 0.01f &&
                   std::abs(C[1] - 22.0f) < 0.01f &&
                   std::abs(C[2] - 43.0f) < 0.01f &&
                   std::abs(C[3] - 50.0f) < 0.01f);

    std::cout << "\n=== Test " << (success ? "PASSED" : "FAILED") << " ===\n";
    return success ? 0 : 1;
}

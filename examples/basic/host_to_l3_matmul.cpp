/**
 * @file host_to_l3_matmul.cpp
 * @brief Full functional matmul: Host → L3 → Shadow → Compute → Local
 *
 * This example demonstrates the complete data flow for a matrix multiplication
 * where data starts in host memory, is DMA'd directly into L3, a shadow copy
 * is written to local (external) memory for persistence, and the result is
 * computed and stored back to local memory.
 *
 * Data Flow:
 * ==========
 *
 *   Host Memory (A, B)     External Memory (Local)
 *        |                        ↑
 *        | dma_host_to_l3()       | dma_l3_to_external() [shadow + result]
 *        ↓                        |
 *   +----L3 Tiles----+            |
 *        |                        |
 *        | start_block_transfer() |
 *        ↓                        |
 *   +----L2 Banks----+←-----------+
 *        |
 *        | start_row_stream() / start_column_stream()
 *        ↓
 *   +----L1 Buffers--+
 *        |
 *        | start_matmul()
 *        ↓
 *   +--Compute Fabric-+
 *        |
 *        | [drain path: L1 → L2 → L3 → External]
 *        ↓
 *   Result in External Memory
 *
 * Key Concepts:
 * - Direct Host→L3 DMA bypasses local memory for input data
 * - Shadow copy to local memory provides backup/persistence
 * - L3 acts as the working set cache for compute
 * - Result flows back through the hierarchy to local memory
 */

#include <sw/kpu/kpu_simulator.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <atomic>
#include <functional>

using namespace sw::kpu;

// ============================================================================
// Utility Functions
// ============================================================================

void print_separator(const std::string& title = "") {
    if (title.empty()) {
        std::cout << std::string(70, '-') << "\n";
    } else {
        std::cout << "\n=== " << title << " " << std::string(65 - title.length(), '=') << "\n";
    }
}

std::string format_bytes(Size bytes) {
    if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

// Fill with random values
void fill_random(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    static std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& v : data) {
        v = dist(gen);
    }
}

// Reference matmul for verification
void reference_matmul(const float* A, const float* B, float* C,
                      Size M, Size N, Size K) {
    for (Size i = 0; i < M; ++i) {
        for (Size j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify results
bool verify_result(const float* computed, const float* reference, Size count,
                   float tolerance = 1e-4f) {
    Size errors = 0;
    for (Size i = 0; i < count; ++i) {
        float diff = std::abs(computed[i] - reference[i]);
        if (diff > tolerance) {
            if (errors < 5) {
                std::cout << "  Mismatch at [" << i << "]: computed=" << computed[i]
                          << " expected=" << reference[i] << " diff=" << diff << "\n";
            }
            errors++;
        }
    }
    if (errors > 0) {
        std::cout << "  Total mismatches: " << errors << " / " << count << "\n";
    }
    return errors == 0;
}

// ============================================================================
// Main Example: Host → L3 → Shadow → Compute → Local
// ============================================================================

int main() {
    std::cout << R"(
================================================================================
     Host → L3 → Shadow → Compute → Local Memory MatMul Example
================================================================================

This example demonstrates the full data flow for matrix multiplication:

  1. Initialize matrices A and B in host memory
  2. DMA transfer: Host → L3 (direct PCIe, bypassing local memory)
  3. Shadow copy: L3 → External (local) memory for persistence
  4. Block move: L3 → L2
  5. Streaming: L2 → L1 (rows for A, columns for B)
  6. Compute: Systolic array performs C = A × B
  7. Drain: L1 → L2 → L3
  8. Store: L3 → External (local) memory for result

================================================================================
)";

    print_separator("1. Configure KPU Simulator");

    // Create a minimal configuration for testing
    KPUSimulator::Config config;
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 256;
    config.host_memory_bandwidth_gbps = 50;
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 256;
    config.memory_bandwidth_gbps = 25;
    config.memory_controller_count = 1;
    config.page_buffer_count = 2;
    config.page_buffer_capacity_kb = 32;
    config.l3_layer.tile_groups = { {"l3", {128}, 1} };
    config.l2_layer.bank_groups = { {"l2", {64}, 4} };
    config.l1_layer.buffer_groups = { {"l1", {64 * 1024}, 64} };
    config.compute_tile_count = 1;
    config.processor_array_rows = 8;
    config.processor_array_cols = 8;
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;
    config.dma_engine_count = 2;
    config.l3_layer.block_mover_count = 2;
    config.streamer_count = 4;

    // Print configuration
    std::cout << "\nKPU Configuration:\n";
    std::cout << "  Host Memory:     " << config.host_memory_region_count << " regions × "
              << config.host_memory_region_capacity_mb << " MB\n";
    std::cout << "  External Memory: " << config.memory_bank_count << " banks × "
              << config.memory_bank_capacity_mb << " MB\n";
    std::cout << "  L3 Tiles:        " << config.l3_layer.total_tiles() << " × "
              << config.l3_layer.tile_groups[0].tile.capacity_kb << " KB\n";
    std::cout << "  L2 Banks:        " << config.l2_layer.total_banks() << " × "
              << config.l2_layer.bank_groups[0].bank.capacity_kb << " KB\n";
    std::cout << "  L1 Buffers:      " << config.l1_layer.total_buffers() << " × "
              << config.l1_layer.buffer_groups[0].buffer.capacity_bytes / 1024 << " KB\n";
    std::cout << "  Systolic Array:  " << config.processor_array_rows << "×"
              << config.processor_array_cols << "\n";

    // Create simulator
    KPUSimulator kpu(config);

    print_separator("2. Initialize Host Memory");

    // Matrix dimensions (must fit in L2/L3 for this simple example)
    const Size M = 32, N = 32, K = 32;
    const Size elem_size = sizeof(float);
    const Size A_bytes = M * K * elem_size;
    const Size B_bytes = K * N * elem_size;
    const Size C_bytes = M * N * elem_size;

    std::cout << "\nMatrix Dimensions:\n";
    std::cout << "  C[" << M << "," << N << "] = A[" << M << "," << K << "] × B["
              << K << "," << N << "]\n";
    std::cout << "  A: " << format_bytes(A_bytes) << "\n";
    std::cout << "  B: " << format_bytes(B_bytes) << "\n";
    std::cout << "  C: " << format_bytes(C_bytes) << "\n";
    std::cout << "  Total: " << format_bytes(A_bytes + B_bytes + C_bytes) << "\n";

    // Create and initialize host data
    std::vector<float> A_host(M * K);
    std::vector<float> B_host(K * N);
    std::vector<float> C_host(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    fill_random(A_host, -1.0f, 1.0f);
    fill_random(B_host, -1.0f, 1.0f);

    std::cout << "\nHost data initialized (random values [-1, 1])\n";
    std::cout << "  A[0,0]=" << A_host[0] << "  A[0,1]=" << A_host[1] << "\n";
    std::cout << "  B[0,0]=" << B_host[0] << "  B[0,1]=" << B_host[1] << "\n";

    // Compute reference result for verification
    reference_matmul(A_host.data(), B_host.data(), C_ref.data(), M, N, K);
    std::cout << "\nReference result computed.\n";

    // Write to host memory region
    Address host_region_base = kpu.get_host_memory_region_base(0);
    Address host_A = host_region_base;
    Address host_B = host_A + A_bytes;

    kpu.write_host_memory(0, 0, A_host.data(), A_bytes);
    kpu.write_host_memory(0, A_bytes, B_host.data(), B_bytes);

    std::cout << "  Written to host memory at 0x" << std::hex << host_A
              << " and 0x" << host_B << std::dec << "\n";

    print_separator("3. DMA: Host → L3 (Direct PCIe Transfer)");

    // Get L3 tile base addresses
    Address l3_base = kpu.get_l3_tile_base(0);
    Address l3_A = l3_base;
    Address l3_B = l3_A + A_bytes;
    Address l3_C = l3_B + B_bytes;

    std::cout << "\nL3 Tile Layout:\n";
    std::cout << "  L3 Base: 0x" << std::hex << l3_base << std::dec << "\n";
    std::cout << "  A @ L3:  0x" << std::hex << l3_A << std::dec << "\n";
    std::cout << "  B @ L3:  0x" << std::hex << l3_B << std::dec << "\n";
    std::cout << "  C @ L3:  0x" << std::hex << l3_C << std::dec << "\n";

    // Track DMA completions
    std::atomic<int> dma_complete{0};

    auto dma_callback = [&dma_complete]() {
        dma_complete++;
    };

    // DMA A from host to L3
    std::cout << "\nStarting DMA transfers...\n";
    kpu.dma_host_to_l3(0, host_A, l3_A, A_bytes, dma_callback);
    std::cout << "  DMA[0]: Host → L3 (matrix A) started\n";

    // DMA B from host to L3 (using second DMA engine if available)
    size_t dma_b = (kpu.get_dma_engine_count() > 1) ? 1 : 0;
    kpu.dma_host_to_l3(dma_b, host_B, l3_B, B_bytes, dma_callback);
    std::cout << "  DMA[" << dma_b << "]: Host → L3 (matrix B) started\n";

    // Wait for DMA completion
    while (dma_complete < 2) {
        kpu.step();
    }
    std::cout << "\nDMA transfers complete at cycle " << kpu.get_current_cycle() << "\n";

    print_separator("4. Shadow Copy: L3 → External (Local) Memory");

    // Get external memory base addresses for shadow copy
    Address ext_base = kpu.get_external_bank_base(0);
    Address ext_A = ext_base;
    Address ext_B = ext_A + A_bytes;
    Address ext_C = ext_B + B_bytes;

    std::cout << "\nExternal Memory Layout:\n";
    std::cout << "  Ext Base: 0x" << std::hex << ext_base << std::dec << "\n";
    std::cout << "  A @ Ext:  0x" << std::hex << ext_A << std::dec << "\n";
    std::cout << "  B @ Ext:  0x" << std::hex << ext_B << std::dec << "\n";
    std::cout << "  C @ Ext:  0x" << std::hex << ext_C << std::dec << "\n";

    // Shadow copy L3 → External
    dma_complete = 0;

    kpu.dma_l3_to_external(0, l3_A, ext_A, A_bytes, dma_callback);
    std::cout << "\n  DMA[0]: L3 → External (shadow A) started\n";

    kpu.dma_l3_to_external(dma_b, l3_B, ext_B, B_bytes, dma_callback);
    std::cout << "  DMA[" << dma_b << "]: L3 → External (shadow B) started\n";

    // Wait for shadow copy completion
    while (dma_complete < 2) {
        kpu.step();
    }
    std::cout << "\nShadow copies complete at cycle " << kpu.get_current_cycle() << "\n";

    print_separator("5. Block Move: L3 → L2");

    // Get L2 bank base addresses
    Address l2_base = kpu.get_l2_bank_base(0);
    Address l2_A = 0;  // Offset within L2 bank
    Address l2_B = A_bytes;
    Address l2_C = A_bytes + B_bytes;

    std::cout << "\nL2 Bank Layout (offsets within bank 0):\n";
    std::cout << "  L2 Base: 0x" << std::hex << l2_base << std::dec << "\n";
    std::cout << "  A @ L2:  offset 0x" << std::hex << l2_A << std::dec << "\n";
    std::cout << "  B @ L2:  offset 0x" << std::hex << l2_B << std::dec << "\n";
    std::cout << "  C @ L2:  offset 0x" << std::hex << l2_C << std::dec << "\n";

    std::atomic<int> bm_complete{0};
    auto bm_callback = [&bm_complete]() {
        bm_complete++;
    };

    // Block move A: L3[0] → L2[0]
    kpu.start_block_transfer(0,          // block_mover_id
                             0,          // src_l3_tile_id
                             0,          // src_offset (offset within L3 tile)
                             0,          // dst_l2_bank_id
                             l2_A,       // dst_offset
                             M,          // block_height
                             K,          // block_width
                             elem_size,  // element_size
                             BlockMover::TransformType::IDENTITY,
                             bm_callback);
    std::cout << "\n  BlockMover[0]: L3[0] → L2[0] (matrix A) started\n";

    // Block move B: L3[0] → L2[0] (or L2[1] if available)
    size_t l2_b_bank = (kpu.get_l2_bank_count() > 1) ? 1 : 0;
    size_t bm_b = (kpu.get_block_mover_count() > 1) ? 1 : 0;
    Address l2_B_offset = (l2_b_bank == 0) ? l2_B : 0;

    kpu.start_block_transfer(bm_b,       // block_mover_id
                             0,          // src_l3_tile_id
                             A_bytes,    // src_offset (B is after A in L3)
                             l2_b_bank,  // dst_l2_bank_id
                             l2_B_offset,// dst_offset
                             K,          // block_height
                             N,          // block_width
                             elem_size,  // element_size
                             BlockMover::TransformType::IDENTITY,
                             bm_callback);
    std::cout << "  BlockMover[" << bm_b << "]: L3[0] → L2[" << l2_b_bank << "] (matrix B) started\n";

    // Wait for block moves
    while (bm_complete < 2) {
        kpu.step();
    }
    std::cout << "\nBlock moves complete at cycle " << kpu.get_current_cycle() << "\n";

    print_separator("6. Streaming: L2 → L1");

    // L1 buffer layout
    Address l1_A = 0;
    Address l1_B = A_bytes;
    Address l1_C = A_bytes + B_bytes;

    std::atomic<int> str_complete{0};
    auto str_callback = [&str_complete]() {
        str_complete++;
    };

    Size systolic_size = kpu.get_systolic_array_rows();

    // Stream A rows: L2[0] → L1
    kpu.start_row_stream(0,          // streamer_id
                         0,          // l2_bank_id
                         0,          // l1_scratchpad_id
                         l2_A,       // l2_base_addr
                         l1_A,       // l1_base_addr
                         M,          // matrix_height
                         K,          // matrix_width
                         elem_size,  // element_size
                         systolic_size,
                         Streamer::StreamDirection::L2_TO_L1,
                         str_callback);
    std::cout << "\n  Streamer[0]: L2[0] → L1 (matrix A rows) started\n";

    // Stream B columns: L2 → L1
    size_t str_b = (kpu.get_streamer_count() > 1) ? 1 : 0;
    kpu.start_column_stream(str_b,      // streamer_id
                            l2_b_bank,  // l2_bank_id
                            0,          // l1_scratchpad_id
                            l2_B_offset,// l2_base_addr
                            l1_B,       // l1_base_addr
                            K,          // matrix_height
                            N,          // matrix_width
                            elem_size,  // element_size
                            systolic_size,
                            Streamer::StreamDirection::L2_TO_L1,
                            str_callback);
    std::cout << "  Streamer[" << str_b << "]: L2[" << l2_b_bank << "] → L1 (matrix B cols) started\n";

    // Wait for streaming
    while (str_complete < 2) {
        kpu.step();
    }
    std::cout << "\nStreaming complete at cycle " << kpu.get_current_cycle() << "\n";

    print_separator("7. Compute: Systolic Array MatMul");

    std::atomic<bool> compute_done{false};
    auto compute_callback = [&compute_done]() {
        compute_done = true;
    };

    kpu.start_matmul(0,           // tile_id
                     0,           // l1_buffer_id
                     M, N, K,     // dimensions
                     l1_A,        // a_addr
                     l1_B,        // b_addr
                     l1_C,        // c_addr
                     compute_callback);
    std::cout << "\n  Compute[0]: MatMul " << M << "×" << N << "×" << K << " started\n";

    // Wait for compute
    Cycle compute_start = kpu.get_current_cycle();
    while (!compute_done) {
        kpu.step();
    }
    Cycle compute_end = kpu.get_current_cycle();
    std::cout << "\nCompute complete at cycle " << compute_end
              << " (took " << (compute_end - compute_start) << " cycles)\n";

    print_separator("8. Drain: L1 → L2 → L3 → External");

    // Note: The drain path requires special handling because:
    // - Streamer supports L1_TO_L2 direction
    // - BlockMover only supports L3→L2 direction (not L2→L3)
    // - We use direct memory operations for L2→L3

    // Drain C from L1 to L2
    str_complete = 0;
    kpu.start_row_stream(0,          // streamer_id
                         0,          // l2_bank_id
                         0,          // l1_scratchpad_id
                         l2_C,       // l2_base_addr
                         l1_C,       // l1_base_addr
                         M,          // matrix_height
                         N,          // matrix_width
                         elem_size,  // element_size
                         systolic_size,
                         Streamer::StreamDirection::L1_TO_L2,
                         str_callback);
    std::cout << "\n  Streamer[0]: L1 → L2 (drain C) started\n";

    while (str_complete < 1) {
        kpu.step();
    }
    std::cout << "  Drain L1→L2 complete at cycle " << kpu.get_current_cycle() << "\n";

    // L2 → L3 transfer using direct memory operations
    // (BlockMover API only supports L3→L2, not the reverse)
    std::vector<float> C_buffer(M * N);
    kpu.read_l2_bank(0, l2_C, C_buffer.data(), C_bytes);
    kpu.write_l3_tile(0, A_bytes + B_bytes, C_buffer.data(), C_bytes);
    std::cout << "  Direct: L2 → L3 (drain C) complete\n";

    // DMA C from L3 to external memory
    dma_complete = 0;
    kpu.dma_l3_to_external(0, l3_C, ext_C, C_bytes, dma_callback);
    std::cout << "  DMA[0]: L3 → External (store C) started\n";

    while (dma_complete < 1) {
        kpu.step();
    }
    std::cout << "  DMA L3→External complete at cycle " << kpu.get_current_cycle() << "\n";

    print_separator("9. Verification");

    // Read result from external memory
    kpu.read_memory_bank(0, ext_C - ext_base, C_host.data(), C_bytes);

    std::cout << "\nResult read from external memory.\n";
    std::cout << "  C[0,0]=" << C_host[0] << " (expected: " << C_ref[0] << ")\n";
    std::cout << "  C[0,1]=" << C_host[1] << " (expected: " << C_ref[1] << ")\n";
    std::cout << "  C[1,0]=" << C_host[N] << " (expected: " << C_ref[N] << ")\n";

    bool success = verify_result(C_host.data(), C_ref.data(), M * N);

    print_separator("Summary");

    std::cout << "\nExecution Statistics:\n";
    std::cout << "  Total Cycles: " << kpu.get_current_cycle() << "\n";
    std::cout << "  Compute Cycles: " << (compute_end - compute_start) << "\n";
    std::cout << "  Overhead Cycles: " << (kpu.get_current_cycle() - (compute_end - compute_start)) << "\n";

    // Calculate performance
    double flops = 2.0 * M * N * K;  // 2 ops per MAC
    double cycles = static_cast<double>(compute_end - compute_start);
    double flops_per_cycle = flops / cycles;

    std::cout << "\nPerformance:\n";
    std::cout << "  FLOPs: " << std::fixed << std::setprecision(0) << flops << "\n";
    std::cout << "  FLOPs/cycle: " << std::setprecision(2) << flops_per_cycle << "\n";

    std::cout << "\nData Movement Summary:\n";
    std::cout << "  Host → L3:       " << format_bytes(A_bytes + B_bytes) << " (direct PCIe)\n";
    std::cout << "  L3 → External:   " << format_bytes(A_bytes + B_bytes + C_bytes) << " (shadow + result)\n";
    std::cout << "  L3 → L2:         " << format_bytes(A_bytes + B_bytes) << "\n";
    std::cout << "  L2 → L1:         " << format_bytes(A_bytes + B_bytes) << "\n";
    std::cout << "  L1 → L2 → L3:    " << format_bytes(C_bytes) << " (drain)\n";

    print_separator("");
    std::cout << "\nResult: " << (success ? "PASS" : "FAIL") << "\n";

    if (success) {
        std::cout << R"(
The example demonstrates the complete data flow:
  1. Host memory → L3 (direct PCIe DMA, bypassing local memory)
  2. L3 → External (shadow copy for persistence)
  3. L3 → L2 (BlockMover)
  4. L2 → L1 (Streamer - rows/columns)
  5. Compute (Systolic array)
  6. L1 → L2 → L3 → External (drain path)

This pattern is useful when:
  - Input data originates on the host (e.g., inference server)
  - Local memory serves as checkpoint/backup
  - L3 provides fast working set access
  - Results must persist in local memory
)";
    }

    return success ? 0 : 1;
}

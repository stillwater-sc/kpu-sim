#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>

#include "../test_utilities.hpp"

using namespace sw;
using namespace sw::kpu;
using namespace sw::trace;

// Test fixture for BlockMover tracing tests
class BlockMoverTracingFixture {
public:
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::unique_ptr<BlockMover> block_mover;
    TraceLogger& logger;

    BlockMoverTracingFixture()
        : logger(TraceLogger::instance())
    {
        // Create 2 L3 tiles of 128KB each
        l3_tiles.emplace_back(0, 128);  // ID 0, 128 KB capacity
        l3_tiles.emplace_back(1, 128);  // ID 1, 128 KB capacity

        // Create 2 L2 banks of 64KB each
        l2_banks.emplace_back(0, 64);  // ID 0, 64 KB capacity
        l2_banks.emplace_back(1, 64);  // ID 1, 64 KB capacity

        // Create BlockMover: engine 0, associated with L3 tile 0, 1 GHz, 100 GB/s
        block_mover = std::make_unique<BlockMover>(0, 0, 1.0, 100.0);

        // Reset and configure tracing
        logger.clear();
        logger.set_enabled(true);
        block_mover->enable_tracing();
    }

    ~BlockMoverTracingFixture() {
        // Leave logger enabled for inspection after tests
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_pattern(size_t size, uint8_t start_value = 0) {
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), start_value);
        return data;
    }

    // Helper to generate float matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }
};

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Single BlockMover Transfer - Identity", "[trace][block_mover]") {
    const size_t block_height = 4;
    const size_t block_width = 4;
    const size_t element_size = sizeof(float);
    const size_t block_size = block_height * block_width * element_size;
    const Address src_addr = 0x1000;
    const Address dst_addr = 0x0;

    // Generate and write test data to L3 tile
    auto test_data = generate_test_pattern(block_size, 0xAA);
    l3_tiles[0].write(src_addr, test_data.data(), block_size);

    // Set initial cycle
    block_mover->set_cycle(1000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue transfer
    bool transfer_complete = false;
    block_mover->enqueue_block_transfer(
        0, src_addr,  // L3 tile 0, source address
        0, dst_addr,  // L2 bank 0, destination address
        block_height, block_width, element_size,
        BlockMover::TransformType::IDENTITY,
        [&transfer_complete]() { transfer_complete = true; }
    );

    // Process the transfer - advance cycle each iteration
    // Traces are logged during processing, not on enqueue
    while (!transfer_complete) {
        block_mover->set_cycle(block_mover->get_cycle() + 1);
        block_mover->process_transfers(l3_tiles, l2_banks);
    }

    // Should have logged issue and completion traces
    REQUIRE(logger.get_trace_count() >= initial_trace_count + 2);

    // Get traces for this BlockMover
    auto bm_traces = logger.get_component_traces(ComponentType::BLOCK_MOVER, 0);
    REQUIRE(bm_traces.size() >= 2);

    // Verify the last two traces (issue and completion)
    auto& issue_trace = bm_traces[bm_traces.size() - 2];
    auto& complete_trace = bm_traces[bm_traces.size() - 1];

    // Verify issue trace
    REQUIRE(issue_trace.component_type == ComponentType::BLOCK_MOVER);
    REQUIRE(issue_trace.component_id == 0);
    REQUIRE(issue_trace.transaction_type == TransactionType::TRANSFER);
    // Cycle may be 1000 or 1001 depending on when transfer starts processing
    REQUIRE(issue_trace.cycle_issue >= 1000);
    REQUIRE(issue_trace.cycle_issue <= 1001);
    REQUIRE(issue_trace.status == TransactionStatus::ISSUED);
    REQUIRE(issue_trace.description.find("IDENTITY") != std::string::npos);

    // Verify completion trace
    REQUIRE(complete_trace.component_type == ComponentType::BLOCK_MOVER);
    REQUIRE(complete_trace.component_id == 0);
    REQUIRE(complete_trace.transaction_type == TransactionType::TRANSFER);
    REQUIRE(complete_trace.status == TransactionStatus::COMPLETED);
    REQUIRE(complete_trace.cycle_complete >= complete_trace.cycle_issue);

    // Verify payload data
    REQUIRE(std::holds_alternative<DMAPayload>(complete_trace.payload));
    const auto& payload = std::get<DMAPayload>(complete_trace.payload);
    REQUIRE(payload.bytes_transferred == block_size);
    REQUIRE(payload.source.address == src_addr);
    REQUIRE(payload.source.type == ComponentType::L3_TILE);
    REQUIRE(payload.destination.address == dst_addr);
    REQUIRE(payload.destination.type == ComponentType::L2_BANK);

    std::cout << "\n=== BlockMover Transfer Trace (IDENTITY) ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Transfer Size: " << block_size << " bytes" << std::endl;
    std::cout << "Bandwidth: " << payload.bandwidth_gb_s << " GB/s" << std::endl;
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: BlockMover Transfer - Transpose", "[trace][block_mover][transpose]") {
    const size_t matrix_rows = 8;
    const size_t matrix_cols = 8;
    const size_t element_size = sizeof(float);
    const size_t matrix_size = matrix_rows * matrix_cols * element_size;
    const Address src_addr = 0x0;
    const Address dst_addr = 0x0;

    // Generate and write test matrix to L3 tile
    auto matrix_data = generate_matrix(matrix_rows, matrix_cols, 1.0f);
    l3_tiles[0].write(src_addr, matrix_data.data(), matrix_size);

    // Set initial cycle
    block_mover->set_cycle(2000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue transfer with TRANSPOSE
    bool transfer_complete = false;
    block_mover->enqueue_block_transfer(
        0, src_addr,
        0, dst_addr,
        matrix_rows, matrix_cols, element_size,
        BlockMover::TransformType::TRANSPOSE,
        [&transfer_complete]() { transfer_complete = true; }
    );

    // Process the transfer - advance cycle each iteration
    // Traces are logged during processing, not on enqueue
    while (!transfer_complete) {
        block_mover->set_cycle(block_mover->get_cycle() + 1);
        block_mover->process_transfers(l3_tiles, l2_banks);
    }

    // Should have logged issue and completion traces
    REQUIRE(logger.get_trace_count() >= initial_trace_count + 2);

    // Get traces for this BlockMover
    auto bm_traces = logger.get_component_traces(ComponentType::BLOCK_MOVER, 0);

    // Find the traces for this specific transfer
    auto& issue_trace = bm_traces[bm_traces.size() - 2];
    auto& complete_trace = bm_traces[bm_traces.size() - 1];

    // Verify transformation type is mentioned in description
    REQUIRE(issue_trace.description.find("TRANSPOSE") != std::string::npos);
    REQUIRE(complete_trace.description.find("TRANSPOSE") != std::string::npos);

    std::cout << "\n=== BlockMover Transfer Trace (TRANSPOSE) ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Matrix: " << matrix_rows << "x" << matrix_cols << " (transpose)" << std::endl;
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Multiple BlockMover Transfers", "[trace][block_mover][queue]") {
    const size_t block_height = 4;
    const size_t block_width = 4;
    const size_t element_size = sizeof(float);
    const size_t block_size = block_height * block_width * element_size;

    // Set initial cycle
    block_mover->set_cycle(3000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue multiple transfers
    int num_transfers = 3;
    int completed_count = 0;
    auto completion_callback = [&completed_count]() { completed_count++; };

    for (int i = 0; i < num_transfers; i++) {
        auto test_data = generate_test_pattern(block_size, static_cast<uint8_t>(i * 0x10));
        l3_tiles[0].write(i * block_size, test_data.data(), block_size);

        block_mover->enqueue_block_transfer(
            0, i * block_size,
            0, i * block_size,
            block_height, block_width, element_size,
            BlockMover::TransformType::IDENTITY,
            completion_callback
        );
    }

    // Process all transfers - advance cycle each iteration
    // Traces are logged during processing, not on enqueue
    while (completed_count < num_transfers) {
        block_mover->set_cycle(block_mover->get_cycle() + 1);
        block_mover->process_transfers(l3_tiles, l2_banks);
    }

    // Should have logged issue and completion traces for all transfers
    REQUIRE(logger.get_trace_count() >= initial_trace_count + (num_transfers * 2));

    // Get all BlockMover traces
    auto bm_traces = logger.get_component_traces(ComponentType::BLOCK_MOVER, 0);

    // Verify all transfers have both issue and complete
    int completed_trace_count = 0;
    for (const auto& trace : bm_traces) {
        if (trace.status == TransactionStatus::COMPLETED) {
            completed_trace_count++;
            REQUIRE(trace.cycle_complete >= trace.cycle_issue);
        }
    }

    REQUIRE(completed_trace_count >= num_transfers);

    std::cout << "\n=== Multiple BlockMover Transfers ===" << std::endl;
    std::cout << "Total traces logged: " << logger.get_trace_count() << std::endl;
    std::cout << "BlockMover 0 traces: " << bm_traces.size() << std::endl;
    std::cout << "Completed transfers: " << completed_trace_count << std::endl;
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Export BlockMover to CSV", "[trace][block_mover][export]") {
    const size_t block_size = 4 * 4 * sizeof(float);

    // Clear previous traces for clean export
    logger.clear();

    // Generate some transfers
    block_mover->set_cycle(5000);

    for (int i = 0; i < 2; i++) {
        auto test_data = generate_test_pattern(block_size);
        l3_tiles[0].write(i * block_size, test_data.data(), block_size);

        bool complete = false;
        block_mover->enqueue_block_transfer(
            0, i * block_size,
            0, i * block_size,
            4, 4, sizeof(float),
            BlockMover::TransformType::IDENTITY,
            [&complete]() { complete = true; }
        );

        while (!complete) {
            block_mover->process_transfers(l3_tiles, l2_banks);
            block_mover->set_cycle(block_mover->get_cycle() + 1);
        }
    }

    // Export traces to CSV
    auto csv_path = sw::test::get_test_output_path("block_mover_trace_test.csv");
    bool csv_export_success = export_logger_traces(csv_path, "csv", logger);
    REQUIRE(csv_export_success);

    std::cout << "\n=== BlockMover Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << csv_path << std::endl;
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Export BlockMover to Chrome Trace Format", "[trace][block_mover][export][chrome]") {
    const size_t block_size = 4 * 4 * sizeof(float);

    // Clear previous traces for cleaner visualization
    logger.clear();

    // Generate transfers with clear cycle progression
    for (int i = 0; i < 5; i++) {
        CycleCount start_cycle = 10000 + i * 1000;
        block_mover->set_cycle(start_cycle);

        auto test_data = generate_test_pattern(block_size);
        l3_tiles[0].write(i * block_size, test_data.data(), block_size);

        // Alternate between IDENTITY and TRANSPOSE
        auto transform = (i % 2 == 0) ? BlockMover::TransformType::IDENTITY
                                       : BlockMover::TransformType::TRANSPOSE;

        bool complete = false;
        block_mover->enqueue_block_transfer(
            0, i * block_size,
            0, i * block_size,
            4, 4, sizeof(float),
            transform,
            [&complete]() { complete = true; }
        );

        while (!complete) {
            block_mover->process_transfers(l3_tiles, l2_banks);
            block_mover->set_cycle(block_mover->get_cycle() + 1);
        }
    }

    // Export traces to Chrome trace format
    auto chrome_path = sw::test::get_test_output_path("block_mover_trace_test.trace");
    bool chrome_export_success = export_logger_traces(chrome_path, "chrome", logger);
    REQUIRE(chrome_export_success);

    std::cout << "\n=== Chrome Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << chrome_path << std::endl;
    std::cout << "Open in chrome://tracing for visualization" << std::endl;
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Cycle Range Query for BlockMover", "[trace][block_mover][query]") {
    // Clear for clean test
    logger.clear();

    // Create transfers at different cycle ranges
    std::vector<CycleCount> start_cycles = {1000, 5000, 10000, 15000};

    for (CycleCount start : start_cycles) {
        block_mover->set_cycle(start);
        auto test_data = generate_test_pattern(64);
        l3_tiles[0].write(0, test_data.data(), 64);

        bool complete = false;
        block_mover->enqueue_block_transfer(
            0, 0, 0, 0,
            2, 2, sizeof(float),
            BlockMover::TransformType::IDENTITY,
            [&complete]() { complete = true; }
        );

        while (!complete) {
            block_mover->process_transfers(l3_tiles, l2_banks);
            block_mover->set_cycle(block_mover->get_cycle() + 1);
        }
    }

    // Query specific cycle ranges
    auto early_traces = logger.get_traces_in_range(0, 6000);
    auto late_traces = logger.get_traces_in_range(6000, 20000);

    std::cout << "\n=== Cycle Range Query ===" << std::endl;
    std::cout << "Early traces (0-6000): " << early_traces.size() << std::endl;
    std::cout << "Late traces (6000-20000): " << late_traces.size() << std::endl;

    // Should have captured traces in both ranges
    REQUIRE(early_traces.size() > 0);
    REQUIRE(late_traces.size() > 0);
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: BlockMover Bandwidth Analysis", "[trace][block_mover][analysis]") {
    // Clear for clean test
    logger.clear();

    std::vector<std::pair<size_t, size_t>> block_configs = {
        {2, 2},   // 16 bytes
        {4, 4},   // 64 bytes
        {8, 8},   // 256 bytes
        {16, 16}  // 1024 bytes
    };

    block_mover->set_cycle(20000);

    for (const auto& [height, width] : block_configs) {
        size_t element_size = sizeof(float);
        size_t block_size = height * width * element_size;

        if (block_size > l2_banks[0].get_capacity()) continue;

        auto test_data = generate_test_pattern(block_size);
        l3_tiles[0].write(0, test_data.data(), block_size);

        bool complete = false;
        block_mover->enqueue_block_transfer(
            0, 0, 0, 0,
            height, width, element_size,
            BlockMover::TransformType::IDENTITY,
            [&complete]() { complete = true; }
        );

        while (!complete) {
            block_mover->process_transfers(l3_tiles, l2_banks);
            block_mover->set_cycle(block_mover->get_cycle() + 1);
        }
    }

    // Analyze bandwidth from traces
    auto bm_traces = logger.get_component_traces(ComponentType::BLOCK_MOVER, 0);

    std::cout << "\n=== BlockMover Bandwidth Analysis ===" << std::endl;
    std::cout << "Block Size (bytes) | Duration (cycles) | Effective BW (GB/s)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (const auto& trace : bm_traces) {
        if (trace.status == TransactionStatus::COMPLETED && std::holds_alternative<DMAPayload>(trace.payload)) {
            const auto& payload = std::get<DMAPayload>(trace.payload);
            uint64_t duration = trace.get_duration_cycles();

            if (duration > 0 && trace.clock_freq_ghz.has_value()) {
                // Effective bandwidth = bytes / (duration_cycles / clock_freq_ghz)
                double duration_ns = static_cast<double>(duration) / trace.clock_freq_ghz.value();
                double effective_bw_gb_s = (payload.bytes_transferred / duration_ns);

                std::cout << payload.bytes_transferred << " | "
                         << duration << " | "
                         << effective_bw_gb_s << std::endl;
            }
        }
    }
}

TEST_CASE_METHOD(BlockMoverTracingFixture, "Trace: Verify Transaction ID Uniqueness", "[trace][block_mover][txn_id]") {
    // Clear for clean test
    logger.clear();

    // Create multiple transfers
    const size_t num_transfers = 10;
    std::vector<bool> completions(num_transfers, false);

    block_mover->set_cycle(30000);

    for (size_t i = 0; i < num_transfers; i++) {
        auto test_data = generate_test_pattern(64);
        l3_tiles[0].write(i * 64, test_data.data(), 64);

        block_mover->enqueue_block_transfer(
            0, i * 64, 0, i * 64,
            2, 2, sizeof(float),
            BlockMover::TransformType::IDENTITY,
            [&completions, i]() { completions[i] = true; }
        );
    }

    // Process all transfers
    while (std::any_of(completions.begin(), completions.end(), [](bool c) { return !c; })) {
        block_mover->process_transfers(l3_tiles, l2_banks);
        block_mover->set_cycle(block_mover->get_cycle() + 1);
    }

    // Get all traces
    auto bm_traces = logger.get_component_traces(ComponentType::BLOCK_MOVER, 0);

    // Collect unique transaction IDs (each transfer has 2 traces with same ID)
    std::set<uint64_t> unique_txn_ids;
    for (const auto& trace : bm_traces) {
        unique_txn_ids.insert(trace.transaction_id);
    }

    // We should have num_transfers unique transaction IDs
    // (each transfer generates 2 traces with the same ID: issue + completion)
    REQUIRE(unique_txn_ids.size() >= static_cast<size_t>(num_transfers));

    std::cout << "\n=== Transaction ID Uniqueness ===" << std::endl;
    std::cout << "Total traces: " << bm_traces.size() << std::endl;
    std::cout << "Unique transaction IDs: " << unique_txn_ids.size() << std::endl;
    std::cout << "Expected unique IDs: " << num_transfers << std::endl;
}

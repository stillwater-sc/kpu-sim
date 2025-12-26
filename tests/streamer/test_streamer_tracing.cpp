#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/streamer.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/kpu/components/l1_buffer.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>

#include "../test_utilities.hpp"

using namespace sw;
using namespace sw::kpu;
using namespace sw::trace;

// Test fixture for Streamer tracing tests
class StreamerTracingFixture {
public:
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;
    std::unique_ptr<Streamer> streamer;
    TraceLogger& logger;

    StreamerTracingFixture()
        : logger(TraceLogger::instance())
    {
        // Create 2 L2 banks of 64KB each
        l2_banks.emplace_back(0, 64);  // ID 0, 64 KB capacity
        l2_banks.emplace_back(1, 64);  // ID 1, 64 KB capacity

        // Create 2 L1 buffers of 16KB each
        l1_buffers.emplace_back(0, 16);  // ID 0, 16 KB capacity
        l1_buffers.emplace_back(1, 16);  // ID 1, 16 KB capacity

        // Create Streamer: streamer 0, 1 GHz, 100 GB/s
        streamer = std::make_unique<Streamer>(0, 1.0, 100.0);

        // Reset and configure tracing
        logger.clear();
        logger.set_enabled(true);
        streamer->enable_tracing();
    }

    ~StreamerTracingFixture() {
        // Leave logger enabled for inspection after tests
    }

    // Helper to generate test data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }
};

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Streamer L2->L1 Row Stream", "[trace][streamer][row]") {
    const size_t matrix_height = 4;
    const size_t matrix_width = 4;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;
    const Address l2_addr = 0x1000;
    const Address l1_addr = 0x0;

    // Generate and write test matrix to L2
    auto matrix_data = generate_matrix(matrix_height, matrix_width, 1.0f);
    l2_banks[0].write(l2_addr, matrix_data.data(), matrix_data.size() * element_size);

    // Set initial cycle
    streamer->set_cycle(1000);

    size_t initial_trace_count = logger.get_trace_count();

    // Configure and enqueue stream
    Streamer::StreamConfig config;
    config.l2_bank_id = 0;
    config.l1_buffer_id = 0;
    config.l2_base_addr = l2_addr;
    config.l1_base_addr = l1_addr;
    config.matrix_height = matrix_height;
    config.matrix_width = matrix_width;
    config.element_size = element_size;
    config.compute_fabric_size = fabric_size;
    config.direction = Streamer::StreamDirection::L2_TO_L1;
    config.stream_type = Streamer::StreamType::ROW_STREAM;
    config.cache_line_size = 64;

    bool stream_complete = false;
    config.completion_callback = [&stream_complete]() { stream_complete = true; };

    streamer->enqueue_stream(config);

    // Should have logged the issue
    REQUIRE(logger.get_trace_count() == initial_trace_count + 1);

    // Process the stream - advance cycle each iteration
    while (!stream_complete) {
        streamer->set_cycle(streamer->get_cycle() + 1);
        streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
    }

    // Should have logged the completion
    REQUIRE(logger.get_trace_count() == initial_trace_count + 2);

    // Get traces for this Streamer
    auto streamer_traces = logger.get_component_traces(ComponentType::STREAMER, 0);
    REQUIRE(streamer_traces.size() >= 2);

    // Verify the last two traces (issue and completion)
    auto& issue_trace = streamer_traces[streamer_traces.size() - 2];
    auto& complete_trace = streamer_traces[streamer_traces.size() - 1];

    // Verify issue trace
    REQUIRE(issue_trace.component_type == ComponentType::STREAMER);
    REQUIRE(issue_trace.component_id == 0);
    REQUIRE(issue_trace.transaction_type == TransactionType::TRANSFER);
    REQUIRE(issue_trace.cycle_issue == 1000);
    REQUIRE(issue_trace.status == TransactionStatus::ISSUED);
    REQUIRE(issue_trace.description.find("L2_TO_L1") != std::string::npos);
    REQUIRE(issue_trace.description.find("ROW_STREAM") != std::string::npos);

    // Verify completion trace
    REQUIRE(complete_trace.component_type == ComponentType::STREAMER);
    REQUIRE(complete_trace.component_id == 0);
    REQUIRE(complete_trace.transaction_type == TransactionType::TRANSFER);
    REQUIRE(complete_trace.status == TransactionStatus::COMPLETED);
    REQUIRE(complete_trace.cycle_complete >= complete_trace.cycle_issue);

    // Verify payload data
    REQUIRE(std::holds_alternative<DMAPayload>(complete_trace.payload));
    const auto& payload = std::get<DMAPayload>(complete_trace.payload);
    REQUIRE(payload.bytes_transferred == matrix_height * matrix_width * element_size);
    REQUIRE(payload.source.type == ComponentType::L2_BANK);
    REQUIRE(payload.destination.type == ComponentType::L1);

    std::cout << "\n=== Streamer L2->L1 Row Stream Trace ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Stream Size: " << payload.bytes_transferred << " bytes" << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Streamer L2->L1 Column Stream", "[trace][streamer][column]") {
    const size_t matrix_height = 8;
    const size_t matrix_width = 8;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;
    const Address l2_addr = 0x0;
    const Address l1_addr = 0x0;

    // Generate and write test matrix to L2
    auto matrix_data = generate_matrix(matrix_height, matrix_width, 1.0f);
    l2_banks[0].write(l2_addr, matrix_data.data(), matrix_data.size() * element_size);

    // Set initial cycle
    streamer->set_cycle(2000);

    size_t initial_trace_count = logger.get_trace_count();

    // Configure and enqueue column stream
    Streamer::StreamConfig config;
    config.l2_bank_id = 0;
    config.l1_buffer_id = 0;
    config.l2_base_addr = l2_addr;
    config.l1_base_addr = l1_addr;
    config.matrix_height = matrix_height;
    config.matrix_width = matrix_width;
    config.element_size = element_size;
    config.compute_fabric_size = fabric_size;
    config.direction = Streamer::StreamDirection::L2_TO_L1;
    config.stream_type = Streamer::StreamType::COLUMN_STREAM;
    config.cache_line_size = 64;

    bool stream_complete = false;
    config.completion_callback = [&stream_complete]() { stream_complete = true; };

    streamer->enqueue_stream(config);

    // Should have logged the issue
    REQUIRE(logger.get_trace_count() == initial_trace_count + 1);

    // Process the stream
    while (!stream_complete) {
        streamer->set_cycle(streamer->get_cycle() + 1);
        streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
    }

    // Should have logged the completion
    REQUIRE(logger.get_trace_count() == initial_trace_count + 2);

    // Get traces
    auto streamer_traces = logger.get_component_traces(ComponentType::STREAMER, 0);

    // Find the traces for this specific stream
    auto& issue_trace = streamer_traces[streamer_traces.size() - 2];
    auto& complete_trace = streamer_traces[streamer_traces.size() - 1];

    // Verify stream type is mentioned in description
    REQUIRE(issue_trace.description.find("COLUMN_STREAM") != std::string::npos);
    REQUIRE(complete_trace.description.find("COLUMN_STREAM") != std::string::npos);

    std::cout << "\n=== Streamer L2->L1 Column Stream Trace ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Matrix: " << matrix_height << "x" << matrix_width << " (column stream)" << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Streamer L1->L2 Row Stream", "[trace][streamer][writeback]") {
    const size_t matrix_height = 4;
    const size_t matrix_width = 4;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;
    const Address l2_addr = 0x1000;
    const Address l1_addr = 0x0;

    // Generate and write test data to L1 buffer
    auto matrix_data = generate_matrix(matrix_height, matrix_width, 10.0f);
    l1_buffers[0].write(l1_addr, matrix_data.data(), matrix_data.size() * element_size);

    // Set initial cycle
    streamer->set_cycle(3000);

    // Configure L1->L2 writeback stream
    Streamer::StreamConfig config;
    config.l2_bank_id = 0;
    config.l1_buffer_id = 0;
    config.l2_base_addr = l2_addr;
    config.l1_base_addr = l1_addr;
    config.matrix_height = matrix_height;
    config.matrix_width = matrix_width;
    config.element_size = element_size;
    config.compute_fabric_size = fabric_size;
    config.direction = Streamer::StreamDirection::L1_TO_L2;
    config.stream_type = Streamer::StreamType::ROW_STREAM;
    config.cache_line_size = 64;

    bool stream_complete = false;
    config.completion_callback = [&stream_complete]() { stream_complete = true; };

    streamer->enqueue_stream(config);

    // Process the stream
    while (!stream_complete) {
        streamer->set_cycle(streamer->get_cycle() + 1);
        streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
    }

    // Verify traces
    auto streamer_traces = logger.get_component_traces(ComponentType::STREAMER, 0);
    auto& complete_trace = streamer_traces[streamer_traces.size() - 1];

    // Verify direction is L1->L2
    REQUIRE(complete_trace.description.find("L1_TO_L2") != std::string::npos);

    // Verify payload shows correct source/destination
    const auto& payload = std::get<DMAPayload>(complete_trace.payload);
    REQUIRE(payload.source.type == ComponentType::L1);
    REQUIRE(payload.destination.type == ComponentType::L2_BANK);

    std::cout << "\n=== Streamer L1->L2 Writeback Trace ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Direction: L1->L2 (writeback)" << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Multiple Streamer Operations", "[trace][streamer][queue]") {
    const size_t matrix_size = 4;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;

    // Set initial cycle
    streamer->set_cycle(4000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue multiple streams
    int num_streams = 3;
    int completed_count = 0;
    auto completion_callback = [&completed_count]() { completed_count++; };

    for (int i = 0; i < num_streams; i++) {
        auto matrix_data = generate_matrix(matrix_size, matrix_size, static_cast<float>(i * 10));
        Address addr = i * matrix_size * matrix_size * element_size;
        l2_banks[0].write(addr, matrix_data.data(), matrix_data.size() * element_size);

        Streamer::StreamConfig config;
        config.l2_bank_id = 0;
        config.l1_buffer_id = 0;
        config.l2_base_addr = addr;
        config.l1_base_addr = addr;
        config.matrix_height = matrix_size;
        config.matrix_width = matrix_size;
        config.element_size = element_size;
        config.compute_fabric_size = fabric_size;
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = Streamer::StreamType::ROW_STREAM;
        config.cache_line_size = 64;
        config.completion_callback = completion_callback;

        streamer->enqueue_stream(config);
    }

    // Should have logged 3 issue traces
    REQUIRE(logger.get_trace_count() == initial_trace_count + num_streams);

    // Process all streams
    while (completed_count < num_streams) {
        streamer->set_cycle(streamer->get_cycle() + 1);
        streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
    }

    // Should have logged 3 additional completion traces (total 6 new traces)
    REQUIRE(logger.get_trace_count() == initial_trace_count + (num_streams * 2));

    // Get all Streamer traces
    auto streamer_traces = logger.get_component_traces(ComponentType::STREAMER, 0);

    // Verify all streams have both issue and complete
    int completed_trace_count = 0;
    for (const auto& trace : streamer_traces) {
        if (trace.status == TransactionStatus::COMPLETED) {
            completed_trace_count++;
            REQUIRE(trace.cycle_complete >= trace.cycle_issue);
        }
    }

    REQUIRE(completed_trace_count >= num_streams);

    std::cout << "\n=== Multiple Streamer Operations ===" << std::endl;
    std::cout << "Total traces logged: " << logger.get_trace_count() << std::endl;
    std::cout << "Streamer 0 traces: " << streamer_traces.size() << std::endl;
    std::cout << "Completed streams: " << completed_trace_count << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Export Streamer to CSV", "[trace][streamer][export]") {
    const size_t matrix_size = 4;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;

    // Clear previous traces for clean export
    logger.clear();

    // Generate some streams
    streamer->set_cycle(5000);

    for (int i = 0; i < 2; i++) {
        auto matrix_data = generate_matrix(matrix_size, matrix_size);
        Address addr = i * matrix_size * matrix_size * element_size;
        l2_banks[0].write(addr, matrix_data.data(), matrix_data.size() * element_size);

        Streamer::StreamConfig config;
        config.l2_bank_id = 0;
        config.l1_buffer_id = 0;
        config.l2_base_addr = addr;
        config.l1_base_addr = addr;
        config.matrix_height = matrix_size;
        config.matrix_width = matrix_size;
        config.element_size = element_size;
        config.compute_fabric_size = fabric_size;
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = Streamer::StreamType::ROW_STREAM;
        config.cache_line_size = 64;

        bool complete = false;
        config.completion_callback = [&complete]() { complete = true; };

        streamer->enqueue_stream(config);

        while (!complete) {
            streamer->set_cycle(streamer->get_cycle() + 1);
            streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
        }
    }

    // Export traces to CSV
    auto csv_path = sw::test::get_test_output_path("streamer_trace_test.csv");
    bool csv_export_success = export_logger_traces(csv_path, "csv", logger);
    REQUIRE(csv_export_success);

    std::cout << "\n=== Streamer Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << csv_path << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Export Streamer to Chrome Format", "[trace][streamer][export][chrome]") {
    const size_t matrix_size = 4;
    const size_t element_size = sizeof(float);
    const size_t fabric_size = 4;

    // Clear previous traces for cleaner visualization
    logger.clear();

    // Generate streams with clear cycle progression
    for (int i = 0; i < 5; i++) {
        CycleCount start_cycle = 10000 + i * 1000;
        streamer->set_cycle(start_cycle);

        auto matrix_data = generate_matrix(matrix_size, matrix_size);
        Address addr = i * matrix_size * matrix_size * element_size;
        l2_banks[0].write(addr, matrix_data.data(), matrix_data.size() * element_size);

        // Alternate between row and column streams
        auto stream_type = (i % 2 == 0) ? Streamer::StreamType::ROW_STREAM
                                        : Streamer::StreamType::COLUMN_STREAM;

        Streamer::StreamConfig config;
        config.l2_bank_id = 0;
        config.l1_buffer_id = 0;
        config.l2_base_addr = addr;
        config.l1_base_addr = addr;
        config.matrix_height = matrix_size;
        config.matrix_width = matrix_size;
        config.element_size = element_size;
        config.compute_fabric_size = fabric_size;
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = stream_type;
        config.cache_line_size = 64;

        bool complete = false;
        config.completion_callback = [&complete]() { complete = true; };

        streamer->enqueue_stream(config);

        while (!complete) {
            streamer->set_cycle(streamer->get_cycle() + 1);
            streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
        }
    }

    // Export traces to Chrome trace format
    auto chrome_path = sw::test::get_test_output_path("streamer_trace_test.trace");
    bool chrome_export_success = export_logger_traces(chrome_path, "chrome", logger);
    REQUIRE(chrome_export_success);

    std::cout << "\n=== Chrome Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << chrome_path << std::endl;
    std::cout << "Open in chrome://tracing for visualization" << std::endl;
}

TEST_CASE_METHOD(StreamerTracingFixture, "Trace: Verify Transaction ID Uniqueness", "[trace][streamer][txn_id]") {
    // Clear for clean test
    logger.clear();

    // Create multiple streams
    const size_t num_streams = 10;
    std::vector<bool> completions(num_streams, false);

    streamer->set_cycle(20000);

    for (size_t i = 0; i < num_streams; i++) {
        auto matrix_data = generate_matrix(4, 4);
        Address addr = i * 64;
        l2_banks[0].write(addr, matrix_data.data(), 64);

        Streamer::StreamConfig config;
        config.l2_bank_id = 0;
        config.l1_buffer_id = 0;
        config.l2_base_addr = addr;
        config.l1_base_addr = addr;
        config.matrix_height = 4;
        config.matrix_width = 4;
        config.element_size = sizeof(float);
        config.compute_fabric_size = 4;
        config.direction = Streamer::StreamDirection::L2_TO_L1;
        config.stream_type = Streamer::StreamType::ROW_STREAM;
        config.cache_line_size = 64;
        config.completion_callback = [&completions, i]() { completions[i] = true; };

        streamer->enqueue_stream(config);
    }

    // Process all streams
    while (std::any_of(completions.begin(), completions.end(), [](bool c) { return !c; })) {
        streamer->set_cycle(streamer->get_cycle() + 1);
        streamer->update(streamer->get_cycle(), l2_banks, l1_buffers);
    }

    // Get all traces
    auto streamer_traces = logger.get_component_traces(ComponentType::STREAMER, 0);

    // Collect unique transaction IDs (each stream has 2 traces with same ID)
    std::set<uint64_t> unique_txn_ids;
    for (const auto& trace : streamer_traces) {
        unique_txn_ids.insert(trace.transaction_id);
    }

    // We should have num_streams unique transaction IDs
    // (each stream generates 2 traces with the same ID: issue + completion)
    REQUIRE(unique_txn_ids.size() >= num_streams);

    std::cout << "\n=== Transaction ID Uniqueness ===" << std::endl;
    std::cout << "Total traces: " << streamer_traces.size() << std::endl;
    std::cout << "Unique transaction IDs: " << unique_txn_ids.size() << std::endl;
    std::cout << "Expected unique IDs: " << num_streams << std::endl;
}

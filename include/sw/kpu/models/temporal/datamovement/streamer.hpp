#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <queue>
#include <array>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::kpu {

// Forward declarations
class L2Bank;
class L1Buffer;

// Streamer for L2-L1 data movement supporting systolic array streaming
class KPU_API Streamer {
public:
    enum class StreamDirection {
        L2_TO_L1,           // L2 → L1 (input data)
        L1_TO_L2            // L1 → L2 (output data)
    };

    enum class StreamType {
        ROW_STREAM,         // Row-wise streaming (A matrix rows)
        COLUMN_STREAM       // Column-wise streaming (B matrix columns)
    };

    struct StreamConfig {
        // Source and destination
        size_t l2_bank_id;
        size_t l1_buffer_id;

        // Memory layout
        Address l2_base_addr;
        Address l1_base_addr;

        // Stream geometry
        Size matrix_height;     // Number of rows in matrix
        Size matrix_width;      // Number of columns in matrix
        Size element_size;      // Size of each element (e.g., 4 for float)
        Size compute_fabric_size; // Size of systolic array (e.g., 16 for 16x16)

        // Streaming parameters
        StreamDirection direction;
        StreamType stream_type;
        Size cache_line_size;   // L2 cache line size (default 64 bytes)

        // Callback for completion
        std::function<void()> completion_callback;

        // Timing and tracing
        Cycle start_cycle = 0;
        Cycle end_cycle = 0;
        uint64_t transaction_id = 0;
    };

private:
    // Stream state management
    struct StreamState {
        StreamConfig config;
        bool is_active;
        Cycle start_cycle;

        // Current streaming position
        Size current_row;
        Size current_col;
        Size elements_streamed_this_cycle;

        // Staggering state for systolic array
        std::vector<Size> row_stagger_offset;    // Per-row stagger tracking
        std::vector<Size> col_stagger_offset;    // Per-column stagger tracking

        // Cache line buffering
        std::vector<uint8_t> cache_line_buffer;
        bool buffer_valid;
        Address buffered_cache_line_addr;
    };

    std::queue<StreamConfig> stream_queue;
    std::unique_ptr<StreamState> current_stream;
    size_t streamer_id;

    // Tracing support
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;
    double clock_freq_ghz_;
    Cycle current_cycle_;
    double bandwidth_gb_s_;

    // Internal streaming engine methods
    void initialize_stream_state(const StreamConfig& config);
    bool advance_stream_cycle(Cycle current_cycle,
                             std::vector<L2Bank>& l2_banks,
                             std::vector<L1Buffer>& l1_buffers);

    // Row streaming implementation
    bool stream_row_l2_to_l1(Cycle current_cycle,
                            std::vector<L2Bank>& l2_banks,
                            std::vector<L1Buffer>& l1_buffers);

    bool stream_row_l1_to_l2(Cycle current_cycle,
                            std::vector<L2Bank>& l2_banks,
                            std::vector<L1Buffer>& l1_buffers);

    // Column streaming implementation
    bool stream_column_l2_to_l1(Cycle current_cycle,
                               std::vector<L2Bank>& l2_banks,
                               std::vector<L1Buffer>& l1_buffers);

    bool stream_column_l1_to_l2(Cycle current_cycle,
                               std::vector<L2Bank>& l2_banks,
                               std::vector<L1Buffer>& l1_buffers);

    // Cache line management
    void fetch_cache_line_if_needed(L2Bank& l2_bank, Address addr);
    void write_cache_line_if_needed(L2Bank& l2_bank, Address addr);

    // Staggering logic for systolic array
    Size calculate_stagger_delay(Size fabric_position, StreamType type) const;
    bool should_stream_this_cycle(Size fabric_position, Cycle current_cycle) const;

    // Address calculation helpers
    Address calculate_row_address(Size row, Size col) const;
    Address calculate_column_address(Size row, Size col) const;

public:
    explicit Streamer(size_t streamer_id, double clock_freq_ghz = 1.0, double bandwidth_gb_s = 100.0);
    ~Streamer() = default;

    // Custom copy and move semantics for std::vector compatibility
    Streamer(const Streamer& other);
    Streamer& operator=(const Streamer& other);
    Streamer(Streamer&&) = default;
    Streamer& operator=(Streamer&&) = default;

    // Tracing control
    void enable_tracing() { tracing_enabled_ = true; }
    void disable_tracing() { tracing_enabled_ = false; }
    bool is_tracing_enabled() const { return tracing_enabled_; }

    // Cycle management for timing simulation
    void set_cycle(Cycle cycle) { current_cycle_ = cycle; }
    Cycle get_cycle() const { return current_cycle_; }

    // Stream configuration and control
    void enqueue_stream(const StreamConfig& config);

    // Update streaming state - called each cycle
    bool update(Cycle current_cycle,
               std::vector<L2Bank>& l2_banks,
               std::vector<L1Buffer>& l1_buffers);

    // Status queries
    bool is_busy() const { return current_stream != nullptr || !stream_queue.empty(); }
    bool is_streaming() const { return current_stream != nullptr && current_stream->is_active; }
    size_t get_queue_size() const { return stream_queue.size(); }
    size_t get_streamer_id() const { return streamer_id; }

    // Configuration helpers
    static Size calculate_stream_cycles(Size matrix_height, Size matrix_width, Size fabric_size);
    static Size calculate_elements_per_cycle(Size fabric_size);

    // Reset and cleanup
    void reset();
    void abort_current_stream();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
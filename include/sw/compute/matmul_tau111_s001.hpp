#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <cstdint>
#include <functional>

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
#include <sw/compute/dot_pe.hpp>
#include <sw/kpu/models/temporal/memory/scratchpad.hpp>

namespace sw::compute {

    using namespace sw::kpu;

// Forward declarations
//class Streamer;


// Systolic Array for matrix multiplication using Tau=[1 1 1] and S=[0 0 1] configuration
template<typename InputType, typename AccumulationType, typename ResultType>
class KPU_API MatmulTau111S001 {
public:
	using DotPE = DotProductAccumulator<InputType, AccumulationType, ResultType>;
    static constexpr Size DEFAULT_ROWS = 16;
    static constexpr Size DEFAULT_COLS = 16;

    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in scratchpad
        size_t scratchpad_id; // Which scratchpad to use
        std::function<void()> completion_callback;
    };

    // Bus directions for data flow
    enum class BusDirection {
        HORIZONTAL,  // A matrix data flows horizontally (left to right)
        VERTICAL,    // B matrix data flows vertically (top to bottom)
        DIAGONAL     // C matrix data flows diagonally (for evacuation)
    };

private:
    // Array configuration
    Size _rows, _cols;

    // Processing elements
    std::vector<std::vector< DotPE > > pe_array;

    // Data buses for systolic flow
    std::vector<std::queue<float>> horizontal_bus; // A data (one per row)
    std::vector<std::queue<float>> vertical_bus;   // B data (one per column)
    std::vector<std::queue<float>> diagonal_bus;   // C data evacuation

    // Operation state
    bool is_computing;
    Cycle compute_start_cycle;
    MatMulConfig current_op;

    // Streaming state
    Size current_a_row, current_a_col;
    Size current_b_row, current_b_col;
    Size current_c_row, current_c_col;
    Size cycles_completed;

    // Staggering for systolic timing
    std::vector<Cycle> row_start_cycles;
    std::vector<Cycle> col_start_cycles;

public:
    explicit MatmulTau111S001(Size rows = DEFAULT_ROWS, Size cols = DEFAULT_COLS) : _rows(rows), _cols(cols), is_computing(false), compute_start_cycle(0),
        current_a_row(0), current_a_col(0), current_b_row(0), current_b_col(0),
        current_c_row(0), current_c_col(0), cycles_completed(0) {

        // Initialize PE array
        pe_array.resize(_rows);
        for (Size row = 0; row < _rows; ++row) {
            pe_array[row].resize(_cols);
            for (Size col = 0; col < _cols; ++col) {
                pe_array[row][col].reset();
            }
        }

        // Initialize buses
        horizontal_bus.resize(_rows);
        vertical_bus.resize(_cols);
        diagonal_bus.resize(_rows + _cols - 1); // Maximum diagonal length

        // Initialize staggering
        row_start_cycles.resize(_rows);
        col_start_cycles.resize(_cols);
    }
    ~MatmulTau111S001() = default;

    // Custom copy and move semantics for std::vector compatibility
    MatmulTau111S001(const MatmulTau111S001& other);
    MatmulTau111S001& operator=(const MatmulTau111S001& other);
    MatmulTau111S001(MatmulTau111S001&&) = default;
    MatmulTau111S001& operator=(MatmulTau111S001&&) = default;

    // Matrix multiplication operations
    void start_matmul(const MatMulConfig& config);
    bool update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_computing; }
    void reset();

    // Configuration
    Size get_rows() const { return _rows; }
    Size get_cols() const { return _cols; }
    Size get_total_pes() const { return _rows * _cols; }

    // Streaming interface for integration with Streamer components
    void stream_a_data(const std::vector<float>& data, Size row_offset);
    void stream_b_data(const std::vector<float>& data, Size col_offset);
    std::vector<float> evacuate_c_data(Size max_elements);

    // Performance metrics
    Cycle estimate_cycles(Size m, Size n, Size k) const;
    Size calculate_throughput() const;

private:
    // Internal processing
    void cycle_pe_array(Cycle current_cycle);
    void load_a_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    void load_b_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    void evacuate_c_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);

    // Bus management
    void propagate_horizontal_bus();
    void propagate_vertical_bus();
    void propagate_diagonal_bus();

    // Timing and staggering
    bool should_start_row(Size row, Cycle current_cycle) const;
    bool should_start_col(Size col, Cycle current_cycle) const;
    Size calculate_stagger_delay(Size position) const;

    // Data loading helpers
    void load_matrix_a_tile(const std::vector<float>& matrix_a, Size tile_row, Size tile_col);
    void load_matrix_b_tile(const std::vector<float>& matrix_b, Size tile_row, Size tile_col);
    void store_matrix_c_tile(std::vector<float>& matrix_c, Size tile_row, Size tile_col);

    // Address calculation
    Address calculate_matrix_address(Address base_addr, Size row, Size col, Size width, Size element_size) const;

    // Temporary simple implementation for testing
    void perform_direct_matrix_multiply(std::vector<Scratchpad>& scratchpads);
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
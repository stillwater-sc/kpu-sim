#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cmath>

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
#include <sw/kpu/components/l1_buffer.hpp>

namespace sw::kpu {

// Forward declarations
class Streamer;

// Processing Element (PE) for systolic array
template<typename Scalar>
class KPU_API ProcessingElement {
public:
    ProcessingElement(size_t row, size_t col) : row_id(row), col_id(col) {}

    // Data inputs
    void set_a_input(Scalar value) { a_input = value; }
    void set_b_input(Scalar value) { b_input = value; }

    // Data outputs (for propagation)
    Scalar get_a_output() const { return a_output; }
    Scalar get_b_output() const { return b_output; }
    Scalar get_c_output() const { return c_accumulator; }

    // Process one cycle
    void cycle() {
        // Output-stationary: accumulate A*B into C, propagate A and B
        if (a_input != Scalar{0} || b_input != Scalar{0}) {
            c_accumulator += a_input * b_input;
            accumulating = true;
        }

        // Propagate data for systolic flow
        a_output = a_input; // Pass A data horizontally (left to right)
        b_output = b_input; // Pass B data vertically (top to bottom)

        // Clear inputs for next cycle
        a_input = Scalar{0};
        b_input = Scalar{0};
    }

    // Reset PE state
    void reset() {
        a_input = a_output = Scalar{0};
        b_input = b_output = Scalar{0};
        c_accumulator = Scalar{0};
        accumulating = false;
        last_valid_cycle = 0;
    }

    // Configuration
    size_t get_row() const { return row_id; }
    size_t get_col() const { return col_id; }

private:
    size_t row_id, col_id;

    // Data registers
    Scalar a_input{}, a_output{};
    Scalar b_input{}, b_output{};
    Scalar c_accumulator{};

    // Control state
    bool accumulating{false};
    Cycle last_valid_cycle{0};
};

// Systolic Array for matrix multiplication using output-stationary schedule
// Template parameter Scalar defines the element type (float, double, posit, etc.)
template<typename Scalar>
class KPU_API SystolicArray {
public:
    static constexpr Size DEFAULT_ROWS = 16;
    static constexpr Size DEFAULT_COLS = 16;

    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in L1 buffer
        size_t l1_buffer_id; // Which L1 buffer to use
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
    Size num_rows, num_cols;

    // Processing elements
    std::vector<std::vector<std::unique_ptr<ProcessingElement<Scalar>>>> pe_array;

    // Data buses for systolic flow (use Scalar type for consistency)
    std::vector<std::queue<Scalar>> horizontal_bus; // A data (one per row)
    std::vector<std::queue<Scalar>> vertical_bus;   // B data (one per column)
    std::vector<std::queue<Scalar>> diagonal_bus;   // C data evacuation

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
    explicit SystolicArray(Size rows = DEFAULT_ROWS, Size cols = DEFAULT_COLS)
        : num_rows(rows), num_cols(cols), is_computing(false), compute_start_cycle(0),
          current_a_row(0), current_a_col(0), current_b_row(0), current_b_col(0),
          current_c_row(0), current_c_col(0), cycles_completed(0) {

        // Initialize PE array
        pe_array.resize(num_rows);
        for (Size row = 0; row < num_rows; ++row) {
            pe_array[row].resize(num_cols);
            for (Size col = 0; col < num_cols; ++col) {
                pe_array[row][col] = std::make_unique<ProcessingElement<Scalar>>(row, col);
            }
        }

        // Initialize buses
        horizontal_bus.resize(num_rows);
        vertical_bus.resize(num_cols);
        diagonal_bus.resize(num_rows + num_cols - 1); // Maximum diagonal length

        // Initialize staggering
        row_start_cycles.resize(num_rows);
        col_start_cycles.resize(num_cols);
    }

    ~SystolicArray() = default;

    // Custom copy semantics for std::vector compatibility
    SystolicArray(const SystolicArray& other)
        : num_rows(other.num_rows), num_cols(other.num_cols),
          is_computing(other.is_computing), compute_start_cycle(other.compute_start_cycle),
          current_op(other.current_op), current_a_row(other.current_a_row),
          current_a_col(other.current_a_col), current_b_row(other.current_b_row),
          current_b_col(other.current_b_col), current_c_row(other.current_c_row),
          current_c_col(other.current_c_col), cycles_completed(other.cycles_completed),
          row_start_cycles(other.row_start_cycles), col_start_cycles(other.col_start_cycles) {

        // Deep copy PE array
        pe_array.resize(num_rows);
        for (Size row = 0; row < num_rows; ++row) {
            pe_array[row].resize(num_cols);
            for (Size col = 0; col < num_cols; ++col) {
                pe_array[row][col] = std::make_unique<ProcessingElement<Scalar>>(*other.pe_array[row][col]);
            }
        }

        // Copy buses (they contain queues, so copy them)
        horizontal_bus = other.horizontal_bus;
        vertical_bus = other.vertical_bus;
        diagonal_bus = other.diagonal_bus;
    }

    SystolicArray& operator=(const SystolicArray& other) {
        if (this != &other) {
            num_rows = other.num_rows;
            num_cols = other.num_cols;
            is_computing = other.is_computing;
            compute_start_cycle = other.compute_start_cycle;
            current_op = other.current_op;
            current_a_row = other.current_a_row;
            current_a_col = other.current_a_col;
            current_b_row = other.current_b_row;
            current_b_col = other.current_b_col;
            current_c_row = other.current_c_row;
            current_c_col = other.current_c_col;
            cycles_completed = other.cycles_completed;
            row_start_cycles = other.row_start_cycles;
            col_start_cycles = other.col_start_cycles;

            // Deep copy PE array
            pe_array.resize(num_rows);
            for (Size row = 0; row < num_rows; ++row) {
                pe_array[row].resize(num_cols);
                for (Size col = 0; col < num_cols; ++col) {
                    pe_array[row][col] = std::make_unique<ProcessingElement<Scalar>>(*other.pe_array[row][col]);
                }
            }

            // Copy buses
            horizontal_bus = other.horizontal_bus;
            vertical_bus = other.vertical_bus;
            diagonal_bus = other.diagonal_bus;
        }
        return *this;
    }

    SystolicArray(SystolicArray&&) = default;
    SystolicArray& operator=(SystolicArray&&) = default;

    // Matrix multiplication operations
    void start_matmul(const MatMulConfig& config) {
        if (is_computing) {
            throw std::runtime_error("SystolicArray is already busy");
        }

        // Validate matrix dimensions
        if (config.m == 0 || config.n == 0 || config.k == 0) {
            throw std::invalid_argument("Invalid matrix dimensions");
        }

        current_op = config;
        is_computing = true;
        compute_start_cycle = 0; // Will be set on first update

        // Reset streaming state
        current_a_row = current_a_col = 0;
        current_b_row = current_b_col = 0;
        current_c_row = current_c_col = 0;
        cycles_completed = 0;

        // Calculate staggered start times for systolic flow
        for (Size row = 0; row < num_rows; ++row) {
            row_start_cycles[row] = calculate_stagger_delay(row);
        }
        for (Size col = 0; col < num_cols; ++col) {
            col_start_cycles[col] = calculate_stagger_delay(col);
        }

        // Reset PE array
        for (Size row = 0; row < num_rows; ++row) {
            for (Size col = 0; col < num_cols; ++col) {
                pe_array[row][col]->reset();
            }
        }

        // Clear buses
        for (auto& bus : horizontal_bus) { while (!bus.empty()) bus.pop(); }
        for (auto& bus : vertical_bus) { while (!bus.empty()) bus.pop(); }
        for (auto& bus : diagonal_bus) { while (!bus.empty()) bus.pop(); }
    }

    bool update(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers) {
        if (!is_computing) {
            return false;
        }

        // Set start cycle on first call
        if (compute_start_cycle == 0) {
            compute_start_cycle = current_cycle;
        }

        // Calculate elapsed cycles
        Cycle cycles_elapsed = current_cycle - compute_start_cycle;

        // Calculate required cycles for this matmul operation
        Cycle required_cycles = estimate_cycles(current_op.m, current_op.n, current_op.k);

        // Check if computation has completed
        if (cycles_elapsed >= required_cycles) {
            // Perform the actual matrix multiplication
            perform_direct_matrix_multiply(l1_buffers);

            // Call completion callback if provided
            if (current_op.completion_callback) {
                current_op.completion_callback();
            }

            is_computing = false;
            return true;
        }

        // Still computing
        return false;
    }

    bool is_busy() const { return is_computing; }

    void reset() {
        is_computing = false;
        compute_start_cycle = 0;
        cycles_completed = 0;

        // Reset all PEs
        for (Size row = 0; row < num_rows; ++row) {
            for (Size col = 0; col < num_cols; ++col) {
                pe_array[row][col]->reset();
            }
        }

        // Clear all buses
        for (auto& bus : horizontal_bus) { while (!bus.empty()) bus.pop(); }
        for (auto& bus : vertical_bus) { while (!bus.empty()) bus.pop(); }
        for (auto& bus : diagonal_bus) { while (!bus.empty()) bus.pop(); }
    }

    // Configuration
    Size get_rows() const { return num_rows; }
    Size get_cols() const { return num_cols; }
    Size get_total_pes() const { return num_rows * num_cols; }

    // Streaming interface for integration with Streamer components
    void stream_a_data(const std::vector<Scalar>& data, Size row_offset) {
        // Interface for external streaming of A matrix data
        for (Size i = 0; i < std::min(data.size(), static_cast<size_t>(num_rows - row_offset)); ++i) {
            if (row_offset + i < horizontal_bus.size()) {
                horizontal_bus[row_offset + i].push(data[i]);
            }
        }
    }

    void stream_b_data(const std::vector<Scalar>& data, Size col_offset) {
        // Interface for external streaming of B matrix data
        for (Size i = 0; i < std::min(data.size(), static_cast<size_t>(num_cols - col_offset)); ++i) {
            if (col_offset + i < vertical_bus.size()) {
                vertical_bus[col_offset + i].push(data[i]);
            }
        }
    }

    std::vector<Scalar> evacuate_c_data(Size max_elements) {
        // Interface for external evacuation of C matrix results
        std::vector<Scalar> results;
        results.reserve(max_elements);

        for (auto& bus : diagonal_bus) {
            while (!bus.empty() && results.size() < max_elements) {
                results.push_back(bus.front());
                bus.pop();
            }
            if (results.size() >= max_elements) break;
        }

        return results;
    }

    // Performance metrics
    Cycle estimate_cycles(Size m, Size n, Size k) const {
        // Systolic array cycles = k (accumulation) + max(m,n) (fill/drain) + stagger delays
        return k + std::max(m, n) + std::max(num_rows, num_cols);
    }

    Size calculate_throughput() const {
        // Theoretical peak: one MAC per PE per cycle
        return num_rows * num_cols;
    }

private:
    // Internal processing
    void cycle_pe_array(Cycle current_cycle) {
        // Cycle all PEs in parallel
        for (Size row = 0; row < num_rows; ++row) {
            for (Size col = 0; col < num_cols; ++col) {
                auto& pe = pe_array[row][col];

                // Feed data from buses to PE
                if (!horizontal_bus[row].empty() && should_start_row(row, current_cycle)) {
                    pe->set_a_input(horizontal_bus[row].front());
                    horizontal_bus[row].pop();
                }

                if (!vertical_bus[col].empty() && should_start_col(col, current_cycle)) {
                    pe->set_b_input(vertical_bus[col].front());
                    vertical_bus[col].pop();
                }

                // Execute PE cycle
                pe->cycle();
            }
        }
    }

    void load_a_data(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers) {
        if (current_op.l1_buffer_id >= l1_buffers.size()) {
            return;
        }

        auto& l1_buffer = l1_buffers[current_op.l1_buffer_id];

        // Simplified loading: stream matrix A row by row with staggering
        (void)current_cycle; // Suppress unused parameter warning for now

        for (Size row = 0; row < std::min(num_rows, current_op.m); ++row) {
            if (should_start_row(row, current_cycle)) {
                // Load one element per cycle per row, staggered
                Size global_row = current_a_row + row;
                Size global_col = current_a_col;

                if (global_row < current_op.m && global_col < current_op.k) {
                    Address addr = calculate_matrix_address(
                        current_op.a_addr, global_row, global_col, current_op.k, sizeof(Scalar));

                    Scalar value;
                    l1_buffer.read(addr, &value, sizeof(Scalar));
                    horizontal_bus[row].push(value);
                }
            }
        }

        // Advance A matrix position
        if (current_cycle % num_rows == 0) { // Every num_rows cycles
            current_a_col++;
            if (current_a_col >= current_op.k) {
                current_a_col = 0;
                current_a_row += num_rows;
            }
        }
    }

    void load_b_data(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers) {
        if (current_op.l1_buffer_id >= l1_buffers.size()) {
            return;
        }

        auto& l1_buffer = l1_buffers[current_op.l1_buffer_id];

        // Simplified loading: stream matrix B column by column with staggering
        for (Size col = 0; col < std::min(num_cols, current_op.n); ++col) {
            if (should_start_col(col, current_cycle)) {
                // Load one element per cycle per column, staggered
                Size global_row = current_b_row;
                Size global_col = current_b_col + col;

                if (global_row < current_op.k && global_col < current_op.n) {
                    Address addr = calculate_matrix_address(
                        current_op.b_addr, global_row, global_col, current_op.n, sizeof(Scalar));

                    Scalar value;
                    l1_buffer.read(addr, &value, sizeof(Scalar));
                    vertical_bus[col].push(value);
                }
            }
        }

        // Advance B matrix position
        if (current_cycle % num_cols == 0) { // Every num_cols cycles
            current_b_row++;
            if (current_b_row >= current_op.k) {
                current_b_row = 0;
                current_b_col += num_cols;
            }
        }
    }

    void evacuate_c_data(Cycle current_cycle, std::vector<L1Buffer>& l1_buffers) {
        if (current_op.l1_buffer_id >= l1_buffers.size()) {
            return;
        }

        auto& l1_buffer = l1_buffers[current_op.l1_buffer_id];

        // Evacuate completed results along diagonal bus
        // Results become available after K cycles of accumulation
        Cycle evacuation_delay = current_op.k + num_rows + num_cols;

        if (current_cycle < evacuation_delay) {
            return; // Too early for results
        }

        // Collect results from PEs that have completed their accumulation
        for (Size row = 0; row < std::min(num_rows, current_op.m); ++row) {
            for (Size col = 0; col < std::min(num_cols, current_op.n); ++col) {
                auto& pe = pe_array[row][col];

                // Check if this PE has accumulated enough for a result
                Size diagonal_idx = row + col;
                if (diagonal_idx < diagonal_bus.size()) {
                    // Move result to diagonal bus for evacuation
                    Scalar result = pe->get_c_output();
                    if (result != Scalar{0}) { // Only evacuate non-zero results
                        diagonal_bus[diagonal_idx].push(result);

                        // Write result back to scratchpad
                        Size global_row = current_c_row + row;
                        Size global_col = current_c_col + col;

                        if (global_row < current_op.m && global_col < current_op.n) {
                            Address addr = calculate_matrix_address(
                                current_op.c_addr, global_row, global_col, current_op.n, sizeof(Scalar));
                            l1_buffer.write(addr, &result, sizeof(Scalar));
                        }
                    }
                }
            }
        }
    }

    // Bus management
    void propagate_horizontal_bus() {
        // A data flows left to right (horizontal)
        // Data in buses is automatically consumed by PEs
    }

    void propagate_vertical_bus() {
        // B data flows top to bottom (vertical)
        // Data in buses is automatically consumed by PEs
    }

    void propagate_diagonal_bus() {
        // C data flows diagonally for evacuation
        // Results are collected from PEs and moved to output
    }

    // Timing and staggering
    bool should_start_row(Size row, Cycle current_cycle) const {
        Cycle relative_cycle = current_cycle - compute_start_cycle;
        return relative_cycle >= row_start_cycles[row];
    }

    bool should_start_col(Size col, Cycle current_cycle) const {
        Cycle relative_cycle = current_cycle - compute_start_cycle;
        return relative_cycle >= col_start_cycles[col];
    }

    Size calculate_stagger_delay(Size position) const {
        // Stagger by one cycle per position for proper systolic timing
        return position;
    }

    // Address calculation
    Address calculate_matrix_address(Address base_addr, Size row, Size col,
                                     Size width, Size element_size) const {
        return base_addr + (row * width + col) * element_size;
    }

    // Direct matrix multiplication (simplified implementation)
    void perform_direct_matrix_multiply(std::vector<L1Buffer>& l1_buffers) {
        if (current_op.l1_buffer_id >= l1_buffers.size()) {
            return;
        }

        auto& l1_buffer = l1_buffers[current_op.l1_buffer_id];

        // Read matrices A and B from L1 buffer
        std::vector<Scalar> matrix_a(current_op.m * current_op.k);
        std::vector<Scalar> matrix_b(current_op.k * current_op.n);
        std::vector<Scalar> matrix_c(current_op.m * current_op.n, Scalar{0});

        // Read matrix A
        l1_buffer.read(current_op.a_addr, matrix_a.data(), matrix_a.size() * sizeof(Scalar));

        // Read matrix B
        l1_buffer.read(current_op.b_addr, matrix_b.data(), matrix_b.size() * sizeof(Scalar));

        // Perform matrix multiplication: C = A * B
        for (Size i = 0; i < current_op.m; ++i) {
            for (Size j = 0; j < current_op.n; ++j) {
                Scalar sum{0};
                for (Size k = 0; k < current_op.k; ++k) {
                    sum += matrix_a[i * current_op.k + k] * matrix_b[k * current_op.n + j];
                }
                matrix_c[i * current_op.n + j] = sum;
            }
        }

        // Write result matrix C back to L1 buffer
        l1_buffer.write(current_op.c_addr, matrix_c.data(), matrix_c.size() * sizeof(Scalar));
    }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

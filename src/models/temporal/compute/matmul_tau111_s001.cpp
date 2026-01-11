#include <algorithm>
#include <stdexcept>
#include <cmath>

#include <sw/kpu/components/systolic_array.hpp>

namespace sw::kpu {

// SystolicArray implementation
SystolicArray::SystolicArray(Size rows, Size cols)
    : num_rows(rows), num_cols(cols), is_computing(false), compute_start_cycle(0),
      current_a_row(0), current_a_col(0), current_b_row(0), current_b_col(0),
      current_c_row(0), current_c_col(0), cycles_completed(0) {

    // Initialize PE array
    pe_array.resize(num_rows);
    for (Size row = 0; row < num_rows; ++row) {
        pe_array[row].resize(num_cols);
        for (Size col = 0; col < num_cols; ++col) {
            pe_array[row][col] = std::make_unique<ProcessingElement>(row, col);
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

SystolicArray::SystolicArray(const SystolicArray& other)
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
            pe_array[row][col] = std::make_unique<ProcessingElement>(*other.pe_array[row][col]);
        }
    }

    // Copy buses (they contain queues, so copy them)
    horizontal_bus = other.horizontal_bus;
    vertical_bus = other.vertical_bus;
    diagonal_bus = other.diagonal_bus;
}

SystolicArray& SystolicArray::operator=(const SystolicArray& other) {
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
                pe_array[row][col] = std::make_unique<ProcessingElement>(*other.pe_array[row][col]);
            }
        }

        // Copy buses
        horizontal_bus = other.horizontal_bus;
        vertical_bus = other.vertical_bus;
        diagonal_bus = other.diagonal_bus;
    }
    return *this;
}

void SystolicArray::start_matmul(const MatMulConfig& config) {
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

bool SystolicArray::update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (!is_computing) {
        return false;
    }

    if (compute_start_cycle == 0) {
        compute_start_cycle = current_cycle;
        // For now, implement a simple direct matrix multiplication
        // Rather than true systolic dataflow
        perform_direct_matrix_multiply(scratchpads);

        if (current_op.completion_callback) {
            current_op.completion_callback();
        }

        is_computing = false;
        return true;
    }

    return false;
}

void SystolicArray::cycle_pe_array(Cycle current_cycle) {
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

void SystolicArray::load_a_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        return;
    }

    auto& scratchpad = scratchpads[current_op.scratchpad_id];

    // Simplified loading: stream matrix A row by row with staggering
    (void)current_cycle; // Suppress unused parameter warning for now

    for (Size row = 0; row < std::min(num_rows, current_op.m); ++row) {
        if (should_start_row(row, current_cycle)) {
            // Load one element per cycle per row, staggered
            Size global_row = current_a_row + row;
            Size global_col = current_a_col;

            if (global_row < current_op.m && global_col < current_op.k) {
                Address addr = calculate_matrix_address(
                    current_op.a_addr, global_row, global_col, current_op.k, sizeof(float));

                float value;
                scratchpad.read(addr, &value, sizeof(float));
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

void SystolicArray::load_b_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        return;
    }

    auto& scratchpad = scratchpads[current_op.scratchpad_id];

    // Simplified loading: stream matrix B column by column with staggering
    for (Size col = 0; col < std::min(num_cols, current_op.n); ++col) {
        if (should_start_col(col, current_cycle)) {
            // Load one element per cycle per column, staggered
            Size global_row = current_b_row;
            Size global_col = current_b_col + col;

            if (global_row < current_op.k && global_col < current_op.n) {
                Address addr = calculate_matrix_address(
                    current_op.b_addr, global_row, global_col, current_op.n, sizeof(float));

                float value;
                scratchpad.read(addr, &value, sizeof(float));
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

void SystolicArray::evacuate_c_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        return;
    }

    auto& scratchpad = scratchpads[current_op.scratchpad_id];

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
                float result = pe->get_c_output();
                if (result != 0.0f) { // Only evacuate non-zero results
                    diagonal_bus[diagonal_idx].push(result);

                    // Write result back to scratchpad
                    Size global_row = current_c_row + row;
                    Size global_col = current_c_col + col;

                    if (global_row < current_op.m && global_col < current_op.n) {
                        Address addr = calculate_matrix_address(
                            current_op.c_addr, global_row, global_col, current_op.n, sizeof(float));
                        scratchpad.write(addr, &result, sizeof(float));
                    }
                }
            }
        }
    }
}

void SystolicArray::propagate_horizontal_bus() {
    // A data flows left to right (horizontal)
    // Data in buses is automatically consumed by PEs
}

void SystolicArray::propagate_vertical_bus() {
    // B data flows top to bottom (vertical)
    // Data in buses is automatically consumed by PEs
}

void SystolicArray::propagate_diagonal_bus() {
    // C data flows diagonally for evacuation
    // Results are collected from PEs and moved to output
}

bool SystolicArray::should_start_row(Size row, Cycle current_cycle) const {
    Cycle relative_cycle = current_cycle - compute_start_cycle;
    return relative_cycle >= row_start_cycles[row];
}

bool SystolicArray::should_start_col(Size col, Cycle current_cycle) const {
    Cycle relative_cycle = current_cycle - compute_start_cycle;
    return relative_cycle >= col_start_cycles[col];
}

Size SystolicArray::calculate_stagger_delay(Size position) const {
    // Stagger by one cycle per position for proper systolic timing
    return position;
}

Address SystolicArray::calculate_matrix_address(Address base_addr, Size row, Size col,
                                               Size width, Size element_size) const {
    return base_addr + (row * width + col) * element_size;
}

Cycle SystolicArray::estimate_cycles(Size m, Size n, Size k) const {
    // Systolic array cycles = k (accumulation) + max(m,n) (fill/drain) + stagger delays
    return k + std::max(m, n) + std::max(num_rows, num_cols);
}

Size SystolicArray::calculate_throughput() const {
    // Theoretical peak: one MAC per PE per cycle
    return num_rows * num_cols;
}

void SystolicArray::stream_a_data(const std::vector<float>& data, Size row_offset) {
    // Interface for external streaming of A matrix data
    for (Size i = 0; i < std::min(data.size(), static_cast<size_t>(num_rows - row_offset)); ++i) {
        if (row_offset + i < horizontal_bus.size()) {
            horizontal_bus[row_offset + i].push(data[i]);
        }
    }
}

void SystolicArray::stream_b_data(const std::vector<float>& data, Size col_offset) {
    // Interface for external streaming of B matrix data
    for (Size i = 0; i < std::min(data.size(), static_cast<size_t>(num_cols - col_offset)); ++i) {
        if (col_offset + i < vertical_bus.size()) {
            vertical_bus[col_offset + i].push(data[i]);
        }
    }
}

std::vector<float> SystolicArray::evacuate_c_data(Size max_elements) {
    // Interface for external evacuation of C matrix results
    std::vector<float> results;
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

void SystolicArray::perform_direct_matrix_multiply(std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        return;
    }

    auto& scratchpad = scratchpads[current_op.scratchpad_id];

    // Read matrices A and B from scratchpad
    std::vector<float> matrix_a(current_op.m * current_op.k);
    std::vector<float> matrix_b(current_op.k * current_op.n);
    std::vector<float> matrix_c(current_op.m * current_op.n, 0.0f);

    // Read matrix A
    scratchpad.read(current_op.a_addr, matrix_a.data(), matrix_a.size() * sizeof(float));

    // Read matrix B
    scratchpad.read(current_op.b_addr, matrix_b.data(), matrix_b.size() * sizeof(float));

    // Perform matrix multiplication: C = A * B
    for (Size i = 0; i < current_op.m; ++i) {
        for (Size j = 0; j < current_op.n; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < current_op.k; ++k) {
                sum += matrix_a[i * current_op.k + k] * matrix_b[k * current_op.n + j];
            }
            matrix_c[i * current_op.n + j] = sum;
        }
    }

    // Write result matrix C back to scratchpad
    scratchpad.write(current_op.c_addr, matrix_c.data(), matrix_c.size() * sizeof(float));
}

void SystolicArray::reset() {
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

} // namespace sw::kpu
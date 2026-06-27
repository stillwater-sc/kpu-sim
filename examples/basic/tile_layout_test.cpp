/**
 * @file tile_layout_test.cpp
 * @brief Test and compare tile layout policies for memory channel interleaving
 *
 * This tool tests all four layout policies and shows:
 * - Channel assignments for each tile
 * - Conflict analysis for concurrent A/B access
 * - Memory utilization and fragmentation
 */

#include <sw/kpu/isa/tile_layout.hpp>
#include <iostream>
#include <iomanip>

using namespace sw::kpu::isa;
using sw::kpu::Size;
using sw::kpu::Address;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

// ============================================================================
// Test a single layout policy
// ============================================================================

void test_layout_policy(LayoutPolicy policy, const LayoutConfig& config) {
    print_separator("Testing: " + layout_policy_to_string(policy));

    try {
        auto layout = create_tile_layout(policy, config);

        // Print the layout's own report
        std::cout << layout->generate_report();

        // Detailed conflict analysis for all iterations
        std::cout << "\n--- Full Conflict Analysis ---\n";
        Size conflicts = 0;
        Size total_iterations = config.m_tiles * config.n_tiles * config.k_tiles;

        for (Size ti = 0; ti < config.m_tiles; ++ti) {
            for (Size tj = 0; tj < config.n_tiles; ++tj) {
                for (Size tk = 0; tk < config.k_tiles; ++tk) {
                    auto a_loc = layout->get_tile_location(MatrixID::A, ti, 0, tk);
                    auto b_loc = layout->get_tile_location(MatrixID::B, 0, tj, tk);

                    if (a_loc.channel == b_loc.channel) {
                        ++conflicts;
                        if (conflicts <= 10) {  // Only print first 10
                            std::cout << "  CONFLICT iter[" << ti << "," << tj << "," << tk << "]: "
                                      << "A[" << ti << "," << tk << "]->Ch" << (int)a_loc.channel
                                      << " B[" << tk << "," << tj << "]->Ch" << (int)b_loc.channel << "\n";
                        }
                    }
                }
            }
        }

        double conflict_rate = 100.0 * static_cast<double>(conflicts) / static_cast<double>(total_iterations);
        std::cout << "\nConflict Summary:\n";
        std::cout << "  Total iterations: " << total_iterations << "\n";
        std::cout << "  Conflicts: " << conflicts << " (" << std::fixed << std::setprecision(1)
                  << conflict_rate << "%)\n";
        std::cout << "  Conflict-free: " << (total_iterations - conflicts)
                  << " (" << (100.0 - conflict_rate) << "%)\n";

        // Channel utilization
        std::cout << "\n--- Channel Utilization ---\n";
        std::vector<Size> a_per_channel(config.num_channels, 0);
        std::vector<Size> b_per_channel(config.num_channels, 0);
        std::vector<Size> c_per_channel(config.num_channels, 0);

        for (Size ti = 0; ti < config.m_tiles; ++ti) {
            for (Size tk = 0; tk < config.k_tiles; ++tk) {
                auto loc = layout->get_tile_location(MatrixID::A, ti, 0, tk);
                a_per_channel[loc.channel]++;
            }
        }
        for (Size tk = 0; tk < config.k_tiles; ++tk) {
            for (Size tj = 0; tj < config.n_tiles; ++tj) {
                auto loc = layout->get_tile_location(MatrixID::B, 0, tj, tk);
                b_per_channel[loc.channel]++;
            }
        }
        for (Size ti = 0; ti < config.m_tiles; ++ti) {
            for (Size tj = 0; tj < config.n_tiles; ++tj) {
                auto loc = layout->get_tile_location(MatrixID::C, ti, tj, 0);
                c_per_channel[loc.channel]++;
            }
        }

        std::cout << std::setw(10) << "Channel" << std::setw(10) << "A tiles"
                  << std::setw(10) << "B tiles" << std::setw(10) << "C tiles"
                  << std::setw(10) << "Total" << "\n";
        std::cout << std::string(50, '-') << "\n";

        for (uint8_t ch = 0; ch < config.num_channels; ++ch) {
            Size total = a_per_channel[ch] + b_per_channel[ch] + c_per_channel[ch];
            std::cout << std::setw(10) << (int)ch
                      << std::setw(10) << a_per_channel[ch]
                      << std::setw(10) << b_per_channel[ch]
                      << std::setw(10) << c_per_channel[ch]
                      << std::setw(10) << total << "\n";
        }

    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
================================================================================
           Tile Layout Policy Test Tool
================================================================================

This tool tests all four memory layout policies for tensor tiles.
Each policy has different trade-offs for channel utilization and conflicts.

For matmul C = A x B, each iteration accesses:
  - A[ti, tk] (input)
  - B[tk, tj] (weights)
  - C[ti, tj] (output, but at different time)

We want A and B to be on DIFFERENT channels for maximum bandwidth.

================================================================================
)";

    // Configure for a small test case
    LayoutConfig config;
    config.num_channels = 4;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.tile_size_bytes = 1024;  // 16x16 float32
    config.element_size = 4;

    // 32x32x32 matmul with 16x16 tiles = 2x2x2 tiles
    config.m_tiles = 2;
    config.n_tiles = 2;
    config.k_tiles = 2;

    // Matrix-partitioned channel assignments
    config.matrix_channels.a_channels = {0, 1};  // A on channels 0,1
    config.matrix_channels.b_channels = {2};     // B on channel 2
    config.matrix_channels.c_channels = {3};     // C on channel 3

    std::cout << "Test Configuration:\n";
    std::cout << "  Channels: " << (int)config.num_channels << "\n";
    std::cout << "  Tile dimensions: " << config.m_tiles << " x " << config.n_tiles
              << " x " << config.k_tiles << "\n";
    std::cout << "  A tiles: " << config.num_a_tiles() << "\n";
    std::cout << "  B tiles: " << config.num_b_tiles() << "\n";
    std::cout << "  C tiles: " << config.num_c_tiles() << "\n";
    std::cout << "  Total tiles: " << config.total_tiles() << "\n";

    // Test Option 1: Matrix-Partitioned
    test_layout_policy(LayoutPolicy::MATRIX_PARTITIONED, config);

    // Test Option 2: Round-Robin
    test_layout_policy(LayoutPolicy::ROUND_ROBIN, config);

    // Test Option 3: Iteration-Aware
    test_layout_policy(LayoutPolicy::ITERATION_AWARE, config);

    // Test Option 4: Hardware-Interleaved
    config.interleave_granularity = 64;
    test_layout_policy(LayoutPolicy::HARDWARE_INTERLEAVED, config);

    // Now test with a larger configuration
    print_separator("Larger Configuration Test (4x4x4 tiles)");

    LayoutConfig large_config = config;
    large_config.m_tiles = 4;
    large_config.n_tiles = 4;
    large_config.k_tiles = 4;

    std::cout << "\nLarger test: " << large_config.m_tiles << "x" << large_config.n_tiles
              << "x" << large_config.k_tiles << " tiles\n";
    std::cout << "Total iterations: " << (large_config.m_tiles * large_config.n_tiles * large_config.k_tiles) << "\n";

    // Compare conflict rates
    std::cout << "\n--- Conflict Rate Comparison ---\n";
    std::cout << std::setw(25) << "Policy" << std::setw(15) << "Conflicts"
              << std::setw(15) << "Rate" << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (auto policy : {LayoutPolicy::MATRIX_PARTITIONED, LayoutPolicy::ROUND_ROBIN,
                        LayoutPolicy::ITERATION_AWARE, LayoutPolicy::HARDWARE_INTERLEAVED}) {
        try {
            auto layout = create_tile_layout(policy, large_config);

            Size conflicts = 0;
            Size total = large_config.m_tiles * large_config.n_tiles * large_config.k_tiles;

            for (Size ti = 0; ti < large_config.m_tiles; ++ti) {
                for (Size tj = 0; tj < large_config.n_tiles; ++tj) {
                    for (Size tk = 0; tk < large_config.k_tiles; ++tk) {
                        if (layout->conflicts(MatrixID::A, ti, 0, tk,
                                             MatrixID::B, 0, tj, tk)) {
                            ++conflicts;
                        }
                    }
                }
            }

            double rate = 100.0 * static_cast<double>(conflicts) / static_cast<double>(total);
            std::cout << std::setw(25) << layout_policy_to_string(policy)
                      << std::setw(15) << conflicts
                      << std::setw(14) << std::fixed << std::setprecision(1) << rate << "%\n";

        } catch (const std::exception& e) {
            std::cout << std::setw(25) << layout_policy_to_string(policy)
                      << std::setw(15) << "ERROR" << "\n";
        }
    }

    print_separator("Summary");

    std::cout << R"(
Layout Policy Trade-offs:

1. MATRIX_PARTITIONED:
   + Simple, predictable
   + Zero conflicts if channels are properly assigned
   - Uneven channel utilization
   - Constrains max tensor size per matrix

2. ROUND_ROBIN:
   + Even distribution over all tiles
   + Simple implementation
   - May have conflicts (A and B on same channel)
   - Conflict rate depends on tile counts

3. ITERATION_AWARE:
   + ZERO conflicts guaranteed (A on even, B on odd)
   + Good for maximum bandwidth
   - Requires even number of channels
   - More complex address calculation

4. HARDWARE_INTERLEAVED:
   + Matches real hardware behavior
   + Uses address bits for channel selection
   - May have fragmentation overhead
   - Tile sizes constrained by interleave granularity

Recommendation:
- Start with MATRIX_PARTITIONED for simplicity
- Use ITERATION_AWARE for maximum bandwidth (production)
- HARDWARE_INTERLEAVED for accurate hardware modeling

)";

    return 0;
}

#pragma once

/**
 * @file processor_array_topology.hpp
 * @brief Processor array topology definitions and L1 buffer derivation
 *
 * L1 Streaming Buffer Architecture:
 * ---------------------------------
 * L1 streaming buffers are part of the compute fabric, providing data paths
 * for input streaming and output extraction at each edge of the processor array.
 *
 * For a RECTANGULAR array (rows × cols):
 *   - Each edge has both ingress (input) and egress (output) buffers
 *   - TOP edge:    cols buffers in (B weights) + cols buffers out (C output)
 *   - BOTTOM edge: cols buffers in (streaming) + cols buffers out (C output)
 *   - LEFT edge:   rows buffers in (A inputs) + rows buffers out (C output)
 *   - RIGHT edge:  rows buffers in (streaming) + rows buffers out (C output)
 *   - Total per tile: 4 × (rows + cols) buffers
 *
 * For a HEXAGONAL array (side_length):
 *   - Each PE has 6 neighbors in a hex grid
 *   - 3 ingress + 3 egress per edge PE
 *   - Total per tile: 6 × side_length × 2 buffers (approximation)
 *
 * The number of L1 buffers is DERIVED from the processor array configuration,
 * not independently configurable.
 *
 * @see models/kpu/host_T1_KPU.cpp for detailed architecture documentation
 */

#include <sw/concepts.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sw::kpu {

/**
 * @brief Processor array topology
 *
 * Different PE array layouts require different L1 buffer configurations.
 */
enum class ProcessorArrayTopology : uint8_t {
    RECTANGULAR = 0,    ///< Standard rows × cols rectangular grid
    HEXAGONAL = 1,      ///< Hexagonal grid (6 neighbors per PE)
    // Future topologies can be added here
};

/**
 * @brief Convert topology enum to string
 */
inline std::string topology_to_string(ProcessorArrayTopology topology) {
    switch (topology) {
        case ProcessorArrayTopology::RECTANGULAR: return "rectangular";
        case ProcessorArrayTopology::HEXAGONAL: return "hexagonal";
        default: return "unknown";
    }
}

/**
 * @brief Parse topology from string
 * @throws std::invalid_argument if string is not recognized
 */
inline ProcessorArrayTopology topology_from_string(const std::string& str) {
    if (str == "rectangular" || str == "rect") {
        return ProcessorArrayTopology::RECTANGULAR;
    } else if (str == "hexagonal" || str == "hex") {
        return ProcessorArrayTopology::HEXAGONAL;
    }
    throw std::invalid_argument("Unknown topology: " + str);
}

/**
 * @brief Compute the number of L1 streaming buffers required for a processor array
 *
 * L1 buffers provide data paths at each edge of the PE array:
 * - Ingress buffers: stream input data (A matrix rows, B matrix columns)
 * - Egress buffers: stream output data (C matrix tiles in any direction)
 *
 * For RECTANGULAR arrays (rows × cols):
 *   Each of the 4 edges (TOP, BOTTOM, LEFT, RIGHT) has:
 *   - Ingress: one buffer per PE on that edge
 *   - Egress: one buffer per PE on that edge
 *   Formula: 4 × (rows + cols) per compute tile
 *   Example: 16×16 array = 4 × 32 = 128 L1 buffers per tile
 *
 * For HEXAGONAL arrays (side_length):
 *   Each edge PE has 3 data directions (vs 2 for rectangular)
 *   Formula: 6 × side_length × 2 per compute tile (approximation)
 *
 * @param topology Processor array topology
 * @param rows Number of rows (or side_length for hexagonal)
 * @param cols Number of columns (ignored for hexagonal)
 * @param compute_tile_count Number of compute tiles
 * @return Total number of L1 streaming buffers required
 */
inline Size compute_l1_buffer_count(
    ProcessorArrayTopology topology,
    Size rows,
    Size cols,
    Size compute_tile_count)
{
    if (compute_tile_count == 0 || rows == 0) {
        return 0;
    }

    Size buffers_per_tile = 0;

    switch (topology) {
        case ProcessorArrayTopology::RECTANGULAR:
            // 4 edges × (row_PEs + col_PEs) × (ingress + egress)
            // = 2 × rows (left + right) + 2 × cols (top + bottom)
            // Each edge has both input and output buffers = × 2
            // Total = 4 × (rows + cols)
            if (cols == 0) cols = rows;  // Square array if cols not specified
            buffers_per_tile = 4 * (rows + cols);
            break;

        case ProcessorArrayTopology::HEXAGONAL:
            // Hexagonal array: 6 edge directions, each with in+out buffers
            // For a hex array with side_length = rows:
            // Perimeter ≈ 6 × side_length, each PE has in+out
            buffers_per_tile = 6 * rows * 2;
            break;

        default:
            throw std::invalid_argument("Unknown processor array topology");
    }

    return compute_tile_count * buffers_per_tile;
}

/**
 * @brief Compute L1 buffers for rectangular array (convenience function)
 */
inline Size compute_l1_buffer_count_rectangular(Size rows, Size cols, Size compute_tile_count) {
    return compute_l1_buffer_count(ProcessorArrayTopology::RECTANGULAR, rows, cols, compute_tile_count);
}

/**
 * @brief Compute L1 buffers for hexagonal array (convenience function)
 */
inline Size compute_l1_buffer_count_hexagonal(Size side_length, Size compute_tile_count) {
    return compute_l1_buffer_count(ProcessorArrayTopology::HEXAGONAL, side_length, 0, compute_tile_count);
}

/**
 * @brief Validate that L1 buffer count matches the processor array configuration
 * @param configured_count The configured L1 buffer count
 * @param topology Processor array topology
 * @param rows Array rows (or side_length for hexagonal)
 * @param cols Array columns
 * @param compute_tile_count Number of compute tiles
 * @return true if the count matches the derived value
 */
inline bool validate_l1_buffer_count(
    Size configured_count,
    ProcessorArrayTopology topology,
    Size rows,
    Size cols,
    Size compute_tile_count)
{
    Size expected = compute_l1_buffer_count(topology, rows, cols, compute_tile_count);
    return configured_count == expected;
}

/**
 * @brief Get a description of the L1 buffer layout for a configuration
 */
inline std::string describe_l1_buffer_layout(
    ProcessorArrayTopology topology,
    Size rows,
    Size cols,
    Size compute_tile_count)
{
    Size total = compute_l1_buffer_count(topology, rows, cols, compute_tile_count);
    Size per_tile = (compute_tile_count > 0) ? total / compute_tile_count : 0;

    std::string desc;
    switch (topology) {
        case ProcessorArrayTopology::RECTANGULAR: {
            Size per_edge_row = rows * 2;  // ingress + egress
            Size per_edge_col = cols * 2;
            desc = std::to_string(total) + " L1 buffers (" +
                   std::to_string(per_tile) + " per tile: " +
                   std::to_string(per_edge_col) + " TOP + " +
                   std::to_string(per_edge_col) + " BOTTOM + " +
                   std::to_string(per_edge_row) + " LEFT + " +
                   std::to_string(per_edge_row) + " RIGHT)";
            break;
        }
        case ProcessorArrayTopology::HEXAGONAL:
            desc = std::to_string(total) + " L1 buffers (" +
                   std::to_string(per_tile) + " per tile, hexagonal layout)";
            break;
        default:
            desc = std::to_string(total) + " L1 buffers";
    }
    return desc;
}

} // namespace sw::kpu

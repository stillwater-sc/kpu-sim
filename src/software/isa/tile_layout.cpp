/**
 * @file tile_layout.cpp
 * @brief Implementation of memory layout strategies for tensor tiles
 */

#include <sw/kpu/isa/tile_layout.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace sw::kpu::isa {

// ============================================================================
// TileLayout Base Class
// ============================================================================

std::string TileLayout::generate_report() const {
    std::ostringstream oss;
    oss << "Layout Policy: " << layout_policy_to_string(policy()) << "\n";
    oss << "Channels: " << (int)config_.num_channels << "\n";
    oss << "Tiles: A=" << config_.num_a_tiles()
        << " B=" << config_.num_b_tiles()
        << " C=" << config_.num_c_tiles() << "\n";
    return oss.str();
}

// ============================================================================
// Option 1: Matrix-Partitioned Layout Implementation
// ============================================================================

MatrixPartitionedLayout::MatrixPartitionedLayout(const LayoutConfig& config)
    : TileLayout(config)
{
    tiles_per_channel_.resize(config_.num_channels, 0);

    // Validate A and B channel assignments don't overlap
    // (They must be on separate channels for concurrent access)
    // C can share channels with A or B since it's accessed at different times
    std::vector<bool> a_used(config_.num_channels, false);
    std::vector<bool> b_used(config_.num_channels, false);

    for (uint8_t ch : config_.matrix_channels.a_channels) {
        if (ch >= config_.num_channels) {
            throw std::invalid_argument("A channel " + std::to_string(ch) + " out of range");
        }
        a_used[ch] = true;
    }
    for (uint8_t ch : config_.matrix_channels.b_channels) {
        if (ch >= config_.num_channels) {
            throw std::invalid_argument("B channel " + std::to_string(ch) + " out of range");
        }
        if (a_used[ch]) {
            throw std::invalid_argument("Channel " + std::to_string(ch) +
                " assigned to both A and B (must be separate for concurrent access)");
        }
        b_used[ch] = true;
    }
    // C channels only need to be in range - can share with A or B
    for (uint8_t ch : config_.matrix_channels.c_channels) {
        if (ch >= config_.num_channels) {
            throw std::invalid_argument("C channel " + std::to_string(ch) + " out of range");
        }
    }
}

const std::vector<uint8_t>& MatrixPartitionedLayout::get_matrix_channels(MatrixID matrix) const {
    switch (matrix) {
        case MatrixID::A: return config_.matrix_channels.a_channels;
        case MatrixID::B: return config_.matrix_channels.b_channels;
        case MatrixID::C: return config_.matrix_channels.c_channels;
        default: throw std::invalid_argument("Invalid matrix ID");
    }
}

Size MatrixPartitionedLayout::get_local_tile_index(MatrixID matrix,
                                                    Size ti, Size tj, Size tk) const {
    switch (matrix) {
        case MatrixID::A:
            // A[M,K] indexed by (ti, tk)
            return ti * config_.k_tiles + tk;
        case MatrixID::B:
            // B[K,N] indexed by (tk, tj)
            return tk * config_.n_tiles + tj;
        case MatrixID::C:
            // C[M,N] indexed by (ti, tj)
            return ti * config_.n_tiles + tj;
        default:
            return 0;
    }
}

TileLocation MatrixPartitionedLayout::get_tile_location(MatrixID matrix,
                                                         Size ti, Size tj, Size tk) const {
    TileLocation loc;

    const auto& channels = get_matrix_channels(matrix);
    Size local_idx = get_local_tile_index(matrix, ti, tj, tk);

    // Distribute tiles across assigned channels
    Size channel_idx = local_idx % channels.size();
    loc.channel = channels[channel_idx];

    // Address within channel: tiles on same channel are packed sequentially
    Size tiles_before_on_channel = local_idx / channels.size();
    loc.address = tiles_before_on_channel * config_.tile_size_bytes;

    // L3 and L2 assignments follow channel
    loc.l3_tile_id = loc.channel % config_.num_l3_tiles;
    loc.l2_bank_id = loc.channel % config_.num_l2_banks;

    return loc;
}

std::string MatrixPartitionedLayout::generate_report() const {
    std::ostringstream oss;
    oss << "\n========================================\n";
    oss << "Matrix-Partitioned Layout Report\n";
    oss << "========================================\n\n";

    oss << "Configuration:\n";
    oss << "  Channels: " << (int)config_.num_channels << "\n";
    oss << "  Tile size: " << config_.tile_size_bytes << " bytes\n";
    oss << "  Tiles: M=" << config_.m_tiles << " N=" << config_.n_tiles
        << " K=" << config_.k_tiles << "\n\n";

    oss << "Channel Assignments:\n";
    oss << "  Matrix A channels: [";
    for (size_t i = 0; i < config_.matrix_channels.a_channels.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (int)config_.matrix_channels.a_channels[i];
    }
    oss << "] (" << config_.num_a_tiles() << " tiles)\n";

    oss << "  Matrix B channels: [";
    for (size_t i = 0; i < config_.matrix_channels.b_channels.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (int)config_.matrix_channels.b_channels[i];
    }
    oss << "] (" << config_.num_b_tiles() << " tiles)\n";

    oss << "  Matrix C channels: [";
    for (size_t i = 0; i < config_.matrix_channels.c_channels.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (int)config_.matrix_channels.c_channels[i];
    }
    oss << "] (" << config_.num_c_tiles() << " tiles)\n\n";

    // Show sample tile placements
    oss << "Sample Tile Placements:\n";
    for (Size ti = 0; ti < std::min(Size(2), config_.m_tiles); ++ti) {
        for (Size tk = 0; tk < std::min(Size(2), config_.k_tiles); ++tk) {
            auto loc = get_tile_location(MatrixID::A, ti, 0, tk);
            oss << "  A[" << ti << "," << tk << "] -> " << loc.to_string() << "\n";
        }
    }
    for (Size tk = 0; tk < std::min(Size(2), config_.k_tiles); ++tk) {
        for (Size tj = 0; tj < std::min(Size(2), config_.n_tiles); ++tj) {
            auto loc = get_tile_location(MatrixID::B, 0, tj, tk);
            oss << "  B[" << tk << "," << tj << "] -> " << loc.to_string() << "\n";
        }
    }

    // Check for conflicts in typical iteration
    oss << "\nConflict Analysis (iteration ti=0, tj=0, tk=0):\n";
    auto a_loc = get_tile_location(MatrixID::A, 0, 0, 0);
    auto b_loc = get_tile_location(MatrixID::B, 0, 0, 0);
    oss << "  A[0,0] on channel " << (int)a_loc.channel << "\n";
    oss << "  B[0,0] on channel " << (int)b_loc.channel << "\n";
    oss << "  Conflict: " << (a_loc.channel == b_loc.channel ? "YES" : "NO") << "\n";

    return oss.str();
}

// ============================================================================
// Option 2: Round-Robin Layout Implementation
// ============================================================================

RoundRobinLayout::RoundRobinLayout(const LayoutConfig& config)
    : TileLayout(config)
{
    a_offset_ = 0;
    b_offset_ = config_.num_a_tiles();
    c_offset_ = config_.num_a_tiles() + config_.num_b_tiles();
}

Size RoundRobinLayout::get_global_tile_index(MatrixID matrix,
                                              Size ti, Size tj, Size tk) const {
    Size local_idx;
    Size offset;

    switch (matrix) {
        case MatrixID::A:
            local_idx = ti * config_.k_tiles + tk;
            offset = a_offset_;
            break;
        case MatrixID::B:
            local_idx = tk * config_.n_tiles + tj;
            offset = b_offset_;
            break;
        case MatrixID::C:
            local_idx = ti * config_.n_tiles + tj;
            offset = c_offset_;
            break;
        default:
            return 0;
    }

    return offset + local_idx;
}

TileLocation RoundRobinLayout::get_tile_location(MatrixID matrix,
                                                  Size ti, Size tj, Size tk) const {
    TileLocation loc;

    Size global_idx = get_global_tile_index(matrix, ti, tj, tk);

    // Channel = global_index % num_channels
    loc.channel = global_idx % config_.num_channels;

    // Address = sequential within channel
    Size tiles_before_on_channel = global_idx / config_.num_channels;
    loc.address = tiles_before_on_channel * config_.tile_size_bytes;

    // L3 and L2 follow channel
    loc.l3_tile_id = loc.channel % config_.num_l3_tiles;
    loc.l2_bank_id = loc.channel % config_.num_l2_banks;

    return loc;
}

std::string RoundRobinLayout::generate_report() const {
    std::ostringstream oss;
    oss << "\n========================================\n";
    oss << "Round-Robin Layout Report\n";
    oss << "========================================\n\n";

    oss << "Configuration:\n";
    oss << "  Channels: " << (int)config_.num_channels << "\n";
    oss << "  Tile size: " << config_.tile_size_bytes << " bytes\n";
    oss << "  Total tiles: " << config_.total_tiles() << "\n";
    oss << "  A tiles: " << config_.num_a_tiles() << " (indices 0-" << (a_offset_ + config_.num_a_tiles() - 1) << ")\n";
    oss << "  B tiles: " << config_.num_b_tiles() << " (indices " << b_offset_ << "-" << (b_offset_ + config_.num_b_tiles() - 1) << ")\n";
    oss << "  C tiles: " << config_.num_c_tiles() << " (indices " << c_offset_ << "-" << (c_offset_ + config_.num_c_tiles() - 1) << ")\n\n";

    // Show sample placements
    oss << "Sample Tile Placements:\n";
    for (Size ti = 0; ti < std::min(Size(2), config_.m_tiles); ++ti) {
        for (Size tk = 0; tk < std::min(Size(2), config_.k_tiles); ++tk) {
            auto loc = get_tile_location(MatrixID::A, ti, 0, tk);
            Size global = get_global_tile_index(MatrixID::A, ti, 0, tk);
            oss << "  A[" << ti << "," << tk << "] global=" << global
                << " -> " << loc.to_string() << "\n";
        }
    }
    for (Size tk = 0; tk < std::min(Size(2), config_.k_tiles); ++tk) {
        for (Size tj = 0; tj < std::min(Size(2), config_.n_tiles); ++tj) {
            auto loc = get_tile_location(MatrixID::B, 0, tj, tk);
            Size global = get_global_tile_index(MatrixID::B, 0, tj, tk);
            oss << "  B[" << tk << "," << tj << "] global=" << global
                << " -> " << loc.to_string() << "\n";
        }
    }

    // Conflict analysis
    oss << "\nConflict Analysis for first few iterations:\n";
    for (Size ti = 0; ti < std::min(Size(2), config_.m_tiles); ++ti) {
        for (Size tj = 0; tj < std::min(Size(2), config_.n_tiles); ++tj) {
            for (Size tk = 0; tk < std::min(Size(2), config_.k_tiles); ++tk) {
                auto a_loc = get_tile_location(MatrixID::A, ti, 0, tk);
                auto b_loc = get_tile_location(MatrixID::B, 0, tj, tk);
                bool conflict = (a_loc.channel == b_loc.channel);
                oss << "  iter[" << ti << "," << tj << "," << tk << "]: "
                    << "A->Ch" << (int)a_loc.channel
                    << " B->Ch" << (int)b_loc.channel
                    << (conflict ? " CONFLICT!" : " OK") << "\n";
            }
        }
    }

    return oss.str();
}

// ============================================================================
// Option 3: Iteration-Aware Layout Implementation
// ============================================================================

IterationAwareLayout::IterationAwareLayout(const LayoutConfig& config)
    : TileLayout(config)
{
    if (config_.num_channels < 2 || config_.num_channels % 2 != 0) {
        throw std::invalid_argument("Iteration-aware layout requires even number of channels >= 2");
    }
    half_channels_ = config_.num_channels / 2;

    channel_placements_.resize(config_.num_channels);
    precompute_placements();
}

void IterationAwareLayout::precompute_placements() {
    // Clear existing placements
    for (auto& ch_placements : channel_placements_) {
        ch_placements.clear();
    }

    // Place A tiles on even channels
    for (Size ti = 0; ti < config_.m_tiles; ++ti) {
        for (Size tk = 0; tk < config_.k_tiles; ++tk) {
            uint8_t channel = 2 * ((ti + tk) % half_channels_);
            Size idx = channel_placements_[channel].size();
            Address addr = idx * config_.tile_size_bytes;
            channel_placements_[channel].push_back({MatrixID::A, ti, 0, tk, addr});
        }
    }

    // Place B tiles on odd channels
    for (Size tk = 0; tk < config_.k_tiles; ++tk) {
        for (Size tj = 0; tj < config_.n_tiles; ++tj) {
            uint8_t channel = 2 * ((tk + tj) % half_channels_) + 1;
            Size idx = channel_placements_[channel].size();
            Address addr = idx * config_.tile_size_bytes;
            channel_placements_[channel].push_back({MatrixID::B, 0, tj, tk, addr});
        }
    }

    // Place C tiles on even channels (not concurrent with A/B loads)
    for (Size ti = 0; ti < config_.m_tiles; ++ti) {
        for (Size tj = 0; tj < config_.n_tiles; ++tj) {
            uint8_t channel = 2 * ((ti + tj) % half_channels_);
            Size idx = channel_placements_[channel].size();
            Address addr = idx * config_.tile_size_bytes;
            channel_placements_[channel].push_back({MatrixID::C, ti, tj, 0, addr});
        }
    }
}

Address IterationAwareLayout::lookup_address(MatrixID matrix,
                                              Size ti, Size tj, Size tk) const {
    // Determine which channel this tile is on
    uint8_t channel;
    switch (matrix) {
        case MatrixID::A:
            channel = 2 * ((ti + tk) % half_channels_);
            break;
        case MatrixID::B:
            channel = 2 * ((tk + tj) % half_channels_) + 1;
            break;
        case MatrixID::C:
            channel = 2 * ((ti + tj) % half_channels_);
            break;
        default:
            return 0;
    }

    // Search for the tile in channel placements
    for (const auto& p : channel_placements_[channel]) {
        if (p.matrix == matrix) {
            bool match = false;
            switch (matrix) {
                case MatrixID::A:
                    match = (p.ti == ti && p.tk == tk);
                    break;
                case MatrixID::B:
                    match = (p.tj == tj && p.tk == tk);
                    break;
                case MatrixID::C:
                    match = (p.ti == ti && p.tj == tj);
                    break;
            }
            if (match) {
                return p.address;
            }
        }
    }

    return 0;  // Not found
}

TileLocation IterationAwareLayout::get_tile_location(MatrixID matrix,
                                                      Size ti, Size tj, Size tk) const {
    TileLocation loc;

    // Channel assignment: A on even, B on odd
    switch (matrix) {
        case MatrixID::A:
            loc.channel = 2 * ((ti + tk) % half_channels_);
            break;
        case MatrixID::B:
            loc.channel = 2 * ((tk + tj) % half_channels_) + 1;
            break;
        case MatrixID::C:
            loc.channel = 2 * ((ti + tj) % half_channels_);
            break;
    }

    loc.address = lookup_address(matrix, ti, tj, tk);
    loc.l3_tile_id = loc.channel % config_.num_l3_tiles;
    loc.l2_bank_id = loc.channel % config_.num_l2_banks;

    return loc;
}

std::string IterationAwareLayout::generate_report() const {
    std::ostringstream oss;
    oss << "\n========================================\n";
    oss << "Iteration-Aware Layout Report\n";
    oss << "========================================\n\n";

    oss << "Configuration:\n";
    oss << "  Channels: " << (int)config_.num_channels << " (even for A/C, odd for B)\n";
    oss << "  Half channels: " << half_channels_ << "\n";
    oss << "  Tile size: " << config_.tile_size_bytes << " bytes\n\n";

    oss << "Channel Assignment Rules:\n";
    oss << "  A[ti,tk] -> Channel 2 * ((ti + tk) % " << half_channels_ << ") = even\n";
    oss << "  B[tk,tj] -> Channel 2 * ((tk + tj) % " << half_channels_ << ") + 1 = odd\n";
    oss << "  C[ti,tj] -> Channel 2 * ((ti + tj) % " << half_channels_ << ") = even\n\n";

    oss << "Tiles per Channel:\n";
    for (uint8_t ch = 0; ch < config_.num_channels; ++ch) {
        Size a_count = 0, b_count = 0, c_count = 0;
        for (const auto& p : channel_placements_[ch]) {
            switch (p.matrix) {
                case MatrixID::A: ++a_count; break;
                case MatrixID::B: ++b_count; break;
                case MatrixID::C: ++c_count; break;
            }
        }
        oss << "  Channel " << (int)ch << ": A=" << a_count
            << " B=" << b_count << " C=" << c_count
            << " total=" << channel_placements_[ch].size() << "\n";
    }

    // Verify no conflicts
    oss << "\nConflict Verification (A and B should NEVER conflict):\n";
    bool any_conflict = false;
    for (Size ti = 0; ti < std::min(Size(3), config_.m_tiles); ++ti) {
        for (Size tj = 0; tj < std::min(Size(3), config_.n_tiles); ++tj) {
            for (Size tk = 0; tk < std::min(Size(3), config_.k_tiles); ++tk) {
                auto a_loc = get_tile_location(MatrixID::A, ti, 0, tk);
                auto b_loc = get_tile_location(MatrixID::B, 0, tj, tk);
                bool conflict = (a_loc.channel == b_loc.channel);
                if (conflict) {
                    any_conflict = true;
                    oss << "  CONFLICT at iter[" << ti << "," << tj << "," << tk << "]: "
                        << "A->Ch" << (int)a_loc.channel
                        << " B->Ch" << (int)b_loc.channel << "\n";
                }
            }
        }
    }
    if (!any_conflict) {
        oss << "  No conflicts detected - A always on even, B always on odd channels.\n";
    }

    return oss.str();
}

// ============================================================================
// Option 4: Hardware-Interleaved Layout Implementation
// ============================================================================

HardwareInterleavedLayout::HardwareInterleavedLayout(const LayoutConfig& config)
    : TileLayout(config)
{
    precompute_addresses();
}

uint8_t HardwareInterleavedLayout::address_to_channel(Address addr) const {
    return (addr / config_.interleave_granularity) % config_.num_channels;
}

Address HardwareInterleavedLayout::find_address_on_channel(Address min_addr,
                                                            uint8_t target_channel) const {
    // Align to interleave granularity
    Address aligned = (min_addr / config_.interleave_granularity) * config_.interleave_granularity;
    if (aligned < min_addr) {
        aligned += config_.interleave_granularity;
    }

    // Find address on target channel
    while (address_to_channel(aligned) != target_channel) {
        aligned += config_.interleave_granularity;
    }

    return aligned;
}

Size HardwareInterleavedLayout::get_local_tile_index(MatrixID matrix,
                                                      Size ti, Size tj, Size tk) const {
    switch (matrix) {
        case MatrixID::A:
            return ti * config_.k_tiles + tk;
        case MatrixID::B:
            return tk * config_.n_tiles + tj;
        case MatrixID::C:
            return ti * config_.n_tiles + tj;
        default:
            return 0;
    }
}

void HardwareInterleavedLayout::precompute_addresses() {
    a_addresses_.resize(config_.num_a_tiles());
    b_addresses_.resize(config_.num_b_tiles());
    c_addresses_.resize(config_.num_c_tiles());

    Address next_addr = 0;
    Size half_channels = config_.num_channels / 2;
    if (half_channels == 0) half_channels = 1;

    // A tiles on even channels (0, 2, 4, ...)
    for (Size ti = 0; ti < config_.m_tiles; ++ti) {
        for (Size tk = 0; tk < config_.k_tiles; ++tk) {
            Size local_idx = ti * config_.k_tiles + tk;
            uint8_t target_channel = 2 * ((ti + tk) % half_channels);
            if (target_channel >= config_.num_channels) {
                target_channel = 0;  // Fallback for odd channel counts
            }

            Address addr = find_address_on_channel(next_addr, target_channel);
            a_addresses_[local_idx] = addr;
            next_addr = addr + config_.tile_size_bytes;
        }
    }

    // B tiles on odd channels (1, 3, 5, ...)
    for (Size tk = 0; tk < config_.k_tiles; ++tk) {
        for (Size tj = 0; tj < config_.n_tiles; ++tj) {
            Size local_idx = tk * config_.n_tiles + tj;
            uint8_t target_channel = 2 * ((tk + tj) % half_channels) + 1;
            if (target_channel >= config_.num_channels) {
                target_channel = 1 % config_.num_channels;  // Fallback
            }

            Address addr = find_address_on_channel(next_addr, target_channel);
            b_addresses_[local_idx] = addr;
            next_addr = addr + config_.tile_size_bytes;
        }
    }

    // C tiles - use any channel
    for (Size ti = 0; ti < config_.m_tiles; ++ti) {
        for (Size tj = 0; tj < config_.n_tiles; ++tj) {
            Size local_idx = ti * config_.n_tiles + tj;
            c_addresses_[local_idx] = next_addr;
            next_addr += config_.tile_size_bytes;
        }
    }
}

TileLocation HardwareInterleavedLayout::get_tile_location(MatrixID matrix,
                                                           Size ti, Size tj, Size tk) const {
    TileLocation loc;
    Size local_idx = get_local_tile_index(matrix, ti, tj, tk);

    switch (matrix) {
        case MatrixID::A:
            loc.address = a_addresses_[local_idx];
            break;
        case MatrixID::B:
            loc.address = b_addresses_[local_idx];
            break;
        case MatrixID::C:
            loc.address = c_addresses_[local_idx];
            break;
    }

    loc.channel = address_to_channel(loc.address);
    loc.l3_tile_id = loc.channel % config_.num_l3_tiles;
    loc.l2_bank_id = loc.channel % config_.num_l2_banks;

    return loc;
}

std::string HardwareInterleavedLayout::generate_report() const {
    std::ostringstream oss;
    oss << "\n========================================\n";
    oss << "Hardware-Interleaved Layout Report\n";
    oss << "========================================\n\n";

    oss << "Configuration:\n";
    oss << "  Channels: " << (int)config_.num_channels << "\n";
    oss << "  Interleave granularity: " << config_.interleave_granularity << " bytes\n";
    oss << "  Tile size: " << config_.tile_size_bytes << " bytes\n";
    oss << "  Channel selection: (addr / " << config_.interleave_granularity
        << ") % " << (int)config_.num_channels << "\n\n";

    // Sample addresses
    oss << "Sample Tile Addresses:\n";
    for (Size i = 0; i < std::min(Size(4), config_.num_a_tiles()); ++i) {
        Size ti = i / config_.k_tiles;
        Size tk = i % config_.k_tiles;
        auto loc = get_tile_location(MatrixID::A, ti, 0, tk);
        oss << "  A[" << ti << "," << tk << "] addr=0x" << std::hex << loc.address
            << std::dec << " -> Ch" << (int)loc.channel << "\n";
    }
    for (Size i = 0; i < std::min(Size(4), config_.num_b_tiles()); ++i) {
        Size tk = i / config_.n_tiles;
        Size tj = i % config_.n_tiles;
        auto loc = get_tile_location(MatrixID::B, 0, tj, tk);
        oss << "  B[" << tk << "," << tj << "] addr=0x" << std::hex << loc.address
            << std::dec << " -> Ch" << (int)loc.channel << "\n";
    }

    // Memory utilization
    Address total_used = 0;
    if (!c_addresses_.empty()) {
        total_used = c_addresses_.back() + config_.tile_size_bytes;
    }
    Address minimum = config_.total_tiles() * config_.tile_size_bytes;
    double overhead = 100.0 * (total_used - minimum) / minimum;

    oss << "\nMemory Utilization:\n";
    oss << "  Total address space used: " << total_used << " bytes\n";
    oss << "  Minimum required: " << minimum << " bytes\n";
    oss << "  Fragmentation overhead: " << std::fixed << std::setprecision(1)
        << overhead << "%\n";

    return oss.str();
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<TileLayout> create_tile_layout(LayoutPolicy policy,
                                                const LayoutConfig& config) {
    switch (policy) {
        case LayoutPolicy::MATRIX_PARTITIONED:
            return std::make_unique<MatrixPartitionedLayout>(config);
        case LayoutPolicy::ROUND_ROBIN:
            return std::make_unique<RoundRobinLayout>(config);
        case LayoutPolicy::ITERATION_AWARE:
            return std::make_unique<IterationAwareLayout>(config);
        case LayoutPolicy::HARDWARE_INTERLEAVED:
            return std::make_unique<HardwareInterleavedLayout>(config);
        default:
            throw std::invalid_argument("Unknown layout policy");
    }
}

} // namespace sw::kpu::isa

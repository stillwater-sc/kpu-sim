// ============================================================================
// src/noc/noc_adapters.cpp
// NoC Adapter Implementations
// ============================================================================

#include <sw/kpu/models/temporal/noc/noc_adapters.hpp>

#include <stdexcept>

namespace sw::kpu::noc {

// ============================================================================
// WormholeNoCAdapter Implementation
// ============================================================================

WormholeNoCAdapter::WormholeNoCAdapter(const NoCConfig& config)
    : config_(config)
    , noc_([&config]() {
        WormholeNoC::Config wh_config;
        wh_config.rows = config.rows;
        wh_config.cols = config.cols;
        wh_config.dma_on_north_edge = config.dma_on_north_edge;
        wh_config.dma_on_south_edge = config.dma_on_south_edge;
        wh_config.dma_on_east_edge = config.dma_on_east_edge;
        wh_config.dma_on_west_edge = config.dma_on_west_edge;
        return wh_config;
    }())
{
}

InjectResult WormholeNoCAdapter::inject_tile(uint8_t src_router, uint8_t dst_router,
                                              const TileDescriptor& tile, uint64_t cycle) {
    auto result = noc_.inject_tile(src_router, dst_router, tile, cycle);

    // Convert WormholeNoC::InjectResult to our InjectResult
    switch (result) {
        case WormholeNoC::InjectResult::SUCCESS:
            return InjectResult::SUCCESS;
        case WormholeNoC::InjectResult::BUSY:
            return InjectResult::BUSY;
        case WormholeNoC::InjectResult::INVALID:
        default:
            return InjectResult::INVALID;
    }
}

bool WormholeNoCAdapter::can_inject(uint8_t router_id) const {
    return noc_.can_inject(router_id);
}

void WormholeNoCAdapter::step(uint64_t cycle) {
    noc_.step(cycle);
}

void WormholeNoCAdapter::drain(uint64_t& cycle) {
    while (!is_idle()) {
        step(cycle);
        cycle++;

        // Safety limit
        if (cycle > 10000000) {
            throw std::runtime_error("WormholeNoCAdapter::drain() timeout");
        }
    }
}

bool WormholeNoCAdapter::is_idle() const {
    return noc_.is_idle();
}

void WormholeNoCAdapter::set_delivery_callback(uint8_t router_id, DeliveryCallback cb) {
    // Wrap the callback to match WormholeNoC's signature
    noc_.set_delivery_callback(router_id,
        [this, router_id, cb](const TileDescriptor& tile, uint8_t src_router, uint64_t cycle) {
            if (cb) {
                // WormholeNoC doesn't track inject_cycle in its callback,
                // so we pass 0 (caller can track if needed)
                cb(tile, src_router, router_id, 0, cycle);
            }
            if (global_callback_) {
                global_callback_(tile, src_router, router_id, 0, cycle);
            }
        });
}

void WormholeNoCAdapter::set_global_delivery_callback(DeliveryCallback cb) {
    global_callback_ = std::move(cb);

    // If no per-router callbacks set, install a callback on each router
    // to forward to global
    for (uint8_t i = 0; i < num_routers(); ++i) {
        auto existing = noc_.router(i);  // Check if callback exists - we can't easily check this
        // For now, set on all routers
        noc_.set_delivery_callback(i,
            [this, i](const TileDescriptor& tile, uint8_t src_router, uint64_t cycle) {
                if (global_callback_) {
                    global_callback_(tile, src_router, i, 0, cycle);
                }
            });
    }
}

NoCStats WormholeNoCAdapter::stats() const {
    const auto& wh_stats = noc_.stats();
    NoCStats s;
    s.tiles_injected = wh_stats.tiles_injected;
    s.tiles_delivered = wh_stats.tiles_delivered;
    s.total_flits = wh_stats.total_flits;
    s.total_bytes = wh_stats.total_bytes;
    s.total_latency_cycles = wh_stats.total_latency_cycles;
    s.max_latency_cycles = wh_stats.max_latency_cycles;
    return s;
}

void WormholeNoCAdapter::reset_stats() {
    noc_.reset_stats();
}

// ============================================================================
// DataflowNoCAdapter Implementation
// ============================================================================

DataflowNoCAdapter::DataflowNoCAdapter(const NoCConfig& config)
    : config_(config)
    , noc_([&config]() {
        DataflowNoC::Config df_config;
        df_config.rows = config.rows;
        df_config.cols = config.cols;
        df_config.buffer_depth = config.buffer_depth;
        df_config.flit_size = config.flit_size_bytes;
        df_config.link_bandwidth_bytes_per_cycle = config.link_bandwidth_bytes_per_cycle;
        return df_config;
    }())
    , per_router_callbacks_(config.num_routers())
{
    // Set up internal delivery callback to dispatch to per-router callbacks
    noc_.set_delivery_callback(
        [this](uint8_t dst_router, const TileTag& tag, uint64_t inject_cycle, uint64_t complete_cycle) {
            // Create a TileDescriptor from the tag
            TileDescriptor tile;
            tile.tensor = tag.tensor;
            tile.m_tile = tag.m_tile;
            tile.n_tile = tag.n_tile;
            tile.k_tile = tag.k_tile;

            // Look up source router (we track this on injection)
            uint8_t src_router = 0;
            TileTagHash hasher;
            uint64_t hash = hasher(tag);
            auto it = tile_sources_.find(hash);
            if (it != tile_sources_.end()) {
                src_router = it->second;
                tile_sources_.erase(it);
            }

            // Invoke per-router callback
            if (dst_router < per_router_callbacks_.size() && per_router_callbacks_[dst_router]) {
                per_router_callbacks_[dst_router](tile, src_router, dst_router,
                                                   inject_cycle, complete_cycle);
            }

            // Invoke global callback
            if (global_callback_) {
                global_callback_(tile, src_router, dst_router, inject_cycle, complete_cycle);
            }
        });
}

InjectResult DataflowNoCAdapter::inject_tile(uint8_t src_router, uint8_t dst_router,
                                              const TileDescriptor& tile, uint64_t cycle) {
    bool success = noc_.inject_tile(src_router, dst_router, tile, cycle);

    if (success) {
        // Track source for delivery callback
        TileTag tag(tile);
        TileTagHash hasher;
        tile_sources_[hasher(tag)] = src_router;
        return InjectResult::SUCCESS;
    }

    return InjectResult::BUSY;
}

bool DataflowNoCAdapter::can_inject(uint8_t router_id) const {
    if (router_id >= num_routers()) {
        return false;
    }
    return noc_.router(router_id).can_accept(Direction::LOCAL);
}

void DataflowNoCAdapter::step(uint64_t cycle) {
    noc_.step(cycle);
}

void DataflowNoCAdapter::drain(uint64_t& cycle) {
    noc_.drain(cycle);
}

bool DataflowNoCAdapter::is_idle() const {
    return !noc_.has_inflight_tiles() && !noc_.has_pending_flits();
}

void DataflowNoCAdapter::set_delivery_callback(uint8_t router_id, DeliveryCallback cb) {
    if (router_id < per_router_callbacks_.size()) {
        per_router_callbacks_[router_id] = std::move(cb);
    }
}

void DataflowNoCAdapter::set_global_delivery_callback(DeliveryCallback cb) {
    global_callback_ = std::move(cb);
}

NoCStats DataflowNoCAdapter::stats() const {
    const auto& df_stats = noc_.stats();
    NoCStats s;
    s.tiles_injected = df_stats.total_tiles_injected;
    s.tiles_delivered = df_stats.total_tiles_delivered;
    s.total_flits = df_stats.total_flits_transferred;
    s.total_bytes = df_stats.total_flits_transferred * config_.flit_size_bytes;
    s.total_latency_cycles = df_stats.total_latency_cycles;
    s.max_latency_cycles = df_stats.max_latency_cycles;
    return s;
}

void DataflowNoCAdapter::reset_stats() {
    // DataflowNoC doesn't have reset_stats, would need to add
    // For now, this is a no-op
}

// ============================================================================
// Factory Implementation
// ============================================================================

std::unique_ptr<INoC> create_noc(NoCType type, const NoCConfig& config) {
    switch (type) {
        case NoCType::WORMHOLE:
            return std::make_unique<WormholeNoCAdapter>(config);
        case NoCType::DATAFLOW:
            return std::make_unique<DataflowNoCAdapter>(config);
        default:
            throw std::invalid_argument("Unknown NoC type");
    }
}

std::unique_ptr<INoC> create_noc(NoCType type) {
    return create_noc(type, NoCConfig{});
}

} // namespace sw::kpu::noc

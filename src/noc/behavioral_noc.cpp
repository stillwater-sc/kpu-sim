// ============================================================================
// src/noc/behavioral_noc.cpp
// Behavioral (functional) NoC implementation
// ============================================================================

#include <sw/kpu/noc/behavioral_noc.hpp>

#include <cstdlib>

namespace sw::kpu::noc {

BehavioralNoC::BehavioralNoC(const NoCConfig& config)
    : config_(config)
{
}

InjectResult BehavioralNoC::inject_tile(
    uint8_t src_router, uint8_t dst_router,
    const TileDescriptor& tile, uint64_t cycle)
{
    // Validate routers
    if (src_router >= config_.num_routers() ||
        dst_router >= config_.num_routers()) {
        return InjectResult::INVALID;
    }

    // Calculate delivery time
    uint32_t hops = calculate_hop_count(src_router, dst_router);
    uint64_t delivery_cycle = cycle + hops * latency_per_hop_;

    // Queue delivery
    pending_deliveries_.push(PendingDelivery{
        .tile = tile,
        .src_router = src_router,
        .dst_router = dst_router,
        .inject_cycle = cycle,
        .deliver_cycle = delivery_cycle
    });

    // Update statistics
    stats_.tiles_injected++;
    stats_.total_bytes += tile.size;

    return InjectResult::SUCCESS;
}

void BehavioralNoC::step(uint64_t cycle) {
    // Process deliveries that are ready
    while (!pending_deliveries_.empty() &&
           pending_deliveries_.top().deliver_cycle <= cycle) {

        PendingDelivery delivery = pending_deliveries_.top();
        pending_deliveries_.pop();

        // Update statistics
        stats_.tiles_delivered++;
        uint64_t latency = delivery.deliver_cycle - delivery.inject_cycle;
        stats_.total_latency_cycles += latency;
        if (latency > stats_.max_latency_cycles) {
            stats_.max_latency_cycles = latency;
        }

        // Invoke callbacks
        if (global_callback_) {
            global_callback_(delivery.tile,
                           delivery.src_router,
                           delivery.dst_router,
                           delivery.inject_cycle,
                           delivery.deliver_cycle);
        }

        auto it = router_callbacks_.find(delivery.dst_router);
        if (it != router_callbacks_.end() && it->second) {
            it->second(delivery.tile,
                      delivery.src_router,
                      delivery.dst_router,
                      delivery.inject_cycle,
                      delivery.deliver_cycle);
        }
    }
}

void BehavioralNoC::drain(uint64_t& cycle) {
    while (!is_idle()) {
        cycle++;
        step(cycle);
    }
}

void BehavioralNoC::set_delivery_callback(uint8_t router_id, DeliveryCallback cb) {
    router_callbacks_[router_id] = cb;
}

void BehavioralNoC::set_global_delivery_callback(DeliveryCallback cb) {
    global_callback_ = cb;
}

uint32_t BehavioralNoC::calculate_hop_count(uint8_t src, uint8_t dst) const {
    // Manhattan distance in 2D mesh
    uint8_t src_row = src / config_.cols;
    uint8_t src_col = src % config_.cols;
    uint8_t dst_row = dst / config_.cols;
    uint8_t dst_col = dst % config_.cols;

    uint32_t row_hops = std::abs(static_cast<int>(dst_row) - static_cast<int>(src_row));
    uint32_t col_hops = std::abs(static_cast<int>(dst_col) - static_cast<int>(src_col));

    return row_hops + col_hops;
}

} // namespace sw::kpu::noc

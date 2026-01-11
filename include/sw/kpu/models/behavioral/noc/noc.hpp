// ============================================================================
// include/sw/kpu/models/behavioral/noc/noc.hpp
// Behavioral (functional) NoC - instant/fixed latency
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <sw/kpu/models/interfaces/noc_interface.hpp>

#include <map>
#include <queue>
#include <vector>

namespace sw::kpu::noc {

/// Behavioral NoC with instant or fixed latency delivery
///
/// This is the fastest simulation mode, providing:
/// - Functional correctness (tiles are delivered correctly)
/// - Optional fixed hop-count-based latency
/// - No buffer contention, no routing delays
///
/// Use for:
/// - Software bring-up and functional verification
/// - Unit testing
/// - CI/CD pipelines where speed matters more than timing accuracy
class BehavioralNoC : public INoC {
public:
    explicit BehavioralNoC(const NoCConfig& config);
    ~BehavioralNoC() override = default;

    // Disable copying, allow moving
    BehavioralNoC(const BehavioralNoC&) = delete;
    BehavioralNoC& operator=(const BehavioralNoC&) = delete;
    BehavioralNoC(BehavioralNoC&&) = default;
    BehavioralNoC& operator=(BehavioralNoC&&) = default;

    // ========================================================================
    // INoC Interface Implementation
    // ========================================================================

    InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                             const TileDescriptor& tile, uint64_t cycle) override;

    bool can_inject(uint8_t router_id) const override {
        (void)router_id;
        return true;  // Always can inject
    }

    void step(uint64_t cycle) override;
    void drain(uint64_t& cycle) override;
    bool is_idle() const override { return pending_deliveries_.empty(); }

    void set_delivery_callback(uint8_t router_id, DeliveryCallback cb) override;
    void set_global_delivery_callback(DeliveryCallback cb) override;

    const NoCConfig& config() const override { return config_; }
    NoCStats stats() const override { return stats_; }
    void reset_stats() override { stats_ = NoCStats{}; }

    uint8_t num_routers() const override { return config_.num_routers(); }
    NoCType type() const override { return NoCType::WORMHOLE; }  // Behaves as simplified wormhole

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set fixed latency per hop (0 = instant)
    void set_latency_per_hop(uint32_t cycles) { latency_per_hop_ = cycles; }

    /// Get latency per hop
    uint32_t latency_per_hop() const { return latency_per_hop_; }

private:
    NoCConfig config_;
    NoCStats stats_;

    // Fixed latency modeling
    uint32_t latency_per_hop_ = 1;  // 1 cycle per hop by default

    // Pending deliveries
    struct PendingDelivery {
        TileDescriptor tile;
        uint8_t src_router;
        uint8_t dst_router;
        uint64_t inject_cycle;
        uint64_t deliver_cycle;

        bool operator>(const PendingDelivery& other) const {
            return deliver_cycle > other.deliver_cycle;
        }
    };
    std::priority_queue<PendingDelivery,
                       std::vector<PendingDelivery>,
                       std::greater<PendingDelivery>> pending_deliveries_;

    // Callbacks
    std::map<uint8_t, DeliveryCallback> router_callbacks_;
    DeliveryCallback global_callback_;

    // Helpers
    uint32_t calculate_hop_count(uint8_t src, uint8_t dst) const;
};

} // namespace sw::kpu::noc

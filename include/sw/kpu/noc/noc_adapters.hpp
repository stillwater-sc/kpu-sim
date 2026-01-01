// ============================================================================
// include/sw/kpu/noc/noc_adapters.hpp
// NoC Adapters - Make existing NoC implementations conform to INoC interface
// ============================================================================

#pragma once

#include <sw/kpu/noc/noc_interface.hpp>
#include <sw/kpu/noc/wormhole_router.hpp>
#include <sw/kpu/noc/dataflow_noc.hpp>

#include <unordered_map>

namespace sw::kpu::noc {

// ============================================================================
// WormholeNoCAdapter
// ============================================================================

/// Adapter that makes WormholeNoC conform to the INoC interface
class WormholeNoCAdapter : public INoC {
public:
    explicit WormholeNoCAdapter(const NoCConfig& config);

    // === INoC Interface ===

    InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                             const TileDescriptor& tile, uint64_t cycle) override;

    bool can_inject(uint8_t router_id) const override;

    void step(uint64_t cycle) override;

    void drain(uint64_t& cycle) override;

    bool is_idle() const override;

    void set_delivery_callback(uint8_t router_id, DeliveryCallback cb) override;

    void set_global_delivery_callback(DeliveryCallback cb) override;

    const NoCConfig& config() const override { return config_; }

    NoCStats stats() const override;

    void reset_stats() override;

    uint8_t num_routers() const override { return noc_.num_routers(); }

    NoCType type() const override { return NoCType::WORMHOLE; }

    // === Direct access (for advanced usage) ===
    WormholeNoC& underlying() { return noc_; }
    const WormholeNoC& underlying() const { return noc_; }

private:
    NoCConfig config_;
    WormholeNoC noc_;
    DeliveryCallback global_callback_;

    // Track injection info for delivery callbacks
    struct InjectionInfo {
        uint8_t src_router;
        uint8_t dst_router;
        uint64_t inject_cycle;
    };
    std::unordered_map<uint64_t, InjectionInfo> pending_injections_;
    uint64_t next_injection_id_ = 0;
};

// ============================================================================
// DataflowNoCAdapter
// ============================================================================

/// Adapter that makes DataflowNoC conform to the INoC interface
class DataflowNoCAdapter : public INoC {
public:
    explicit DataflowNoCAdapter(const NoCConfig& config);

    // === INoC Interface ===

    InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                             const TileDescriptor& tile, uint64_t cycle) override;

    bool can_inject(uint8_t router_id) const override;

    void step(uint64_t cycle) override;

    void drain(uint64_t& cycle) override;

    bool is_idle() const override;

    void set_delivery_callback(uint8_t router_id, DeliveryCallback cb) override;

    void set_global_delivery_callback(DeliveryCallback cb) override;

    const NoCConfig& config() const override { return config_; }

    NoCStats stats() const override;

    void reset_stats() override;

    uint8_t num_routers() const override { return noc_.num_routers(); }

    NoCType type() const override { return NoCType::DATAFLOW; }

    // === Direct access (for advanced usage) ===
    DataflowNoC& underlying() { return noc_; }
    const DataflowNoC& underlying() const { return noc_; }

private:
    NoCConfig config_;
    DataflowNoC noc_;
    std::vector<DeliveryCallback> per_router_callbacks_;
    DeliveryCallback global_callback_;

    // Track source router for deliveries
    std::unordered_map<uint64_t, uint8_t> tile_sources_;  // TileTag hash -> src_router
};

} // namespace sw::kpu::noc

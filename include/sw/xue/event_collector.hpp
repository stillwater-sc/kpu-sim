/**
 * @file event_collector.hpp
 * @brief XUE Event Collector for Simulation Instrumentation
 * @version 0.3.3
 *
 * The EventCollector is the primary interface for instrumenting
 * simulation code with XUE events. It provides:
 *
 *   - Singleton access for global event collection
 *   - Scoped event recording (RAII)
 *   - Kernel-level and tile-level scoping
 *   - Integration with simulation cycles
 *
 * Usage:
 *   // Get global collector
 *   auto& xue = EventCollector::instance();
 *
 *   // Record events
 *   xue.record_matmul(M, N, K);
 *   xue.record_dram_read(bytes);
 *
 *   // Use scoped recording
 *   {
 *       XUE_KERNEL_SCOPE("matmul_1024x1024");
 *       // ... kernel execution ...
 *   }  // Automatically records KERNEL_END
 */

#pragma once

#include <sw/xue/event_hierarchy.hpp>
#include <sw/xue/event_counter.hpp>
#include <memory>
#include <string>
#include <stack>
#include <chrono>
#include <functional>

namespace sw::xue {

/**
 * @brief Scope identifier for hierarchical event tracking
 */
struct EventScope {
    std::string name;
    uint64_t start_cycle;
    EventType start_event;
    EventType end_event;

    EventScope(std::string n, uint64_t cycle, EventType start, EventType end)
        : name(std::move(n)), start_cycle(cycle), start_event(start), end_event(end) {}
};

/**
 * @brief Central event collector for XUE instrumentation
 */
class EventCollector {
public:
    /**
     * @brief Get the global EventCollector instance
     */
    static EventCollector& instance() {
        static EventCollector instance;
        return instance;
    }

    /**
     * @brief Enable or disable collection
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    /**
     * @brief Set the current simulation cycle
     */
    void set_cycle(uint64_t cycle) { current_cycle_ = cycle; }
    uint64_t get_cycle() const { return current_cycle_; }

    /**
     * @brief Advance the simulation cycle
     */
    void advance_cycle(uint64_t delta = 1) { current_cycle_ += delta; }

    // ========== Low-Level Event Recording ==========

    /**
     * @brief Record a raw event
     */
    void record(EventType type, const EventMetadata& meta = EventMetadata{}) {
        if (!enabled_) return;
        uint64_t cycles = current_cycle_ - last_event_cycle_;
        counter_.record(type, meta, cycles);
        last_event_cycle_ = current_cycle_;
    }

    // ========== Compute Events ==========

    /**
     * @brief Record a matrix multiply operation
     *
     * Records MATMUL_16x16 events based on the number of 16x16 tiles
     * required for the given dimensions.
     */
    void record_matmul(uint32_t M, uint32_t N, uint32_t K, uint64_t cycles = 0) {
        if (!enabled_) return;

        // Calculate number of systolic tile operations
        uint32_t tiles_m = (M + KPUConstants::SYSTOLIC_SIZE - 1) / KPUConstants::SYSTOLIC_SIZE;
        uint32_t tiles_n = (N + KPUConstants::SYSTOLIC_SIZE - 1) / KPUConstants::SYSTOLIC_SIZE;
        uint32_t tiles_k = (K + KPUConstants::SYSTOLIC_SIZE - 1) / KPUConstants::SYSTOLIC_SIZE;

        // Each tile contributes 2*16*16*16 FLOPs (for K=16)
        uint64_t total_tiles = static_cast<uint64_t>(tiles_m) * tiles_n * tiles_k;
        // Note: flops = 2ULL * M * N * K; // Used for validation

        // Record tile-level events
        for (uint64_t t = 0; t < total_tiles; ++t) {
            counter_.record_compute(EventType::MATMUL_16x16, KPUConstants::TILE_FLOPS,
                                   KPUConstants::SYSTOLIC_SIZE,
                                   KPUConstants::SYSTOLIC_SIZE,
                                   KPUConstants::SYSTOLIC_SIZE,
                                   cycles / total_tiles);
        }

        // Record accumulate and writeback
        uint64_t output_tiles = static_cast<uint64_t>(tiles_m) * tiles_n;
        for (uint64_t t = 0; t < output_tiles * (tiles_k - 1); ++t) {
            counter_.record(EventType::MATMUL_ACCUMULATE);
        }
        for (uint64_t t = 0; t < output_tiles; ++t) {
            counter_.record(EventType::MATMUL_WRITEBACK);
        }
    }

    /**
     * @brief Record an elementwise operation
     */
    void record_elementwise(EventType op, uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_compute(op, elements, 0, 0, 0, cycles);
    }

    void record_relu(uint64_t elements, uint64_t cycles = 0) {
        record_elementwise(EventType::ELEM_RELU, elements, cycles);
    }

    void record_bias_add(uint64_t elements, uint64_t cycles = 0) {
        record_elementwise(EventType::ELEM_BIAS, elements, cycles);
    }

    void record_add(uint64_t elements, uint64_t cycles = 0) {
        record_elementwise(EventType::ELEM_ADD, elements, cycles);
    }

    void record_mul(uint64_t elements, uint64_t cycles = 0) {
        record_elementwise(EventType::ELEM_MUL, elements, cycles);
    }

    /**
     * @brief Record a reduction operation
     */
    void record_softmax(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // Softmax = exp + sum + div
        counter_.record_compute(EventType::REDUCE_SOFTMAX, elements * 3, 0, 0, 0, cycles);
    }

    void record_layernorm(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // LayerNorm = mean + variance + normalize
        counter_.record_compute(EventType::REDUCE_LAYERNORM, elements * 4, 0, 0, 0, cycles);
    }

    // ========== Memory Events ==========

    /**
     * @brief Record DRAM read
     */
    void record_dram_read(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::DRAM_READ, bytes, 0, 0, cycles);
    }

    /**
     * @brief Record DRAM write
     */
    void record_dram_write(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::DRAM_WRITE, bytes, 0, 0, cycles);
    }

    /**
     * @brief Record L3 tile push (arrival)
     */
    void record_l3_push(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L3_TILE_PUSH, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record L3 tile pop (consumption)
     */
    void record_l3_pop(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L3_TILE_POP, bytes, buffer_id, 0, cycles);
    }

    /**
     * @brief Record L3 credit return
     */
    void record_l3_credit_return(uint16_t buffer_id = 0) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.source_id = buffer_id;
        counter_.record(EventType::L3_CREDIT_RETURN, meta);
    }

    /**
     * @brief Record L2 tile push
     */
    void record_l2_push(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L2_TILE_PUSH, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record L2 tile pop
     */
    void record_l2_pop(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L2_TILE_POP, bytes, buffer_id, 0, cycles);
    }

    /**
     * @brief Record L1 stream feed
     */
    void record_l1_feed(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L1_STREAM_FEED, bytes, 0, stream_id, cycles);
    }

    // ========== Data Movement Events ==========

    /**
     * @brief Record DMA transfer
     */
    void record_dma_transfer(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::DMA_TRANSFER_START, bytes, 0, 0, cycles);
        counter_.record(EventType::DMA_TRANSFER_COMPLETE);
    }

    /**
     * @brief Record BlockMover push L3->L2
     */
    void record_blockmover_push(uint64_t bytes, uint16_t from_buffer = 0,
                                uint16_t to_buffer = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::BM_PUSH_L3_L2, bytes, from_buffer, to_buffer, cycles);
    }

    /**
     * @brief Record Streamer feed L2->L1
     */
    void record_streamer_feed(uint64_t bytes, uint16_t from_buffer = 0,
                             uint16_t to_stream = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_FEED_L2_L1, bytes, from_buffer, to_stream, cycles);
    }

    // ========== Synchronization Events ==========

    /**
     * @brief Record credit stall (waiting for downstream buffer)
     */
    void record_credit_stall(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;  // Store stall duration in bytes field
        counter_.record(EventType::SYNC_CREDIT_STALL, meta, cycles);
    }

    /**
     * @brief Record data dependency stall
     */
    void record_dependency_stall(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::SYNC_DATA_DEPENDENCY, meta, cycles);
    }

    // ========== Scoped Recording ==========

    /**
     * @brief Begin a kernel scope
     */
    void begin_kernel(const std::string& name) {
        if (!enabled_) return;
        scopes_.emplace(name, current_cycle_,
                       EventType::KERNEL_START, EventType::KERNEL_END);
        counter_.record(EventType::KERNEL_START);
    }

    /**
     * @brief End a kernel scope
     */
    void end_kernel() {
        if (!enabled_ || scopes_.empty()) return;
        auto scope = scopes_.top();
        scopes_.pop();
        counter_.record(EventType::KERNEL_END);
    }

    /**
     * @brief Begin a tile iteration scope
     */
    void begin_tile_iteration() {
        if (!enabled_) return;
        scopes_.emplace("tile", current_cycle_,
                       EventType::TILE_ITERATION_START, EventType::TILE_ITERATION_END);
        counter_.record(EventType::TILE_ITERATION_START);
    }

    /**
     * @brief End a tile iteration scope
     */
    void end_tile_iteration() {
        if (!enabled_ || scopes_.empty()) return;
        auto scope = scopes_.top();
        scopes_.pop();
        counter_.record(EventType::TILE_ITERATION_END);
    }

    // ========== Access to Counters ==========

    /**
     * @brief Get the underlying event counter
     */
    const EventCounter& counters() const { return counter_; }
    EventCounter& counters() { return counter_; }

    /**
     * @brief Get JSON representation
     */
    std::string to_json() const { return counter_.to_json(); }

    /**
     * @brief Get human-readable summary
     */
    std::string summary() const { return counter_.summary(); }

    /**
     * @brief Reset all counters
     */
    void reset() {
        counter_.reset();
        current_cycle_ = 0;
        last_event_cycle_ = 0;
        while (!scopes_.empty()) scopes_.pop();
    }

private:
    EventCollector() = default;
    EventCollector(const EventCollector&) = delete;
    EventCollector& operator=(const EventCollector&) = delete;

    bool enabled_ = true;
    uint64_t current_cycle_ = 0;
    uint64_t last_event_cycle_ = 0;
    EventCounter counter_;
    std::stack<EventScope> scopes_;
};

/**
 * @brief RAII scope guard for kernel recording
 */
class KernelScope {
public:
    explicit KernelScope(const std::string& name) {
        EventCollector::instance().begin_kernel(name);
    }

    ~KernelScope() {
        EventCollector::instance().end_kernel();
    }

    KernelScope(const KernelScope&) = delete;
    KernelScope& operator=(const KernelScope&) = delete;
};

/**
 * @brief RAII scope guard for tile iteration recording
 */
class TileScope {
public:
    TileScope() {
        EventCollector::instance().begin_tile_iteration();
    }

    ~TileScope() {
        EventCollector::instance().end_tile_iteration();
    }

    TileScope(const TileScope&) = delete;
    TileScope& operator=(const TileScope&) = delete;
};

// Convenience macros
#define XUE_KERNEL_SCOPE(name) sw::xue::KernelScope _xue_kernel_scope_(name)
#define XUE_TILE_SCOPE() sw::xue::TileScope _xue_tile_scope_

// Global access
inline EventCollector& xue() { return EventCollector::instance(); }

} // namespace sw::xue

/**
 * @file event_collector.hpp
 * @brief XUE Observation Architecture - Event Collector
 * @version 0.4.0
 *
 * The EventCollector is the primary interface for instrumenting
 * simulation code with XUE events. It provides:
 *
 *   - Singleton access for global event collection
 *   - Scoped event recording (RAII)
 *   - Kernel-level and tile-level scoping
 *   - Integration with simulation cycles
 *   - Hierarchical event recording (ALU primitives + named operations)
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
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
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

    /**
     * @brief Tick the observation cycle (must call each clock cycle for OA)
     *
     * This accumulates occupancy for all event types.
     */
    void tick() {
        if (!enabled_) return;
        counter_.accumulate_all_occupancy();
    }

    // ========================================================================
    // Low-Level Event Recording
    // ========================================================================

    /**
     * @brief Record a raw event
     */
    void record(EventType type, const EventMetadata& meta = EventMetadata{}) {
        if (!enabled_) return;
        uint64_t cycles = current_cycle_ - last_event_cycle_;
        counter_.record(type, meta, cycles);
        last_event_cycle_ = current_cycle_;
    }

    // ========================================================================
    // Compute Events
    // ========================================================================

    /**
     * @brief Record a matrix multiply operation
     *
     * Records both the named operation (OP_MATMUL) and the underlying
     * ALU_MAC primitives for hierarchical analysis.
     */
    void record_matmul(uint32_t M, uint32_t N, uint32_t K, uint64_t cycles = 0) {
        if (!enabled_) return;

        // Total FLOPs and MACs for the matmul
        uint64_t total_macs = static_cast<uint64_t>(M) * N * K;
        uint64_t total_flops = 2 * total_macs;

        // Record the named operation (high-level view)
        // FLOPs are tracked ONLY at the named operation level
        counter_.record_compute(EventType::OP_MATMUL, total_flops, M, N, K, cycles);

        // Record ALU primitive counts (for drill-down analysis)
        // NOTE: FLOPs are NOT set here to avoid double-counting
        // The primitive count enables drill-down to see how many MACs
        EventMetadata mac_meta;
        mac_meta.flops = 0;  // FLOPs tracked at named operation level only
        for (uint64_t i = 0; i < total_macs; ++i) {
            counter_.record(EventType::ALU_MAC, mac_meta, 0);
        }
    }

    /**
     * @brief Record a GEMM operation (C = αAB + βC)
     */
    void record_gemm(uint32_t M, uint32_t N, uint32_t K, uint64_t cycles = 0) {
        if (!enabled_) return;

        uint64_t total_macs = static_cast<uint64_t>(M) * N * K;
        uint64_t scale_ops = static_cast<uint64_t>(M) * N * 2;  // α scaling + β scaling
        uint64_t total_flops = 2 * total_macs + scale_ops;

        // FLOPs tracked at named operation level only
        counter_.record_compute(EventType::OP_GEMM, total_flops, M, N, K, cycles);

        // Record underlying ALU op counts (no FLOPs to avoid double-counting)
        EventMetadata mac_meta;
        mac_meta.flops = 0;
        for (uint64_t i = 0; i < total_macs; ++i) {
            counter_.record(EventType::ALU_MAC, mac_meta, 0);
        }
        EventMetadata mul_meta;
        mul_meta.flops = 0;
        for (uint64_t i = 0; i < scale_ops; ++i) {
            counter_.record(EventType::ALU_MUL, mul_meta, 0);
        }
    }

    /**
     * @brief Record an ALU primitive operation
     */
    void record_alu(EventType op, uint64_t count, uint64_t cycles = 0) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.flops = count;  // 1 FLOP per operation
        counter_.record(op, meta, cycles);
    }

    /**
     * @brief Record ReLU activation
     */
    void record_relu(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // Record named operation (FLOPs tracked here)
        counter_.record_compute(EventType::OP_RELU, elements, 0, 0, 0, cycles);
        // Record ALU primitive counts (no FLOPs to avoid double-counting)
        EventMetadata meta;
        meta.flops = 0;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_RELU, meta, 0);
        }
    }

    /**
     * @brief Record GELU activation
     */
    void record_gelu(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // ~10 operations per element (FLOPs tracked at named operation level)
        counter_.record_compute(EventType::OP_GELU, elements * 10, 0, 0, 0, cycles);
        // Record ALU primitive counts (no FLOPs to avoid double-counting)
        EventMetadata meta;
        meta.flops = 0;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_GELU, meta, 0);
        }
    }

    /**
     * @brief Record sigmoid activation
     */
    void record_sigmoid(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // FLOPs tracked at named operation level (4 ops/element: exp + div + etc.)
        counter_.record_compute(EventType::OP_SIGMOID, elements * 4, 0, 0, 0, cycles);
        // Record ALU primitive counts (no FLOPs to avoid double-counting)
        EventMetadata meta;
        meta.flops = 0;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_SIGMOID, meta, 0);
        }
    }

    /**
     * @brief Record tanh activation
     */
    void record_tanh(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // FLOPs tracked at named operation level (4 ops/element)
        counter_.record_compute(EventType::OP_TANH, elements * 4, 0, 0, 0, cycles);
        // Record ALU primitive counts (no FLOPs to avoid double-counting)
        EventMetadata meta;
        meta.flops = 0;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_TANH, meta, 0);
        }
    }

    /**
     * @brief Record bias addition
     */
    void record_bias_add(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // FLOPs tracked at named operation level
        counter_.record_compute(EventType::OP_BIAS_ADD, elements, 0, 0, 0, cycles);
        // Record ALU primitive counts (no FLOPs to avoid double-counting)
        EventMetadata meta;
        meta.flops = 0;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_ADD, meta, 0);
        }
    }

    /**
     * @brief Record elementwise add
     */
    void record_add(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.flops = 1;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_ADD, meta, cycles / elements);
        }
    }

    /**
     * @brief Record elementwise multiply
     */
    void record_mul(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.flops = 1;
        for (uint64_t i = 0; i < elements; ++i) {
            counter_.record(EventType::ALU_MUL, meta, cycles / elements);
        }
    }

    /**
     * @brief Record softmax operation
     */
    void record_softmax(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // Softmax = exp + sum + div
        counter_.record_compute(EventType::REDUCE_SOFTMAX, elements * 3, 0, 0, 0, cycles);
    }

    /**
     * @brief Record layer normalization
     */
    void record_layernorm(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        // LayerNorm = mean + variance + normalize
        counter_.record_compute(EventType::OP_LAYERNORM, elements * 4, 0, 0, 0, cycles);
    }

    /**
     * @brief Record batch normalization
     */
    void record_batchnorm(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_compute(EventType::OP_BATCHNORM, elements * 4, 0, 0, 0, cycles);
    }

    // ========================================================================
    // Memory Events (DRAM)
    // ========================================================================

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
     * @brief Record DRAM activate (row open)
     */
    void record_dram_activate(uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record(EventType::DRAM_ACTIVATE, EventMetadata{}, cycles);
    }

    /**
     * @brief Record DRAM precharge (row close)
     */
    void record_dram_precharge(uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record(EventType::DRAM_PRECHARGE, EventMetadata{}, cycles);
    }

    /**
     * @brief Record DRAM page hit
     */
    void record_dram_page_hit() {
        if (!enabled_) return;
        counter_.record(EventType::DRAM_PAGE_HIT);
    }

    /**
     * @brief Record DRAM page miss
     */
    void record_dram_page_miss() {
        if (!enabled_) return;
        counter_.record(EventType::DRAM_PAGE_MISS);
    }

    // ========================================================================
    // Memory Events (L3 Tile Store)
    // ========================================================================

    /**
     * @brief Record L3 tile write (tile arrives at L3)
     */
    void record_l3_write(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L3_TILE_WRITE, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record L3 tile read (tile consumed from L3)
     */
    void record_l3_read(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L3_TILE_READ, bytes, buffer_id, 0, cycles);
    }

    // Legacy aliases for backward compatibility
    void record_l3_push(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        record_l3_write(bytes, buffer_id, cycles);
    }
    void record_l3_pop(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        record_l3_read(bytes, buffer_id, cycles);
    }

    // ========================================================================
    // Memory Events (L2 Tile Store)
    // ========================================================================

    /**
     * @brief Record L2 tile write
     */
    void record_l2_write(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L2_TILE_WRITE, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record L2 tile read
     */
    void record_l2_read(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L2_TILE_READ, bytes, buffer_id, 0, cycles);
    }

    // Legacy aliases
    void record_l2_push(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        record_l2_write(bytes, buffer_id, cycles);
    }
    void record_l2_pop(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        record_l2_read(bytes, buffer_id, cycles);
    }

    // ========================================================================
    // Memory Events (L1 Stream Buffers)
    // ========================================================================

    /**
     * @brief Record L1 vector write (vector arrives at L1 buffer)
     */
    void record_l1_vector_write(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L1_VECTOR_WRITE, bytes, 0, stream_id, cycles);
    }

    /**
     * @brief Record L1 vector read (vector consumed from L1 buffer)
     */
    void record_l1_vector_read(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::L1_VECTOR_READ, bytes, stream_id, 0, cycles);
    }

    /**
     * @brief Record elements sent to compute
     */
    void record_l1_to_compute(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_elements(EventType::L1_ELEMENT_TO_COMPUTE, elements, cycles);
    }

    /**
     * @brief Record elements received from compute
     */
    void record_l1_from_compute(uint64_t elements, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_elements(EventType::L1_ELEMENT_FROM_COMPUTE, elements, cycles);
    }

    // Legacy alias
    void record_l1_feed(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        record_l1_vector_write(bytes, stream_id, cycles);
    }

    // ========================================================================
    // Data Movement Events (DMA)
    // ========================================================================

    /**
     * @brief Record DMA read from DRAM
     */
    void record_dma_read_dram(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::DMA_READ_DRAM, bytes, 0, 0, cycles);
    }

    /**
     * @brief Record DMA write to L3
     */
    void record_dma_write_l3(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::DMA_WRITE_L3, bytes, 0, 0, cycles);
    }

    /**
     * @brief Record DMA transfer (read from DRAM, write to L3)
     */
    void record_dma_transfer(uint64_t bytes, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record(EventType::DMA_TRANSFER_START);
        counter_.record_memory(EventType::DMA_READ_DRAM, bytes, 0, 0, cycles / 2);
        counter_.record_memory(EventType::DMA_WRITE_L3, bytes, 0, 0, cycles / 2);
        counter_.record(EventType::DMA_TRANSFER_COMPLETE);
    }

    // ========================================================================
    // Data Movement Events (BlockMover)
    // ========================================================================

    /**
     * @brief Record BlockMover read from L3
     */
    void record_blockmover_read_l3(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::BM_READ_L3, bytes, buffer_id, 0, cycles);
    }

    /**
     * @brief Record BlockMover write to L2
     */
    void record_blockmover_write_l2(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::BM_WRITE_L2, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record BlockMover push L3->L2 (read from L3, write to L2)
     */
    void record_blockmover_push(uint64_t bytes, uint16_t from_buffer = 0,
                                uint16_t to_buffer = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::BM_READ_L3, bytes, from_buffer, 0, cycles / 2);
        counter_.record_memory(EventType::BM_WRITE_L2, bytes, 0, to_buffer, cycles / 2);
    }

    // ========================================================================
    // Data Movement Events (Streamer)
    // ========================================================================

    /**
     * @brief Record Streamer read from L2
     */
    void record_streamer_read_l2(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_READ_L2, bytes, buffer_id, 0, cycles);
    }

    /**
     * @brief Record Streamer write to stream buffer
     */
    void record_streamer_write_buffer(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_WRITE_BUFFER, bytes, 0, stream_id, cycles);
    }

    /**
     * @brief Record Streamer read from stream buffer
     */
    void record_streamer_read_buffer(uint64_t bytes, uint16_t stream_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_READ_BUFFER, bytes, stream_id, 0, cycles);
    }

    /**
     * @brief Record Streamer write to L2
     */
    void record_streamer_write_l2(uint64_t bytes, uint16_t buffer_id = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_WRITE_L2, bytes, 0, buffer_id, cycles);
    }

    /**
     * @brief Record Streamer feed L2->L1 (legacy)
     */
    void record_streamer_feed(uint64_t bytes, uint16_t from_buffer = 0,
                             uint16_t to_stream = 0, uint64_t cycles = 0) {
        if (!enabled_) return;
        counter_.record_memory(EventType::STR_READ_L2, bytes, from_buffer, 0, cycles / 2);
        counter_.record_memory(EventType::STR_WRITE_BUFFER, bytes, 0, to_stream, cycles / 2);
    }

    // ========================================================================
    // Synchronization Events (Credits)
    // ========================================================================

    /**
     * @brief Record L3 credit update
     */
    void record_l3_credit_update() {
        if (!enabled_) return;
        counter_.record(EventType::CREDIT_L3_UPDATE);
    }

    /**
     * @brief Record L3 credit stall
     */
    void record_l3_credit_stall(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::CREDIT_L3_STALL, meta, cycles);
    }

    /**
     * @brief Record L2 credit update
     */
    void record_l2_credit_update() {
        if (!enabled_) return;
        counter_.record(EventType::CREDIT_L2_UPDATE);
    }

    /**
     * @brief Record L2 credit stall
     */
    void record_l2_credit_stall(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::CREDIT_L2_STALL, meta, cycles);
    }

    /**
     * @brief Record L1 credit update
     */
    void record_l1_credit_update() {
        if (!enabled_) return;
        counter_.record(EventType::CREDIT_L1_UPDATE);
    }

    /**
     * @brief Record L1 credit stall
     */
    void record_l1_credit_stall(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::CREDIT_L1_STALL, meta, cycles);
    }

    /**
     * @brief Record credit stall (generic - uses L3 by default)
     */
    void record_credit_stall(uint64_t cycles = 1) {
        record_l3_credit_stall(cycles);
    }

    // Legacy alias
    void record_l3_credit_return(uint16_t buffer_id = 0) {
        (void)buffer_id;
        record_l3_credit_update();
    }

    // ========================================================================
    // Synchronization Events (Dependencies)
    // ========================================================================

    /**
     * @brief Record data dependency wait
     */
    void record_data_dependency_wait(uint64_t cycles = 1) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::DEP_DATA_WAIT, meta, cycles);
    }

    /**
     * @brief Record data dependency ready
     */
    void record_data_dependency_ready() {
        if (!enabled_) return;
        counter_.record(EventType::DEP_DATA_READY);
    }

    // Legacy alias
    void record_dependency_stall(uint64_t cycles = 1) {
        record_data_dependency_wait(cycles);
    }

    // ========================================================================
    // Barrier Events
    // ========================================================================

    /**
     * @brief Record barrier enter
     */
    void record_barrier_enter() {
        if (!enabled_) return;
        counter_.record(EventType::BARRIER_ENTER);
    }

    /**
     * @brief Record barrier exit
     */
    void record_barrier_exit() {
        if (!enabled_) return;
        counter_.record(EventType::BARRIER_EXIT);
    }

    /**
     * @brief Record barrier wait cycles
     */
    void record_barrier_wait(uint64_t cycles) {
        if (!enabled_) return;
        EventMetadata meta;
        meta.bytes = cycles;
        counter_.record(EventType::BARRIER_WAIT_CYCLES, meta, cycles);
    }

    // ========================================================================
    // Scoped Recording
    // ========================================================================

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

    // ========================================================================
    // Access to Counters
    // ========================================================================

    /**
     * @brief Get the underlying event counter
     */
    const EventCounter& counters() const { return counter_; }
    EventCounter& counters() { return counter_; }

    /**
     * @brief Get observation cycles
     */
    uint64_t observation_cycles() const { return counter_.observation_cycles(); }

    /**
     * @brief Calculate XUE metrics
     */
    XUEMetrics calculate_xue_metrics(double clock_ghz = 1.0,
                                      double peak_flops_per_cycle = 512.0,
                                      double peak_bytes_per_cycle = 64.0) const {
        return counter_.calculate_xue_metrics(clock_ghz, peak_flops_per_cycle, peak_bytes_per_cycle);
    }

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

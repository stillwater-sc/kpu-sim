/**
 * @file stats_collector.hpp
 * @brief Central Statistics Collector for KPU Simulator
 * @version 0.3.4
 *
 * Provides a unified interface for collecting and reporting simulation statistics:
 *   - Cycle breakdown (compute, memory, stall, idle)
 *   - Memory traffic (per hierarchy level)
 *   - Resource utilization (per component type)
 *
 * Integrates with the XUE (X, U, E) Observation Architecture for operational
 * analysis and performance validation.
 *
 * Thread-safe for concurrent simulation contexts.
 */

#pragma once

#include <sw/kpu/stats/cycle_breakdown.hpp>
#include <sw/kpu/stats/memory_traffic.hpp>
#include <sw/kpu/stats/utilization.hpp>

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <chrono>

namespace sw::kpu::stats {

/**
 * @brief Configuration for the statistics collector
 */
struct StatsConfig {
    double clock_frequency_ghz = 1.0;   // For bandwidth calculations
    double peak_gflops = 1000.0;        // Peak compute throughput
    double external_bw_gbs = 64.0;      // Peak external bandwidth
    double l3_bw_gbs = 128.0;           // Peak L3 bandwidth
    double l2_bw_gbs = 256.0;           // Peak L2 bandwidth
    double l1_bw_gbs = 512.0;           // Peak L1 bandwidth
    bool enable_detailed_tracking = true;
};

/**
 * @brief Summary of collected statistics
 */
struct StatsSummary {
    // Timing
    uint64_t total_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t stall_cycles = 0;
    uint64_t idle_cycles = 0;
    double wall_time_seconds = 0.0;

    // Compute
    uint64_t total_flops = 0;
    double achieved_gflops = 0.0;
    double compute_efficiency = 0.0;

    // Memory
    uint64_t external_bytes = 0;
    uint64_t total_traffic_bytes = 0;
    double external_bw_gbs = 0.0;
    double traffic_amplification = 0.0;

    // Utilization
    double overall_utilization = 0.0;
    double systolic_utilization = 0.0;
    double memory_controller_utilization = 0.0;

    // Derived metrics
    double arithmetic_intensity = 0.0;  // FLOP/byte
    bool is_compute_bound = false;
    bool is_memory_bound = false;

    std::string to_string() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Statistics Summary\n";
        ss << "==================\n";
        ss << "Timing:\n";
        ss << "  Total Cycles:     " << total_cycles << "\n";
        ss << "  Compute Cycles:   " << compute_cycles
           << " (" << (total_cycles > 0 ? 100.0 * compute_cycles / total_cycles : 0) << "%)\n";
        ss << "  Stall Cycles:     " << stall_cycles
           << " (" << (total_cycles > 0 ? 100.0 * stall_cycles / total_cycles : 0) << "%)\n";
        ss << "  Idle Cycles:      " << idle_cycles
           << " (" << (total_cycles > 0 ? 100.0 * idle_cycles / total_cycles : 0) << "%)\n";
        ss << "\nCompute:\n";
        ss << "  Total FLOPs:      " << total_flops << "\n";
        ss << "  Achieved GFLOPS:  " << achieved_gflops << "\n";
        ss << "  Compute Efficiency: " << (compute_efficiency * 100) << "%\n";
        ss << "\nMemory:\n";
        ss << "  External Traffic: " << (external_bytes / 1e6) << " MB\n";
        ss << "  Total Traffic:    " << (total_traffic_bytes / 1e6) << " MB\n";
        ss << "  External BW:      " << external_bw_gbs << " GB/s\n";
        ss << "  Traffic Amp:      " << traffic_amplification << "x\n";
        ss << "\nUtilization:\n";
        ss << "  Overall:          " << (overall_utilization * 100) << "%\n";
        ss << "  Systolic Array:   " << (systolic_utilization * 100) << "%\n";
        ss << "  Memory Controller:" << (memory_controller_utilization * 100) << "%\n";
        ss << "\nAnalysis:\n";
        ss << "  Arithmetic Intensity: " << arithmetic_intensity << " FLOP/byte\n";
        ss << "  Bottleneck: " << (is_compute_bound ? "COMPUTE" : (is_memory_bound ? "MEMORY" : "BALANCED")) << "\n";
        return ss.str();
    }
};

/**
 * @brief Central statistics collector for KPU simulation
 *
 * Thread-safe collector that aggregates:
 * - Cycle breakdown by category
 * - Memory traffic by hierarchy level
 * - Resource utilization by component
 *
 * Provides JSON/text output and integrates with XUE analysis.
 */
class StatsCollector {
public:
    explicit StatsCollector(const StatsConfig& config = StatsConfig{})
        : config_(config) {
        // Configure memory traffic with peak bandwidths
        memory_.set_peak_bandwidth(MemoryLevel::EXTERNAL, config.external_bw_gbs);
        memory_.set_peak_bandwidth(MemoryLevel::L3, config.l3_bw_gbs);
        memory_.set_peak_bandwidth(MemoryLevel::L2, config.l2_bw_gbs);
        memory_.set_peak_bandwidth(MemoryLevel::L1, config.l1_bw_gbs);
    }

    // Copy constructor
    StatsCollector(const StatsCollector& other)
        : config_(other.config_)
        , cycles_(other.cycles_)
        , memory_(other.memory_)
        , utilization_(other.utilization_)
        , total_flops_(other.total_flops_.load(std::memory_order_relaxed)) {}

    // Move constructor
    StatsCollector(StatsCollector&& other) noexcept
        : config_(std::move(other.config_))
        , cycles_(std::move(other.cycles_))
        , memory_(std::move(other.memory_))
        , utilization_(std::move(other.utilization_))
        , total_flops_(other.total_flops_.load(std::memory_order_relaxed)) {}

    // Copy assignment
    StatsCollector& operator=(const StatsCollector& other) {
        if (this != &other) {
            config_ = other.config_;
            cycles_ = other.cycles_;
            memory_ = other.memory_;
            utilization_ = other.utilization_;
            total_flops_.store(other.total_flops_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // Move assignment
    StatsCollector& operator=(StatsCollector&& other) noexcept {
        if (this != &other) {
            config_ = std::move(other.config_);
            cycles_ = std::move(other.cycles_);
            memory_ = std::move(other.memory_);
            utilization_ = std::move(other.utilization_);
            total_flops_.store(other.total_flops_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // =========================================================================
    // Cycle Recording
    // =========================================================================

    void record_compute_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::COMPUTE, cycles);
    }

    void record_memory_access_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::MEMORY_ACCESS, cycles);
    }

    void record_memory_stall_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::MEMORY_STALL, cycles);
    }

    void record_credit_stall_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::CREDIT_STALL, cycles);
    }

    void record_data_stall_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::DATA_STALL, cycles);
    }

    void record_sync_stall_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::SYNC_STALL, cycles);
    }

    void record_idle_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::IDLE, cycles);
    }

    void record_overhead_cycles(uint64_t cycles) {
        cycles_.record(CycleCategory::OVERHEAD, cycles);
    }

    // =========================================================================
    // Memory Traffic Recording
    // =========================================================================

    void record_external_read(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_read(MemoryLevel::EXTERNAL, bytes, cycles);
    }

    void record_external_write(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_write(MemoryLevel::EXTERNAL, bytes, cycles);
    }

    void record_l3_read(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_read(MemoryLevel::L3, bytes, cycles);
    }

    void record_l3_write(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_write(MemoryLevel::L3, bytes, cycles);
    }

    void record_l2_read(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_read(MemoryLevel::L2, bytes, cycles);
    }

    void record_l2_write(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_write(MemoryLevel::L2, bytes, cycles);
    }

    void record_l1_read(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_read(MemoryLevel::L1, bytes, cycles);
    }

    void record_l1_write(uint64_t bytes, uint64_t cycles = 0) {
        memory_.record_write(MemoryLevel::L1, bytes, cycles);
    }

    // =========================================================================
    // Utilization Recording
    // =========================================================================

    void record_systolic_busy(uint64_t cycles) {
        utilization_.record_busy(ResourceType::SYSTOLIC_ARRAY, 0, cycles);
    }

    void record_systolic_idle(uint64_t cycles) {
        utilization_.record_idle(ResourceType::SYSTOLIC_ARRAY, 0, cycles);
    }

    void record_systolic_stall(uint64_t cycles) {
        utilization_.record_stall(ResourceType::SYSTOLIC_ARRAY, 0, cycles);
    }

    void record_systolic_operation(uint64_t flops = 1) {
        utilization_.record_operation(ResourceType::SYSTOLIC_ARRAY, 0, flops);
        total_flops_.fetch_add(flops, std::memory_order_relaxed);
    }

    void record_vector_engine_busy(uint64_t cycles, uint32_t instance = 0) {
        utilization_.record_busy(ResourceType::VECTOR_ENGINE, instance, cycles);
    }

    void record_memory_controller_busy(uint64_t cycles, uint32_t instance = 0) {
        utilization_.record_busy(ResourceType::MEMORY_CONTROLLER, instance, cycles);
    }

    void record_dma_busy(uint64_t cycles, uint32_t instance = 0) {
        utilization_.record_busy(ResourceType::DMA_ENGINE, instance, cycles);
    }

    void record_block_mover_busy(uint64_t cycles, uint32_t instance = 0) {
        utilization_.record_busy(ResourceType::BLOCK_MOVER, instance, cycles);
    }

    void record_streamer_busy(uint64_t cycles, uint32_t instance = 0) {
        utilization_.record_busy(ResourceType::STREAMER, instance, cycles);
    }

    void record_noc_busy(uint64_t cycles, uint32_t link_id = 0) {
        utilization_.record_busy(ResourceType::NOC_LINK, link_id, cycles);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    const CycleBreakdown& cycles() const { return cycles_; }
    const MemoryTraffic& memory() const { return memory_; }
    const UtilizationTracker& utilization() const { return utilization_; }

    CycleBreakdown& cycles() { return cycles_; }
    MemoryTraffic& memory() { return memory_; }
    UtilizationTracker& utilization() { return utilization_; }

    uint64_t total_flops() const {
        return total_flops_.load(std::memory_order_relaxed);
    }

    const StatsConfig& config() const { return config_; }

    // =========================================================================
    // Analysis
    // =========================================================================

    /**
     * @brief Generate summary of all collected statistics
     */
    StatsSummary summarize() const {
        StatsSummary s;

        // Timing
        s.total_cycles = cycles_.total_cycles();
        s.compute_cycles = cycles_.compute_cycles();
        s.stall_cycles = cycles_.stall_cycles();
        s.idle_cycles = cycles_.get(CycleCategory::IDLE);

        // Compute
        s.total_flops = total_flops_.load();
        s.achieved_gflops = s.total_cycles > 0 ?
            (static_cast<double>(s.total_flops) / s.total_cycles) * config_.clock_frequency_ghz : 0.0;
        s.compute_efficiency = config_.peak_gflops > 0 ?
            s.achieved_gflops / config_.peak_gflops : 0.0;

        // Memory
        s.external_bytes = memory_.external_bytes();
        s.total_traffic_bytes = memory_.total_traffic();
        s.external_bw_gbs = memory_.achieved_bandwidth(
            MemoryLevel::EXTERNAL, config_.clock_frequency_ghz, s.total_cycles);
        s.traffic_amplification = memory_.traffic_amplification();

        // Utilization
        s.overall_utilization = utilization_.overall_utilization();
        s.systolic_utilization = utilization_.average_utilization(ResourceType::SYSTOLIC_ARRAY);
        s.memory_controller_utilization = utilization_.average_utilization(ResourceType::MEMORY_CONTROLLER);

        // Derived
        s.arithmetic_intensity = s.external_bytes > 0 ?
            static_cast<double>(s.total_flops) / s.external_bytes : 0.0;

        // Determine bottleneck using roofline analysis
        double ridge_point = config_.peak_gflops / config_.external_bw_gbs;
        s.is_memory_bound = s.arithmetic_intensity < ridge_point;
        s.is_compute_bound = s.arithmetic_intensity > ridge_point * 1.5;  // Some margin

        return s;
    }

    /**
     * @brief Calculate arithmetic intensity (FLOP/byte)
     */
    double arithmetic_intensity() const {
        uint64_t external = memory_.external_bytes();
        uint64_t flops = total_flops_.load();
        return external > 0 ? static_cast<double>(flops) / external : 0.0;
    }

    /**
     * @brief Check if workload is memory-bound
     */
    bool is_memory_bound() const {
        double ai = arithmetic_intensity();
        double ridge_point = config_.peak_gflops / config_.external_bw_gbs;
        return ai < ridge_point;
    }

    /**
     * @brief Check if workload is compute-bound
     */
    bool is_compute_bound() const {
        double ai = arithmetic_intensity();
        double ridge_point = config_.peak_gflops / config_.external_bw_gbs;
        return ai > ridge_point;
    }

    /**
     * @brief Get ridge point (FLOP/byte where compute = memory bound)
     */
    double ridge_point() const {
        return config_.external_bw_gbs > 0 ?
            config_.peak_gflops / config_.external_bw_gbs : 0.0;
    }

    // =========================================================================
    // Reset
    // =========================================================================

    void reset() {
        cycles_.reset();
        memory_.reset();
        utilization_.reset();
        total_flops_.store(0);
    }

    // =========================================================================
    // Output Formats
    // =========================================================================

    /**
     * @brief Generate JSON representation of all statistics
     */
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);

        auto summary = summarize();

        ss << "{\n";
        ss << "  \"version\": \"0.3.4\",\n";
        ss << "  \"config\": {\n";
        ss << "    \"clock_frequency_ghz\": " << config_.clock_frequency_ghz << ",\n";
        ss << "    \"peak_gflops\": " << config_.peak_gflops << ",\n";
        ss << "    \"external_bw_gbs\": " << config_.external_bw_gbs << ",\n";
        ss << "    \"ridge_point\": " << ridge_point() << "\n";
        ss << "  },\n";

        ss << "  \"summary\": {\n";
        ss << "    \"total_cycles\": " << summary.total_cycles << ",\n";
        ss << "    \"total_flops\": " << summary.total_flops << ",\n";
        ss << "    \"achieved_gflops\": " << summary.achieved_gflops << ",\n";
        ss << "    \"compute_efficiency\": " << summary.compute_efficiency << ",\n";
        ss << "    \"arithmetic_intensity\": " << summary.arithmetic_intensity << ",\n";
        ss << "    \"is_memory_bound\": " << (summary.is_memory_bound ? "true" : "false") << ",\n";
        ss << "    \"is_compute_bound\": " << (summary.is_compute_bound ? "true" : "false") << ",\n";
        ss << "    \"overall_utilization\": " << summary.overall_utilization << "\n";
        ss << "  },\n";

        ss << "  \"cycles\": " << cycles_.to_json() << ",\n";
        ss << "  \"memory\": " << memory_.to_json(config_.clock_frequency_ghz, cycles_.total_cycles()) << ",\n";
        ss << "  \"utilization\": " << utilization_.to_json() << "\n";
        ss << "}\n";

        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary() const {
        std::ostringstream ss;
        ss << "================================================================================\n";
        ss << "                     KPU Simulation Statistics (v0.3.4)\n";
        ss << "================================================================================\n\n";

        ss << summarize().to_string() << "\n";

        ss << "--------------------------------------------------------------------------------\n";
        ss << cycles_.summary() << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << memory_.summary(config_.clock_frequency_ghz, cycles_.total_cycles()) << "\n";
        ss << "--------------------------------------------------------------------------------\n";
        ss << utilization_.summary();
        ss << "================================================================================\n";

        return ss.str();
    }

    /**
     * @brief Generate CSV header for statistics export
     */
    static std::string csv_header() {
        return "total_cycles,compute_cycles,stall_cycles,idle_cycles,"
               "total_flops,achieved_gflops,compute_efficiency,"
               "external_bytes,total_traffic_bytes,arithmetic_intensity,"
               "overall_utilization,systolic_utilization,is_memory_bound\n";
    }

    /**
     * @brief Generate CSV row for statistics export
     */
    std::string to_csv_row() const {
        auto s = summarize();
        std::ostringstream ss;
        ss << s.total_cycles << ","
           << s.compute_cycles << ","
           << s.stall_cycles << ","
           << s.idle_cycles << ","
           << s.total_flops << ","
           << s.achieved_gflops << ","
           << s.compute_efficiency << ","
           << s.external_bytes << ","
           << s.total_traffic_bytes << ","
           << s.arithmetic_intensity << ","
           << s.overall_utilization << ","
           << s.systolic_utilization << ","
           << (s.is_memory_bound ? 1 : 0) << "\n";
        return ss.str();
    }

private:
    StatsConfig config_;
    CycleBreakdown cycles_;
    MemoryTraffic memory_;
    UtilizationTracker utilization_;
    std::atomic<uint64_t> total_flops_{0};
};

/**
 * @brief RAII scope for timing a block of code
 */
class ScopedStatsRecorder {
public:
    ScopedStatsRecorder(StatsCollector& collector, CycleCategory category,
                        uint64_t& cycle_counter)
        : collector_(collector), category_(category),
          cycle_ref_(cycle_counter), start_cycle_(cycle_counter) {}

    ~ScopedStatsRecorder() {
        uint64_t elapsed = cycle_ref_ - start_cycle_;
        collector_.cycles().record(category_, elapsed);
    }

    ScopedStatsRecorder(const ScopedStatsRecorder&) = delete;
    ScopedStatsRecorder& operator=(const ScopedStatsRecorder&) = delete;

private:
    StatsCollector& collector_;
    CycleCategory category_;
    uint64_t& cycle_ref_;
    uint64_t start_cycle_;
};

#define STATS_SCOPE(collector, category, cycle_var) \
    sw::kpu::stats::ScopedStatsRecorder _stats_recorder_##__LINE__(collector, category, cycle_var)

/**
 * @brief Global statistics collector singleton (optional)
 */
class GlobalStats {
public:
    static StatsCollector& instance() {
        static StatsCollector collector;
        return collector;
    }

    static void configure(const StatsConfig& config) {
        std::lock_guard<std::mutex> lock(mutex());
        instance() = StatsCollector(config);
    }

    static void reset() {
        instance().reset();
    }

private:
    static std::mutex& mutex() {
        static std::mutex m;
        return m;
    }
};

} // namespace sw::kpu::stats

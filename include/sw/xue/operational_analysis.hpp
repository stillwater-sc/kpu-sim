/**
 * @file operational_analysis.hpp
 * @brief XUE Operational Analysis for Performance Prediction
 * @version 0.3.3
 *
 * Operational Analysis uses event counts to predict performance
 * without detailed cycle-accurate simulation. This follows the
 * methodology from:
 *
 *   - Roofline model (Williams et al., 2009)
 *   - I/O complexity theory (Hong & Kung, 1981)
 *   - Communication-avoiding algorithms (Demmel et al., 2012)
 *
 * The key insight is that for memory-bound or compute-bound
 * workloads, performance can be accurately predicted from:
 *   - Total FLOPs (from compute events)
 *   - Total data movement (from memory events)
 *   - Hardware characteristics (peak FLOPS, bandwidth)
 *
 * This enables rapid design space exploration without requiring
 * full timing simulation.
 */

#pragma once

#include <sw/xue/event_counter.hpp>
#include <sw/benchmark/benchmark.hpp>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

namespace sw::xue {

/**
 * @brief Hardware model for operational analysis
 *
 * Default values match the KPU simulator at 1 GHz reference clock:
 * - 16x16 systolic array = 256 MACs/cycle = 512 FLOPs/cycle (FMA)
 * - Peak GFLOPS = 512 at 1 GHz reference
 */
struct HardwareModel {
    // Compute capability
    double peak_gflops = 512.0;               // 16x16 systolic @ 1 GHz (256 FMA = 512 FLOP)
    double clock_ghz = 1.0;

    // Memory bandwidth (GB/s)
    double dram_bandwidth_gbs = 64.0;         // External memory
    double l3_bandwidth_gbs = 128.0;
    double l2_bandwidth_gbs = 256.0;
    double l1_bandwidth_gbs = 512.0;

    // Latencies (cycles)
    double dram_latency_cycles = 100.0;
    double l3_latency_cycles = 20.0;
    double l2_latency_cycles = 10.0;
    double l1_latency_cycles = 2.0;

    // Derived metrics
    double ridge_point_dram() const {
        return peak_gflops / dram_bandwidth_gbs;
    }

    double ridge_point_l3() const {
        return peak_gflops / l3_bandwidth_gbs;
    }

    double ridge_point_l2() const {
        return peak_gflops / l2_bandwidth_gbs;
    }

    /**
     * @brief Predict GFLOPS based on arithmetic intensity
     */
    double predict_gflops(double arithmetic_intensity) const {
        // Roofline model: min(peak, AI * bandwidth)
        double mem_limited = arithmetic_intensity * dram_bandwidth_gbs;
        return std::min(peak_gflops, mem_limited);
    }

    /**
     * @brief Determine bottleneck level
     */
    std::string bottleneck(double arithmetic_intensity) const {
        if (arithmetic_intensity >= ridge_point_dram()) return "compute";
        if (arithmetic_intensity >= ridge_point_l3()) return "dram";
        if (arithmetic_intensity >= ridge_point_l2()) return "l3";
        return "l2";
    }
};

/**
 * @brief Results of operational analysis
 */
struct OperationalResult {
    // Input metrics
    uint64_t total_flops = 0;
    uint64_t dram_bytes = 0;
    uint64_t l3_bytes = 0;
    uint64_t l2_bytes = 0;
    uint64_t l1_bytes = 0;

    // Derived metrics
    double arithmetic_intensity = 0.0;      // FLOP/byte (DRAM)
    double l3_arithmetic_intensity = 0.0;   // FLOP/byte (L3)

    // Predictions
    double predicted_gflops = 0.0;
    double predicted_cycles = 0.0;
    double predicted_runtime_us = 0.0;
    std::string predicted_bottleneck;

    // Efficiency
    double roofline_efficiency = 0.0;       // Achieved / Roofline prediction

    // Event breakdown
    uint64_t matmul_events = 0;
    uint64_t elementwise_events = 0;
    uint64_t reduction_events = 0;
    uint64_t memory_events = 0;
    uint64_t sync_events = 0;
};

/**
 * @brief Operational analysis engine
 *
 * Takes event counts and produces performance predictions
 * using the roofline model and I/O complexity theory.
 */
class OperationalAnalyzer {
public:
    explicit OperationalAnalyzer(const HardwareModel& hw = HardwareModel{})
        : hw_(hw) {}

    /**
     * @brief Analyze event counts and produce predictions
     */
    OperationalResult analyze(const EventCounter& events) const {
        OperationalResult result;

        // Extract metrics from events
        result.total_flops = events.total_flops();
        result.dram_bytes = events.dram_bytes();

        // Get L3/L2/L1 traffic
        auto l3_stats = events.get_memory_subcategory_stats(MemorySubcategory::L3);
        auto l2_stats = events.get_memory_subcategory_stats(MemorySubcategory::L2);
        auto l1_stats = events.get_memory_subcategory_stats(MemorySubcategory::L1);

        result.l3_bytes = l3_stats.total_bytes;
        result.l2_bytes = l2_stats.total_bytes;
        result.l1_bytes = l1_stats.total_bytes;

        // Calculate arithmetic intensities
        if (result.dram_bytes > 0) {
            result.arithmetic_intensity =
                static_cast<double>(result.total_flops) / result.dram_bytes;
        }
        if (result.l3_bytes > 0) {
            result.l3_arithmetic_intensity =
                static_cast<double>(result.total_flops) / result.l3_bytes;
        }

        // Roofline prediction
        result.predicted_gflops = hw_.predict_gflops(result.arithmetic_intensity);
        result.predicted_bottleneck = hw_.bottleneck(result.arithmetic_intensity);

        // Predict execution time
        if (result.predicted_gflops > 0) {
            double seconds = result.total_flops / (result.predicted_gflops * 1e9);
            result.predicted_runtime_us = seconds * 1e6;
            result.predicted_cycles = seconds * hw_.clock_ghz * 1e9;
        }

        // Event breakdown
        auto compute_stats = events.get_category_stats(EventCategory::COMPUTE);
        auto matmul_stats = events.get_compute_subcategory_stats(ComputeSubcategory::MATMUL);
        auto elem_stats = events.get_compute_subcategory_stats(ComputeSubcategory::ELEMENTWISE);
        auto reduce_stats = events.get_compute_subcategory_stats(ComputeSubcategory::REDUCTION);
        auto mem_stats = events.get_category_stats(EventCategory::MEMORY);
        auto sync_stats = events.get_category_stats(EventCategory::SYNCHRONIZATION);

        result.matmul_events = matmul_stats.total_events;
        result.elementwise_events = elem_stats.total_events;
        result.reduction_events = reduce_stats.total_events;
        result.memory_events = mem_stats.total_events;
        result.sync_events = sync_stats.total_events;

        return result;
    }

    /**
     * @brief Analyze and compare with actual simulation result
     */
    struct ValidationResult {
        OperationalResult prediction;
        double actual_gflops = 0.0;
        double actual_cycles = 0.0;
        double gflops_error_percent = 0.0;
        double cycles_error_percent = 0.0;
        bool within_10_percent = false;
    };

    ValidationResult validate(const EventCounter& events,
                             double actual_gflops,
                             uint64_t actual_cycles) const {
        ValidationResult result;
        result.prediction = analyze(events);
        result.actual_gflops = actual_gflops;
        result.actual_cycles = actual_cycles;

        // Calculate errors
        if (actual_gflops > 0) {
            result.gflops_error_percent = 100.0 *
                std::abs(result.prediction.predicted_gflops - actual_gflops) / actual_gflops;
        }
        if (actual_cycles > 0) {
            result.cycles_error_percent = 100.0 *
                std::abs(result.prediction.predicted_cycles - actual_cycles) / actual_cycles;
        }

        result.within_10_percent =
            result.gflops_error_percent <= 10.0 && result.cycles_error_percent <= 10.0;

        // Update roofline efficiency
        if (result.prediction.predicted_gflops > 0) {
            result.prediction.roofline_efficiency =
                actual_gflops / result.prediction.predicted_gflops;
        }

        return result;
    }

    /**
     * @brief Generate JSON report
     */
    std::string to_json(const OperationalResult& result) const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"version\": \"0.3.3\",\n";
        ss << "  \"hardware\": {\n";
        ss << "    \"peak_gflops\": " << hw_.peak_gflops << ",\n";
        ss << "    \"dram_bandwidth_gbs\": " << hw_.dram_bandwidth_gbs << ",\n";
        ss << "    \"ridge_point_dram\": " << hw_.ridge_point_dram() << ",\n";
        ss << "    \"ridge_point_l3\": " << hw_.ridge_point_l3() << ",\n";
        ss << "    \"ridge_point_l2\": " << hw_.ridge_point_l2() << "\n";
        ss << "  },\n";
        ss << "  \"workload\": {\n";
        ss << "    \"total_flops\": " << result.total_flops << ",\n";
        ss << "    \"dram_bytes\": " << result.dram_bytes << ",\n";
        ss << "    \"l3_bytes\": " << result.l3_bytes << ",\n";
        ss << "    \"l2_bytes\": " << result.l2_bytes << ",\n";
        ss << "    \"l1_bytes\": " << result.l1_bytes << ",\n";
        ss << "    \"arithmetic_intensity\": " << result.arithmetic_intensity << "\n";
        ss << "  },\n";
        ss << "  \"prediction\": {\n";
        ss << "    \"gflops\": " << result.predicted_gflops << ",\n";
        ss << "    \"cycles\": " << result.predicted_cycles << ",\n";
        ss << "    \"runtime_us\": " << result.predicted_runtime_us << ",\n";
        ss << "    \"bottleneck\": \"" << result.predicted_bottleneck << "\"\n";
        ss << "  },\n";
        ss << "  \"events\": {\n";
        ss << "    \"matmul\": " << result.matmul_events << ",\n";
        ss << "    \"elementwise\": " << result.elementwise_events << ",\n";
        ss << "    \"reduction\": " << result.reduction_events << ",\n";
        ss << "    \"memory\": " << result.memory_events << ",\n";
        ss << "    \"sync\": " << result.sync_events << "\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    /**
     * @brief Generate validation report JSON
     */
    std::string to_json(const ValidationResult& result) const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"version\": \"0.3.3\",\n";
        ss << "  \"prediction\": " << to_json(result.prediction);
        // Remove closing brace and newline from prediction JSON
        std::string pred_json = to_json(result.prediction);
        pred_json = pred_json.substr(0, pred_json.size() - 2);
        ss.str("");  // Clear stream
        ss << "{\n";
        ss << "  \"version\": \"0.3.3\",\n";
        ss << "  \"actual\": {\n";
        ss << "    \"gflops\": " << result.actual_gflops << ",\n";
        ss << "    \"cycles\": " << result.actual_cycles << "\n";
        ss << "  },\n";
        ss << "  \"predicted\": {\n";
        ss << "    \"gflops\": " << result.prediction.predicted_gflops << ",\n";
        ss << "    \"cycles\": " << result.prediction.predicted_cycles << "\n";
        ss << "  },\n";
        ss << "  \"error\": {\n";
        ss << "    \"gflops_percent\": " << result.gflops_error_percent << ",\n";
        ss << "    \"cycles_percent\": " << result.cycles_error_percent << "\n";
        ss << "  },\n";
        ss << "  \"roofline_efficiency\": " << result.prediction.roofline_efficiency << ",\n";
        ss << "  \"within_10_percent\": " << (result.within_10_percent ? "true" : "false") << ",\n";
        ss << "  \"bottleneck\": \"" << result.prediction.predicted_bottleneck << "\",\n";
        ss << "  \"arithmetic_intensity\": " << result.prediction.arithmetic_intensity << "\n";
        ss << "}\n";
        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary(const OperationalResult& result) const {
        std::ostringstream ss;
        ss << "=== XUE Operational Analysis (v0.3.3) ===\n\n";

        ss << "Workload Characteristics:\n";
        ss << "  Total FLOPs:           " << result.total_flops << "\n";
        ss << "  DRAM Traffic:          " << result.dram_bytes << " bytes\n";
        ss << "  Arithmetic Intensity:  " << result.arithmetic_intensity << " FLOP/byte\n\n";

        ss << "Performance Prediction (Roofline Model):\n";
        ss << "  Predicted GFLOPS:      " << result.predicted_gflops << "\n";
        ss << "  Predicted Cycles:      " << result.predicted_cycles << "\n";
        ss << "  Predicted Runtime:     " << result.predicted_runtime_us << " us\n";
        ss << "  Bottleneck:            " << result.predicted_bottleneck << "\n\n";

        ss << "Event Breakdown:\n";
        ss << "  Matmul Operations:     " << result.matmul_events << "\n";
        ss << "  Elementwise Ops:       " << result.elementwise_events << "\n";
        ss << "  Reduction Ops:         " << result.reduction_events << "\n";
        ss << "  Memory Events:         " << result.memory_events << "\n";
        ss << "  Sync Events:           " << result.sync_events << "\n";

        return ss.str();
    }

    /**
     * @brief Generate validation summary
     */
    std::string summary(const ValidationResult& result) const {
        std::ostringstream ss;
        ss << summary(result.prediction);

        ss << "\nValidation Against Simulation:\n";
        ss << "  Actual GFLOPS:         " << result.actual_gflops << "\n";
        ss << "  Actual Cycles:         " << result.actual_cycles << "\n";
        ss << "  GFLOPS Error:          " << result.gflops_error_percent << "%\n";
        ss << "  Cycles Error:          " << result.cycles_error_percent << "%\n";
        ss << "  Roofline Efficiency:   " << (result.prediction.roofline_efficiency * 100) << "%\n";
        ss << "  Within 10% Target:     " << (result.within_10_percent ? "YES" : "NO") << "\n";

        return ss.str();
    }

    const HardwareModel& hardware() const { return hw_; }

private:
    HardwareModel hw_;
};

/**
 * @brief I/O Complexity Analysis
 *
 * Implements Hong-Kung I/O complexity theory for
 * analyzing memory access lower bounds.
 */
class IOComplexityAnalyzer {
public:
    /**
     * @brief Calculate Hong-Kung I/O lower bound for matrix multiply
     *
     * For C = A * B where A is MxK, B is KxN:
     *   Q >= Ω(MNK / √M_fast)
     *
     * where M_fast is the fast memory (cache/scratchpad) size.
     */
    static uint64_t matmul_io_lower_bound(uint64_t M, uint64_t N, uint64_t K,
                                          uint64_t fast_memory_bytes,
                                          size_t element_size = 4) {
        // M_fast in terms of matrix elements
        uint64_t M_fast = fast_memory_bytes / element_size;

        // Q >= MNK / √M_fast
        double sqrt_M = std::sqrt(static_cast<double>(M_fast));
        uint64_t MNK = M * N * K;

        return static_cast<uint64_t>(std::ceil(MNK / sqrt_M));
    }

    /**
     * @brief Calculate communication-optimal tile size
     *
     * For a fast memory of size M_fast (in elements), the optimal
     * tile size is approximately √(M_fast / 3) for square tiles.
     */
    static uint64_t optimal_tile_size(uint64_t fast_memory_bytes,
                                      size_t element_size = 4) {
        uint64_t M_fast = fast_memory_bytes / element_size;
        // Need to fit 3 tiles: A, B, C
        return static_cast<uint64_t>(std::sqrt(M_fast / 3.0));
    }

    /**
     * @brief Calculate reuse factor achieved vs theoretical minimum
     *
     * Reuse factor = (Total FLOPs) / (Actual I/O)
     * Optimal reuse = √M_fast for matmul
     */
    static double reuse_factor(uint64_t total_flops, uint64_t actual_io_bytes,
                              size_t element_size = 4) {
        if (actual_io_bytes == 0) return 0.0;
        uint64_t io_elements = actual_io_bytes / element_size;
        return static_cast<double>(total_flops) / io_elements;
    }

    /**
     * @brief Calculate efficiency vs Hong-Kung optimal
     *
     * Returns actual_reuse / optimal_reuse
     */
    static double io_efficiency(uint64_t total_flops, uint64_t actual_io_bytes,
                               uint64_t fast_memory_bytes,
                               size_t element_size = 4) {
        uint64_t M_fast = fast_memory_bytes / element_size;
        double optimal_reuse = std::sqrt(static_cast<double>(M_fast));
        double actual_reuse = reuse_factor(total_flops, actual_io_bytes, element_size);
        return actual_reuse / optimal_reuse;
    }
};

} // namespace sw::xue

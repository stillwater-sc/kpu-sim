// ============================================================================
// include/sw/kpu/harness/schedule_validator.hpp
// Validation utilities for data movement schedules
//
// Provides both static (pre-execution) and runtime validation:
// - Schedule correctness checking
// - Resource availability verification
// - Invariant checking (credit flow, ordering)
// - Data integrity validation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/harness_config.hpp>
#include <sw/kpu/harness/pipeline_harness.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <functional>

namespace sw::kpu::harness {

// ============================================================================
// Schedule Validation Result
// ============================================================================

/// Detailed validation result with categorized issues
struct ScheduleValidationResult {
    bool passed = true;

    // Errors by category
    std::vector<std::string> resource_errors;      // Buffer/bank availability
    std::vector<std::string> dependency_errors;    // Missing dependencies
    std::vector<std::string> ordering_errors;      // Incorrect operation order
    std::vector<std::string> invariant_errors;     // Credit/protocol violations

    // Warnings by category
    std::vector<std::string> efficiency_warnings;  // Potential inefficiencies
    std::vector<std::string> resource_warnings;    // Resource usage issues

    /// Add a resource error
    void add_resource_error(const std::string& msg) {
        passed = false;
        resource_errors.push_back(msg);
    }

    /// Add a dependency error
    void add_dependency_error(const std::string& msg) {
        passed = false;
        dependency_errors.push_back(msg);
    }

    /// Add an ordering error
    void add_ordering_error(const std::string& msg) {
        passed = false;
        ordering_errors.push_back(msg);
    }

    /// Add an invariant error
    void add_invariant_error(const std::string& msg) {
        passed = false;
        invariant_errors.push_back(msg);
    }

    /// Add efficiency warning
    void add_efficiency_warning(const std::string& msg) {
        efficiency_warnings.push_back(msg);
    }

    /// Add resource warning
    void add_resource_warning(const std::string& msg) {
        resource_warnings.push_back(msg);
    }

    /// Get total error count
    size_t error_count() const {
        return resource_errors.size() + dependency_errors.size() +
               ordering_errors.size() + invariant_errors.size();
    }

    /// Get total warning count
    size_t warning_count() const {
        return efficiency_warnings.size() + resource_warnings.size();
    }

    /// Check if validation passed
    explicit operator bool() const { return passed; }

    /// Generate summary string
    std::string summary() const;

    /// Print detailed results
    void print(std::ostream& os = std::cerr) const;
};

// ============================================================================
// Schedule Validator
// ============================================================================

/// Validates data movement schedules before and during execution
class ScheduleValidator {
public:
    /// Configuration for validation
    struct Config {
        uint32_t l3_buffer_count = 8;
        uint32_t l2_bank_count = 16;
        uint32_t l1_depth = 4;
        bool strict_ordering = true;
        bool check_efficiency = true;
    };

    /// Default constructor with default config
    ScheduleValidator() : config_{} {}

    /// Constructor with custom config
    explicit ScheduleValidator(const Config& config)
        : config_(config) {}

    // ========================================================================
    // Static Validation (Before Execution)
    // ========================================================================

    /// Validate schedule before execution
    ScheduleValidationResult validate_schedule(const PipelineSchedule& schedule) const;

    /// Check resource requirements
    ScheduleValidationResult check_resource_requirements(
        const PipelineSchedule& schedule) const;

    /// Check dependency graph
    ScheduleValidationResult check_dependencies(
        const PipelineSchedule& schedule) const;

    /// Check operation ordering
    ScheduleValidationResult check_ordering(
        const PipelineSchedule& schedule) const;

    // ========================================================================
    // Runtime Invariant Checking
    // ========================================================================

    /// Check credit invariant (credits never negative)
    bool check_credit_invariant(HarnessComponent comp, int32_t credits) const {
        return credits >= 0;
    }

    /// Check ordering invariant (tile must visit stages in order)
    /// L3 before L2, L2 before L1
    bool check_ordering_invariant(const TileID& tile,
                                   HarnessComponent current_stage,
                                   const std::set<HarnessComponent>& visited_stages) const;

    /// Check data integrity
    bool check_data_integrity(const TileID& tile,
                               const void* expected, const void* actual,
                               size_t size, double tolerance = 1e-5) const;

    // ========================================================================
    // Post-Execution Validation
    // ========================================================================

    /// Validate output against expected data
    ValidationResult validate_output(
        const void* actual, const void* expected,
        size_t size, double tolerance = 1e-5) const;

    /// Validate tile journey completeness
    ValidationResult validate_journeys(
        const std::vector<TileJourney>& journeys) const;

private:
    Config config_;

    /// Build dependency graph from schedule
    std::unordered_map<TileID, std::set<TileID>, TileIDHash>
    build_dependency_graph(const PipelineSchedule& schedule) const;

    /// Check for dependency cycles
    bool has_cycles(const std::unordered_map<TileID, std::set<TileID>, TileIDHash>& graph) const;
};

// ============================================================================
// ScheduleValidationResult Implementation
// ============================================================================

inline std::string ScheduleValidationResult::summary() const {
    std::ostringstream ss;
    if (passed) {
        ss << "PASSED";
    } else {
        ss << "FAILED";
    }
    ss << " (" << error_count() << " errors, " << warning_count() << " warnings)";
    return ss.str();
}

inline void ScheduleValidationResult::print(std::ostream& os) const {
    os << "Schedule Validation " << summary() << "\n";
    os << std::string(60, '=') << "\n";

    if (!resource_errors.empty()) {
        os << "\nResource Errors:\n";
        for (const auto& err : resource_errors) {
            os << "  [E] " << err << "\n";
        }
    }

    if (!dependency_errors.empty()) {
        os << "\nDependency Errors:\n";
        for (const auto& err : dependency_errors) {
            os << "  [E] " << err << "\n";
        }
    }

    if (!ordering_errors.empty()) {
        os << "\nOrdering Errors:\n";
        for (const auto& err : ordering_errors) {
            os << "  [E] " << err << "\n";
        }
    }

    if (!invariant_errors.empty()) {
        os << "\nInvariant Errors:\n";
        for (const auto& err : invariant_errors) {
            os << "  [E] " << err << "\n";
        }
    }

    if (!efficiency_warnings.empty()) {
        os << "\nEfficiency Warnings:\n";
        for (const auto& warn : efficiency_warnings) {
            os << "  [W] " << warn << "\n";
        }
    }

    if (!resource_warnings.empty()) {
        os << "\nResource Warnings:\n";
        for (const auto& warn : resource_warnings) {
            os << "  [W] " << warn << "\n";
        }
    }
}

// ============================================================================
// ScheduleValidator Implementation
// ============================================================================

inline ScheduleValidationResult ScheduleValidator::validate_schedule(
    const PipelineSchedule& schedule) const
{
    ScheduleValidationResult result;

    // Check resource requirements
    auto resource_result = check_resource_requirements(schedule);
    if (!resource_result.passed) {
        result.resource_errors.insert(result.resource_errors.end(),
                                       resource_result.resource_errors.begin(),
                                       resource_result.resource_errors.end());
        result.passed = false;
    }

    // Check dependencies
    auto dep_result = check_dependencies(schedule);
    if (!dep_result.passed) {
        result.dependency_errors.insert(result.dependency_errors.end(),
                                         dep_result.dependency_errors.begin(),
                                         dep_result.dependency_errors.end());
        result.passed = false;
    }

    // Check ordering
    if (config_.strict_ordering) {
        auto order_result = check_ordering(schedule);
        if (!order_result.passed) {
            result.ordering_errors.insert(result.ordering_errors.end(),
                                           order_result.ordering_errors.begin(),
                                           order_result.ordering_errors.end());
            result.passed = false;
        }
    }

    return result;
}

inline ScheduleValidationResult ScheduleValidator::check_resource_requirements(
    const PipelineSchedule& schedule) const
{
    ScheduleValidationResult result;

    // Track peak resource usage
    uint32_t max_l3_buffers = 0;
    uint32_t max_l2_banks = 0;
    uint32_t current_l3 = 0;
    uint32_t current_l2 = 0;

    std::set<TileID> in_l3;
    std::set<TileID> in_l2;

    for (const auto& op : schedule.operations) {
        switch (op.type) {
            case ScheduleOperation::Type::DMA_LOAD:
                in_l3.insert(op.tile_id);
                current_l3 = static_cast<uint32_t>(in_l3.size());
                max_l3_buffers = std::max(max_l3_buffers, current_l3);
                break;

            case ScheduleOperation::Type::BM_L3_TO_L2:
                in_l3.erase(op.tile_id);
                in_l2.insert(op.tile_id);
                current_l2 = static_cast<uint32_t>(in_l2.size());
                max_l2_banks = std::max(max_l2_banks, current_l2);
                break;

            case ScheduleOperation::Type::STREAM_ROWS:
            case ScheduleOperation::Type::STREAM_COLS:
                // L2 data still needed
                break;

            case ScheduleOperation::Type::DRAIN:
                // Output tile takes L2 bank
                current_l2++;
                max_l2_banks = std::max(max_l2_banks, current_l2);
                break;

            default:
                break;
        }
    }

    if (max_l3_buffers > config_.l3_buffer_count) {
        result.add_resource_error("Schedule requires " +
                                   std::to_string(max_l3_buffers) +
                                   " L3 buffers, but only " +
                                   std::to_string(config_.l3_buffer_count) +
                                   " available");
    }

    if (max_l2_banks > config_.l2_bank_count) {
        result.add_resource_error("Schedule requires " +
                                   std::to_string(max_l2_banks) +
                                   " L2 banks, but only " +
                                   std::to_string(config_.l2_bank_count) +
                                   " available");
    }

    return result;
}

inline ScheduleValidationResult ScheduleValidator::check_dependencies(
    const PipelineSchedule& schedule) const
{
    ScheduleValidationResult result;

    auto graph = build_dependency_graph(schedule);

    // Check for cycles
    if (has_cycles(graph)) {
        result.add_dependency_error("Dependency graph contains cycles");
    }

    // Check all dependencies are satisfied
    std::set<TileID> completed;
    for (const auto& op : schedule.operations) {
        for (const auto& dep : op.dependencies) {
            if (completed.find(dep) == completed.end()) {
                result.add_dependency_error(
                    "Tile " + op.tile_id.to_string() +
                    " depends on uncompleted tile " + dep.to_string());
            }
        }
        completed.insert(op.tile_id);
    }

    return result;
}

inline ScheduleValidationResult ScheduleValidator::check_ordering(
    const PipelineSchedule& schedule) const
{
    ScheduleValidationResult result;

    // Track which tiles have been loaded, moved, streamed
    std::set<TileID> dma_loaded;
    std::set<TileID> bm_moved;
    std::set<TileID> streamed;

    for (const auto& op : schedule.operations) {
        switch (op.type) {
            case ScheduleOperation::Type::DMA_LOAD:
                dma_loaded.insert(op.tile_id);
                break;

            case ScheduleOperation::Type::BM_L3_TO_L2:
                if (dma_loaded.find(op.tile_id) == dma_loaded.end()) {
                    result.add_ordering_error(
                        "BlockMover for tile " + op.tile_id.to_string() +
                        " before DMA load");
                }
                bm_moved.insert(op.tile_id);
                break;

            case ScheduleOperation::Type::STREAM_ROWS:
            case ScheduleOperation::Type::STREAM_COLS:
                if (bm_moved.find(op.tile_id) == bm_moved.end()) {
                    result.add_ordering_error(
                        "Streamer for tile " + op.tile_id.to_string() +
                        " before BlockMover");
                }
                streamed.insert(op.tile_id);
                break;

            default:
                break;
        }
    }

    return result;
}

inline bool ScheduleValidator::check_ordering_invariant(
    const TileID& tile,
    HarnessComponent current_stage,
    const std::set<HarnessComponent>& visited_stages) const
{
    // Define valid stage orderings
    auto stage_order = [](HarnessComponent c) -> int {
        switch (c) {
            case HarnessComponent::MEMORY_CONTROLLER: return 0;
            case HarnessComponent::L3_BUFFER: return 1;
            case HarnessComponent::L2_BANK: return 2;
            case HarnessComponent::L1_BUFFER: return 3;
            case HarnessComponent::COMPUTE_FABRIC: return 4;
            default: return -1;
        }
    };

    int current_order = stage_order(current_stage);
    if (current_order < 0) return true;

    for (const auto& visited : visited_stages) {
        int visited_order = stage_order(visited);
        if (visited_order > current_order) {
            // Already visited a later stage - invalid
            return false;
        }
    }

    return true;
}

inline bool ScheduleValidator::check_data_integrity(
    const TileID& tile,
    const void* expected, const void* actual,
    size_t size, double tolerance) const
{
    (void)tile;  // For future logging

    const float* exp = static_cast<const float*>(expected);
    const float* act = static_cast<const float*>(actual);
    size_t count = size / sizeof(float);

    for (size_t i = 0; i < count; ++i) {
        if (std::abs(exp[i] - act[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

inline ValidationResult ScheduleValidator::validate_output(
    const void* actual, const void* expected,
    size_t size, double tolerance) const
{
    ValidationResult result;

    const float* exp = static_cast<const float*>(expected);
    const float* act = static_cast<const float*>(actual);
    size_t count = size / sizeof(float);

    size_t mismatches = 0;
    double max_diff = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double diff = std::abs(exp[i] - act[i]);
        if (diff > tolerance) {
            mismatches++;
            max_diff = std::max(max_diff, diff);
        }
    }

    if (mismatches > 0) {
        result.add_error("Data mismatch: " + std::to_string(mismatches) +
                         "/" + std::to_string(count) +
                         " elements differ (max diff: " +
                         std::to_string(max_diff) + ")");
    }

    return result;
}

inline ValidationResult ScheduleValidator::validate_journeys(
    const std::vector<TileJourney>& journeys) const
{
    ValidationResult result;

    for (const auto& j : journeys) {
        // Check complete journeys have all stages
        if (j.is_complete()) {
            if (j.l3_arrival == 0) {
                result.add_warning("Tile " + j.tile_id.to_string() +
                                    " completed without L3 arrival recorded");
            }
            if (j.l2_arrival == 0) {
                result.add_warning("Tile " + j.tile_id.to_string() +
                                    " completed without L2 arrival recorded");
            }
        }

        // Check timing consistency
        if (j.l3_arrival > 0 && j.dram_fetch_start > 0 &&
            j.l3_arrival < j.dram_fetch_start) {
            result.add_error("Tile " + j.tile_id.to_string() +
                              " arrived at L3 before DRAM fetch started");
        }

        if (j.l2_arrival > 0 && j.l3_arrival > 0 &&
            j.l2_arrival < j.l3_arrival) {
            result.add_error("Tile " + j.tile_id.to_string() +
                              " arrived at L2 before L3");
        }
    }

    return result;
}

inline std::unordered_map<TileID, std::set<TileID>, TileIDHash>
ScheduleValidator::build_dependency_graph(const PipelineSchedule& schedule) const {
    std::unordered_map<TileID, std::set<TileID>, TileIDHash> graph;

    for (const auto& op : schedule.operations) {
        for (const auto& dep : op.dependencies) {
            graph[op.tile_id].insert(dep);
        }
    }

    return graph;
}

inline bool ScheduleValidator::has_cycles(
    const std::unordered_map<TileID, std::set<TileID>, TileIDHash>& graph) const
{
    std::set<TileID> visited;
    std::set<TileID> rec_stack;

    std::function<bool(const TileID&)> dfs = [&](const TileID& node) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        auto it = graph.find(node);
        if (it != graph.end()) {
            for (const auto& dep : it->second) {
                // Self-dependencies are allowed (operation on tile A waiting
                // for previous operation on tile A to complete)
                if (dep == node) continue;

                if (visited.find(dep) == visited.end()) {
                    if (dfs(dep)) return true;
                } else if (rec_stack.find(dep) != rec_stack.end()) {
                    return true;
                }
            }
        }

        rec_stack.erase(node);
        return false;
    };

    for (const auto& [node, deps] : graph) {
        (void)deps;
        if (visited.find(node) == visited.end()) {
            if (dfs(node)) return true;
        }
    }

    return false;
}

} // namespace sw::kpu::harness

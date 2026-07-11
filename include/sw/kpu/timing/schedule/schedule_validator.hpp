// ============================================================================
// include/sw/kpu/timing/schedule/schedule_validator.hpp
// Static schedule validation for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::timing::schedule {

// ============================================================================
// Validation Severity
// ============================================================================

enum class ValidationSeverity {
    INFO,       ///< Informational message
    WARNING,    ///< Potential issue but may be intentional
    ERROR       ///< Definite problem that will cause failure
};

inline const char* to_string(ValidationSeverity severity) {
    switch (severity) {
        case ValidationSeverity::INFO:    return "INFO";
        case ValidationSeverity::WARNING: return "WARNING";
        case ValidationSeverity::ERROR:   return "ERROR";
        default:                          return "UNKNOWN";
    }
}

// ============================================================================
// Validation Issue
// ============================================================================

/**
 * @brief A single validation issue found in a schedule
 */
struct ValidationIssue {
    ValidationSeverity severity;
    std::string rule_id;        ///< Rule identifier (e.g., "VAL-001")
    std::string message;        ///< Human-readable description
    size_t operation_index = 0; ///< Index of problematic operation (if applicable)

    /**
     * @brief Format issue as string
     */
    [[nodiscard]] std::string to_string() const {
        std::ostringstream ss;
        ss << "[" << schedule::to_string(severity) << "] " << rule_id << ": " << message;
        if (operation_index > 0) {
            ss << " (op #" << operation_index << ")";
        }
        return ss.str();
    }
};

// ============================================================================
// Validation Result
// ============================================================================

/**
 * @brief Result of schedule validation
 */
struct ValidationResult {
    bool valid = true;                          ///< True if no errors found
    std::vector<ValidationIssue> issues;        ///< All issues found

    /**
     * @brief Check if validation passed
     */
    [[nodiscard]] explicit operator bool() const {
        return valid;
    }

    /**
     * @brief Count issues by severity
     */
    [[nodiscard]] size_t count_errors() const {
        size_t count = 0;
        for (const auto& issue : issues) {
            if (issue.severity == ValidationSeverity::ERROR) ++count;
        }
        return count;
    }

    [[nodiscard]] size_t count_warnings() const {
        size_t count = 0;
        for (const auto& issue : issues) {
            if (issue.severity == ValidationSeverity::WARNING) ++count;
        }
        return count;
    }

    /**
     * @brief Format all issues as string
     */
    [[nodiscard]] std::string to_string() const {
        std::ostringstream ss;
        ss << "Validation " << (valid ? "PASSED" : "FAILED");
        ss << " (" << count_errors() << " errors, " << count_warnings() << " warnings)\n";
        for (const auto& issue : issues) {
            ss << "  " << issue.to_string() << "\n";
        }
        return ss.str();
    }
};

// ============================================================================
// Schedule Validator
// ============================================================================

/**
 * @brief Static validator for operation schedules
 *
 * The ScheduleValidator performs static analysis on schedules to detect
 * potential issues before execution. This enables fast validation without
 * running the full simulation.
 *
 * Validation rules:
 * - VAL-001: Schedule must not be empty
 * - VAL-002: Each tile must have required operations (load->move->feed or drain->writeback->store)
 * - VAL-003: Schedule should be interleaved to prevent livelock
 * - VAL-004: Consecutive same-matrix operations should not exceed threshold
 * - VAL-005: Resource usage should not exceed configuration limits
 * - VAL-006: Dependencies between operations must be satisfiable
 *
 * Usage:
 * ```cpp
 * ScheduleValidator validator;
 * auto result = validator.validate(schedule);
 * if (!result) {
 *     std::cerr << result.to_string();
 * }
 * ```
 */
class ScheduleValidator {
public:
    /**
     * @brief Validation configuration
     */
    struct Config {
        /// Maximum consecutive same-matrix operations before warning
        size_t max_consecutive_threshold = 4;

        /// Require interleaved scheduling
        bool require_interleaving = false;

        /// Maximum A/B ratio before warning (e.g., 3.0 means 3:1 ratio)
        double max_ab_ratio = 3.0;

        /// Enable dependency checking
        bool check_dependencies = true;

        /// Maximum operations in schedule (0 = unlimited)
        size_t max_operations = 0;
    };

    /**
     * @brief Construct with default configuration
     */
    ScheduleValidator() = default;

    /**
     * @brief Construct with custom configuration
     */
    explicit ScheduleValidator(const Config& config) : config_(config) {}

    /**
     * @brief Validate a schedule
     * @param schedule The schedule to validate
     * @return ValidationResult with all issues found
     */
    ValidationResult validate(const ScheduleResult& schedule) const {
        ValidationResult result;

        // VAL-001: Schedule must not be empty
        if (schedule.operations.empty()) {
            result.issues.push_back({
                ValidationSeverity::ERROR,
                "VAL-001",
                "Schedule is empty",
                0
            });
            result.valid = false;
            return result;
        }

        // Perform all checks
        check_tile_completeness(schedule, result);
        check_interleaving(schedule, result);
        check_ab_balance(schedule, result);
        check_operation_order(schedule, result);

        if (config_.max_operations > 0) {
            check_size_limits(schedule, result);
        }

        // Set overall validity
        result.valid = (result.count_errors() == 0);
        return result;
    }

    /**
     * @brief Validate a schedule for livelock safety
     * @param schedule The schedule to validate
     * @param available_credits Number of available credits at each level
     * @return ValidationResult focused on livelock issues
     */
    ValidationResult validate_livelock_safety(
        const ScheduleResult& schedule,
        size_t l3_credits,
        size_t l2_credits) const {

        ValidationResult result;

        auto analysis = ScheduleAnalysis::analyze(schedule);

        // Check if consecutive operations exceed available credits
        if (analysis.max_consecutive_a > l3_credits / 2) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-LL1",
                "Consecutive A operations (" + std::to_string(analysis.max_consecutive_a) +
                ") may exceed A credit partition (" + std::to_string(l3_credits / 3) + ")",
                0
            });
        }

        if (analysis.max_consecutive_b > l3_credits / 2) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-LL2",
                "Consecutive B operations (" + std::to_string(analysis.max_consecutive_b) +
                ") may exceed B credit partition (" + std::to_string(l3_credits / 3) + ")",
                0
            });
        }

        if (!analysis.is_interleaved && l3_credits < 12) {
            result.issues.push_back({
                ValidationSeverity::ERROR,
                "VAL-LL3",
                "Non-interleaved schedule with limited credits (" +
                std::to_string(l3_credits) + ") is livelock-prone",
                0
            });
            result.valid = false;
        }

        // Simulate credit usage to detect potential exhaustion
        simulate_credit_usage(schedule, l3_credits, l2_credits, result);

        if (result.count_errors() == 0) {
            result.valid = true;
        }

        return result;
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

    /**
     * @brief Update configuration
     */
    void set_config(const Config& config) { config_ = config; }

private:
    Config config_;

    /**
     * @brief VAL-002: Check that each tile has complete operation sequence
     */
    void check_tile_completeness(const ScheduleResult& schedule, ValidationResult& result) const {
        // Track operations per tile
        std::map<TileID, std::set<ScheduleOpType>, TileIDCompare> tile_ops;

        for (const auto& op : schedule.operations) {
            tile_ops[op.tile.tile_id].insert(op.type);
        }

        // Check A tiles: need LOAD, MOVE, FEED
        // Check B tiles: need LOAD, MOVE, FEED
        // Check C tiles: need DRAIN, WRITEBACK, STORE

        for (const auto& [tile_id, ops] : tile_ops) {
            bool has_load = ops.count(ScheduleOpType::LOAD) > 0;
            bool has_move = ops.count(ScheduleOpType::MOVE) > 0;
            bool has_feed = ops.count(ScheduleOpType::FEED) > 0;
            bool has_drain = ops.count(ScheduleOpType::DRAIN) > 0;
            bool has_writeback = ops.count(ScheduleOpType::WRITEBACK) > 0;
            bool has_store = ops.count(ScheduleOpType::STORE) > 0;

            if (tile_id.matrix == isa::MatrixID::A || tile_id.matrix == isa::MatrixID::B) {
                // Input tiles need load->move->feed
                if (has_load && !has_move) {
                    result.issues.push_back({
                        ValidationSeverity::WARNING,
                        "VAL-002",
                        "Tile " + tile_id.to_string() + " has LOAD but no MOVE",
                        0
                    });
                }
                if (has_move && !has_feed) {
                    result.issues.push_back({
                        ValidationSeverity::WARNING,
                        "VAL-002",
                        "Tile " + tile_id.to_string() + " has MOVE but no FEED",
                        0
                    });
                }
            } else {
                // Output tiles need drain->writeback->store
                if (has_drain && !has_writeback) {
                    result.issues.push_back({
                        ValidationSeverity::WARNING,
                        "VAL-002",
                        "Tile " + tile_id.to_string() + " has DRAIN but no WRITEBACK",
                        0
                    });
                }
                if (has_writeback && !has_store) {
                    result.issues.push_back({
                        ValidationSeverity::WARNING,
                        "VAL-002",
                        "Tile " + tile_id.to_string() + " has WRITEBACK but no STORE",
                        0
                    });
                }
            }
        }
    }

    /**
     * @brief VAL-003: Check interleaving
     */
    void check_interleaving(const ScheduleResult& schedule, ValidationResult& result) const {
        auto analysis = ScheduleAnalysis::analyze(schedule);

        if (!analysis.is_interleaved) {
            ValidationSeverity severity = config_.require_interleaving
                ? ValidationSeverity::ERROR
                : ValidationSeverity::WARNING;

            result.issues.push_back({
                severity,
                "VAL-003",
                "Schedule is not interleaved (max consecutive A: " +
                std::to_string(analysis.max_consecutive_a) +
                ", B: " + std::to_string(analysis.max_consecutive_b) + ")",
                0
            });

            if (severity == ValidationSeverity::ERROR) {
                result.valid = false;
            }
        }
    }

    /**
     * @brief VAL-004: Check consecutive same-matrix operations
     */
    void check_ab_balance(const ScheduleResult& schedule, ValidationResult& result) const {
        auto analysis = ScheduleAnalysis::analyze(schedule);

        if (analysis.max_consecutive_a > config_.max_consecutive_threshold) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-004",
                "Too many consecutive A operations: " +
                std::to_string(analysis.max_consecutive_a) +
                " (threshold: " + std::to_string(config_.max_consecutive_threshold) + ")",
                0
            });
        }

        if (analysis.max_consecutive_b > config_.max_consecutive_threshold) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-004",
                "Too many consecutive B operations: " +
                std::to_string(analysis.max_consecutive_b) +
                " (threshold: " + std::to_string(config_.max_consecutive_threshold) + ")",
                0
            });
        }

        // Check A/B ratio
        if (analysis.a_ops > 0 && analysis.b_ops > 0) {
            double ratio = static_cast<double>(analysis.a_ops) / static_cast<double>(analysis.b_ops);
            if (ratio > config_.max_ab_ratio || ratio < 1.0 / config_.max_ab_ratio) {
                result.issues.push_back({
                    ValidationSeverity::WARNING,
                    "VAL-004",
                    "Unbalanced A/B ratio: " + std::to_string(ratio) +
                    " (max: " + std::to_string(config_.max_ab_ratio) + ":1)",
                    0
                });
            }
        }
    }

    /**
     * @brief VAL-006: Check operation ordering dependencies
     */
    void check_operation_order(const ScheduleResult& schedule, ValidationResult& result) const {
        if (!config_.check_dependencies) return;

        // Track first occurrence of each operation type per tile
        std::map<TileID, std::map<ScheduleOpType, size_t>, TileIDCompare> first_occurrence;

        for (size_t i = 0; i < schedule.operations.size(); ++i) {
            const auto& op = schedule.operations[i];
            auto& tile_ops = first_occurrence[op.tile.tile_id];
            if (tile_ops.find(op.type) == tile_ops.end()) {
                tile_ops[op.type] = i;
            }
        }

        // Check that LOAD comes before MOVE, MOVE before FEED, etc.
        for (const auto& [tile_id, ops] : first_occurrence) {
            auto has_op = [&ops](ScheduleOpType type) { return ops.count(type) > 0; };
            auto get_idx = [&ops](ScheduleOpType type) { return ops.at(type); };

            // For input tiles
            if (tile_id.matrix == isa::MatrixID::A || tile_id.matrix == isa::MatrixID::B) {
                if (has_op(ScheduleOpType::LOAD) && has_op(ScheduleOpType::MOVE)) {
                    if (get_idx(ScheduleOpType::MOVE) < get_idx(ScheduleOpType::LOAD)) {
                        result.issues.push_back({
                            ValidationSeverity::ERROR,
                            "VAL-006",
                            "Tile " + tile_id.to_string() + ": MOVE before LOAD",
                            get_idx(ScheduleOpType::MOVE)
                        });
                    }
                }
                if (has_op(ScheduleOpType::MOVE) && has_op(ScheduleOpType::FEED)) {
                    if (get_idx(ScheduleOpType::FEED) < get_idx(ScheduleOpType::MOVE)) {
                        result.issues.push_back({
                            ValidationSeverity::ERROR,
                            "VAL-006",
                            "Tile " + tile_id.to_string() + ": FEED before MOVE",
                            get_idx(ScheduleOpType::FEED)
                        });
                    }
                }
            } else {
                // For output tiles
                if (has_op(ScheduleOpType::DRAIN) && has_op(ScheduleOpType::WRITEBACK)) {
                    if (get_idx(ScheduleOpType::WRITEBACK) < get_idx(ScheduleOpType::DRAIN)) {
                        result.issues.push_back({
                            ValidationSeverity::ERROR,
                            "VAL-006",
                            "Tile " + tile_id.to_string() + ": WRITEBACK before DRAIN",
                            get_idx(ScheduleOpType::WRITEBACK)
                        });
                    }
                }
                if (has_op(ScheduleOpType::WRITEBACK) && has_op(ScheduleOpType::STORE)) {
                    if (get_idx(ScheduleOpType::STORE) < get_idx(ScheduleOpType::WRITEBACK)) {
                        result.issues.push_back({
                            ValidationSeverity::ERROR,
                            "VAL-006",
                            "Tile " + tile_id.to_string() + ": STORE before WRITEBACK",
                            get_idx(ScheduleOpType::STORE)
                        });
                    }
                }
            }
        }
    }

    /**
     * @brief Check schedule size limits
     */
    void check_size_limits(const ScheduleResult& schedule, ValidationResult& result) const {
        if (schedule.operations.size() > config_.max_operations) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-005",
                "Schedule size (" + std::to_string(schedule.operations.size()) +
                ") exceeds limit (" + std::to_string(config_.max_operations) + ")",
                0
            });
        }
    }

    /**
     * @brief Simulate credit usage to detect potential exhaustion
     */
    void simulate_credit_usage(
        const ScheduleResult& schedule,
        size_t l3_credits,
        size_t l2_credits,
        ValidationResult& result) const {

        // Simplified simulation: track outstanding operations per level
        size_t l3_outstanding = 0;
        size_t l2_outstanding = 0;
        size_t max_l3_outstanding = 0;
        size_t max_l2_outstanding = 0;

        for (size_t i = 0; i < schedule.operations.size(); ++i) {
            const auto& op = schedule.operations[i];

            switch (op.type) {
                case ScheduleOpType::LOAD:
                    l3_outstanding++;
                    break;
                case ScheduleOpType::MOVE:
                    // MOVE consumes L3, produces L2
                    if (l3_outstanding > 0) l3_outstanding--;
                    l2_outstanding++;
                    break;
                case ScheduleOpType::FEED:
                    if (l2_outstanding > 0) l2_outstanding--;
                    break;
                case ScheduleOpType::COMPUTE:
                    // COMPUTE doesn't affect memory credits
                    break;
                case ScheduleOpType::DRAIN:
                    l2_outstanding++;
                    break;
                case ScheduleOpType::WRITEBACK:
                    if (l2_outstanding > 0) l2_outstanding--;
                    l3_outstanding++;
                    break;
                case ScheduleOpType::STORE:
                    if (l3_outstanding > 0) l3_outstanding--;
                    break;
            }

            max_l3_outstanding = std::max(max_l3_outstanding, l3_outstanding);
            max_l2_outstanding = std::max(max_l2_outstanding, l2_outstanding);
        }

        if (max_l3_outstanding > l3_credits) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-LL4",
                "Peak L3 usage (" + std::to_string(max_l3_outstanding) +
                ") exceeds available credits (" + std::to_string(l3_credits) + ")",
                0
            });
        }

        if (max_l2_outstanding > l2_credits) {
            result.issues.push_back({
                ValidationSeverity::WARNING,
                "VAL-LL5",
                "Peak L2 usage (" + std::to_string(max_l2_outstanding) +
                ") exceeds available credits (" + std::to_string(l2_credits) + ")",
                0
            });
        }
    }

    // Comparator for TileID in map
    struct TileIDCompare {
        bool operator()(const TileID& a, const TileID& b) const {
            return a < b;
        }
    };
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Validate a schedule with default settings
 */
inline ValidationResult validate_schedule(const ScheduleResult& schedule) {
    ScheduleValidator validator;
    return validator.validate(schedule);
}

/**
 * @brief Validate a schedule for livelock safety
 */
inline ValidationResult validate_livelock_safety(
    const ScheduleResult& schedule,
    size_t l3_credits,
    size_t l2_credits) {

    ScheduleValidator validator;
    return validator.validate_livelock_safety(schedule, l3_credits, l2_credits);
}

/**
 * @brief Quick check if schedule is livelock-safe (capacity-blind heuristic)
 *
 * Uses the fixed interleaving threshold (max 3 consecutive same-matrix ops).
 * Prefer the capacity-aware overload when the resource envelope is known:
 * it checks the constructive property the generators guarantee.
 */
inline bool is_livelock_safe(const ScheduleResult& schedule) {
    auto analysis = ScheduleAnalysis::analyze(schedule);
    return analysis.is_interleaved;
}

/**
 * @brief Capacity-aware livelock-safety check (issue #67)
 *
 * Verifies the constructive residency bound the generators emit against:
 * each matrix's longest burst must fit its share of the credit pools
 * (a quarter of the smaller pool, so an A burst and a B burst can always
 * be resident simultaneously with headroom for drains and C writebacks).
 * This is an invariant check of the schedule's working set against the
 * envelope - not a fixed-threshold heuristic.
 */
inline bool is_livelock_safe(const ScheduleResult& schedule,
                             size_t l3_credits,
                             size_t l2_credits) {
    auto analysis = ScheduleAnalysis::analyze(schedule);
    size_t share = per_matrix_burst_share(static_cast<Size>(l3_credits),
                                          static_cast<Size>(l2_credits));
    // ScheduleAnalysis counts consecutive OPERATIONS; a resident tile
    // contributes at most two ops to a burst (LOAD + MOVE), so a burst of
    // `share` tiles appears as up to 2*share consecutive same-matrix ops.
    size_t share_ops = 2 * share;
    return analysis.max_consecutive_a <= share_ops &&
           analysis.max_consecutive_b <= share_ops;
}

} // namespace sw::kpu::timing::schedule

// ============================================================================
// include/sw/kpu/timing/schedule/schedule_executor.hpp
// Schedule executor for applying schedules to ConcurrentTimingExecutor
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing::schedule {

// ============================================================================
// Schedule Execution Result
// ============================================================================

/**
 * @brief Result of schedule execution
 */
struct ExecutionResult {
    bool success = false;               ///< Whether execution completed successfully
    Cycle total_cycles = 0;             ///< Total cycles taken
    std::string error_message;          ///< Error message if failed
    std::vector<std::string> warnings;  ///< Non-fatal issues (e.g. envelope mismatch, #91)

    // Per-operation statistics
    size_t ops_completed = 0;           ///< Operations successfully executed
    size_t ops_total = 0;               ///< Total operations in schedule

    // Timing breakdown
    Cycle dma_cycles = 0;
    Cycle bm_cycles = 0;
    Cycle str_cycles = 0;

    // Livelock detection
    bool livelock_detected = false;     ///< True if livelock was detected
    Cycle stall_duration = 0;           ///< Cycles spent stalled

    /**
     * @brief Check if execution succeeded
     */
    [[nodiscard]] explicit operator bool() const {
        return success;
    }

    /**
     * @brief Calculate throughput (ops/cycle)
     */
    [[nodiscard]] double throughput() const {
        return total_cycles > 0 ? static_cast<double>(ops_completed) / static_cast<double>(total_cycles) : 0.0;
    }
};

// ============================================================================
// Schedule Executor
// ============================================================================

/**
 * @brief Executes a schedule on a ConcurrentTimingExecutor
 *
 * The ScheduleExecutor bridges generated schedules to the timing executor.
 * It maps ScheduleOperations to the executor's schedule_* API and tracks
 * execution progress.
 *
 * Usage:
 * ```cpp
 * ConcurrentTimingExecutor executor(config);
 * ScheduleExecutor scheduler(executor);
 *
 * auto schedule = generator.generate();
 * auto result = scheduler.execute(schedule);
 *
 * if (result.success) {
 *     std::cout << "Completed in " << result.total_cycles << " cycles\n";
 * }
 * ```
 */
class ScheduleExecutor {
public:
    /**
     * @brief Configuration for schedule execution
     */
    struct Config {
        /// Enable progress reporting
        bool enable_progress_reporting = false;

        /// Progress callback interval (operations)
        size_t progress_interval = 100;

        /// Enable detailed tracing
        bool enable_tracing = false;

        /// Abort on first error
        bool abort_on_error = true;
    };

    /**
     * @brief Progress callback type
     */
    using ProgressCallback = std::function<void(size_t current, size_t total)>;

    /**
     * @brief Optional value-plane binding for COMPUTE operations
     *
     * When set, each COMPUTE operation is offered to the binder; returning a
     * FunctionalComputeSpec dispatches it as a value-producing compute (the
     * #66 payload machinery) instead of a timing-only compute. Returning
     * nullopt keeps the timing-only path for that operation. Timing behavior
     * is identical either way: the functional path derives its dependency
     * accounting from the same scheduled-feed counts.
     */
    using FunctionalComputeBinder =
        std::function<std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec>(
            const ScheduleOperation&)>;

    /**
     * @brief Construct executor with a timing executor reference
     * @param executor The ConcurrentTimingExecutor to use
     */
    explicit ScheduleExecutor(ConcurrentTimingExecutor& executor)
        : executor_(executor) {}

    /**
     * @brief Construct with configuration
     */
    ScheduleExecutor(ConcurrentTimingExecutor& executor, const Config& config)
        : executor_(executor), config_(config) {}

    /**
     * @brief Set progress callback
     */
    void set_progress_callback(ProgressCallback callback) {
        progress_callback_ = std::move(callback);
    }

    /**
     * @brief Install a value-plane binder for COMPUTE operations
     */
    void set_functional_compute_binder(FunctionalComputeBinder binder) {
        functional_binder_ = std::move(binder);
    }

    /**
     * @brief Execute a schedule
     * @param schedule The schedule to execute
     * @return ExecutionResult with timing and status information
     */
    ExecutionResult execute(const ScheduleResult& schedule) {
        ExecutionResult result;
        result.ops_total = schedule.operations.size();

        if (!schedule.valid) {
            result.success = false;
            result.error_message = "Invalid schedule: " + schedule.error_message;
            return result;
        }

        check_envelope_mismatch(schedule, result);

        // Reset executor state
        executor_.reset();

        // Enqueue all operations
        for (size_t i = 0; i < schedule.operations.size(); ++i) {
            const auto& op = schedule.operations[i];

            try {
                enqueue_operation(op);
                result.ops_completed++;
            } catch (const std::exception& e) {
                result.error_message = "Failed to enqueue operation " +
                                       std::to_string(i) + ": " + e.what();
                if (config_.abort_on_error) {
                    return result;
                }
            }

            // Report progress
            if (config_.enable_progress_reporting && progress_callback_ &&
                (i % config_.progress_interval == 0)) {
                progress_callback_(i, result.ops_total);
            }
        }

        // Run simulation to completion
        bool completed = executor_.run();

        result.total_cycles = executor_.current_cycle();
        result.success = completed;

        if (!completed) {
            // Check if it was livelock or max cycles
            if (executor_.current_cycle() >= executor_.config().max_cycles) {
                result.error_message = "Execution exceeded max cycles";
            } else {
                result.livelock_detected = true;
                result.error_message = "Livelock detected";
            }
        }

        // Collect statistics
        auto stats = executor_.get_statistics();
        result.dma_cycles = stats.dma_credit_stalls;
        result.bm_cycles = stats.bm_tag_stalls + stats.bm_credit_stalls;
        result.str_cycles = stats.str_tag_stalls + stats.str_credit_stalls;

        return result;
    }

    /**
     * @brief Execute a schedule with a custom livelock callback
     * @param schedule The schedule to execute
     * @param on_livelock Callback invoked if livelock is detected
     * @return ExecutionResult
     */
    ExecutionResult execute_with_livelock_handler(
        const ScheduleResult& schedule,
        std::function<void(Cycle)> on_livelock) {

        auto result = execute(schedule);

        if (result.livelock_detected && on_livelock) {
            on_livelock(result.total_cycles);
        }

        return result;
    }

    /**
     * @brief Get underlying executor
     */
    [[nodiscard]] ConcurrentTimingExecutor& executor() { return executor_; }
    [[nodiscard]] const ConcurrentTimingExecutor& executor() const { return executor_; }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

    /**
     * @brief Update configuration
     */
    void set_config(const Config& config) { config_ = config; }

private:
    ConcurrentTimingExecutor& executor_;
    Config config_;
    ProgressCallback progress_callback_;
    FunctionalComputeBinder functional_binder_;

    /**
     * @brief Warn when the schedule's generation envelope differs from the
     *        executor's configured credit pools (issue #91)
     *
     * The #67/#90 constructive-safety guarantees (burst bounds, working-set
     * fit) were established against the generation envelope. Executing under
     * SMALLER pools voids them - the schedule may wedge; larger pools are
     * benign but still flagged so the mismatch is visible. Schedules with no
     * recorded envelope (hand-built or legacy, metadata fields 0) are not
     * checked.
     */
    void check_envelope_mismatch(const ScheduleResult& schedule,
                                 ExecutionResult& result) const {
        const auto& meta = schedule.metadata;
        if (meta.l3_buffer_count == 0 && meta.l2_bank_count == 0) {
            return;  // envelope not recorded
        }
        const auto& exec_config = executor_.config();
        if (meta.l3_buffer_count == exec_config.l3_buffer_count &&
            meta.l2_bank_count == exec_config.l2_bank_count) {
            return;  // envelopes agree
        }
        const bool shrunk =
            exec_config.l3_buffer_count < meta.l3_buffer_count ||
            exec_config.l2_bank_count < meta.l2_bank_count;
        std::string warning =
            "schedule '" + meta.name + "' was generated against envelope L3=" +
            std::to_string(meta.l3_buffer_count) + "/L2=" +
            std::to_string(meta.l2_bank_count) +
            " but the executor is configured with L3=" +
            std::to_string(exec_config.l3_buffer_count) + "/L2=" +
            std::to_string(exec_config.l2_bank_count);
        warning += shrunk
            ? " - the executor pools are SMALLER: the schedule's working-set "
              "and burst-safety guarantees do not hold and it may wedge"
            : " - the executor pools are larger; execution is safe but "
              "regenerate against the actual envelope for faithful timing";
        result.warnings.push_back(std::move(warning));
    }

    /**
     * @brief Enqueue a single operation on the executor
     */
    void enqueue_operation(const ScheduleOperation& op) {
        switch (op.type) {
            case ScheduleOpType::LOAD:
                executor_.schedule_load(op.tile, op.engine_id);
                break;

            case ScheduleOpType::STORE:
                executor_.schedule_store(op.tile, op.engine_id);
                break;

            case ScheduleOpType::MOVE:
                executor_.schedule_move(op.tile, op.transpose, op.mover_id);
                break;

            case ScheduleOpType::WRITEBACK:
                executor_.schedule_writeback(op.tile, op.mover_id);
                break;

            case ScheduleOpType::FEED:
                executor_.schedule_feed(op.tile, op.streamer_id);
                break;

            case ScheduleOpType::COMPUTE:
                if (functional_binder_) {
                    if (auto spec = functional_binder_(op)) {
                        executor_.schedule_functional_compute(op.tile, *spec);
                        break;
                    }
                }
                if (!op.resident_tiles.empty()) {
                    // Timing-tier compute with compute-resident inputs (#155)
                    executor_.schedule_compute(op.tile, op.dependency_tiles,
                                               op.resident_tiles);
                } else if (!op.dependency_tiles.empty()) {
                    executor_.schedule_compute(op.tile, op.dependency_tiles);
                } else {
                    executor_.schedule_compute(op.tile, op.dependency_tile);
                }
                break;

            case ScheduleOpType::DRAIN:
                executor_.schedule_drain(op.tile, op.streamer_id);
                break;

            default:
                throw std::runtime_error("Unknown schedule operation type");
        }
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Execute a schedule and return result
 *
 * Convenience function for one-shot execution.
 */
inline ExecutionResult execute_schedule(
    ConcurrentTimingExecutor& executor,
    const ScheduleResult& schedule) {

    ScheduleExecutor sched_exec(executor);
    return sched_exec.execute(schedule);
}

/**
 * @brief Generate and execute a schedule in one step
 */
inline ExecutionResult generate_and_execute(
    ConcurrentTimingExecutor& executor,
    IScheduleGenerator& generator) {

    auto schedule = generator.generate();
    return execute_schedule(executor, schedule);
}

} // namespace sw::kpu::timing::schedule

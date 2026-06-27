// patterns/memory/gddr6/common/multi_fidelity.hpp
//
// Multi-fidelity comparison framework for GDDR6 memory controller
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>

#include "gddr6_configs.hpp"
#include "gddr6_harness.hpp"
#include "workloads.hpp"

#include <sw/kpu/fidelity/simulation_fidelity.hpp>

namespace sw::kpu::patterns::gddr6 {

// ============================================================================
// Latency Statistics
// ============================================================================

struct LatencyStats {
    uint64_t count = 0;
    double min = 0.0;
    double max = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    std::vector<uint64_t> samples;

    void add_sample(uint64_t latency) {
        double lat = static_cast<double>(latency);
        if (count == 0) {
            min = lat;
            max = lat;
        } else {
            min = std::min(min, lat);
            max = std::max(max, lat);
        }
        sum += lat;
        sum_sq += lat * lat;
        samples.push_back(latency);
        ++count;
    }

    double mean() const { return count > 0 ? sum / static_cast<double>(count) : 0.0; }
    double variance() const {
        if (count < 2) return 0.0;
        return (sum_sq - (sum * sum / static_cast<double>(count))) / static_cast<double>(count - 1);
    }
    double stddev() const { return std::sqrt(variance()); }
};

// ============================================================================
// Fidelity Run Results
// ============================================================================

struct FidelityRunResult {
    SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL;
    std::string workload_name;

    uint64_t total_cycles = 0;
    LatencyStats read_latencies;
    LatencyStats write_latencies;
    LatencyStats all_latencies;

    uint64_t page_hits = 0;
    uint64_t page_empty = 0;
    uint64_t page_conflicts = 0;
};

// ============================================================================
// Fidelity Comparison
// ============================================================================

struct FidelityComparison {
    FidelityRunResult behavioral;
    FidelityRunResult transactional;
    FidelityRunResult cycle_accurate;

    double behavioral_cycle_error_pct = 0.0;
    double transactional_cycle_error_pct = 0.0;
    double behavioral_latency_error_pct = 0.0;
    double transactional_latency_error_pct = 0.0;

    void compute_errors() {
        if (cycle_accurate.total_cycles > 0) {
            behavioral_cycle_error_pct = 100.0 *
                std::abs(static_cast<double>(behavioral.total_cycles) -
                        static_cast<double>(cycle_accurate.total_cycles)) /
                static_cast<double>(cycle_accurate.total_cycles);

            transactional_cycle_error_pct = 100.0 *
                std::abs(static_cast<double>(transactional.total_cycles) -
                        static_cast<double>(cycle_accurate.total_cycles)) /
                static_cast<double>(cycle_accurate.total_cycles);
        }

        double ca_mean = cycle_accurate.all_latencies.mean();
        if (ca_mean > 0) {
            behavioral_latency_error_pct = 100.0 *
                std::abs(behavioral.all_latencies.mean() - ca_mean) / ca_mean;
            transactional_latency_error_pct = 100.0 *
                std::abs(transactional.all_latencies.mean() - ca_mean) / ca_mean;
        }
    }
};

// ============================================================================
// Calibration Parameters
// ============================================================================

struct BehavioralCalibration {
    uint32_t fixed_read_latency = 100;
    uint32_t fixed_write_latency = 100;

    static BehavioralCalibration from_cycle_accurate(const FidelityRunResult& ca) {
        BehavioralCalibration cal;
        cal.fixed_read_latency = static_cast<uint32_t>(std::round(ca.all_latencies.mean()));
        cal.fixed_write_latency = static_cast<uint32_t>(std::round(ca.all_latencies.mean()));
        return cal;
    }
};

struct TransactionalCalibration {
    uint32_t mean_read_latency = 100;
    uint32_t mean_write_latency = 80;
    uint32_t latency_variance = 20;
    double page_hit_latency_factor = 0.6;
    double page_empty_latency_factor = 1.0;
    double page_conflict_latency_factor = 1.4;

    static TransactionalCalibration from_cycle_accurate(const FidelityRunResult& ca) {
        TransactionalCalibration cal;
        double mean = ca.all_latencies.mean();
        cal.mean_read_latency = static_cast<uint32_t>(std::round(mean));
        cal.mean_write_latency = static_cast<uint32_t>(std::round(mean * 0.8));  // Writes typically faster
        cal.latency_variance = static_cast<uint32_t>(std::round(ca.all_latencies.stddev()));

        // Derive factors from GDDR6 theoretical latencies relative to measured mean
        if (mean > 0) {
            cal.page_hit_latency_factor = PAGE_HIT_READ_LATENCY / mean;
            cal.page_empty_latency_factor = PAGE_EMPTY_READ_LATENCY / mean;
            cal.page_conflict_latency_factor = PAGE_CONFLICT_READ_LATENCY / mean;
        }
        return cal;
    }
};

// ============================================================================
// GDDR6 Behavioral Memory Controller
// ============================================================================

/// Behavioral (functional) GDDR6 memory controller with fixed latency
/// Fastest simulation mode - no queue contention, no timing constraints
class GDDR6BehavioralMemoryController {
public:
    struct Config {
        uint32_t fixed_read_latency;
        uint32_t fixed_write_latency;
        uint8_t num_channels;
        uint8_t banks_per_channel;

        Config()
            : fixed_read_latency(100), fixed_write_latency(100)
            , num_channels(2), banks_per_channel(16) {}

        Config(uint32_t read_lat, uint32_t write_lat, uint8_t channels = 2, uint8_t banks = 16)
            : fixed_read_latency(read_lat), fixed_write_latency(write_lat)
            , num_channels(channels), banks_per_channel(banks) {}
    };

    GDDR6BehavioralMemoryController()
        : config_() {}

    explicit GDDR6BehavioralMemoryController(const Config& config)
        : config_(config) {}

    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }

    uint64_t submit_read(uint64_t /*address*/, uint32_t /*size*/ = 64) {
        uint64_t latency = config_.fixed_read_latency;
        uint64_t complete = current_cycle_ + latency;
        pending_.push_back({complete, latency, true});
        ++total_reads_;
        return complete;
    }

    uint64_t submit_write(uint64_t /*address*/, uint32_t /*size*/ = 64) {
        uint64_t latency = config_.fixed_write_latency;
        uint64_t complete = current_cycle_ + latency;
        pending_.push_back({complete, latency, false});
        ++total_writes_;
        return complete;
    }

    void tick() {
        ++current_cycle_;
        auto it = pending_.begin();
        while (it != pending_.end()) {
            if (it->complete_cycle <= current_cycle_) {
                completed_.push_back({it->latency, it->is_read});
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool has_pending() const { return !pending_.empty(); }
    uint64_t current_cycle() const { return current_cycle_; }

    const auto& completed() const { return completed_; }

    uint64_t total_reads() const { return total_reads_; }
    uint64_t total_writes() const { return total_writes_; }

    void reset() {
        current_cycle_ = 0;
        pending_.clear();
        completed_.clear();
        total_reads_ = 0;
        total_writes_ = 0;
    }

private:
    struct Pending {
        uint64_t complete_cycle;
        uint64_t latency;
        bool is_read;
    };

    Config config_;
    uint64_t current_cycle_ = 0;
    std::vector<Pending> pending_;
    std::vector<std::pair<uint64_t, bool>> completed_;
    uint64_t total_reads_ = 0;
    uint64_t total_writes_ = 0;
};

// ============================================================================
// GDDR6 Transactional Memory Controller
// ============================================================================

/// Transactional GDDR6 memory controller with queue-based statistical timing
/// Medium fidelity - models page hits/misses, bank contention, turnarounds
class GDDR6TransactionalMemoryController {
public:
    struct Config {
        uint32_t mean_read_latency;
        uint32_t mean_write_latency;
        uint32_t latency_variance;
        double page_hit_factor;
        double page_empty_factor;
        double page_conflict_factor;
        uint8_t num_channels;
        uint8_t banks_per_channel;
        uint8_t bank_groups;
        uint32_t seed;

        Config()
            : mean_read_latency(100), mean_write_latency(80), latency_variance(20)
            , page_hit_factor(0.6), page_empty_factor(1.0), page_conflict_factor(1.4)
            , num_channels(2), banks_per_channel(16), bank_groups(4), seed(12345) {}

        Config(uint32_t read_lat, uint32_t write_lat, uint32_t variance,
               double hit_f, double empty_f, double conflict_f,
               uint8_t channels = 2, uint8_t banks = 16, uint8_t groups = 4,
               uint32_t s = 12345)
            : mean_read_latency(read_lat), mean_write_latency(write_lat)
            , latency_variance(variance)
            , page_hit_factor(hit_f), page_empty_factor(empty_f), page_conflict_factor(conflict_f)
            , num_channels(channels), banks_per_channel(banks), bank_groups(groups)
            , seed(s) {}
    };

    GDDR6TransactionalMemoryController()
        : config_(), rng_(config_.seed) {
        init_banks();
    }

    explicit GDDR6TransactionalMemoryController(const Config& config)
        : config_(config), rng_(config.seed) {
        init_banks();
    }

    void set_config(const Config& config) {
        config_ = config;
        rng_.seed(config.seed);
        init_banks();
    }

    const Config& config() const { return config_; }

    uint64_t submit_read(uint64_t address, uint32_t /*size*/ = 64) {
        auto [channel, bank, row] = decode_address(address);
        auto pr = estimate_page_result(channel, bank, row);

        double base = config_.mean_read_latency;
        base *= get_factor(pr);

        // Add bank contention delay
        uint64_t bank_available = get_bank_available(channel, bank);
        uint64_t start_cycle = std::max(current_cycle_, bank_available);

        // Add variance
        int32_t variance = static_cast<int32_t>(
            (rng_() % (2 * config_.latency_variance + 1)) - config_.latency_variance);
        uint64_t latency = static_cast<uint64_t>(std::max(1.0, base + variance));

        uint64_t complete = start_cycle + latency;
        set_bank_available(channel, bank, complete);
        update_row_buffer(channel, bank, row);

        pending_.push_back({complete, latency, true, pr, channel, bank});
        ++total_reads_;

        // Track turnaround
        if (last_op_type_ == OpType::WRITE && current_cycle_ - last_op_cycle_ < tRTW) {
            ++write_to_read_turnarounds_;
        }
        last_op_type_ = OpType::READ;
        last_op_cycle_ = current_cycle_;

        return complete;
    }

    uint64_t submit_write(uint64_t address, uint32_t /*size*/ = 64) {
        auto [channel, bank, row] = decode_address(address);
        auto pr = estimate_page_result(channel, bank, row);

        double base = config_.mean_write_latency;
        base *= get_factor(pr);

        // Add bank contention delay
        uint64_t bank_available = get_bank_available(channel, bank);
        uint64_t start_cycle = std::max(current_cycle_, bank_available);

        // Add variance
        int32_t variance = static_cast<int32_t>(
            (rng_() % (2 * config_.latency_variance + 1)) - config_.latency_variance);
        uint64_t latency = static_cast<uint64_t>(std::max(1.0, base + variance));

        uint64_t complete = start_cycle + latency;
        set_bank_available(channel, bank, complete);
        update_row_buffer(channel, bank, row);

        pending_.push_back({complete, latency, false, pr, channel, bank});
        ++total_writes_;

        // Track turnaround
        if (last_op_type_ == OpType::READ && current_cycle_ - last_op_cycle_ < tWTR_L) {
            ++read_to_write_turnarounds_;
        }
        last_op_type_ = OpType::WRITE;
        last_op_cycle_ = current_cycle_;

        return complete;
    }

    void tick() {
        ++current_cycle_;
        auto it = pending_.begin();
        while (it != pending_.end()) {
            if (it->complete_cycle <= current_cycle_) {
                completed_.push_back({it->latency, it->is_read, it->page_result});
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool has_pending() const { return !pending_.empty(); }
    uint64_t current_cycle() const { return current_cycle_; }

    uint64_t page_hits() const {
        return std::count_if(completed_.begin(), completed_.end(),
            [](const auto& c) { return c.page_result == PageResult::HIT; });
    }
    uint64_t page_empty() const {
        return std::count_if(completed_.begin(), completed_.end(),
            [](const auto& c) { return c.page_result == PageResult::EMPTY; });
    }
    uint64_t page_conflicts() const {
        return std::count_if(completed_.begin(), completed_.end(),
            [](const auto& c) { return c.page_result == PageResult::CONFLICT; });
    }

    uint64_t read_to_write_turnarounds() const { return read_to_write_turnarounds_; }
    uint64_t write_to_read_turnarounds() const { return write_to_read_turnarounds_; }

    uint64_t total_reads() const { return total_reads_; }
    uint64_t total_writes() const { return total_writes_; }

    const auto& completed() const { return completed_; }

    void reset() {
        current_cycle_ = 0;
        pending_.clear();
        completed_.clear();
        row_buffers_.clear();
        bank_available_.clear();
        init_banks();
        total_reads_ = 0;
        total_writes_ = 0;
        read_to_write_turnarounds_ = 0;
        write_to_read_turnarounds_ = 0;
        last_op_type_ = OpType::READ;
        last_op_cycle_ = 0;
    }

private:
    enum class PageResult { HIT, EMPTY, CONFLICT };
    enum class OpType { READ, WRITE };

    struct Pending {
        uint64_t complete_cycle;
        uint64_t latency;
        bool is_read;
        PageResult page_result;
        uint8_t channel;
        uint8_t bank;
    };

    struct Completed {
        uint64_t latency;
        bool is_read;
        PageResult page_result;
    };

    void init_banks() {
        row_buffers_.clear();
        bank_available_.clear();
        for (uint8_t ch = 0; ch < config_.num_channels; ++ch) {
            for (uint8_t b = 0; b < config_.banks_per_channel; ++b) {
                uint32_t key = make_bank_key(ch, b);
                row_buffers_[key] = UINT32_MAX;  // No row open
                bank_available_[key] = 0;
            }
        }
    }

    uint32_t make_bank_key(uint8_t channel, uint8_t bank) const {
        return (static_cast<uint32_t>(channel) << 8) | bank;
    }

    std::tuple<uint8_t, uint8_t, uint32_t> decode_address(uint64_t address) const {
        // Match GDDR6 address format from gddr6_configs.hpp
        uint8_t channel = (address >> 5) & 0x1;
        uint8_t bank = (address >> (5 + 1 + 10)) & 0xF;
        uint32_t row = (address >> (5 + 1 + 10 + 4)) & 0xFFFF;
        return {channel, bank, row};
    }

    PageResult estimate_page_result(uint8_t channel, uint8_t bank, uint32_t row) {
        uint32_t key = make_bank_key(channel, bank);
        auto it = row_buffers_.find(key);
        if (it == row_buffers_.end() || it->second == UINT32_MAX) {
            return PageResult::EMPTY;
        }
        return it->second == row ? PageResult::HIT : PageResult::CONFLICT;
    }

    void update_row_buffer(uint8_t channel, uint8_t bank, uint32_t row) {
        uint32_t key = make_bank_key(channel, bank);
        row_buffers_[key] = row;
    }

    uint64_t get_bank_available(uint8_t channel, uint8_t bank) const {
        uint32_t key = make_bank_key(channel, bank);
        auto it = bank_available_.find(key);
        return it != bank_available_.end() ? it->second : 0;
    }

    void set_bank_available(uint8_t channel, uint8_t bank, uint64_t cycle) {
        uint32_t key = make_bank_key(channel, bank);
        bank_available_[key] = cycle;
    }

    double get_factor(PageResult pr) const {
        switch (pr) {
            case PageResult::HIT: return config_.page_hit_factor;
            case PageResult::EMPTY: return config_.page_empty_factor;
            case PageResult::CONFLICT: return config_.page_conflict_factor;
        }
        return 1.0;
    }

    Config config_;
    std::minstd_rand rng_;
    uint64_t current_cycle_ = 0;
    std::vector<Pending> pending_;
    std::vector<Completed> completed_;
    std::map<uint32_t, uint32_t> row_buffers_;    // bank_key -> open_row
    std::map<uint32_t, uint64_t> bank_available_; // bank_key -> available_cycle
    uint64_t total_reads_ = 0;
    uint64_t total_writes_ = 0;
    uint64_t read_to_write_turnarounds_ = 0;
    uint64_t write_to_read_turnarounds_ = 0;
    OpType last_op_type_ = OpType::READ;
    uint64_t last_op_cycle_ = 0;
};

// ============================================================================
// Multi-Fidelity Harness
// ============================================================================

class MultiFidelityHarness {
public:
    explicit MultiFidelityHarness(const GDDR6MemoryController::Config& ca_config)
        : ca_config_(ca_config)
        , behavioral_mc_(GDDR6BehavioralMemoryController::Config{})
        , transactional_mc_(GDDR6TransactionalMemoryController::Config{})
    {}

    /// Run workload at all three fidelity levels and compare
    FidelityComparison run_comparison(const Workload& workload, bool verbose = true) {
        FidelityComparison comp;

        if (verbose) {
            std::cout << "\n=== Running Workload: " << workload.name << " ===" << std::endl;
            std::cout << "Description: " << workload.description << std::endl;
            std::cout << "Accesses: " << workload.accesses.size() << std::endl;
        }

        comp.cycle_accurate = run_cycle_accurate(workload);
        comp.behavioral = run_behavioral(workload);
        comp.transactional = run_transactional(workload);
        comp.compute_errors();

        if (verbose) {
            print_comparison(comp);
        }

        return comp;
    }

    /// Run workload with calibrated behavioral/transactional controllers
    FidelityComparison run_calibrated(const Workload& workload, bool verbose = true) {
        FidelityComparison comp;

        // First run cycle-accurate to get calibration data
        comp.cycle_accurate = run_cycle_accurate(workload);

        // Calibrate behavioral controller
        auto b_cal = BehavioralCalibration::from_cycle_accurate(comp.cycle_accurate);
        behavioral_mc_.set_config({
            b_cal.fixed_read_latency,
            b_cal.fixed_write_latency,
            ca_config_.num_channels,
            ca_config_.banks_per_channel
        });

        // Calibrate transactional controller
        auto t_cal = TransactionalCalibration::from_cycle_accurate(comp.cycle_accurate);
        transactional_mc_.set_config({
            t_cal.mean_read_latency,
            t_cal.mean_write_latency,
            t_cal.latency_variance,
            t_cal.page_hit_latency_factor,
            t_cal.page_empty_latency_factor,
            t_cal.page_conflict_latency_factor,
            ca_config_.num_channels,
            ca_config_.banks_per_channel,
            ca_config_.bank_groups,
            12345
        });

        // Run calibrated simulations
        comp.behavioral = run_behavioral(workload);
        comp.transactional = run_transactional(workload);
        comp.compute_errors();

        if (verbose) {
            std::cout << "\n=== Calibrated: " << workload.name << " ===" << std::endl;
            print_comparison(comp);
        }

        return comp;
    }

    /// Run all workloads and report summary
    void run_all_workloads(const std::vector<Workload>& workloads, bool calibrated = true) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "GDDR6 Multi-Fidelity Comparison" << std::endl;
        std::cout << "========================================" << std::endl;

        double total_behavioral_error = 0;
        double total_transactional_error = 0;

        for (const auto& w : workloads) {
            FidelityComparison comp;
            if (calibrated) {
                comp = run_calibrated(w, true);
            } else {
                comp = run_comparison(w, true);
            }
            total_behavioral_error += comp.behavioral_cycle_error_pct;
            total_transactional_error += comp.transactional_cycle_error_pct;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Summary (" << workloads.size() << " workloads)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Avg Behavioral Error:    "
                  << std::fixed << std::setprecision(1)
                  << (total_behavioral_error / static_cast<double>(workloads.size())) << "%" << std::endl;
        std::cout << "Avg Transactional Error: "
                  << (total_transactional_error / static_cast<double>(workloads.size())) << "%" << std::endl;
    }

private:
    FidelityRunResult run_behavioral(const Workload& workload) {
        FidelityRunResult result;
        result.fidelity = SimulationFidelity::BEHAVIORAL;
        result.workload_name = workload.name;

        behavioral_mc_.reset();
        for (const auto& access : workload.accesses) {
            if (access.type == AccessType::READ) {
                behavioral_mc_.submit_read(access.address, access.size);
            } else {
                behavioral_mc_.submit_write(access.address, access.size);
            }
        }
        while (behavioral_mc_.has_pending()) behavioral_mc_.tick();

        result.total_cycles = behavioral_mc_.current_cycle();
        for (const auto& [lat, is_read] : behavioral_mc_.completed()) {
            result.all_latencies.add_sample(lat);
            if (is_read) {
                result.read_latencies.add_sample(lat);
            } else {
                result.write_latencies.add_sample(lat);
            }
        }

        return result;
    }

    FidelityRunResult run_transactional(const Workload& workload) {
        FidelityRunResult result;
        result.fidelity = SimulationFidelity::TRANSACTIONAL;
        result.workload_name = workload.name;

        transactional_mc_.reset();
        for (const auto& access : workload.accesses) {
            if (access.type == AccessType::READ) {
                transactional_mc_.submit_read(access.address, access.size);
            } else {
                transactional_mc_.submit_write(access.address, access.size);
            }
        }
        while (transactional_mc_.has_pending()) transactional_mc_.tick();

        result.total_cycles = transactional_mc_.current_cycle();
        result.page_hits = transactional_mc_.page_hits();
        result.page_empty = transactional_mc_.page_empty();
        result.page_conflicts = transactional_mc_.page_conflicts();

        for (const auto& c : transactional_mc_.completed()) {
            result.all_latencies.add_sample(c.latency);
            if (c.is_read) {
                result.read_latencies.add_sample(c.latency);
            } else {
                result.write_latencies.add_sample(c.latency);
            }
        }

        return result;
    }

    FidelityRunResult run_cycle_accurate(const Workload& workload) {
        FidelityRunResult result;
        result.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        result.workload_name = workload.name;

        GDDR6Harness harness(ca_config_);
        for (const auto& access : workload.accesses) {
            if (access.type == AccessType::READ) {
                harness.submit_read(access.address, access.size);
            } else {
                harness.submit_write(access.address, access.size);
            }
        }
        harness.run_until_complete();

        result.total_cycles = harness.current_cycle();
        const auto& stats = harness.stats();
        result.page_hits = stats.page_hits;
        result.page_empty = stats.page_empty;
        result.page_conflicts = stats.page_conflicts;

        // Estimate per-request latencies from aggregate stats
        uint64_t total = stats.reads + stats.writes;
        if (total > 0 && stats.total_latency > 0) {
            double avg = static_cast<double>(stats.total_latency) / static_cast<double>(total);
            for (size_t i = 0; i < total; ++i) {
                result.all_latencies.add_sample(static_cast<uint64_t>(avg));
            }
        }

        return result;
    }

    void print_comparison(const FidelityComparison& comp) {
        std::cout << std::setw(20) << "Metric"
                  << std::setw(14) << "BEHAVIORAL"
                  << std::setw(14) << "TRANSACT."
                  << std::setw(14) << "CYCLE_ACC" << std::endl;
        std::cout << std::string(62, '-') << std::endl;

        std::cout << std::setw(20) << "Total Cycles"
                  << std::setw(14) << comp.behavioral.total_cycles
                  << std::setw(14) << comp.transactional.total_cycles
                  << std::setw(14) << comp.cycle_accurate.total_cycles << std::endl;

        std::cout << std::setw(20) << "Avg Latency"
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << comp.behavioral.all_latencies.mean()
                  << std::setw(14) << comp.transactional.all_latencies.mean()
                  << std::setw(14) << comp.cycle_accurate.all_latencies.mean() << std::endl;

        std::cout << std::setw(20) << "Page Hits"
                  << std::setw(14) << "-"
                  << std::setw(14) << comp.transactional.page_hits
                  << std::setw(14) << comp.cycle_accurate.page_hits << std::endl;

        std::cout << std::setw(20) << "Page Empty"
                  << std::setw(14) << "-"
                  << std::setw(14) << comp.transactional.page_empty
                  << std::setw(14) << comp.cycle_accurate.page_empty << std::endl;

        std::cout << std::setw(20) << "Page Conflicts"
                  << std::setw(14) << "-"
                  << std::setw(14) << comp.transactional.page_conflicts
                  << std::setw(14) << comp.cycle_accurate.page_conflicts << std::endl;

        std::cout << std::setw(20) << "Cycle Error %"
                  << std::setw(13) << comp.behavioral_cycle_error_pct << "%"
                  << std::setw(13) << comp.transactional_cycle_error_pct << "%"
                  << std::setw(14) << "ref" << std::endl;
    }

    GDDR6MemoryController::Config ca_config_;
    GDDR6BehavioralMemoryController behavioral_mc_;
    GDDR6TransactionalMemoryController transactional_mc_;
};

} // namespace sw::kpu::patterns::gddr6

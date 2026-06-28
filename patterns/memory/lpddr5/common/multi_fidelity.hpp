// patterns/memory/lpddr5/common/multi_fidelity.hpp
//
// Multi-fidelity comparison framework for LPDDR5 memory controller
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

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

#include "lpddr5_configs.hpp"
#include "lpddr5_harness.hpp"
#include "workloads.hpp"

#include <sw/kpu/fidelity/simulation_fidelity.hpp>

namespace sw::kpu::patterns::lpddr5 {

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
    uint32_t mean_read_latency = 80;
    uint32_t mean_write_latency = 90;
    uint32_t latency_variance = 20;
    double page_hit_latency_factor = 0.6;
    double page_empty_latency_factor = 1.0;
    double page_conflict_latency_factor = 1.4;

    static TransactionalCalibration from_cycle_accurate(const FidelityRunResult& ca) {
        TransactionalCalibration cal;
        cal.mean_read_latency = static_cast<uint32_t>(std::round(ca.all_latencies.mean()));
        cal.mean_write_latency = static_cast<uint32_t>(std::round(ca.all_latencies.mean()));
        cal.latency_variance = static_cast<uint32_t>(std::round(ca.all_latencies.stddev()));

        double mean = ca.all_latencies.mean();
        if (mean > 0) {
            cal.page_hit_latency_factor = PAGE_HIT_READ_LATENCY / mean;
            cal.page_empty_latency_factor = PAGE_EMPTY_READ_LATENCY / mean;
            cal.page_conflict_latency_factor = PAGE_CONFLICT_READ_LATENCY / mean;
        }
        return cal;
    }
};

// ============================================================================
// Behavioral Memory Controller
// ============================================================================

class BehavioralMemoryController {
public:
    struct Config {
        uint32_t fixed_read_latency = 100;
        uint32_t fixed_write_latency = 100;
    };

    explicit BehavioralMemoryController(const Config& config) : config_(config) {}
    void set_config(const Config& config) { config_ = config; }

    uint64_t submit_read(uint64_t, uint32_t) {
        uint64_t complete = current_cycle_ + config_.fixed_read_latency;
        pending_.push_back({complete, config_.fixed_read_latency, true});
        return complete;
    }

    uint64_t submit_write(uint64_t, uint32_t) {
        uint64_t complete = current_cycle_ + config_.fixed_write_latency;
        pending_.push_back({complete, config_.fixed_write_latency, false});
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
    void reset() { current_cycle_ = 0; pending_.clear(); completed_.clear(); }

private:
    struct Pending { uint64_t complete_cycle; uint64_t latency; bool is_read; };
    Config config_;
    uint64_t current_cycle_ = 0;
    std::vector<Pending> pending_;
    std::vector<std::pair<uint64_t, bool>> completed_;
};

// ============================================================================
// Transactional Memory Controller
// ============================================================================

class TransactionalMemoryController {
public:
    struct Config {
        uint32_t mean_read_latency = 80;
        uint32_t mean_write_latency = 90;
        uint32_t latency_variance = 20;
        double page_hit_factor = 0.6;
        double page_empty_factor = 1.0;
        double page_conflict_factor = 1.4;
        uint32_t seed = 12345;
    };

    explicit TransactionalMemoryController(const Config& config)
        : config_(config), rng_(config.seed) {}

    void set_config(const Config& config) {
        config_ = config;
        rng_.seed(config.seed);
    }

    uint64_t submit_read(uint64_t address, uint32_t) {
        auto pr = estimate_page_result(address);
        double base = config_.mean_read_latency;
        base *= get_factor(pr);
        base += (rng_() % (2 * config_.latency_variance + 1)) - config_.latency_variance;
        uint64_t latency = static_cast<uint64_t>(std::max(1.0, base));
        uint64_t complete = current_cycle_ + latency;
        pending_.push_back({complete, latency, true, pr});
        update_locality(address);
        return complete;
    }

    uint64_t submit_write(uint64_t address, uint32_t) {
        auto pr = estimate_page_result(address);
        double base = config_.mean_write_latency;
        base *= get_factor(pr);
        base += (rng_() % (2 * config_.latency_variance + 1)) - config_.latency_variance;
        uint64_t latency = static_cast<uint64_t>(std::max(1.0, base));
        uint64_t complete = current_cycle_ + latency;
        pending_.push_back({complete, latency, false, pr});
        update_locality(address);
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

    const auto& completed() const { return completed_; }
    void reset() {
        current_cycle_ = 0;
        pending_.clear();
        completed_.clear();
        last_rows_.clear();
    }

private:
    enum class PageResult { HIT, EMPTY, CONFLICT };

    PageResult estimate_page_result(uint64_t address) {
        uint64_t row = (address >> 6) & 0x3FFF;
        uint8_t bank = (address >> 20) & 0xF;
        auto it = last_rows_.find(bank);
        if (it == last_rows_.end()) return PageResult::EMPTY;
        return it->second == row ? PageResult::HIT : PageResult::CONFLICT;
    }

    void update_locality(uint64_t address) {
        uint64_t row = (address >> 6) & 0x3FFF;
        uint8_t bank = (address >> 20) & 0xF;
        last_rows_[bank] = row;
    }

    double get_factor(PageResult pr) {
        switch (pr) {
            case PageResult::HIT: return config_.page_hit_factor;
            case PageResult::EMPTY: return config_.page_empty_factor;
            case PageResult::CONFLICT: return config_.page_conflict_factor;
        }
        return 1.0;
    }

    struct Pending {
        uint64_t complete_cycle;
        uint64_t latency;
        bool is_read;
        PageResult page_result;
    };
    struct Completed {
        uint64_t latency;
        bool is_read;
        PageResult page_result;
    };

    Config config_;
    std::minstd_rand rng_;
    uint64_t current_cycle_ = 0;
    std::vector<Pending> pending_;
    std::vector<Completed> completed_;
    std::map<uint8_t, uint64_t> last_rows_;
};

// ============================================================================
// Multi-Fidelity Harness
// ============================================================================

class MultiFidelityHarness {
public:
    explicit MultiFidelityHarness(const LPDDR5MemoryController::Config& ca_config)
        : ca_config_(ca_config)
        , behavioral_mc_(BehavioralMemoryController::Config{})
        , transactional_mc_(TransactionalMemoryController::Config{})
    {}

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

    FidelityComparison run_calibrated(const Workload& workload, bool verbose = true) {
        FidelityComparison comp;

        comp.cycle_accurate = run_cycle_accurate(workload);

        auto b_cal = BehavioralCalibration::from_cycle_accurate(comp.cycle_accurate);
        auto t_cal = TransactionalCalibration::from_cycle_accurate(comp.cycle_accurate);

        behavioral_mc_.set_config({b_cal.fixed_read_latency, b_cal.fixed_write_latency});
        transactional_mc_.set_config({
            t_cal.mean_read_latency, t_cal.mean_write_latency,
            t_cal.latency_variance,
            t_cal.page_hit_latency_factor, t_cal.page_empty_latency_factor,
            t_cal.page_conflict_latency_factor, 12345
        });

        comp.behavioral = run_behavioral(workload);
        comp.transactional = run_transactional(workload);
        comp.compute_errors();

        if (verbose) {
            std::cout << "\n=== Calibrated: " << workload.name << " ===" << std::endl;
            print_comparison(comp);
        }

        return comp;
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
        }

        return result;
    }

    FidelityRunResult run_cycle_accurate(const Workload& workload) {
        FidelityRunResult result;
        result.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        result.workload_name = workload.name;

        LPDDR5Harness harness(ca_config_);
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
        std::cout << std::setw(18) << "Metric"
                  << std::setw(14) << "BEHAVIORAL"
                  << std::setw(14) << "TRANSACT."
                  << std::setw(14) << "CYCLE_ACC" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::cout << std::setw(18) << "Total Cycles"
                  << std::setw(14) << comp.behavioral.total_cycles
                  << std::setw(14) << comp.transactional.total_cycles
                  << std::setw(14) << comp.cycle_accurate.total_cycles << std::endl;

        std::cout << std::setw(18) << "Avg Latency"
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << comp.behavioral.all_latencies.mean()
                  << std::setw(14) << comp.transactional.all_latencies.mean()
                  << std::setw(14) << comp.cycle_accurate.all_latencies.mean() << std::endl;

        std::cout << std::setw(18) << "Cycle Error %"
                  << std::setw(13) << comp.behavioral_cycle_error_pct << "%"
                  << std::setw(13) << comp.transactional_cycle_error_pct << "%"
                  << std::setw(14) << "ref" << std::endl;
    }

    LPDDR5MemoryController::Config ca_config_;
    BehavioralMemoryController behavioral_mc_;
    TransactionalMemoryController transactional_mc_;
};

} // namespace sw::kpu::patterns::lpddr5

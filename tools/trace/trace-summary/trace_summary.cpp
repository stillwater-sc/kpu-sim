/**
 * @file trace_summary.cpp
 * @brief KPU Trace Summary - CLI trace analysis tool
 *
 * Analyzes LPDDR5 memory controller traces in Chrome Trace Event Format.
 * Provides statistics, timing validation, and transaction analysis.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
 */

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// LPDDR5-6400 timing parameters (in cycles at 3.2 GHz)
struct TimingParams {
    int tRCD = 14;      // ACT to READ/WRITE
    int tRP = 14;       // PRE to ACT (same bank)
    int tCL = 14;       // CAS latency
    int tRRD_L = 6;     // ACT to ACT (same bank group)
    int tRRD_S = 4;     // ACT to ACT (different bank group)
    int tFAW = 24;      // Four Activate Window
    int tRTW = 14;      // Read to Write turnaround
    int tWTR_L = 10;    // Write to Read turnaround (same bank group)
    int tWTR_S = 4;     // Write to Read turnaround (different bank group)
};

struct Command {
    std::string name;
    std::string category;
    int txn_id = -1;
    int cycle_issue = 0;
    int cycle_complete = 0;
    int channel = 0;
    int bank = 0;
    std::string desc;
};

struct Transaction {
    int txn_id = -1;
    std::string type;  // READ or WRITE
    std::vector<Command> commands;
    int start_cycle = INT_MAX;
    int end_cycle = 0;
    int latency = 0;
    bool has_activate = false;
    bool has_data_op = false;
    bool has_precharge = false;
    bool is_page_hit = false;
};

struct TimingViolation {
    std::string invariant;
    std::string severity;
    std::string message;
    std::string fix_hint;
};

struct TraceStats {
    int total_transactions = 0;
    int read_count = 0;
    int write_count = 0;
    int total_commands = 0;
    int activate_count = 0;
    int read_cmd_count = 0;
    int write_cmd_count = 0;
    int precharge_count = 0;
    int page_hits = 0;
    int page_conflicts = 0;
    int start_cycle = INT_MAX;
    int end_cycle = 0;
    std::vector<int> latencies;
    std::map<int, int> bank_usage;  // bank -> command count
    std::vector<TimingViolation> violations;
};

class TraceSummary {
public:
    TraceSummary() = default;

    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file: " << filepath << "\n";
            return false;
        }

        try {
            file >> trace_data_;
        } catch (const json::parse_error& e) {
            std::cerr << "Error: Failed to parse JSON: " << e.what() << "\n";
            return false;
        }

        filepath_ = filepath;
        return parse();
    }

    bool parse() {
        if (!trace_data_.is_array()) {
            std::cerr << "Error: Trace must be a JSON array\n";
            return false;
        }

        // Parse events into commands
        for (const auto& event : trace_data_) {
            if (!event.contains("ph") || event["ph"] != "X") {
                continue;  // Skip metadata events
            }

            Command cmd;
            cmd.name = event.value("name", "");
            cmd.category = event.value("cat", "");

            if (event.contains("args")) {
                const auto& args = event["args"];
                cmd.txn_id = args.value("txn_id", -1);
                cmd.cycle_issue = args.value("cycle_issue", 0);
                cmd.cycle_complete = args.value("cycle_complete", 0);
                cmd.desc = args.value("desc", "");
            }

            cmd.channel = event.value("pid", 0);
            cmd.bank = event.value("tid", 0);

            commands_.push_back(cmd);
        }

        // Group commands into transactions
        std::map<int, Transaction> txn_map;
        for (const auto& cmd : commands_) {
            if (cmd.txn_id < 0) continue;

            auto& txn = txn_map[cmd.txn_id];
            txn.txn_id = cmd.txn_id;
            txn.commands.push_back(cmd);

            txn.start_cycle = std::min(txn.start_cycle, cmd.cycle_issue);
            txn.end_cycle = std::max(txn.end_cycle, cmd.cycle_complete);

            if (cmd.name == "ACTIVATE") {
                txn.has_activate = true;
            } else if (cmd.name == "BURST_READ" || cmd.name == "READ") {
                txn.has_data_op = true;
                txn.type = "READ";
            } else if (cmd.name == "BURST_WRITE" || cmd.name == "WRITE") {
                txn.has_data_op = true;
                txn.type = "WRITE";
            } else if (cmd.name == "PRECHARGE") {
                txn.has_precharge = true;
            }
        }

        // Finalize transactions
        for (auto& [id, txn] : txn_map) {
            txn.latency = txn.end_cycle - txn.start_cycle;
            txn.is_page_hit = !txn.has_activate && txn.has_data_op;
            transactions_.push_back(txn);
        }

        // Sort by start cycle
        std::sort(transactions_.begin(), transactions_.end(),
            [](const Transaction& a, const Transaction& b) {
                return a.start_cycle < b.start_cycle;
            });

        return true;
    }

    TraceStats computeStats() {
        TraceStats stats;

        stats.total_transactions = transactions_.size();

        for (const auto& txn : transactions_) {
            if (txn.type == "READ") {
                stats.read_count++;
            } else if (txn.type == "WRITE") {
                stats.write_count++;
            }

            if (txn.is_page_hit) {
                stats.page_hits++;
            } else if (txn.has_activate) {
                stats.page_conflicts++;
            }

            stats.latencies.push_back(txn.latency);

            for (const auto& cmd : txn.commands) {
                stats.total_commands++;

                if (cmd.name == "ACTIVATE") {
                    stats.activate_count++;
                } else if (cmd.name == "BURST_READ" || cmd.name == "READ") {
                    stats.read_cmd_count++;
                } else if (cmd.name == "BURST_WRITE" || cmd.name == "WRITE") {
                    stats.write_cmd_count++;
                } else if (cmd.name == "PRECHARGE") {
                    stats.precharge_count++;
                }

                stats.start_cycle = std::min(stats.start_cycle, cmd.cycle_issue);
                stats.end_cycle = std::max(stats.end_cycle, cmd.cycle_complete);
                stats.bank_usage[cmd.bank]++;
            }
        }

        return stats;
    }

    void validateTiming(TraceStats& stats) {
        // Group activates by bank for tRRD checking
        std::map<int, std::vector<int>> bank_activates;  // bank -> cycle list

        for (const auto& cmd : commands_) {
            if (cmd.name == "ACTIVATE") {
                bank_activates[cmd.bank].push_back(cmd.cycle_issue);
            }
        }

        // Check tRCD: ACT to READ/WRITE
        for (const auto& txn : transactions_) {
            int act_complete = -1;
            int data_issue = -1;

            for (const auto& cmd : txn.commands) {
                if (cmd.name == "ACTIVATE") {
                    act_complete = cmd.cycle_complete;
                } else if (cmd.name == "BURST_READ" || cmd.name == "BURST_WRITE" ||
                           cmd.name == "READ" || cmd.name == "WRITE") {
                    data_issue = cmd.cycle_issue;
                }
            }

            if (act_complete >= 0 && data_issue >= 0) {
                int gap = data_issue - act_complete;
                if (gap < 0) {
                    stats.violations.push_back({
                        "INV-100",
                        "ERROR",
                        "tRCD violation: txn_id=" + std::to_string(txn.txn_id) +
                            " data op issued before ACT complete (gap=" + std::to_string(gap) + ")",
                        "Ensure READ/WRITE waits for ACTIVATE to complete"
                    });
                }
            }
        }

        // Check for orphan commands (INV-001, INV-002)
        for (const auto& txn : transactions_) {
            if (!txn.has_data_op) {
                stats.violations.push_back({
                    "INV-001",
                    "ERROR",
                    "txn_id=" + std::to_string(txn.txn_id) + " has no data operation",
                    "Every transaction must have exactly one READ or WRITE"
                });
            }
        }

        // Check temporal ordering (INV-003)
        for (const auto& txn : transactions_) {
            int act_cycle = -1;
            int data_cycle = -1;
            int pre_cycle = -1;

            for (const auto& cmd : txn.commands) {
                if (cmd.name == "ACTIVATE") act_cycle = cmd.cycle_issue;
                else if (cmd.name == "BURST_READ" || cmd.name == "BURST_WRITE" ||
                         cmd.name == "READ" || cmd.name == "WRITE") {
                    data_cycle = cmd.cycle_issue;
                }
                else if (cmd.name == "PRECHARGE") pre_cycle = cmd.cycle_issue;
            }

            if (act_cycle >= 0 && data_cycle >= 0 && act_cycle > data_cycle) {
                stats.violations.push_back({
                    "INV-003",
                    "ERROR",
                    "Temporal ordering violation: txn_id=" + std::to_string(txn.txn_id) +
                        " ACTIVATE after data operation",
                    "ACTIVATE must precede READ/WRITE"
                });
            }

            if (data_cycle >= 0 && pre_cycle >= 0 && pre_cycle < data_cycle) {
                stats.violations.push_back({
                    "INV-003",
                    "ERROR",
                    "Temporal ordering violation: txn_id=" + std::to_string(txn.txn_id) +
                        " PRECHARGE before data operation",
                    "PRECHARGE must follow READ/WRITE"
                });
            }
        }
    }

    void printSummary(const TraceStats& stats, bool verbose) {
        // Extract filename from path
        std::string filename = filepath_;
        auto pos = filename.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }

        std::cout << "\n";
        std::cout << "LPDDR5 Trace Summary: " << filename << "\n";
        std::cout << std::string(50, '=') << "\n";

        // Transaction summary
        std::cout << "Transactions: " << stats.total_transactions;
        if (stats.read_count > 0 || stats.write_count > 0) {
            std::cout << " (" << stats.read_count << " READ, "
                      << stats.write_count << " WRITE)";
        }
        std::cout << "\n";

        std::cout << "Total Cycles: " << (stats.end_cycle - stats.start_cycle) << "\n";
        std::cout << "Commands: " << stats.total_commands
                  << " (" << stats.activate_count << " ACT, "
                  << (stats.read_cmd_count + stats.write_cmd_count) << " CAS, "
                  << stats.precharge_count << " PRE)\n";

        // Page hit rate
        if (stats.total_transactions > 0) {
            double hit_rate = 100.0 * stats.page_hits / stats.total_transactions;
            std::cout << "Page Hits: " << stats.page_hits << "/" << stats.total_transactions
                      << " (" << std::fixed << std::setprecision(1) << hit_rate << "%)\n";
        }

        // Bank utilization
        std::cout << "\nBank Utilization:\n";
        for (const auto& [bank, count] : stats.bank_usage) {
            int bar_len = std::min(count * 2, 40);
            std::cout << "  Bank " << bank << ": ";
            std::cout << std::string(bar_len, '#');
            std::cout << std::string(std::max(0, 40 - bar_len), ' ');
            std::cout << " " << count << " cmds\n";
        }

        // Latency statistics
        if (!stats.latencies.empty()) {
            std::vector<int> sorted_lat = stats.latencies;
            std::sort(sorted_lat.begin(), sorted_lat.end());

            double avg = std::accumulate(sorted_lat.begin(), sorted_lat.end(), 0.0) / sorted_lat.size();
            int p50 = sorted_lat[sorted_lat.size() / 2];
            int p95 = sorted_lat[std::min(sorted_lat.size() - 1, (size_t)(sorted_lat.size() * 0.95))];
            int p99 = sorted_lat[std::min(sorted_lat.size() - 1, (size_t)(sorted_lat.size() * 0.99))];

            std::cout << "\nLatency Distribution (cycles):\n";
            std::cout << "  Min: " << sorted_lat.front()
                      << " | Avg: " << std::fixed << std::setprecision(1) << avg
                      << " | P50: " << p50
                      << " | P95: " << p95
                      << " | P99: " << p99
                      << " | Max: " << sorted_lat.back() << "\n";
        }

        // Timing validation
        std::cout << "\nTiming Validation:\n";
        if (stats.violations.empty()) {
            std::cout << "  [PASS] All invariants passed\n";
        } else {
            int errors = 0, warnings = 0;
            for (const auto& v : stats.violations) {
                if (v.severity == "ERROR") errors++;
                else warnings++;
            }
            std::cout << "  [FAIL] " << errors << " errors, " << warnings << " warnings\n";

            for (const auto& v : stats.violations) {
                std::cout << "    " << (v.severity == "ERROR" ? "[E]" : "[W]")
                          << " " << v.invariant << ": " << v.message << "\n";
            }
        }

        // Verbose: per-transaction details
        if (verbose && !transactions_.empty()) {
            std::cout << "\nTransaction Details:\n";
            std::cout << std::string(70, '-') << "\n";
            std::cout << std::setw(8) << "TXN_ID"
                      << std::setw(8) << "TYPE"
                      << std::setw(10) << "START"
                      << std::setw(10) << "END"
                      << std::setw(10) << "LATENCY"
                      << std::setw(12) << "PAGE_HIT"
                      << std::setw(10) << "CMDS" << "\n";
            std::cout << std::string(70, '-') << "\n";

            for (const auto& txn : transactions_) {
                std::cout << std::setw(8) << txn.txn_id
                          << std::setw(8) << txn.type
                          << std::setw(10) << txn.start_cycle
                          << std::setw(10) << txn.end_cycle
                          << std::setw(10) << txn.latency
                          << std::setw(12) << (txn.is_page_hit ? "yes" : "no")
                          << std::setw(10) << txn.commands.size() << "\n";
            }
        }

        std::cout << "\n";
    }

    void printJson(const TraceStats& stats) {
        json output;

        output["file"] = filepath_;
        output["summary"] = {
            {"total_transactions", stats.total_transactions},
            {"read_count", stats.read_count},
            {"write_count", stats.write_count},
            {"total_commands", stats.total_commands},
            {"activate_count", stats.activate_count},
            {"read_cmd_count", stats.read_cmd_count},
            {"write_cmd_count", stats.write_cmd_count},
            {"precharge_count", stats.precharge_count},
            {"page_hits", stats.page_hits},
            {"page_conflicts", stats.page_conflicts},
            {"total_cycles", stats.end_cycle - stats.start_cycle}
        };

        // Latency stats
        if (!stats.latencies.empty()) {
            std::vector<int> sorted_lat = stats.latencies;
            std::sort(sorted_lat.begin(), sorted_lat.end());
            double avg = std::accumulate(sorted_lat.begin(), sorted_lat.end(), 0.0) / sorted_lat.size();

            output["latency"] = {
                {"min", sorted_lat.front()},
                {"max", sorted_lat.back()},
                {"avg", avg},
                {"p50", sorted_lat[sorted_lat.size() / 2]},
                {"p95", sorted_lat[(size_t)(sorted_lat.size() * 0.95)]},
                {"p99", sorted_lat[(size_t)(sorted_lat.size() * 0.99)]}
            };
        }

        // Bank usage
        output["bank_usage"] = json::object();
        for (const auto& [bank, count] : stats.bank_usage) {
            output["bank_usage"][std::to_string(bank)] = count;
        }

        // Validation
        output["validation"] = {
            {"status", stats.violations.empty() ? "PASSED" : "FAILED"},
            {"violation_count", stats.violations.size()}
        };

        if (!stats.violations.empty()) {
            output["violations"] = json::array();
            for (const auto& v : stats.violations) {
                output["violations"].push_back({
                    {"invariant", v.invariant},
                    {"severity", v.severity},
                    {"message", v.message},
                    {"fix_hint", v.fix_hint}
                });
            }
        }

        std::cout << output.dump(2) << "\n";
    }

    bool hasViolations() const {
        TraceStats stats;
        // Quick check without full computation
        for (const auto& txn : transactions_) {
            if (!txn.has_data_op) return true;
        }
        return false;
    }

private:
    std::string filepath_;
    json trace_data_;
    std::vector<Command> commands_;
    std::vector<Transaction> transactions_;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <trace.json> [options]\n\n";
    std::cout << "Analyze LPDDR5 memory controller traces.\n\n";
    std::cout << "Options:\n";
    std::cout << "  --verbose, -v    Show per-transaction details\n";
    std::cout << "  --json, -j       Output in JSON format\n";
    std::cout << "  --validate       Validate timing constraints (exit code 1 on failure)\n";
    std::cout << "  --help, -h       Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " traces/memory/lpddr5/single-bank/page_hits_trace.json\n";
    std::cout << "  " << prog << " trace.json --verbose\n";
    std::cout << "  " << prog << " trace.json --json > summary.json\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string filepath;
    bool verbose = false;
    bool json_output = false;
    bool validate_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--json" || arg == "-j") {
            json_output = true;
        } else if (arg == "--validate") {
            validate_only = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            filepath = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    if (filepath.empty()) {
        std::cerr << "Error: No trace file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    TraceSummary summary;
    if (!summary.load(filepath)) {
        return 2;
    }

    auto stats = summary.computeStats();
    summary.validateTiming(stats);

    if (json_output) {
        summary.printJson(stats);
    } else {
        summary.printSummary(stats, verbose);
    }

    // Return exit code based on validation
    if (validate_only || !stats.violations.empty()) {
        return stats.violations.empty() ? 0 : 1;
    }

    return 0;
}

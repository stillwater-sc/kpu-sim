// src/calibration/calibration_storage.cpp
//
// Calibration parameter storage implementation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <sw/kpu/calibration/calibration_storage.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace sw::kpu::calibration {

using json = nlohmann::json;

// ============================================================================
// JSON Serialization Helpers
// ============================================================================

namespace {

// Helper to get value with default
template<typename T>
T get_or(const json& j, const std::string& key, T default_value) {
    if (j.contains(key) && !j[key].is_null()) {
        return j[key].get<T>();
    }
    return default_value;
}

CycleAccurateReference parse_ca_reference(const json& j) {
    CycleAccurateReference ref;
    ref.total_requests = get_or<uint64_t>(j, "total_requests", 0);
    ref.total_cycles = get_or<uint64_t>(j, "total_cycles", 0);
    ref.mean_read_latency_cycles = get_or<double>(j, "mean_read_latency_cycles", 0.0);
    ref.mean_write_latency_cycles = get_or<double>(j, "mean_write_latency_cycles", 0.0);
    ref.latency_std_dev_cycles = get_or<double>(j, "latency_std_dev_cycles", 0.0);
    ref.page_hit_rate = get_or<double>(j, "page_hit_rate", 0.0);
    ref.page_empty_rate = get_or<double>(j, "page_empty_rate", 0.0);
    ref.page_conflict_rate = get_or<double>(j, "page_conflict_rate", 0.0);
    return ref;
}

json serialize_ca_reference(const CycleAccurateReference& ref) {
    return json{
        {"total_requests", ref.total_requests},
        {"total_cycles", ref.total_cycles},
        {"mean_read_latency_cycles", ref.mean_read_latency_cycles},
        {"mean_write_latency_cycles", ref.mean_write_latency_cycles},
        {"latency_std_dev_cycles", ref.latency_std_dev_cycles},
        {"page_hit_rate", ref.page_hit_rate},
        {"page_empty_rate", ref.page_empty_rate},
        {"page_conflict_rate", ref.page_conflict_rate}
    };
}

BehavioralCalibration parse_behavioral(const json& j) {
    BehavioralCalibration cal;
    cal.fixed_read_latency_cycles = get_or<uint32_t>(j, "fixed_read_latency_cycles", 100);
    cal.fixed_write_latency_cycles = get_or<uint32_t>(j, "fixed_write_latency_cycles", 100);
    return cal;
}

json serialize_behavioral(const BehavioralCalibration& cal) {
    return json{
        {"fixed_read_latency_cycles", cal.fixed_read_latency_cycles},
        {"fixed_write_latency_cycles", cal.fixed_write_latency_cycles}
    };
}

TransactionalCalibration parse_transactional(const json& j) {
    TransactionalCalibration cal;
    cal.mean_read_latency_cycles = get_or<uint32_t>(j, "mean_read_latency_cycles", 80);
    cal.mean_write_latency_cycles = get_or<uint32_t>(j, "mean_write_latency_cycles", 90);
    cal.latency_std_dev_cycles = get_or<uint32_t>(j, "latency_std_dev_cycles", 20);
    cal.page_hit_factor = get_or<double>(j, "page_hit_factor", 0.6);
    cal.page_empty_factor = get_or<double>(j, "page_empty_factor", 1.0);
    cal.page_conflict_factor = get_or<double>(j, "page_conflict_factor", 1.4);
    // Per-scenario latencies (preferred when available)
    cal.page_hit_latency_cycles = get_or<uint32_t>(j, "page_hit_latency_cycles", 0);
    cal.page_empty_latency_cycles = get_or<uint32_t>(j, "page_empty_latency_cycles", 0);
    cal.page_conflict_latency_cycles = get_or<uint32_t>(j, "page_conflict_latency_cycles", 0);
    return cal;
}

json serialize_transactional(const TransactionalCalibration& cal) {
    return json{
        {"mean_read_latency_cycles", cal.mean_read_latency_cycles},
        {"mean_write_latency_cycles", cal.mean_write_latency_cycles},
        {"latency_std_dev_cycles", cal.latency_std_dev_cycles},
        {"page_hit_factor", cal.page_hit_factor},
        {"page_empty_factor", cal.page_empty_factor},
        {"page_conflict_factor", cal.page_conflict_factor},
        {"page_hit_latency_cycles", cal.page_hit_latency_cycles},
        {"page_empty_latency_cycles", cal.page_empty_latency_cycles},
        {"page_conflict_latency_cycles", cal.page_conflict_latency_cycles}
    };
}

ValidationResults parse_validation(const json& j) {
    ValidationResults val;
    val.behavioral_latency_error_pct = get_or<double>(j, "behavioral_latency_error_pct", 0.0);
    val.behavioral_cycle_error_pct = get_or<double>(j, "behavioral_cycle_error_pct", 0.0);
    val.transactional_latency_error_pct = get_or<double>(j, "transactional_latency_error_pct", 0.0);
    val.transactional_cycle_error_pct = get_or<double>(j, "transactional_cycle_error_pct", 0.0);
    val.max_acceptable_error_pct = get_or<double>(j, "max_acceptable_error_pct", 5.0);
    val.status = get_or<std::string>(j, "status", "NOT_VALIDATED");
    return val;
}

json serialize_validation(const ValidationResults& val) {
    return json{
        {"behavioral_latency_error_pct", val.behavioral_latency_error_pct},
        {"behavioral_cycle_error_pct", val.behavioral_cycle_error_pct},
        {"transactional_latency_error_pct", val.transactional_latency_error_pct},
        {"transactional_cycle_error_pct", val.transactional_cycle_error_pct},
        {"max_acceptable_error_pct", val.max_acceptable_error_pct},
        {"status", val.status}
    };
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

std::optional<CalibrationData> load_calibration(const std::filesystem::path& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            return std::nullopt;
        }

        json j = json::parse(file);

        CalibrationData data;
        data.version = get_or<std::string>(j, "version", "1.0");
        data.technology = get_or<std::string>(j, "technology", "LPDDR5");
        data.speed_grade_mt_s = get_or<uint32_t>(j, "speed_grade_mt_s", 6400);
        data.calibration_date = get_or<std::string>(j, "calibration_date", "");
        data.description = get_or<std::string>(j, "description", "");

        if (j.contains("source_patterns") && j["source_patterns"].is_array()) {
            for (const auto& p : j["source_patterns"]) {
                data.source_patterns.push_back(p.get<std::string>());
            }
        }

        if (j.contains("cycle_accurate_reference")) {
            data.cycle_accurate_reference = parse_ca_reference(j["cycle_accurate_reference"]);
        }

        if (j.contains("behavioral")) {
            data.behavioral = parse_behavioral(j["behavioral"]);
        }

        if (j.contains("transactional")) {
            data.transactional = parse_transactional(j["transactional"]);
        }

        if (j.contains("validation")) {
            data.validation = parse_validation(j["validation"]);
        }

        return data;

    } catch (const std::exception&) {
        return std::nullopt;
    }
}

bool save_calibration(const CalibrationData& data, const std::filesystem::path& path) {
    try {
        json j;
        j["version"] = data.version;
        j["technology"] = data.technology;
        j["speed_grade_mt_s"] = data.speed_grade_mt_s;
        j["calibration_date"] = data.calibration_date;
        j["description"] = data.description;
        j["source_patterns"] = data.source_patterns;
        j["cycle_accurate_reference"] = serialize_ca_reference(data.cycle_accurate_reference);
        j["behavioral"] = serialize_behavioral(data.behavioral);
        j["transactional"] = serialize_transactional(data.transactional);
        j["validation"] = serialize_validation(data.validation);

        // Create parent directories if needed
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }

        file << j.dump(2);
        return true;

    } catch (const std::exception&) {
        return false;
    }
}

std::optional<std::filesystem::path> find_calibration_file(
    const std::filesystem::path& config_dir,
    const std::string& technology,
    uint32_t speed_grade)
{
    auto filename = make_calibration_filename(technology, speed_grade);
    auto path = config_dir / filename;

    if (std::filesystem::exists(path)) {
        return path;
    }

    return std::nullopt;
}

std::filesystem::path get_calibration_dir() {
    // Check environment variable first
    const char* env_dir = std::getenv("KPUSIM_CALIBRATION_DIR");
    if (env_dir != nullptr) {
        return std::filesystem::path(env_dir);
    }

    // Default to configs/calibration relative to current directory
    return std::filesystem::current_path() / "configs" / "calibration";
}

std::string make_calibration_filename(const std::string& technology, uint32_t speed_grade) {
    std::string tech_lower = technology;
    std::transform(tech_lower.begin(), tech_lower.end(), tech_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return tech_lower + "_" + std::to_string(speed_grade) + ".json";
}

} // namespace sw::kpu::calibration

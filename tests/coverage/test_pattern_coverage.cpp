// ============================================================================
// tests/coverage/test_pattern_coverage.cpp
// Machine-checkable CSP pattern-coverage matrix (issue #93)
//
// Enforces the schema and consistency of tests/coverage/pattern_coverage.json
// and asserts milestone gates: a milestone marked "achieved" must have every
// required (operator, stage) at "done". Pattern epics update the manifest as
// capabilities land; the model-validation epic (E18, #87) asserts the full
// matrix through these gates.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <set>
#include <string>

namespace {

nlohmann::json load_manifest() {
    std::ifstream file(COVERAGE_MANIFEST_PATH);
    REQUIRE(file.is_open());
    nlohmann::json manifest;
    file >> manifest;
    return manifest;
}

} // namespace

TEST_CASE("Coverage manifest is schema-consistent", "[coverage]") {
    auto manifest = load_manifest();

    // Pattern classes P1..P9 are all defined
    const auto& patterns = manifest.at("pattern_classes");
    for (int p = 1; p <= 9; ++p) {
        std::string key = "P" + std::to_string(p);
        INFO("pattern class " << key);
        REQUIRE(patterns.contains(key));
    }

    std::set<std::string> valid_stages;
    for (const auto& s : manifest.at("stages")) {
        valid_stages.insert(s.get<std::string>());
    }
    std::set<std::string> valid_statuses;
    for (const auto& s : manifest.at("statuses")) {
        valid_statuses.insert(s.get<std::string>());
    }
    REQUIRE(valid_stages.size() == 5);
    REQUIRE(valid_statuses == std::set<std::string>{"done", "partial", "missing"});

    std::set<std::string> names;
    std::set<std::string> referenced_patterns;
    const std::set<std::string> valid_models{"cv", "llm", "jepa"};

    for (const auto& op : manifest.at("operators")) {
        std::string name = op.at("name").get<std::string>();
        INFO("operator " << name);

        // Unique names
        REQUIRE(names.insert(name).second);

        // Valid, non-empty pattern references
        REQUIRE_FALSE(op.at("patterns").empty());
        for (const auto& p : op.at("patterns")) {
            REQUIRE(patterns.contains(p.get<std::string>()));
            referenced_patterns.insert(p.get<std::string>());
        }

        // Epic reference is a plausible issue number
        REQUIRE(op.at("epic").get<int>() > 0);

        // At least one model family
        REQUIRE_FALSE(op.at("models").empty());
        for (const auto& m : op.at("models")) {
            REQUIRE(valid_models.count(m.get<std::string>()) == 1);
        }

        // Every stage present with a valid status
        const auto& stages = op.at("stages");
        REQUIRE(stages.size() == valid_stages.size());
        for (const auto& stage : valid_stages) {
            INFO("stage " << stage);
            REQUIRE(stages.contains(stage));
            REQUIRE(valid_statuses.count(
                        stages.at(stage).get<std::string>()) == 1);
        }
    }

    // Every pattern class is exercised by at least one operator
    for (int p = 1; p <= 9; ++p) {
        std::string key = "P" + std::to_string(p);
        INFO("pattern class " << key << " must be referenced by an operator");
        REQUIRE(referenced_patterns.count(key) == 1);
    }
}

TEST_CASE("Milestone gates hold", "[coverage][milestones]") {
    auto manifest = load_manifest();

    // Index operator stages by name
    std::map<std::string, nlohmann::json> ops;
    for (const auto& op : manifest.at("operators")) {
        ops[op.at("name").get<std::string>()] = op.at("stages");
    }

    for (const auto& milestone : manifest.at("milestones")) {
        std::string id = milestone.at("id").get<std::string>();
        bool achieved = milestone.at("achieved").get<bool>();

        for (const auto& req : milestone.at("requires")) {
            std::string op_name = req.at("operator").get<std::string>();
            std::string stage = req.at("stage").get<std::string>();
            INFO("milestone " << id << " requires " << op_name << "." << stage);

            // Requirements must reference real operators and stages
            REQUIRE(ops.count(op_name) == 1);
            REQUIRE(ops[op_name].contains(stage));

            // An achieved milestone's requirements must all be done -
            // this is the gate that keeps the matrix honest: you cannot
            // mark a milestone achieved while a required capability is
            // partial or missing, and you cannot regress a capability
            // below an achieved milestone without this test failing.
            if (achieved) {
                REQUIRE(ops[op_name].at(stage).get<std::string>() == "done");
            }
        }
    }
}

TEST_CASE("Coverage summary is reportable", "[coverage]") {
    auto manifest = load_manifest();

    size_t total = 0;
    size_t done = 0;
    for (const auto& op : manifest.at("operators")) {
        for (const auto& [stage, status] : op.at("stages").items()) {
            ++total;
            if (status.get<std::string>() == "done") ++done;
        }
    }
    // The matrix must never be empty, and the baseline (matmul + M1) work
    // guarantees a nonzero floor of completed cells
    REQUIRE(total >= 100);
    REQUIRE(done >= 6);

    WARN("Pattern coverage: " << done << "/" << total
         << " operator-stage cells done ("
         << (100.0 * static_cast<double>(done) / static_cast<double>(total))
         << "%)");
}

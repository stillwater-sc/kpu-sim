// ============================================================================
// tests/program/test_domain_flow_record.cpp
// The domain-flow record (docs/plans/domain-flow-debugger.md step 2; #363).
//   - It is written by the oracle's own run, and watching changes no value.
//   - Its steps are the L-B stepper's steps (the same computation, repeated).
//   - It round-trips through write_dflow / read_dflow.
//   - A step limit cuts steps.bin and nothing else.
//   - A program L-B refuses is recorded up to the refusal, which the record names.
//   - Values (step 4): every operand's final tile values are L-B's result, bit for bit; a selected
//     tile is recorded wherever each action leaves it; a byte cap stops them and says where.
//   - The L-T1 overlay (step 5): L-T1's legs joined on the action, every slot wait a Load's and
//     adding up to L-T1's, and each final tile's proximity to L-B (tile_check, per tile).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/csp_step.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace lang = sw::kpu::program::csp::lang;
using record::DomainFlowRecord;
using record::DomainFlowRecorder;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

const char* kPrograms[] = {"matmul_48x48x48_t16.csp", "linear_64_t16_relu.csp", "lu_64_t16.csp"};

struct Recorded {
    DomainFlowRecord rec;
    driver::CspReference ref;
};

Recorded record_of(const std::string& text, std::uint64_t limit = 0, record::ValueSelection values = {}) {
    const lang::Program ast = lang::parse(text);
    DomainFlowRecorder r(ast, text, limit, values);
    Recorded out;
    out.ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000, &r);
    out.rec = r.finish(out.ref.values ? &*out.ref.values : nullptr);
    return out;
}

// A snapshot's elements.
std::vector<float> elements_of(const DomainFlowRecord& r, const DomainFlowRecord::Snapshot& s) {
    return {r.elements.begin() + static_cast<std::ptrdiff_t>(s.first),
            r.elements.begin() + static_cast<std::ptrdiff_t>(s.first + std::uint64_t(s.rows) * s.cols)};
}

// The tile's values in an operand.
std::vector<float> tile_of(const TensorOperand& o, Dim ti, Dim tj) {
    std::vector<float> v;
    for (Dim r = o.row_begin(ti); r < o.row_end(ti); ++r)
        for (Dim c = o.col_begin(tj); c < o.col_end(tj); ++c) v.push_back(o.at(r, c));
    return v;
}

std::filesystem::path scratch(const std::string& name) {
    auto p = std::filesystem::temp_directory_path() / ("dflow-test-" + name);
    std::filesystem::remove_all(p);
    return p;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

}  // namespace

TEST_CASE("dflow: the record is the oracle's run, and watching it changes nothing", "[dflow][record]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        const std::string text = slurp(std::string("tests/program/corpus/") + name);
        const lang::Program ast = lang::parse(text);
        const auto plain = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
        const Recorded r = record_of(text);
        REQUIRE(r.ref.values.has_value());
        CHECK(r.ref.replay_mismatch.empty());
        CHECK(r.rec.actions == plain.actions);
        CHECK(r.rec.steps.size() == r.rec.actions);
        CHECK_FALSE(r.rec.cut);
        CHECK_FALSE(r.rec.refusal.has_value());
        for (const auto& op : plain.values->operand_order())
            CHECK(same_bits(r.ref.values->operand(op).values, plain.values->operand(op).values));
        // The trace is unchanged too: the observer only reads.
        REQUIRE(r.ref.trace.has_value());
        CHECK(r.ref.trace->ops().size() == plain.trace->ops().size());
    }
}

TEST_CASE("dflow: every step is the L-B stepper's step", "[dflow][record]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        const std::string text = slurp(std::string("tests/program/corpus/") + name);
        const lang::Program ast = lang::parse(text);
        const Recorded r = record_of(text);
        driver::CspBehavioralStepper st(ast, driver::csp_inputs(ast), driver::CspBehavioralStepper::Count::Lazy);
        std::size_t i = 0;
        while (st.step()) {
            const auto& s = st.current();
            REQUIRE(i < r.rec.steps.size());
            const auto& d = r.rec.steps[i];
            CAPTURE(i);
            CHECK(d.action == s.action);
            CHECK(d.kind == static_cast<std::uint8_t>(s.kind));
            CHECK(r.rec.tile_name(d.tile) == s.tile.to_string());
            CHECK((d.process == record::kNone8) == !s.process.has_value());
            CHECK((d.from == record::kNone8) == !s.from.has_value());
            CHECK(d.residency == (s.residency == csp::kNone ? record::kNone64 : s.residency));
            CHECK(bool(d.flags & DomainFlowRecord::kOpens) == s.opens);
            CHECK(bool(d.flags & DomainFlowRecord::kCloses) == s.closes);
            CHECK(bool(d.flags & DomainFlowRecord::kRetains) == s.retains);
            CHECK(bool(d.flags & DomainFlowRecord::kHasPlace) == s.place.has_value());
            if (s.place) CHECK(d.slot == s.place->slot);
            for (std::size_t c = 0; c < csp::kChannels; ++c) CHECK(d.held[c] == s.held[c]);
            // The position, through the interned tables.
            const auto& w = r.rec.wheres.at(d.where);
            CHECK(static_cast<int>(w.line) == s.where->line);
            REQUIRE(w.loop_count == s.where->loops.size());
            for (std::size_t k = 0; k < w.loop_count; ++k) {
                const auto& l = r.rec.loops.at(w.loop_first + k);
                CHECK(r.rec.vars.at(l.var) == s.where->loops[k].var);
                CHECK(l.value == s.where->loops[k].value);
            }
            ++i;
        }
        CHECK(i == r.rec.steps.size());
    }
}

TEST_CASE("dflow: the bundle round-trips", "[dflow][record]") {
    const std::string text = slurp("tests/program/corpus/linear_64_t16_relu.csp");
    const DomainFlowRecord a = record_of(text).rec;
    const auto dir = scratch("roundtrip");
    record::write_dflow(a, dir.string());
    const DomainFlowRecord b = record::read_dflow(dir.string());
    CHECK(b.program_text == a.program_text);
    CHECK(b.program_name == a.program_name);
    CHECK((b.l3 == a.l3 && b.banks == a.banks && b.place == a.place));
    CHECK((b.actions == a.actions && b.loads == a.loads && b.stores == a.stores));
    CHECK(b.vars == a.vars);
    REQUIRE(b.steps.size() == a.steps.size());
    for (std::size_t i = 0; i < a.steps.size(); ++i) {
        const auto &x = a.steps[i], &y = b.steps[i];
        CHECK((x.action == y.action && x.kind == y.kind && x.process == y.process && x.from == y.from &&
               x.to == y.to && x.tile == y.tile && x.residency == y.residency && x.flags == y.flags &&
               x.where == y.where && x.held == y.held && x.l3_tile == y.l3_tile && x.slot == y.slot &&
               x.bank == y.bank));
    }
    REQUIRE(b.residencies.size() == a.residencies.size());
    for (std::size_t i = 0; i < a.residencies.size(); ++i) {
        const auto &x = a.residencies[i], &y = b.residencies[i];
        CHECK((x.tile == y.tile && x.open == y.open && x.close == y.close && x.slot == y.slot &&
               x.bank == y.bank && x.flags == y.flags));
    }
    REQUIRE(b.iterations.size() == a.iterations.size());
    for (std::size_t i = 0; i < a.iterations.size(); ++i) {
        const auto &x = a.iterations[i], &y = b.iterations[i];
        CHECK((x.depth == y.depth && x.line == y.line && x.value == y.value && x.first == y.first &&
               x.last == y.last && x.actions == y.actions && x.loads == y.loads && x.stores == y.stores &&
               x.peak == y.peak));
    }
    CHECK(b.wheres.size() == a.wheres.size());
    CHECK(b.loops.size() == a.loops.size());
    CHECK(b.tiles.size() == a.tiles.size());
    std::filesystem::remove_all(dir);
}

TEST_CASE("dflow: values round-trip, bit for bit", "[dflow][record][values]") {
    const std::string text = slurp("tests/program/corpus/linear_64_t16_relu.csp");
    const DomainFlowRecord a = record_of(text, 0, record::parse_value_selection("C[1,2],b")).rec;
    REQUIRE_FALSE(a.snapshots.empty());
    const auto dir = scratch("values");
    record::write_dflow(a, dir.string());
    const DomainFlowRecord b = record::read_dflow(dir.string());
    REQUIRE(b.snapshots.size() == a.snapshots.size());
    for (std::size_t i = 0; i < a.snapshots.size(); ++i) {
        const auto &x = a.snapshots[i], &y = b.snapshots[i];
        CHECK((x.step == y.step && x.first == y.first && x.tile == y.tile && x.rows == y.rows && x.cols == y.cols &&
               x.chan == y.chan));
    }
    CHECK(same_bits(b.elements, a.elements));
    CHECK(b.values_selected == "C[1,2],b");
    CHECK(b.values_cap == a.values_cap);
    CHECK_FALSE(b.values_cut_at.has_value());
    CHECK(b.finals_note.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("dflow: every operand's final values are L-B's result", "[dflow][record][values]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        const Recorded r = record_of(slurp(std::string("tests/program/corpus/") + name));
        REQUIRE(r.rec.finals_note.empty());
        std::size_t finals = 0, tiles = 0;
        for (const auto& op : r.ref.values->operand_order()) {
            const auto& o = r.ref.values->operand(op);
            tiles += o.n_tile_rows() * o.n_tile_cols();
        }
        for (const auto& s : r.rec.snapshots) {
            REQUIRE(s.step == record::kNone64);      // nothing selected: finals only
            ++finals;
            const auto& t = r.rec.tiles.at(s.tile);
            const auto& o = r.ref.values->operand(r.rec.operands.at(t.operand).name);
            CHECK(same_bits(elements_of(r.rec, s), tile_of(o, t.ti, t.tj)));
        }
        CHECK(finals == tiles);
    }
}

TEST_CASE("dflow: a selected tile is recorded wherever each action leaves it", "[dflow][record][values]") {
    // matmul_48x48x48_t16: C[1,2] is three accumulating calls, then drained, written back, stored.
    const Recorded r = record_of(slurp("tests/program/corpus/matmul_48x48x48_t16.csp"), 0,
                                 record::parse_value_selection("C[1,2]"));
    std::vector<const DomainFlowRecord::Snapshot*> mine;
    const DomainFlowRecord::Snapshot* final = nullptr;
    for (const auto& s : r.rec.snapshots) {
        if (r.rec.tile_name(s.tile) != "C[1,2]") {
            CHECK(s.step == record::kNone64);        // only the selected tile has step snapshots
            continue;
        }
        if (s.step == record::kNone64) final = &s; else mine.push_back(&s);
    }
    REQUIRE(final != nullptr);
    std::vector<std::string> kinds;
    for (const auto* s : mine) kinds.push_back(csp::to_string(static_cast<csp::Action::Kind>(r.rec.steps.at(s->step).kind)));
    CHECK(kinds == std::vector<std::string>{"CALL", "CALL", "CALL", "DRAIN", "WRITEBACK", "STORE"});
    CHECK(mine[0]->chan == static_cast<std::uint8_t>(csp::Chan::Cf));
    CHECK(mine[3]->chan == static_cast<std::uint8_t>(csp::Chan::L2));
    CHECK(mine[4]->chan == static_cast<std::uint8_t>(csp::Chan::L3));
    CHECK(mine[5]->chan == static_cast<std::uint8_t>(csp::Chan::Dram));
    // The accumulator changes with each call, and what is stored is the result.
    CHECK_FALSE(same_bits(elements_of(r.rec, *mine[0]), elements_of(r.rec, *mine[1])));
    CHECK(same_bits(elements_of(r.rec, *mine[2]), elements_of(r.rec, *mine[5])));
    CHECK(same_bits(elements_of(r.rec, *mine[5]), elements_of(r.rec, *final)));
}

TEST_CASE("dflow: the values cap stops snapshots and says where; finals come first", "[dflow][record][values]") {
    const std::string text = slurp("tests/program/corpus/matmul_48x48x48_t16.csp");
    // Finals are 3 x 48 x 48 floats = 27,648 bytes; a 16x16 snapshot is 1,024.
    auto sel = record::parse_value_selection("all");
    sel.cap_bytes = 27'648 + 5 * 1'024;
    const DomainFlowRecord r = record_of(text, 0, sel).rec;
    CHECK(r.finals_note.empty());
    REQUIRE(r.values_cut_at.has_value());
    std::size_t steps = 0;
    for (const auto& s : r.snapshots) steps += s.step != record::kNone64;
    CHECK(steps == 5);
    // Below the finals, nothing fits: no finals, and the note says why.
    auto tiny = record::parse_value_selection("all");
    tiny.cap_bytes = 1'000;
    const DomainFlowRecord t = record_of(text, 0, tiny).rec;
    CHECK_THAT(t.finals_note, ContainsSubstring("cap"));
    for (const auto& s : t.snapshots) CHECK(s.step != record::kNone64);
}

TEST_CASE("dflow: the value selection parses tiles, operands and all", "[dflow][record][values]") {
    const auto v = record::parse_value_selection("C[2,3], A[02,0],B");
    CHECK_FALSE(v.all);
    CHECK(v.tiles == std::vector<std::string>{"C[2,3]", "A[2,0]"});
    CHECK(v.operands == std::vector<std::string>{"B"});
    CHECK(v.selects(TileCoord{"B", 4, 4}));
    CHECK(v.selects(TileCoord{"C", 2, 3}));
    CHECK_FALSE(v.selects(TileCoord{"C", 3, 2}));
    CHECK(record::parse_value_selection("all").all);
    CHECK_THROWS_AS(record::parse_value_selection("C[2]"), std::invalid_argument);
    CHECK_THROWS_AS(record::parse_value_selection("C[2,x]"), std::invalid_argument);
    CHECK_THROWS_AS(record::parse_value_selection("C[1,1],,A"), std::invalid_argument);
    CHECK_THROWS_AS(record::parse_value_selection("C[99999999999999999999,0]"), std::invalid_argument);
    const lang::Program ast = lang::parse(slurp("tests/program/corpus/matmul_48x48x48_t16.csp"));
    CHECK_THROWS_WITH(DomainFlowRecorder(ast, "", 0, record::parse_value_selection("Q[0,0]")),
                      ContainsSubstring("names no declared operand"));
}

TEST_CASE("dflow: the iteration summary covers the run", "[dflow][record]") {
    // matmul_48x48x48_t16: for j in 0..3 { ... for i in 0..3 { ... for k in 0..3 { call } } }
    const DomainFlowRecord r = record_of(slurp("tests/program/corpus/matmul_48x48x48_t16.csp")).rec;
    std::size_t by_depth[4] = {};
    for (const auto& it : r.iterations) {
        REQUIRE(it.depth < 4);
        ++by_depth[it.depth];
    }
    CHECK(by_depth[0] == 1);
    CHECK(by_depth[1] == 3);
    CHECK(by_depth[2] == 9);
    CHECK(by_depth[3] == 27);
    const auto& program = r.iterations.front();
    CHECK(program.depth == 0);
    CHECK((program.first == 0 && program.last + 1 == r.actions && program.actions == r.actions));
    CHECK((program.loads == r.loads && program.stores == r.stores));
    std::uint64_t j_actions = 0, j_loads = 0;
    for (const auto& it : r.iterations)
        if (it.depth == 1) {
            j_actions += it.actions;
            j_loads += it.loads;
        }
    CHECK(j_actions == r.actions);          // every action of this program is inside a j iteration
    CHECK(j_loads == r.loads);
}

TEST_CASE("dflow: a step limit cuts steps.bin, and only it", "[dflow][record]") {
    const std::string text = slurp("tests/program/corpus/matmul_48x48x48_t16.csp");
    const DomainFlowRecord all = record_of(text).rec;
    const DomainFlowRecord cut = record_of(text, 50).rec;
    CHECK(cut.cut);
    CHECK(cut.steps.size() == 50);
    CHECK(cut.actions == all.actions);
    CHECK(cut.residencies.size() == all.residencies.size());
    REQUIRE(cut.iterations.size() == all.iterations.size());
    for (std::size_t i = 0; i < all.iterations.size(); ++i) CHECK(cut.iterations[i].actions == all.iterations[i].actions);
    for (std::size_t i = 0; i < 50; ++i) CHECK(cut.steps[i].held == all.steps[i].held);
}

TEST_CASE("dflow: a refused program is recorded up to its refusal", "[dflow][record]") {
    const std::string text = R"(csp 1.0
program bad machine flat(l3 = 2) {
  tensor A[16,16] tile 16x16 in;
  resident A[0, 0];
  release A[0, 0];
  release A[0, 0];
}
)";
    const lang::Program ast = lang::parse(text);
    DomainFlowRecorder r(ast, text, 0);
    CHECK_THROWS_WITH(driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000, &r),
                      ContainsSubstring("not resident"));
    REQUIRE(r.refused_at().has_value());
    CHECK(*r.refused_at() == 2);
    const DomainFlowRecord rec = r.finish();
    REQUIRE(rec.refusal.has_value());
    CHECK(rec.refusal->action == 2);
    CHECK_THAT(rec.refusal->what, ContainsSubstring("line 6"));
    CHECK(rec.steps.size() == 2);
    CHECK(rec.actions == 2);
    REQUIRE(rec.residencies.size() == 1);
    CHECK(rec.residencies[0].close == 1);
    const auto dir = scratch("refused");
    record::write_dflow(rec, dir.string());
    CHECK(record::read_dflow(dir.string()).refusal->action == 2);
    std::filesystem::remove_all(dir);
}

TEST_CASE("dflow: the reader refuses what is not a version-1 bundle", "[dflow][record]") {
    const auto dir = scratch("bad");
    std::filesystem::create_directories(dir);
    {
        std::ofstream(dir / "manifest.json") << R"({"format": "kpu-tflow", "version": 1})";
    }
    CHECK_THROWS_WITH(record::read_dflow(dir.string()), ContainsSubstring("not a .dflow bundle"));
    {
        std::ofstream(dir / "manifest.json") << R"({"format": "kpu-dflow", "version": 99})";
    }
    CHECK_THROWS_WITH(record::read_dflow(dir.string()), ContainsSubstring("version 99"));
    CHECK_THROWS_AS(record::read_dflow((dir / "missing").string()), record::RecordError);
    std::filesystem::remove_all(dir);
}

namespace {

// A program's record with its L-T1 overlay, on kpu_s1's lanes (where a slot can decide a start).
struct Overlaid {
    DomainFlowRecord rec;
    driver::CspLevelOutcome lt1;
};

Overlaid overlaid(const std::string& text) {
    const lang::Program ast = lang::parse(text);
    const auto spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    Recorded r = record_of(text);
    Overlaid out;
    out.lt1 = driver::csp_run_level(driver::ExecutionLevel::BlockSequential, ast, &spec.device(0), driver::csp_inputs(ast));
    out.rec = std::move(r.rec);
    record::attach_lt1(out.rec, out.lt1, *r.ref.values, "test");
    return out;
}

}  // namespace

TEST_CASE("dflow: the L-T1 overlay joins L-T1's legs on the action", "[dflow][record][lt1]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        const Overlaid o = overlaid(slurp(std::string("tests/program/corpus/") + name));
        REQUIRE(o.rec.lt1);
        CHECK(o.rec.legs.size() == o.lt1.records.size());
        CHECK(o.rec.lt1_makespan == o.lt1.makespan);
        std::vector<std::size_t> legs(o.rec.actions, 0);
        std::uint64_t latest = 0;
        for (const auto& l : o.rec.legs) {
            REQUIRE(l.action < o.rec.actions);
            CHECK(l.kind == o.rec.steps.at(l.action).kind);      // the join: the same action
            CHECK(l.leg == legs[l.action]++);                    // legs numbered in order
            CHECK(l.start <= l.finish);
            latest = std::max(latest, l.finish);
        }
        for (std::size_t a = 0; a < legs.size(); ++a) CHECK(legs[a] >= 1);
        CHECK(latest == o.rec.lt1_makespan);
        // L-T1 refines L-B: every final tile within the bound (bit-identical by construction).
        CHECK(o.rec.lt1_pass);
        std::size_t tiles = 0;
        for (const auto& op : o.rec.operands) tiles += ((op.rows + op.tile_rows - 1) / op.tile_rows) * ((op.cols + op.tile_cols - 1) / op.tile_cols);
        CHECK(o.rec.proximity.size() == tiles);
        for (const auto& p : o.rec.proximity) {
            CHECK(p.pass);
            CHECK(p.max_abs == 0.0);
        }
    }
}

TEST_CASE("dflow: the overlay's slot waits are a Load's, and add up to L-T1's", "[dflow][record][lt1]") {
    const Overlaid o = overlaid(slurp("tests/program/dflow/streamed_lowest.csp"));
    REQUIRE(o.rec.lt1_slot_bound > 0);
    std::uint64_t waits = 0, waited = 0;
    for (const auto& l : o.rec.legs)
        if (l.slot_wait) {
            CHECK(l.kind == static_cast<std::uint8_t>(csp::Action::Kind::Load));
            waits += l.slot_wait;
            ++waited;
        }
    CHECK(waits == o.rec.lt1_slot_wait_cycles);
    CHECK(waited == o.rec.lt1_slot_bound);
    CHECK(o.rec.lt1_slot_wait_cycles == 704);           // l3-placement-at-lb step 5a's measurement
}

TEST_CASE("dflow: the overlay round-trips", "[dflow][record][lt1]") {
    const Overlaid o = overlaid(slurp("tests/program/dflow/streamed_lowest.csp"));
    const auto dir = scratch("lt1");
    record::write_dflow(o.rec, dir.string());
    const DomainFlowRecord b = record::read_dflow(dir.string());
    REQUIRE(b.lt1);
    CHECK((b.lt1_makespan == o.rec.lt1_makespan && b.lt1_slot_bound == o.rec.lt1_slot_bound &&
           b.lt1_slot_wait_cycles == o.rec.lt1_slot_wait_cycles && b.lt1_pass == o.rec.lt1_pass));
    CHECK(b.lt1_lanes == o.rec.lt1_lanes);
    REQUIRE(b.legs.size() == o.rec.legs.size());
    for (std::size_t i = 0; i < b.legs.size(); ++i) {
        const auto &x = o.rec.legs[i], &y = b.legs[i];
        CHECK((x.action == y.action && x.start == y.start && x.finish == y.finish && x.slot_wait == y.slot_wait &&
               x.lane == y.lane && x.kind == y.kind && x.proc == y.proc && x.leg == y.leg));
    }
    REQUIRE(b.proximity.size() == o.rec.proximity.size());
    for (std::size_t i = 0; i < b.proximity.size(); ++i)
        CHECK((b.proximity[i].tile == o.rec.proximity[i].tile && b.proximity[i].pass == o.rec.proximity[i].pass));
    // A record without an overlay says so, and reads back without one.
    const DomainFlowRecord plain = record_of(slurp("tests/program/corpus/matmul_48x48x48_t16.csp")).rec;
    record::write_dflow(plain, dir.string());
    CHECK_FALSE(record::read_dflow(dir.string()).lt1);
    std::filesystem::remove_all(dir);
}

TEST_CASE("tile_check reports every tile's proximity, not only the misses", "[dflow][tile_check]") {
    TensorOperand a("X", 4, 4, 2, 2), o("X", 4, 4, 2, 2);
    for (std::size_t i = 0; i < 16; ++i) a.values[i] = o.values[i] = 1.0f + static_cast<float>(i);
    a.at(3, 3) += 1.0f;                                 // one element of tile (1,1) out of bound
    const TileCheck c = check_tiles(a, o, ElementFormat{}, Proximity{});
    REQUIRE(c.per_tile.size() == 4);
    for (const auto& t : c.per_tile) {
        CAPTURE(t.tile.to_string());
        const bool off = t.tile.ti == 1 && t.tile.tj == 1;
        CHECK(t.pass == !off);
        CHECK((t.max_abs > 0.0) == off);
    }
    CHECK(c.misses.size() == 1);
}

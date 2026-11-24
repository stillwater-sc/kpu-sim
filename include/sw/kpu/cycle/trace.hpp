#pragma once
// ============================================================================
// include/sw/kpu/cycle/trace.hpp
// The cycle core's trace, in the Chrome trace-event JSON that Perfetto (ui.perfetto.dev) and
// chrome://tracing open (docs/plans/cycle-based-kpu-prototype.md §3.3, decision 2).
//
// What it records, once attached (Simulator::attach); nothing, and no cost but a null test, when not:
//   a component     one track (a thread of process 1) per component, its STATE as runs of
//                   complete events ("X"): "busy", or "stall: <cause>"; idle cycles are not drawn
//   a link          its BEATS as async slices ("b"/"e", one async track per link, named by it), each
//                   from its push to its pop -- beats overlap on a link, which complete events on one
//                   thread may not -- and its OCCUPANCY as a counter track ("C"), at each change. The
//                   slices' ids are process-local ("id2": {"local": ...}), so Perfetto files the link
//                   tracks under the process, beside its components, not among its global events
// Times are the domains' edges in picoseconds, written in microseconds (the format's unit), so
// domains at different clocks line up. A cycle c spans [edge(c), edge(c + 1)).
//
// The trace keeps at most `limit` events; past that it counts what it dropped and says so in the
// file's metadata, so a long run never fills memory unnoticed.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/counters.hpp>
#include <sw/kpu/cycle/time.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class Tracer {
public:
    explicit Tracer(std::size_t limit = 1'000'000) : limit_(limit) {}

    // A track per component (a thread) or per link (an async track and a counter track).
    std::uint32_t add_track(std::string name, std::string domain, bool link) {
        tracks_.push_back({std::move(name), std::move(domain), link});
        runs_.push_back({});
        return static_cast<std::uint32_t>(tracks_.size() - 1);
    }

    // A component's state for the cycle [from, to): runs of one state become one event.
    void state(std::uint32_t track, State s, Stall cause, Picoseconds from, Picoseconds to) {
        Run& r = runs_[track];
        if (r.open && r.state == s && (s != State::Stalled || r.cause == cause) && r.to == from) {
            r.to = to;
            return;
        }
        close(track);
        if (s == State::Idle) return;
        r = {true, s, cause, from, to};
    }
    // A beat's stay on a link, from its push to its pop.
    void beat(std::uint32_t track, std::uint64_t id, Picoseconds from, Picoseconds to) {
        const std::uint64_t key = next_async_++;
        add({Event::Kind::AsyncBegin, track, "beat " + std::to_string(id), from, 0, key, 0});
        add({Event::Kind::AsyncEnd, track, "beat " + std::to_string(id), to, 0, key, 0});
    }
    // A link's occupancy, written when it changes.
    void occupancy(std::uint32_t track, Picoseconds at, std::uint64_t value) {
        Run& r = runs_[track];
        if (r.has_value && r.value == value) return;
        r.has_value = true;
        r.value = value;
        add({Event::Kind::Counter, track, "occupancy", at, 0, 0, value});
    }

    // Closes every open run: call before writing, at the end of a run.
    void finish() {
        for (std::uint32_t t = 0; t < runs_.size(); ++t) close(t);
    }

    [[nodiscard]] std::size_t events() const { return events_.size(); }
    [[nodiscard]] std::size_t dropped() const { return dropped_; }

    void write(const std::string& path) {
        finish();
        std::ofstream out(path);
        if (!out) throw std::runtime_error("cycle::Tracer: cannot write " + path);
        out << "{\"displayTimeUnit\":\"ns\",\"otherData\":{\"source\":\"kpu-sim cycle core\",\"events\":"
            << events_.size() << ",\"dropped\":" << dropped_ << "},\"traceEvents\":[\n";
        bool first = true;
        auto sep = [&] {
            if (!first) out << ",\n";
            first = false;
        };
        sep();
        out << R"({"ph":"M","pid":1,"tid":0,"name":"process_name","args":{"name":"cycle core"}})";
        // Every track is named: a link's events carry its thread id too, so it is named "link ...".
        for (std::uint32_t t = 0; t < tracks_.size(); ++t) {
            sep();
            out << R"({"ph":"M","pid":1,"tid":)" << tid(t) << R"(,"name":"thread_name","args":{"name":")"
                << escape((tracks_[t].link ? "link " : "") + tracks_[t].name + " (" + tracks_[t].domain + ")") << "\"}}";
            sep();
            out << R"({"ph":"M","pid":1,"tid":)" << tid(t) << R"(,"name":"thread_sort_index","args":{"sort_index":)"
                << t << "}}";
        }
        char ts[64], dur[64];
        for (const Event& e : events_) {
            sep();
            const Track& k = tracks_[e.track];
            std::snprintf(ts, sizeof ts, "%.6f", static_cast<double>(e.at) / 1e6);
            switch (e.kind) {
                case Event::Kind::Complete:
                    std::snprintf(dur, sizeof dur, "%.6f", static_cast<double>(e.dur) / 1e6);
                    out << R"({"ph":"X","pid":1,"tid":)" << tid(e.track) << R"(,"name":")" << escape(e.name)
                        << R"(","ts":)" << ts << R"(,"dur":)" << dur << "}";
                    break;
                case Event::Kind::AsyncBegin:
                case Event::Kind::AsyncEnd:
                    out << R"({"ph":")" << (e.kind == Event::Kind::AsyncBegin ? "b" : "e")
                        << R"(","pid":1,"tid":)" << tid(e.track) << R"(,"cat":"link","name":")" << escape(k.name)
                        << R"(","id2":{"local":")" << e.key << R"("},"ts":)" << ts << R"(,"args":{"beat":")" << escape(e.name)
                        << "\"}}";
                    break;
                case Event::Kind::Counter:
                    out << R"({"ph":"C","pid":1,"tid":)" << tid(e.track) << R"(,"name":")"
                        << escape(k.name + " occupancy") << R"(","ts":)" << ts << R"(,"args":{"beats":)" << e.value
                        << "}}";
                    break;
            }
        }
        out << "\n]}\n";
        if (!out) throw std::runtime_error("cycle::Tracer: writing " + path + " failed");
    }

private:
    struct Track {
        std::string name, domain;
        bool link = false;
    };
    struct Run {
        bool open = false;
        State state = State::Idle;
        Stall cause = Stall::NoCredit;
        Picoseconds from = 0, to = 0;
        bool has_value = false;      // a link's last occupancy written
        std::uint64_t value = 0;
    };
    struct Event {
        enum class Kind : std::uint8_t { Complete, AsyncBegin, AsyncEnd, Counter } kind;
        std::uint32_t track;
        std::string name;
        Picoseconds at, dur;
        std::uint64_t key, value;
    };

    void close(std::uint32_t track) {
        Run& r = runs_[track];
        if (!r.open) return;
        const std::string name = r.state == State::Busy ? "busy" : std::string("stall: ") + to_string(r.cause);
        add({Event::Kind::Complete, track, name, r.from, r.to - r.from, 0, 0});
        r.open = false;
    }
    void add(Event e) {
        if (events_.size() >= limit_) {
            ++dropped_;
            return;
        }
        events_.push_back(std::move(e));
    }
    static std::uint32_t tid(std::uint32_t track) { return track + 1; }
    static std::string escape(const std::string& s) {
        std::string o;
        for (const char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (static_cast<unsigned char>(c) < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                o += b;
                continue;
            }
            o += c;
        }
        return o;
    }

    std::size_t limit_;
    std::vector<Track> tracks_;
    std::vector<Run> runs_;
    std::vector<Event> events_;
    std::size_t dropped_ = 0;
    std::uint64_t next_async_ = 1;
};

}  // namespace sw::kpu::cycle

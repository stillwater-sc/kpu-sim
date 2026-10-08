// ============================================================================
// include/sw/kpu/timing/memory_side_harness.hpp
// The memory side of the KPU on its own: DRAM, memory controllers, DMA engines and their
// store buffers, with every NoC port replaced by a stub (docs/plans/memory-side-debugger.md
// §3.2, step 2).
//
// Nothing here is a new memory model. The harness wires the CSP processes the executor uses
// (MemoryControllerProcess hosting the LPDDR5 controller, DMAEngineProcess with its burst window
// and store buffer) exactly as csp_config_from / csp_noc_wiring do for a deployment, and stands in
// for everything north of the ports:
//
//   REQUEST MODELS   generate the DMA reads (loads) and the ejections (stores) a port carries:
//                    stream, strided, random, matrix tiles (row segments of a pitched matrix),
//                    and replay (the LOAD/STORE tiles of a schedule).
//   LOAD SINK        the port's injection bus: one block per `block_cycles`, from a per-engine
//                    input queue of `input_queue_blocks`. A full queue refuses the DMA engine,
//                    which holds its block (and its L3 credit) and offers it again next cycle.
//                    A delivered load lands in the stand-in L3; `consume_latency` cycles later the
//                    consumer frees its slot. `infinite` lands every load at once.
//   EJECTION MODEL   the port's output side: a block arrives every `eject_interval` cycles per
//                    engine, takes a store-buffer slot (waiting while none is free), crosses the
//                    ejection bus (one block per `block_cycles`), and is delivered under its
//                    store's ticket. The engine writes it to DRAM.
//
// The stand-in L3 is one credit pool: loads still take a credit before their first burst
// (dram-bank-model.md §2), and the sink's consumer returns it.
//
// Everything that happens is kept for the .mflow record (step 3): the controllers' commands and
// bursts (MemoryControllerProcess::Config::record), each request's lifetime, store-buffer
// occupancy, and the port stubs' offers, refusals and waits.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/dma_engine_process.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>
#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing::memside {

// ----------------------------------------------------------------------------
// Request models
// ----------------------------------------------------------------------------
struct RequestModel {
    enum class Kind : std::uint8_t { Stream, Strided, Random, MatrixTiles, Replay };
    Kind kind = Kind::Stream;
    bool is_load = true;                // a DMA read toward L3; false = an ejection to DRAM
    std::uint64_t base = 0;
    std::uint32_t count = 0;            // Stream / Strided / Random: how many requests
    std::uint32_t bytes = 4096;         // Stream / Strided / Random: bytes per request
    std::uint64_t stride = 0;           // Strided: start-to-start distance (>= bytes)
    std::uint64_t region = 0;           // Random: requests start in [base, base + region)
    std::uint64_t seed = 1;             // Random
    // MatrixTiles: a rows x cols matrix of element_bytes elements, rows pitch bytes apart, cut
    // into tile_rows x tile_cols tiles in row-major tile order. Each tile row is one request.
    std::uint32_t rows = 0, cols = 0, element_bytes = 4, tile_rows = 0, tile_cols = 0;
    std::uint64_t pitch = 0;            // 0 = cols * element_bytes
    std::vector<TileDescriptor> replay; // Replay: the requests themselves

    // The LOAD (or STORE) tiles of a schedule, in schedule order.
    static RequestModel replay_of(const schedule::ScheduleResult& s, bool loads) {
        RequestModel m;
        m.kind = Kind::Replay;
        m.is_load = loads;
        for (const auto& op : s.operations)
            if (op.type == (loads ? schedule::ScheduleOpType::LOAD : schedule::ScheduleOpType::STORE))
                m.replay.push_back(op.tile);
        return m;
    }

    // The requests, in issue order: (address, bytes).
    std::vector<std::pair<std::uint64_t, std::uint32_t>> generate() const {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> out;
        switch (kind) {
            case Kind::Stream:
                for (std::uint32_t i = 0; i < count; ++i) out.emplace_back(base + std::uint64_t{i} * bytes, bytes);
                break;
            case Kind::Strided: {
                if (stride < bytes) throw std::invalid_argument("strided: stride below the request size");
                for (std::uint32_t i = 0; i < count; ++i) out.emplace_back(base + std::uint64_t{i} * stride, bytes);
                break;
            }
            case Kind::Random: {
                if (bytes == 0) throw std::invalid_argument("random: bytes must be greater than zero");
                if (region < bytes) throw std::invalid_argument("random: region below the request size");
                std::uint64_t s = seed ? seed : 1;
                const std::uint64_t slots = region / bytes;
                for (std::uint32_t i = 0; i < count; ++i) {
                    s = s * 6364136223846793005ull + 1442695040888963407ull;   // LCG: deterministic
                    out.emplace_back(base + ((s >> 33) % slots) * bytes, bytes);
                }
                break;
            }
            case Kind::MatrixTiles: {
                if (!rows || !cols || !tile_rows || !tile_cols)
                    throw std::invalid_argument("matrix_tiles: rows, cols and the tile shape must be set");
                const std::uint64_t p = pitch ? pitch : std::uint64_t{cols} * element_bytes;
                if (p < std::uint64_t{cols} * element_bytes)
                    throw std::invalid_argument("matrix_tiles: pitch below a row");
                for (std::uint32_t tr = 0; tr * tile_rows < rows; ++tr)
                    for (std::uint32_t tc = 0; tc * tile_cols < cols; ++tc)
                        for (std::uint32_t r = 0; r < tile_rows && tr * tile_rows + r < rows; ++r) {
                            const std::uint32_t c0 = tc * tile_cols;
                            const std::uint32_t w = std::min(tile_cols, cols - c0);
                            out.emplace_back(base + std::uint64_t{tr * tile_rows + r} * p +
                                                 std::uint64_t{c0} * element_bytes,
                                             w * element_bytes);
                        }
                break;
            }
            case Kind::Replay:
                for (const auto& t : replay) out.emplace_back(t.dram_address, t.size_bytes);
                break;
        }
        return out;
    }
};

// ----------------------------------------------------------------------------
// The harness
// ----------------------------------------------------------------------------
class MemorySideHarness {
public:
    struct PortStub {
        bool infinite = false;              // loads land at once, ejections arrive at once
        Cycle block_cycles = 0;             // 0 = from the spec: ceil(bytes / noc link rate)
        std::size_t input_queue_blocks = 0; // per engine; 0 = noc.port.input_queue_blocks (or 2)
        Cycle consume_latency = 0;          // landing to the stand-in L3 slot being freed
        Cycle eject_interval = 1;           // an ejection arrives every N cycles per engine
    };

    // One request model on one port: its requests are dealt round-robin over the port's engines.
    struct Stream {
        unsigned port = 0;
        RequestModel model;
        Cycle issue_interval = 0;           // loads: one per N cycles per engine (0 = all at once)
    };

    struct Config {
        std::size_t window = 0;             // 0 = the spec's dma.window (required)
        std::size_t store_buffer_blocks = 0;// 0 = noc.port.output_queue_blocks, else 2
        std::size_t l3_slots = 0;           // stand-in L3 credits; 0 = l3.capacity_tiles
        PortStub ports;
        std::vector<Stream> streams;
        Cycle max_cycles = 10'000'000;
    };

    // One request's lifetime (the .mflow requests table).
    struct Request {
        std::uint32_t id = 0;
        unsigned engine = 0, port = 0;
        TileID tile;
        bool is_load = true;
        std::uint64_t address = 0;
        std::uint32_t bytes = 0;
        Cycle offered = 0;                  // given to the engine (load) / arrived at the port (store)
        Cycle credit = 0;                   // load: L3 credit; store: store-buffer slot
        Cycle first_burst = 0, last_burst = 0;
        Cycle retired = 0;                  // load: landed in L3; store: written to DRAM
        bool done = false;
    };
    struct BufferSample { Cycle t; unsigned engine; std::size_t held, staged; };
    struct PortEvent {
        enum class Kind : std::uint8_t { Offer, Accept, Refuse, Land, EjectArrive, EjectWait, EjectDeliver };
        Cycle t; unsigned port, engine; Kind kind; std::uint32_t request;
    };

    MemorySideHarness(const sw::kpu::program::platform::DeviceSpecification& d, Config config)
        : config_(std::move(config)) {
        std::string why;
        auto csp = csp_config_from(d, &why);
        if (!csp) throw std::invalid_argument("memory-side harness: " + why);
        const auto& c = csp->config;
        if (!c.dram) throw std::invalid_argument("memory-side harness: the device declares no memory.dram");
        window_ = config_.window ? config_.window : c.dma_window;
        if (!window_) throw std::invalid_argument("memory-side harness: no burst window (dma.window or Config::window)");
        const std::size_t buffers = config_.store_buffer_blocks ? config_.store_buffer_blocks
                                  : (d.noc && d.noc->port.output_queue_blocks ? d.noc->port.output_queue_blocks : 2);
        input_queue_ = config_.ports.input_queue_blocks ? config_.ports.input_queue_blocks
                     : (d.noc ? d.noc->port.input_queue_blocks : 2);
        noc_bytes_per_cycle_ = d.movers.noc_bytes_per_cycle;
        const std::size_t slots = config_.l3_slots ? config_.l3_slots : c.l3_buffer_count;
        l3_credits_ = std::make_unique<CreditPool>(slots);
        l3_cam_ = std::make_unique<TagCAM>(slots);

        for (std::size_t m = 0; m < c.num_memory_controllers; ++m) {
            MemoryControllerProcess::Config mc;
            mc.controller_id = static_cast<std::uint32_t>(m);
            mc.hosted = c.dram;
            mc.clock_ghz = c.clock_ghz;
            mc.request_queue_depth = c.mc_request_queue_depth;
            mc.record = true;
            mcs_.push_back(std::make_unique<MemoryControllerProcess>(mc));
        }
        // Engines: numbered per controller, attached to ports as the floorplan does.
        sw::kpu::program::platform::DeploymentSpec one;
        one.devices = {d};
        const auto attach = sw::kpu::program::platform::dma_port_attachment(one);
        engine_port_.assign(c.num_dma_engines, ~0u);
        const std::size_t per = c.num_dma_engines / c.num_memory_controllers;
        for (const auto& a : attach) engine_port_.at(static_cast<std::size_t>(a.mc) * per + a.engine) = a.port;
        for (std::size_t e = 0; e < c.num_dma_engines; ++e) {
            DMAEngineProcess::Config dc;
            dc.engine_id = static_cast<std::uint32_t>(e);
            dc.queue_depth = c.dma_queue_depth;
            dc.store_buffer_blocks = buffers;
            dc.window = window_;
            dc.name = "DMA" + std::to_string(e);
            dmas_.push_back(std::make_unique<DMAEngineProcess>(
                dc, *mcs_.at(c.dma_engine_controller.at(e)), *l3_credits_, *l3_cam_));
            if (engine_port_[e] != ~0u) ports_[engine_port_[e]].engines.push_back(static_cast<unsigned>(e));
        }
        sink_.harness = this;
        for (auto& dma : dmas_) dma->set_load_route(&sink_);
        for (auto& [k, p] : ports_) p.queues.assign(p.engines.size(), {});

        // Deal each stream's requests over its port's engines.
        std::uint32_t next = 0;
        for (const auto& s : config_.streams) {
            auto it = ports_.find(s.port);
            if (it == ports_.end() || it->second.engines.empty())
                throw std::invalid_argument("memory-side harness: port " + std::to_string(s.port) +
                                            " has no attached DMA engine");
            const auto& engines = it->second.engines;
            const auto reqs = s.model.generate();
            for (std::size_t i = 0; i < reqs.size(); ++i) {
                Request r;
                r.id = next++;
                r.engine = engines[i % engines.size()];
                r.port = s.port;
                r.is_load = s.model.is_load;
                r.address = reqs[i].first;
                r.bytes = reqs[i].second;
                // Unique per request: loads are A tiles, stores C tiles, indexed by request id.
                r.tile = TileID{r.is_load ? isa::MatrixID::A : isa::MatrixID::C, r.id, s.port, 0};
                const Cycle at = s.issue_interval * static_cast<Cycle>(i / engines.size());
                (r.is_load ? loads_due_ : ejects_due_)[r.engine].push_back({at, r.id});
                requests_.push_back(r);
            }
        }
        for (const auto& r : requests_) by_tile_[r.tile] = r.id;
        // Streams are dealt one after another, so an engine's queue is in stream order. Offering
        // stops at the first entry not yet due: sort by due time (stably, keeping stream order on
        // ties) so a later stream's early requests are not held behind an earlier stream's late
        // ones.
        for (auto* due : {&loads_due_, &ejects_due_})
            for (auto& [e, q] : *due)
                std::stable_sort(q.begin(), q.end(),
                                 [](const Due& a, const Due& b) { return a.at < b.at; });
    }
    MemorySideHarness(const MemorySideHarness&) = delete;
    MemorySideHarness& operator=(const MemorySideHarness&) = delete;

    // Run until every request has retired. False if max_cycles came first.
    bool run() {
        while (!finished() && now_ < config_.max_cycles) step();
        return finished();
    }

    void step() {
        offer_loads();
        offer_ejections();
        std::vector<TimingEvent> events;
        for (auto& mc : mcs_) {
            auto e = mc->tick(now_);
            events.insert(events.end(), e.begin(), e.end());
        }
        for (auto& dma : dmas_) {
            auto e = dma->tick(now_);
            events.insert(events.end(), e.begin(), e.end());
        }
        tick_ports(events);
        for (const auto& e : events) observe(e);
        sample_buffers();
        ++now_;
    }

    bool finished() const {
        for (const auto& r : requests_) if (!r.done) return false;
        return true;
    }

    // ---- results (the .mflow record, step 3) ----
    Cycle now() const { return now_; }
    std::size_t window() const { return window_; }
    const std::vector<Request>& requests() const { return requests_; }
    const std::vector<BufferSample>& buffers() const { return buffer_samples_; }
    const std::vector<PortEvent>& port_events() const { return port_events_; }
    const std::vector<std::unique_ptr<MemoryControllerProcess>>& controllers() const { return mcs_; }
    const DMAEngineProcess& engine(std::size_t e) const { return *dmas_.at(e); }
    std::size_t engines() const { return dmas_.size(); }
    unsigned port_of(std::size_t engine) const { return engine_port_.at(engine); }
    const CreditPool& l3_credits() const { return *l3_credits_; }
    // Every controller's data buses at full rate, in bytes per executor cycle: channels x width x
    // data rate, over the executor clock.
    double ceiling_bytes_per_cycle() const {
        double total = 0;
        for (const auto& mc : mcs_) {
            const auto& h = *mc->config().hosted;
            total += static_cast<double>(h.map.channels()) * (h.channel_width_bits / 8.0) *
                     h.data_rate_mtps * 1e6 / (mc->config().clock_ghz * 1e9);
        }
        return total;
    }
    std::uint64_t bytes_moved() const {
        std::uint64_t n = 0;
        for (const auto& r : requests_) if (r.done) n += r.bytes;
        return n;
    }

private:
    struct Due { Cycle at; std::uint32_t id; };
    struct Port {
        std::vector<unsigned> engines;
        std::vector<std::deque<std::pair<std::uint64_t, Cycle>>> queues;   // per engine: (tag, since)
        std::optional<std::pair<std::uint64_t, Cycle>> on_bus;             // load crossing: (tag, ends)
        Cycle eject_bus_free = 0;
    };
    struct Sink : LoadRoute {
        MemorySideHarness* harness = nullptr;
        bool inject(uint32_t engine, const TileDescriptor& tile, uint64_t tag) override {
            return harness->inject(engine, tile, tag);
        }
    } sink_;

    Config config_;
    std::size_t window_ = 0, input_queue_ = 2;
    double noc_bytes_per_cycle_ = 128.0;
    std::unique_ptr<CreditPool> l3_credits_;
    std::unique_ptr<TagCAM> l3_cam_;
    std::vector<std::unique_ptr<MemoryControllerProcess>> mcs_;
    std::vector<std::unique_ptr<DMAEngineProcess>> dmas_;
    std::vector<unsigned> engine_port_;
    std::map<unsigned, Port> ports_;
    std::vector<Request> requests_;
    std::unordered_map<TileID, std::uint32_t, TileIDHash> by_tile_;
    std::map<unsigned, std::deque<Due>> loads_due_, ejects_due_;
    std::deque<std::pair<Cycle, std::uint32_t>> consuming_;                     // (free at, request)
    std::map<unsigned, std::optional<std::uint32_t>> waiting_eject_;            // per engine
    std::map<unsigned, Cycle> next_eject_;
    std::map<unsigned, std::uint32_t> wait_logged_;     // per engine: request id + 1 last logged
    std::vector<BufferSample> buffer_samples_;
    std::vector<std::pair<std::size_t, std::size_t>> last_buffer_;
    std::vector<PortEvent> port_events_;
    Cycle now_ = 0;

    TileDescriptor descriptor(const Request& r) const {
        TileDescriptor t;
        t.tile_id = r.tile;
        t.dram_address = r.address;
        t.size_bytes = r.bytes;
        return t;
    }
    Cycle block_cycles(std::uint32_t bytes) const {
        if (config_.ports.block_cycles) return config_.ports.block_cycles;
        return std::max<Cycle>(1, static_cast<Cycle>(std::ceil(bytes / noc_bytes_per_cycle_)));
    }

    // Loads: due requests go to their engine (the engine's pending list is a staging queue).
    void offer_loads() {
        for (auto& [e, q] : loads_due_)
            while (!q.empty() && q.front().at <= now_) {
                Request& r = requests_[q.front().id];
                r.offered = now_;
                dmas_[e]->schedule_load(descriptor(r));
                port_events_.push_back({now_, r.port, e, PortEvent::Kind::Offer, r.id});
                q.pop_front();
            }
    }

    // Ejections: one per engine at a time; it needs a store-buffer slot, then the ejection bus.
    void offer_ejections() {
        for (auto& [e, q] : ejects_due_) {
            auto& waiting = waiting_eject_[e];
            if (!waiting) {
                if (q.empty() || q.front().at > now_ || now_ < next_eject_[e]) continue;
                waiting = q.front().id;
                q.pop_front();
                Request& r = requests_[*waiting];
                r.offered = now_;
                port_events_.push_back({now_, r.port, e, PortEvent::Kind::EjectArrive, r.id});
            }
            Request& r = requests_[*waiting];
            Port& p = ports_.at(r.port);
            DMAEngineProcess& dma = *dmas_[e];
            if (!config_.ports.infinite && now_ < p.eject_bus_free) continue;
            if (!dma.store_buffer().reserve()) {
                // Logged once, when the ejection first finds the buffer full; it retries silently.
                if (wait_logged_[e] != r.id + 1) {
                    port_events_.push_back({now_, r.port, e, PortEvent::Kind::EjectWait, r.id});
                    wait_logged_[e] = r.id + 1;
                }
                continue;
            }
            r.credit = now_;
            const std::uint64_t ticket = dma.schedule_store(descriptor(r));
            const Cycle cross = config_.ports.infinite ? 0 : block_cycles(r.bytes);
            p.eject_bus_free = now_ + cross;
            pending_delivery_.push_back({now_ + cross, ticket, e, r.id});
            next_eject_[e] = now_ + config_.ports.eject_interval;
            waiting.reset();
        }
        for (auto it = pending_delivery_.begin(); it != pending_delivery_.end();) {
            if (it->at > now_) { ++it; continue; }
            dmas_[it->engine]->store_buffer().deliver(it->ticket);
            port_events_.push_back({now_, requests_[it->request].port, it->engine,
                                    PortEvent::Kind::EjectDeliver, it->request});
            it = pending_delivery_.erase(it);
        }
    }
    struct Delivery { Cycle at; std::uint64_t ticket; unsigned engine; std::uint32_t request; };
    std::vector<Delivery> pending_delivery_;

    // The load sink: the port's input queue for this engine.
    bool inject(uint32_t engine, const TileDescriptor& tile, uint64_t tag) {
        const std::uint32_t id = by_tile_.at(tile.tile_id);
        const unsigned port = engine_port_.at(engine);
        Port& p = ports_.at(port);
        const auto slot = static_cast<std::size_t>(
            std::find(p.engines.begin(), p.engines.end(), engine) - p.engines.begin());
        if (!config_.ports.infinite && p.queues[slot].size() >= input_queue_) {
            port_events_.push_back({now_, port, engine, PortEvent::Kind::Refuse, id});
            return false;
        }
        p.queues[slot].emplace_back(tag, now_);
        tag_request_[tag] = id;
        port_events_.push_back({now_, port, engine, PortEvent::Kind::Accept, id});
        return true;
    }

    void tick_ports(std::vector<TimingEvent>& events) {
        for (auto& [k, p] : ports_) {
            if (config_.ports.infinite) {
                for (std::size_t s = 0; s < p.queues.size(); ++s)
                    while (!p.queues[s].empty()) {
                        land(p.engines[s], p.queues[s].front().first, events);
                        p.queues[s].pop_front();
                    }
                continue;
            }
            // The injection bus: one block at a time, oldest queued head first.
            if (p.on_bus && p.on_bus->second <= now_) {
                land(static_cast<unsigned>(p.on_bus->first >> 40), p.on_bus->first, events);
                p.on_bus.reset();
            }
            if (!p.on_bus) {
                std::optional<std::size_t> pick;
                for (std::size_t s = 0; s < p.queues.size(); ++s)
                    if (!p.queues[s].empty() &&
                        (!pick || p.queues[s].front().second < p.queues[*pick].front().second))
                        pick = s;
                if (pick) {
                    const std::uint64_t tag = p.queues[*pick].front().first;
                    p.queues[*pick].pop_front();
                    const std::uint32_t id = request_of_tag(tag);
                    p.on_bus = std::make_pair(tag, now_ + block_cycles(requests_[id].bytes));
                }
            }
        }
        // The consumer frees each landed load's stand-in L3 slot after consume_latency.
        while (!consuming_.empty() && consuming_.front().first <= now_) {
            const Request& r = requests_[consuming_.front().second];
            if (l3_cam_->invalidate(r.tile)) l3_credits_->release(static_cast<std::size_t>(r.tile.matrix));
            consuming_.pop_front();
        }
    }

    // A load's tag is its engine's request ticket; the sink learns it when the load is offered.
    std::unordered_map<std::uint64_t, std::uint32_t> tag_request_;
    std::uint32_t request_of_tag(std::uint64_t tag) const { return tag_request_.at(tag); }

    void land(unsigned engine, std::uint64_t tag, std::vector<TimingEvent>& events) {
        dmas_[engine]->land_load(tag, now_, events);
        const std::uint32_t id = tag_request_.at(tag);
        port_events_.push_back({now_, requests_[id].port, engine, PortEvent::Kind::Land, id});
        consuming_.emplace_back(now_ + config_.ports.consume_latency, id);
    }

    void observe(const TimingEvent& e) {
        auto it = by_tile_.find(e.tile_id);
        if (it == by_tile_.end()) return;
        Request& r = requests_[it->second];
        switch (e.type) {
            case EventType::CREDIT_ACQUIRED: if (r.is_load) r.credit = e.cycle; break;
            case EventType::DMA_LOAD_START:
            case EventType::DMA_STORE_START: r.first_burst = e.cycle; break;
            case EventType::DMA_LOAD_COMPLETE:
            case EventType::DMA_STORE_COMPLETE: r.last_burst = e.cycle + e.duration; break;
            case EventType::TILE_ARRIVED_L3: r.retired = e.cycle; r.done = true; break;
            case EventType::DMA_STORE_RETIRED: r.retired = e.cycle; r.done = true; break;
            default: break;
        }
    }

    void sample_buffers() {
        if (last_buffer_.size() != dmas_.size()) last_buffer_.assign(dmas_.size(), {~std::size_t{0}, 0});
        for (std::size_t e = 0; e < dmas_.size(); ++e) {
            const auto& b = dmas_[e]->store_buffer();
            const std::pair<std::size_t, std::size_t> now{b.held(), b.staged_count()};
            if (now == last_buffer_[e]) continue;
            last_buffer_[e] = now;
            buffer_samples_.push_back({now_, static_cast<unsigned>(e), now.first, now.second});
        }
    }
};

} // namespace sw::kpu::timing::memside

// ============================================================================
// The run as a .mflow record (docs/plans/memory-side-debugger.md §3.3, step 3)
// ============================================================================
#include <sw/kpu/program/record/memory_flow_record.hpp>

namespace sw::kpu::timing::memside {

inline program::record::MemoryFlowRecord to_record(const MemorySideHarness& h, const std::string& device) {
    using MFR = program::record::MemoryFlowRecord;
    MFR rec;
    rec.device = device;
    rec.makespan = h.now();
    rec.window = static_cast<std::uint32_t>(h.window());
    if (!h.controllers().empty() && h.controllers().front()->bridge()) {
        rec.timing_note = h.controllers().front()->bridge()->timing_note();
        rec.burst_bytes = static_cast<std::uint32_t>(h.controllers().front()->burst_bytes());
        // Every controller of a device runs the same part: one table, one clock ratio.
        rec.ticks_per_cycle = h.controllers().front()->bridge()->ticks_per_cycle();
        rec.dram_timing = h.controllers().front()->bridge()->timing_table();
    }
    rec.ceiling_bytes_per_cycle = h.ceiling_bytes_per_cycle();

    // Stations: banks, buses, engines, store buffers, ports (memory_flow_record.hpp).
    for (std::size_t m = 0; m < h.controllers().size(); ++m) {
        const auto& map = h.controllers()[m]->bridge()->map();
        for (unsigned ch = 0; ch < map.channels(); ++ch)
            for (unsigned bg = 0; bg < map.bank_groups(); ++bg)
                for (unsigned ba = 0; ba < map.banks_per_group(); ++ba)
                    rec.stations.push_back({device + "/mc[" + std::to_string(m) + "]/ch[" + std::to_string(ch) +
                                                "]/bg[" + std::to_string(bg) + "]/ba[" + std::to_string(ba) + "]",
                                            "dram_bank", 1});
        for (unsigned ch = 0; ch < map.channels(); ++ch)
            rec.stations.push_back({device + "/mc[" + std::to_string(m) + "]/ch[" + std::to_string(ch) + "]/bus",
                                    "dram_bus", 1});
    }
    for (std::size_t e = 0; e < h.engines(); ++e)
        rec.stations.push_back({device + "/dma[" + std::to_string(e) + "]", "dma", h.window()});
    for (std::size_t e = 0; e < h.engines(); ++e)
        rec.stations.push_back({device + "/dmabuf[" + std::to_string(e) + "]", "dmabuf",
                                h.engine(e).store_buffer().capacity()});
    std::vector<unsigned> ports;
    for (std::size_t e = 0; e < h.engines(); ++e)
        if (h.port_of(e) != ~0u && std::find(ports.begin(), ports.end(), h.port_of(e)) == ports.end())
            ports.push_back(h.port_of(e));
    std::sort(ports.begin(), ports.end());
    for (unsigned k : ports) rec.stations.push_back({device + "/noc/port[" + std::to_string(k) + "]", "port", 1});

    // Requests, by id; and each request's tile, to tie bursts to it.
    std::unordered_map<TileID, std::uint32_t, TileIDHash> request_of;
    for (const auto& r : h.requests()) {
        request_of[r.tile] = r.id;
        rec.requests.push_back({r.engine, r.port, r.is_load, r.address, r.bytes, r.offered, r.credit,
                                r.first_burst, r.last_burst, r.retired});
    }

    // Bursts, then commands naming them: a command's tag is its controller's burst id.
    constexpr std::uint64_t kBurstTag = std::uint64_t{1} << 63;
    std::vector<std::unordered_map<std::uint64_t, std::uint32_t>> burst_index(h.controllers().size());
    for (std::size_t m = 0; m < h.controllers().size(); ++m)
        for (const auto& b : h.controllers()[m]->recorded_bursts()) {
            burst_index[m][b.id] = static_cast<std::uint32_t>(rec.bursts.size());
            auto it = request_of.find(b.tile);
            MFR::Burst x;
            x.engine = b.submitter_id;
            x.request = it == request_of.end() ? program::record::kNone : it->second;
            x.mc = static_cast<std::uint8_t>(b.coord.mc);
            x.channel = static_cast<std::uint8_t>(b.coord.channel);
            x.rank = static_cast<std::uint8_t>(b.coord.rank);
            x.bank_group = static_cast<std::uint8_t>(b.coord.bank_group);
            x.bank = static_cast<std::uint8_t>(b.coord.bank);
            x.row = static_cast<std::uint32_t>(b.coord.row);
            x.col = static_cast<std::uint32_t>(b.coord.col);
            x.is_load = b.is_load;
            x.outcome = static_cast<MFR::Outcome>(b.outcome);
            x.submitted = b.submitted;
            x.first_command = b.first_command;
            x.data_start = b.data_start;
            x.data_end = b.data_end;
            x.done = b.done;
            rec.bursts.push_back(x);
        }
    for (std::size_t m = 0; m < h.controllers().size(); ++m)
        for (const auto& c : h.controllers()[m]->recorded_commands()) {
            MFR::Command x;
            x.mc = static_cast<std::uint8_t>(c.mc);
            x.channel = static_cast<std::uint8_t>(c.channel);
            x.bank_group = static_cast<std::uint8_t>(c.bank_group);
            x.bank = static_cast<std::uint8_t>(c.bank);
            x.row = static_cast<std::uint32_t>(c.row);
            x.kind = static_cast<MFR::CommandKind>(c.kind);
            if (c.tag && (*c.tag & kBurstTag)) {
                auto it = burst_index[m].find(*c.tag & ~kBurstTag);
                if (it != burst_index[m].end()) x.burst = it->second;
            }
            x.issue = c.issue;
            x.end = c.end;
            x.data_start = c.data_start;
            x.data_end = c.data_end;
            x.tick = c.tick;
            x.tick_end = c.tick_end;
            x.tick_data_start = c.tick_data_start;
            x.tick_data_end = c.tick_data_end;
            rec.commands.push_back(x);
        }

    for (const auto& b : h.buffers())
        rec.buffers.push_back({b.engine, b.t, static_cast<std::uint32_t>(b.held), static_cast<std::uint32_t>(b.staged)});
    for (const auto& p : h.port_events())
        rec.ports.push_back({p.port, p.engine, p.request, static_cast<MFR::PortKind>(p.kind), p.t});
    return rec;
}

} // namespace sw::kpu::timing::memside

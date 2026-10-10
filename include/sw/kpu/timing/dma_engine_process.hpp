// ============================================================================
// include/sw/kpu/timing/dma_engine_process.hpp
// DMA Engine Process for CSP-style concurrent timing simulation
//
// The DMA Engine is a programmable ISA-driven process that:
// - Executes data movement operations (LOAD/STORE tiles)
// - Manages L3 credit acquisition and release
// - Tracks tile arrivals in L3 TagCAM
// - Uses a Memory Controller for actual DRAM access contention
//
// Architecture:
//   DMA Engine (CSP Process) --uses--> Memory Controller (Resource)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/process_interface.hpp>
#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>
#include <sw/kpu/timing/work_queue.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_set>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief A DMA engine's store buffer: tiles ejected out of L3, waiting for their DRAM write
 *
 * Push-with-credit on both sides. A BlockMover reserve()s a slot before it starts an
 * ejection and deliver()s the store's TICKET when the ejection lands. The engine take()s a
 * staged ticket when it starts the DRAM write, and release()s the slot when the write
 * completes. Entries are per store, not per tile: two stores of one tile are two tickets,
 * two slots and two writes.
 */
class DmaStoreBuffer {
public:
    explicit DmaStoreBuffer(size_t blocks) : credits_(blocks) {}

    bool reserve() { return credits_.acquire(); }
    void deliver(uint64_t ticket) { staged_.insert(ticket); }
    [[nodiscard]] bool staged(uint64_t ticket) const { return staged_.count(ticket) != 0; }
    void take(uint64_t ticket) { staged_.erase(ticket); }
    void release() { credits_.release(); }

    [[nodiscard]] size_t capacity() const { return credits_.capacity(); }
    [[nodiscard]] size_t held() const { return credits_.outstanding(); }
    [[nodiscard]] size_t staged_count() const { return staged_.size(); }
    void reset() {
        credits_.reset();
        staged_.clear();
    }

private:
    CreditPool credits_;
    std::unordered_set<uint64_t> staged_;
};

/**
 * @brief Where a load goes once DRAM has delivered it: onto the NoC, toward its home hub
 *
 * Without a route, a load lands in L3 the cycle the MC completes it. With one, it is injected
 * at the engine's port, crosses the NoC, and lands only when the hub delivers it
 * (DMAEngineProcess::land_load); its home-tile L3 credit is held the whole way
 * (docs/plans/noc-port-arbitration.md §4.1, step 4b.4).
 */
class LoadRoute {
public:
    virtual ~LoadRoute() = default;
    /// Offer the block to the NoC. False = the port's input queue has no room this cycle.
    virtual bool inject(uint32_t engine_id, const TileDescriptor& tile, uint64_t tag) = 0;
};

/**
 * @brief DMA Engine Process for DRAM ↔ L3 transfers
 *
 * The DMA engine handles:
 * - Loading tiles from DRAM to L3 (requires L3 credit)
 * - Storing tiles to DRAM from its own STORE BUFFER
 *
 * EVERY HOP IS A PUSH (docs/plans/noc-port-arbitration.md step 3; CSP tier, step 4b.2).
 * A store is not a DMA read of L3, which the machine does not have. A BlockMover ejects
 * the tile out of L3 into this engine's store buffer (BlockMoverProcess::schedule_eject),
 * holding one of the buffer's slots, and freeing the L3 slot when the ejection lands.
 * The engine then writes the buffer to DRAM and frees the buffer slot when the write
 * completes.
 *
 * The DMA engine uses a Memory Controller for actual DRAM access. Multiple
 * DMA engines can share a single MC (realistic for multi-channel configs).
 *
 * Credit flow:
 * - LOAD: Must acquire L3 credit before submitting to MC (buffer space needed)
 * - STORE: the BlockMover takes a store-buffer credit before it ejects; the engine
 *   writes once the tile is in its buffer, and returns the credit on write completion
 *
 * Concurrency:
 * - DMA engine queues requests, MC processes them with contention
 * - Multiple DMA engines can share one MC
 */
class DMAEngineProcess : public IProcess {
public:
    /**
     * @brief DMA Engine configuration
     */
    struct Config {
        uint32_t engine_id = 0;        ///< Unique engine identifier
        size_t queue_depth = 32;       ///< Max requests concurrently submitted to the MC
        size_t l3_credit_reserve = 0;  ///< L3 credits loads may NOT consume (reserved for
                                       ///< downstream writebacks; prevents credit starvation)
        size_t store_buffer_blocks = 2;  ///< Tiles the store buffer holds: ejected out of L3,
                                         ///< waiting for or in their DRAM write
        /// Bursts this engine keeps in flight (docs/plans/dram-bank-model.md step 3). 0 = a
        /// tile is one request to the controller. W > 0 needs a hosted controller: the engine
        /// decomposes each tile into DRAM bursts, keeps at most W in flight across all its
        /// tiles, and a tile completes when its last burst is home. queue_depth still bounds
        /// the tiles in flight; the L3 credit is still taken before the first burst.
        size_t window = 0;
        std::string name = "DMA";      ///< Human-readable name

        /// Generate human-readable name
        std::string display_name() const {
            return "DMA" + std::to_string(engine_id);
        }
    };

    /**
     * @brief Pending request state
     */
    enum class RequestState {
        WAITING_CREDIT,   ///< Load waiting for L3 credit
        WAITING_TAG,      ///< Store waiting for tile in L3
        SUBMITTED,        ///< Submitted to MC, waiting for completion
        TO_INJECT,        ///< Load read from DRAM, waiting to enter the NoC (routed loads)
        IN_TRANSIT,       ///< Load on the NoC, waiting for its hub to deliver it to L3
        COMPLETED         ///< Completed (will be removed)
    };

    /**
     * @brief Pending request tracking
     */
    struct PendingRequest {
        TileDescriptor tile;
        bool is_load;
        RequestState state;
        Cycle enqueue_cycle;
        uint32_t slot_id = 0;    ///< L3 slot for this tile (for loads)
        uint64_t ticket = 0;     ///< Stores: this store's identity in the store buffer
        // Burst window only:
        uint64_t bursts = 0, sent = 0, done = 0;    ///< the tile's bursts: total, in, home
        Cycle start_cycle = 0;                      ///< when its first burst was submitted
    };

    /**
     * @brief Construct a DMA engine process
     * @param config Engine configuration
     * @param mc Reference to shared Memory Controller
     * @param l3_credits Credit pool for L3 buffers (shared)
     * @param l3_tag_cam Tag CAM for L3 tile tracking (shared)
     */
    DMAEngineProcess(const Config& config,
                     MemoryControllerProcess& mc,
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam)
        : DMAEngineProcess(config, mc, std::vector<CreditPool*>{&l3_credits},
                           std::vector<TagCAM*>{&l3_tag_cam}) {}

    /**
     * @brief Construct a DMA engine that loads into several L3 tiles
     * @param l3_credits Each L3 tile's credit pool, by L3 tile index
     * @param l3_tag_cams Each L3 tile's Tag CAM, by L3 tile index
     *
     * A load lands in its tile's home L3 tile (TileDescriptor::l3_tile; -1 = tile 0) and takes
     * that tile's credit (docs/plans/noc-port-arbitration.md §4.1, step 4b.3).
     */
    DMAEngineProcess(const Config& config,
                     MemoryControllerProcess& mc,
                     std::vector<CreditPool*> l3_credits,
                     std::vector<TagCAM*> l3_tag_cams)
        : config_(config),
          mc_(mc),
          l3_credits_(std::move(l3_credits)),
          l3_tag_cams_(std::move(l3_tag_cams)),
          store_buffer_(config.store_buffer_blocks),
          next_slot_id_(0) {
        if (l3_credits_.empty() || l3_credits_.size() != l3_tag_cams_.size())
            throw std::invalid_argument("DMAEngineProcess: one credit pool and one Tag CAM per "
                                        "L3 tile, and at least one tile");
        if (config_.window > 0 && !mc_.hosted())
            throw std::invalid_argument(name() + ": a burst window needs a hosted DRAM "
                                        "controller (bursts need a burst size and an address map)");
    }

    /**
     * @brief Schedule a tile load from DRAM to L3
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be loaded when:
     * 1. An L3 credit is available
     * 2. MC can accept the request
     */
    void schedule_load(const TileDescriptor& tile) {
        // The pending list is a software staging queue and accepts every
        // scheduled request; the hardware constraint (queue_depth) bounds how
        // many requests are concurrently SUBMITTED to the memory controller.
        PendingRequest req;
        req.tile = tile;
        req.is_load = true;
        req.state = RequestState::WAITING_CREDIT;
        req.enqueue_cycle = current_cycle_;
        req.ticket = (static_cast<uint64_t>(config_.engine_id) << 40) | ++next_ticket_;

        pending_requests_.push_back(req);
    }

    /// Route loads over the NoC (step 4b.4). nullptr = they land in L3 on MC completion.
    void set_load_route(LoadRoute* route) { load_route_ = route; }

    /**
     * @brief A routed load has been delivered to its home hub's L3
     * @param tag The load's ticket, as given to LoadRoute::inject
     *
     * The load lands now: its Tag CAM entry appears and TILE_ARRIVED_L3 is emitted, so the
     * DRAM -> L3 copy and every downstream consumer see it at delivery, not at MC completion.
     */
    void land_load(uint64_t tag, Cycle now, std::vector<TimingEvent>& events) {
        for (auto& req : pending_requests_) {
            if (!req.is_load || req.state != RequestState::IN_TRANSIT || req.ticket != tag) continue;
            current_cycle_ = now;
            arrive_in_l3(req, events);
            req.state = RequestState::COMPLETED;
            return;
        }
        throw std::logic_error(name() + ": delivery of an unknown load (ticket " +
                               std::to_string(tag) + ")");
    }

    /**
     * @brief Schedule a tile store from this engine's store buffer to DRAM
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be stored when:
     * 1. A BlockMover has ejected it into store_buffer(), under the returned ticket
     * 2. No other store of the same tile is being written by this engine (the MC's
     *    completion names the tile, not the store, and write-after-write stays ordered)
     * 3. MC can accept the request
     *
     * @return the store's ticket; the BlockMover's ejection must deliver it
     */
    uint64_t schedule_store(const TileDescriptor& tile) {
        // Staging queue accepts every scheduled request (see schedule_load).
        PendingRequest req;
        req.tile = tile;
        req.is_load = false;
        req.state = RequestState::WAITING_TAG;
        req.enqueue_cycle = current_cycle_;
        req.ticket = (static_cast<uint64_t>(config_.engine_id) << 40) | ++next_ticket_;

        pending_requests_.push_back(req);
        return req.ticket;
    }

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfers from MC
        process_mc_completions(events);

        // Step 1b: routed loads read from DRAM enter the NoC (oldest first, as room allows)
        if (load_route_)
            for (auto& req : pending_requests_)
                if (req.state == RequestState::TO_INJECT &&
                    load_route_->inject(config_.engine_id, req.tile, req.ticket))
                    req.state = RequestState::IN_TRANSIT;

        // Step 2: Try to acquire credits and submit new loads
        process_pending_loads(events);

        // Step 3: Try to match tags and submit new stores
        process_pending_stores(events);

        // Step 3b: with a burst window, keep up to W bursts of the submitted tiles in flight
        if (config_.window > 0) issue_bursts(events);

        // Step 4: Remove completed requests
        remove_completed_requests();

        // Direct active-cycle measurement (follow-on 1b): this cycle counts as
        // active iff the MC is transferring on our behalf (>=1 SUBMITTED request).
        // This is measured, not derived from total - stalls, so idle cycles (no
        // request queued) are correctly excluded from utilization.
        if (!is_idle()) ++active_cycles_;

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        // DMA is idle if no requests are submitted to MC
        bool has_submitted = false;
        for (const auto& req : pending_requests_) {
            if (req.state == RequestState::SUBMITTED) {
                has_submitted = true;
                break;
            }
        }
        return !has_submitted;
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !pending_requests_.empty();
    }

    /**
     * @brief Check if DMA engine is complete (no pending or in-flight work)
     */
    [[nodiscard]] bool is_complete() const override {
        return pending_requests_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.engine_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name;
    }

    void reset() override {
        pending_requests_.clear();
        store_buffer_.reset();
        submitted_load_tiles_.clear();
        submitted_store_tiles_.clear();
        next_ticket_ = 0;
        bursts_in_flight_ = 0;
        next_slot_id_ = 0;
        stall_cycles_credit_ = 0;
        stall_cycles_tag_ = 0;
        active_cycles_ = 0;
        total_bytes_loaded_ = 0;
        total_bytes_stored_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t pending_count() const {
        return pending_requests_.size();
    }

    [[nodiscard]] size_t submitted_count() const {
        size_t count = 0;
        for (const auto& req : pending_requests_) {
            if (req.state == RequestState::SUBMITTED) count++;
        }
        return count;
    }

    [[nodiscard]] Cycle stall_cycles_credit() const {
        return stall_cycles_credit_;
    }

    [[nodiscard]] Cycle stall_cycles_tag() const {
        return stall_cycles_tag_;
    }

    [[nodiscard]] Cycle stall_cycles() const {
        return stall_cycles_credit_ + stall_cycles_tag_;
    }

    /// Directly measured cycles the engine was actively transferring (a request
    /// SUBMITTED to the MC). Excludes both stalled and idle cycles.
    [[nodiscard]] Cycle active_cycles() const {
        return active_cycles_;
    }

    [[nodiscard]] size_t total_bytes_loaded() const {
        return total_bytes_loaded_;
    }

    [[nodiscard]] size_t total_bytes_stored() const {
        return total_bytes_stored_;
    }

    [[nodiscard]] const Config& config() const {
        return config_;
    }

    /// Bursts this engine has in the controller right now (burst window; never above it).
    [[nodiscard]] size_t bursts_in_flight() const { return bursts_in_flight_; }

    /// Where a BlockMover ejects this engine's stores (schedule_eject).
    [[nodiscard]] DmaStoreBuffer& store_buffer() { return store_buffer_; }
    [[nodiscard]] const DmaStoreBuffer& store_buffer() const { return store_buffer_; }

private:
    Config config_;
    MemoryControllerProcess& mc_;
    std::vector<CreditPool*> l3_credits_;   // by L3 tile
    std::vector<TagCAM*> l3_tag_cams_;      // by L3 tile

    // The home L3 tile of `tile`: where it lands and whose credit it takes.
    [[nodiscard]] size_t home(const TileDescriptor& tile) const {
        const size_t h = tile.l3_tile < 0 ? 0 : static_cast<size_t>(tile.l3_tile);
        if (h >= l3_credits_.size())
            throw std::out_of_range(name() + ": " + tile.tile_id.to_string() + " is homed on L3 tile " +
                                    std::to_string(h) + "; there are " +
                                    std::to_string(l3_credits_.size()));
        return h;
    }
    [[nodiscard]] CreditPool& l3_credits(const TileDescriptor& t) { return *l3_credits_[home(t)]; }
    [[nodiscard]] TagCAM& l3_cam(const TileDescriptor& t) { return *l3_tag_cams_[home(t)]; }
    DmaStoreBuffer store_buffer_;

    std::vector<PendingRequest> pending_requests_;
    std::unordered_set<TileID, TileIDHash> submitted_load_tiles_;
    std::unordered_set<TileID, TileIDHash> submitted_store_tiles_;   // one write per tile at a time
    uint64_t next_ticket_ = 0;
    LoadRoute* load_route_ = nullptr;
    size_t bursts_in_flight_ = 0;           // burst window: across all this engine's tiles

    Cycle current_cycle_ = 0;
    uint32_t next_slot_id_ = 0;

    // Statistics
    Cycle stall_cycles_credit_ = 0;
    Cycle stall_cycles_tag_ = 0;
    Cycle active_cycles_ = 0;
    size_t total_bytes_loaded_ = 0;
    size_t total_bytes_stored_ = 0;

    /**
     * @brief Process completed transfers from MC
     */
    void process_mc_completions(std::vector<TimingEvent>& events) {
        if (config_.window > 0) {
            // Burst window: count each burst home; the tile completes with its last.
            while (auto b = mc_.get_completed_burst(config_.engine_id)) {
                for (auto& req : pending_requests_) {
                    if (req.state != RequestState::SUBMITTED || req.is_load != b->is_load ||
                        !(req.tile.tile_id == b->tile) || req.done >= req.sent)
                        continue;
                    ++req.done;
                    --bursts_in_flight_;
                    if (req.done == req.bursts) {
                        auto e = TimingEvent::duration_event(
                            req.is_load ? EventType::DMA_LOAD_COMPLETE : EventType::DMA_STORE_COMPLETE,
                            req.start_cycle, current_cycle_ - req.start_cycle, config_.engine_id,
                            req.tile.tile_id, name());
                        e.matrix_base_address = req.tile.matrix_base_address;
                        e.dram_address = req.tile.dram_address;
                        events.push_back(e);
                        finish(req, events);
                    }
                    break;
                }
            }
            return;
        }
        // Only poll for our own completions using our engine_id
        while (auto completed = mc_.get_completed_transfer(config_.engine_id)) {
            // Find matching pending request
            for (auto& req : pending_requests_) {
                if (req.state == RequestState::SUBMITTED &&
                    req.tile.tile_id == completed->tile.tile_id &&
                    req.is_load == completed->is_load) {
                    finish(req, events);
                    break;
                }
            }
        }
    }

    /// Hand a tile to the controller. Tile-level: one request, which the MC may refuse. Burst
    /// window: the tile is accepted and its bursts go out in issue_bursts as the window allows.
    bool submit(PendingRequest& req) {
        if (config_.window == 0) return mc_.submit_request(req.tile, req.is_load, config_.engine_id);
        req.bursts = mc_.bursts_of(req.tile);
        req.sent = req.done = 0;
        return true;
    }

    /// The tile is through DRAM: a load arrives in L3 (or enters the NoC first); a store
    /// retires, freeing its store-buffer slot.
    void finish(PendingRequest& req, std::vector<TimingEvent>& events) {
        if (req.is_load) {
            // Read from DRAM. Unrouted, the tile arrives in L3 now; routed, it enters the NoC
            // first and arrives when its hub delivers it.
            if (load_route_) {
                req.state = RequestState::TO_INJECT;
                return;
            }
            arrive_in_l3(req, events);
        } else {
            // Store complete: the tile is in DRAM, and its store-buffer slot is free. L3 was
            // freed when the BlockMover's ejection landed.
            total_bytes_stored_ += req.tile.size_bytes;
            store_buffer_.release();
            submitted_store_tiles_.erase(req.tile.tile_id);
            events.push_back(TimingEvent(EventType::DMA_STORE_RETIRED, current_cycle_,
                                         config_.engine_id, req.tile.tile_id, name()));
            events.back().store_ticket = req.ticket;
        }
        req.state = RequestState::COMPLETED;
    }

    /// Burst window: post the next bursts of the submitted tiles, oldest tile first, while fewer
    /// than W are in flight. A posted burst counts against the window; the controller grants it
    /// from its next tick, round-robin with the other engines it serves
    /// (MemoryControllerProcess::Config::arbitration).
    void issue_bursts(std::vector<TimingEvent>& events) {
        for (auto& req : pending_requests_) {
            if (req.state != RequestState::SUBMITTED) continue;
            if (req.bursts == 0) {              // a zero-byte tile: nothing to move
                req.start_cycle = current_cycle_;
                finish(req, events);
                continue;
            }
            while (req.sent < req.bursts && bursts_in_flight_ < config_.window) {
                mc_.post_burst(req.tile, req.sent, req.is_load, config_.engine_id);
                if (req.sent++ == 0) {
                    req.start_cycle = current_cycle_;
                    auto e = TimingEvent(req.is_load ? EventType::DMA_LOAD_START
                                                     : EventType::DMA_STORE_START,
                                         current_cycle_, config_.engine_id, req.tile.tile_id, name());
                    e.matrix_base_address = req.tile.matrix_base_address;
                    e.dram_address = req.tile.dram_address;
                    events.push_back(e);
                }
                ++bursts_in_flight_;
            }
            if (bursts_in_flight_ >= config_.window) return;
        }
    }

    /**
     * @brief Try to acquire credits and submit pending loads to MC
     */
    void process_pending_loads(std::vector<TimingEvent>& events) {
        size_t in_flight = submitted_count();
        bool credit_stalled = false;
        TileID stalled_tile{};
        for (auto& req : pending_requests_) {
            if (!req.is_load || req.state != RequestState::WAITING_CREDIT) {
                continue;
            }

            // Check if tile is already in L3 (supports tile reuse)
            TagCAM& cam = l3_cam(req.tile);
            if (req.tile.l3_held &&
                std::any_of(pending_requests_.begin(), pending_requests_.end(), [&](const PendingRequest& o) {
                    return !o.is_load && o.state != RequestState::COMPLETED && o.tile.ordered &&
                           o.tile.program_seq < req.tile.program_seq && o.tile.tile_id == req.tile.tile_id;
                })) {
                // A CSP program's load after its own store of the tile (a result read back): DRAM
                // has the new bytes only when that store retires. Stores and loads of one tile
                // share this engine (tile-affine), so waiting here orders them -- by the
                // program's order, since a driver may hand loads over later than stores.
                continue;
            }
            if (cam.lookup(req.tile.tile_id) && req.tile.l3_held) {
                // A CSP program's load: the program decided this tile comes from DRAM; the copy
                // still in L3 belongs to its previous residency, which the program's Release
                // retires. Wait for it (credits up, data down).
                continue;
            }
            if (cam.lookup(req.tile.tile_id)) {
                // Tile already in its home L3 tile - just increment ref_count, no credit needed
                auto entry = cam.match(req.tile.tile_id);
                cam.insert(req.tile.tile_id, entry->slot_id, current_cycle_);

                events.push_back(TimingEvent(
                    EventType::TILE_ARRIVED_L3,
                    current_cycle_,
                    config_.engine_id,
                    req.tile.tile_id,
                    name()
                ));
                events.back().slot_id = entry->slot_id;
                events.back().matrix_base_address = req.tile.matrix_base_address;
                events.back().dram_address = req.tile.dram_address;

                req.state = RequestState::COMPLETED;
                continue;
            }

            // Defer if a load for the same tile is already in flight: when it
            // completes and inserts into L3, this request resolves via the
            // dedup path above. Submitting a second real load would acquire a
            // second credit for a single TagCAM entry (which releases only
            // one credit), leaking credits on reuse-heavy schedules (#61).
            if (submitted_load_tiles_.count(req.tile.tile_id) > 0) {
                continue;
            }

            // Hardware queue slot required beyond this point
            if (in_flight >= config_.queue_depth) {
                continue;  // All slots occupied - dedup hits above can still complete
            }

            // Need L3 credit (from the tile's matrix partition when the pool
            // is partitioned, per #89 - partitioning makes the downstream
            // protection structural). Loads may not consume the reserved
            // credits - those are kept for C-tile writebacks (BlockMover
            // L2->L3). Without the reserve, greedy prefetch loads re-acquire
            // every freed credit before a writeback can, starving the
            // downstream path and wedging the pipeline (credit cycle livelock).
            const size_t load_part = static_cast<size_t>(req.tile.tile_id.matrix);
            CreditPool& credits = l3_credits(req.tile);
            if (credits.available(load_part) <= config_.l3_credit_reserve ||
                !credits.acquire(load_part)) {
                if (!credit_stalled) {
                    credit_stalled = true;
                    stalled_tile = req.tile.tile_id;
                }
                continue;  // Try other requests
            }

            // Got credit - submit to MC with our engine_id
            req.slot_id = allocate_slot(cam);
            if (!submit(req)) {
                // MC queue full - release credit and retry later
                credits.release(load_part);
                continue;
            }

            req.state = RequestState::SUBMITTED;
            submitted_load_tiles_.insert(req.tile.tile_id);
            ++in_flight;

            events.push_back(TimingEvent(
                EventType::CREDIT_ACQUIRED,
                current_cycle_,
                config_.engine_id,
                req.tile.tile_id,
                name()
            ));
        }

        // Stall accounting: at most one stall cycle per tick (matches BM/STR)
        if (credit_stalled) {
            events.push_back(TimingEvent(
                EventType::DMA_STALL_CREDIT,
                current_cycle_,
                config_.engine_id,
                stalled_tile,
                name()
            ));
            stall_cycles_credit_++;
        }
    }

    /**
     * @brief Try to match tags and submit pending stores to MC
     */
    void process_pending_stores(std::vector<TimingEvent>& events) {
        size_t in_flight = submitted_count();
        bool tag_stalled = false;
        TileID stalled_tile{};
        for (auto& req : pending_requests_) {
            if (req.is_load || req.state != RequestState::WAITING_TAG) {
                continue;
            }
            if (in_flight >= config_.queue_depth) {
                break;  // All hardware queue slots occupied
            }

            // Need this store's ticket in the buffer (a BlockMover ejected it there)
            if (!store_buffer_.staged(req.ticket)) {
                if (!tag_stalled) {
                    tag_stalled = true;
                    stalled_tile = req.tile.tile_id;
                }
                continue;  // Try other requests
            }

            // Another store of this tile is being written by this engine: wait for it
            if (submitted_store_tiles_.count(req.tile.tile_id) > 0) continue;

            // Tile is in the buffer - submit to MC with our engine_id
            if (!submit(req)) {
                // MC queue full - retry later
                continue;
            }
            store_buffer_.take(req.ticket);   // the slot stays held until the write ends
            submitted_store_tiles_.insert(req.tile.tile_id);

            req.state = RequestState::SUBMITTED;
            ++in_flight;
        }

        // Stall accounting: at most one stall cycle per tick (matches BM/STR)
        if (tag_stalled) {
            events.push_back(TimingEvent(
                EventType::DMA_STALL_TAG,
                current_cycle_,
                config_.engine_id,
                stalled_tile,
                name()
            ));
            stall_cycles_tag_++;
        }
    }

    /**
     * @brief Remove completed requests from the list
     */
    void remove_completed_requests() {
        pending_requests_.erase(
            std::remove_if(pending_requests_.begin(), pending_requests_.end(),
                [](const PendingRequest& req) {
                    return req.state == RequestState::COMPLETED;
                }),
            pending_requests_.end()
        );
    }

    /// The load is in its home L3 tile: Tag CAM entry, and TILE_ARRIVED_L3.
    void arrive_in_l3(const PendingRequest& req, std::vector<TimingEvent>& events) {
        // One reference: the legacy path's Move consumes it; a CSP program's Release retires it.
        l3_cam(req.tile).insert(req.tile.tile_id, req.slot_id, current_cycle_);
        submitted_load_tiles_.erase(req.tile.tile_id);
        total_bytes_loaded_ += req.tile.size_bytes;

        events.push_back(TimingEvent(
            EventType::TILE_ARRIVED_L3,
            current_cycle_,
            config_.engine_id,
            req.tile.tile_id,
            name()
        ));
        events.back().slot_id = req.slot_id;
        events.back().matrix_base_address = req.tile.matrix_base_address;
        events.back().dram_address = req.tile.dram_address;
    }

    /**
     * @brief Allocate a buffer slot (round-robin)
     */
    uint32_t allocate_slot(const TagCAM& cam) {
        uint32_t slot = next_slot_id_ % static_cast<uint32_t>(cam.capacity());
        next_slot_id_ = (slot + 1) % static_cast<uint32_t>(cam.capacity());
        return slot;
    }
};

} // namespace sw::kpu::timing

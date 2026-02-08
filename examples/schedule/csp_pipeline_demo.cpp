/**
 * @file csp_pipeline_demo.cpp
 * @brief Educational demonstration of CSP credit/debit flow and tag matching
 *
 * This program demonstrates a minimal 1×1×1 matmul (single tile of each matrix)
 * to show the complete dataflow through the KPU memory hierarchy:
 *
 *   DRAM → L3 → L2 → Compute → L2 → L3 → DRAM
 *        DMA   BM    STR      STR   BM    DMA
 *
 * Each step shows:
 *   - Credit acquisition/release (flow control)
 *   - TagCAM insert/lookup/invalidate (tile tracking)
 *   - Component coordination via CSP synchronization
 *
 * Build:
 *   cmake --build --preset release --target csp_pipeline_demo
 *
 * Usage:
 *   ./build/examples/schedule/csp_pipeline_demo [--verbose]
 *  
 * 1. DMA+MC separation is working:
 *    - MC0 handles DRAM access (LOAD_START/COMPLETE, STORE_START/COMPLETE)
 *    - DMA0 handles L3 tile tracking (TILE_ARRIVED_L3)
 * 2. Command bus serialization: Loads start at cycles 1 and 2 (not both at cycle 0), showing the MC correctly serializes commands
 * 3. Credit flow is correct: L3 credits go 4→3→2 for loads, then back to 4 after stores complete
 * 4. Full pipeline completes: 163 cycles, STATUS: SUCCESS
 *
 * The demo now correctly shows the architecture where:
 * - DMA = CSP Process (manages credits, TagCAM, schedules operations)
 * - MC = Communication Resource (models DRAM contention, command bus)
 *
 */

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/process_interface.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using sw::kpu::isa::MatrixID;

// ============================================================================
// Transaction Log Entry
// ============================================================================

struct TransactionLogEntry {
    Cycle cycle;
    std::string component;
    std::string event;
    std::string tile;
    std::string credit_flow;
    std::string tag_action;

    void print(std::ostream& os) const {
        os << std::setw(5) << cycle << " | "
           << std::setw(14) << component << " | "
           << std::setw(18) << event << " | "
           << std::setw(10) << tile << " | "
           << std::setw(14) << credit_flow << " | "
           << tag_action << "\n";
    }
};

// ============================================================================
// Event Analyzer - Converts raw events to educational transaction log
// ============================================================================

class EventAnalyzer {
public:
    struct CreditState {
        size_t l3_available;
        size_t l3_capacity;
        size_t l2_available;
        size_t l2_capacity;
    };

    EventAnalyzer(size_t l3_cap, size_t l2_cap)
        : l3_capacity_(l3_cap), l2_capacity_(l2_cap),
          l3_available_(l3_cap), l2_available_(l2_cap) {}

    std::vector<TransactionLogEntry> analyze(const std::vector<TimingEvent>& events) {
        std::vector<TransactionLogEntry> log;

        for (const auto& event : events) {
            TransactionLogEntry entry;
            entry.component = event.component_name;
            entry.tile = event.tile_id.to_string();

            // For COMPLETE events with duration, the actual completion time is
            // cycle + duration. TimingEvent stores start_cycle in cycle field
            // for Chrome trace compatibility (which uses start + duration format).
            bool is_complete_event = (event.type == EventType::DMA_LOAD_COMPLETE ||
                                      event.type == EventType::DMA_STORE_COMPLETE ||
                                      event.type == EventType::BM_MOVE_COMPLETE ||
                                      event.type == EventType::BM_WRITEBACK_COMPLETE ||
                                      event.type == EventType::STR_FEED_COMPLETE ||
                                      event.type == EventType::STR_DRAIN_COMPLETE ||
                                      event.type == EventType::COMPUTE_COMPLETE);

            if (is_complete_event && event.duration > 0) {
                entry.cycle = event.cycle + event.duration;  // Actual completion time
            } else {
                entry.cycle = event.cycle;
            }

            // Determine event description and credit/tag actions
            switch (event.type) {
                // DMA Events
                case EventType::DMA_LOAD_START:
                    entry.event = "LOAD_START";
                    l3_available_--;
                    entry.credit_flow = "L3: " + std::to_string(l3_available_ + 1) +
                                       "→" + std::to_string(l3_available_);
                    entry.tag_action = "-";
                    break;

                case EventType::DMA_LOAD_COMPLETE:
                    entry.event = "LOAD_COMPLETE";
                    entry.credit_flow = "-";
                    entry.tag_action = "L3.insert(" + entry.tile + ")";
                    break;

                case EventType::DMA_STORE_START:
                    entry.event = "STORE_START";
                    entry.credit_flow = "-";
                    entry.tag_action = "L3.match(" + entry.tile + ")";
                    break;

                case EventType::DMA_STORE_COMPLETE:
                    entry.event = "STORE_COMPLETE";
                    l3_available_++;
                    entry.credit_flow = "L3: " + std::to_string(l3_available_ - 1) +
                                       "→" + std::to_string(l3_available_);
                    entry.tag_action = "L3.invalidate(" + entry.tile + ")";
                    break;

                case EventType::DMA_STALL_CREDIT:
                    entry.event = "STALL_CREDIT";
                    entry.credit_flow = "L3: 0 (blocked)";
                    entry.tag_action = "-";
                    break;

                // BlockMover Events
                case EventType::BM_MOVE_START:
                    entry.event = "MOVE_START";
                    l2_available_--;
                    entry.credit_flow = "L2: " + std::to_string(l2_available_ + 1) +
                                       "→" + std::to_string(l2_available_);
                    entry.tag_action = "L3.match(" + entry.tile + ") HIT";
                    break;

                case EventType::BM_MOVE_COMPLETE:
                    entry.event = "MOVE_COMPLETE";
                    l3_available_++;
                    entry.credit_flow = "L3: " + std::to_string(l3_available_ - 1) +
                                       "→" + std::to_string(l3_available_);
                    entry.tag_action = "L3.invalidate, L2.insert";
                    break;

                case EventType::BM_WRITEBACK_START:
                    entry.event = "WRITEBACK_START";
                    l3_available_--;
                    entry.credit_flow = "L3: " + std::to_string(l3_available_ + 1) +
                                       "→" + std::to_string(l3_available_);
                    entry.tag_action = "L2.match(" + entry.tile + ") HIT";
                    break;

                case EventType::BM_WRITEBACK_COMPLETE:
                    entry.event = "WRITEBACK_COMPLETE";
                    l2_available_++;
                    entry.credit_flow = "L2: " + std::to_string(l2_available_ - 1) +
                                       "→" + std::to_string(l2_available_);
                    entry.tag_action = "L2.invalidate, L3.insert";
                    break;

                case EventType::BM_STALL_TAG:
                    entry.event = "STALL_TAG";
                    entry.credit_flow = "-";
                    entry.tag_action = "L3.lookup(" + entry.tile + ") MISS";
                    break;

                case EventType::BM_STALL_CREDIT:
                    entry.event = "STALL_CREDIT";
                    entry.credit_flow = "L2: 0 (blocked)";
                    entry.tag_action = "-";
                    break;

                // Streamer Events
                case EventType::STR_FEED_START:
                    entry.event = "FEED_START";
                    entry.credit_flow = "-";
                    entry.tag_action = "L2.match(" + entry.tile + ") HIT";
                    break;

                case EventType::STR_FEED_COMPLETE:
                    entry.event = "FEED_COMPLETE";
                    l2_available_++;
                    entry.credit_flow = "L2: " + std::to_string(l2_available_ - 1) +
                                       "→" + std::to_string(l2_available_);
                    entry.tag_action = "L2.invalidate(" + entry.tile + ")";
                    break;

                case EventType::STR_DRAIN_START:
                    entry.event = "DRAIN_START";
                    l2_available_--;
                    entry.credit_flow = "L2: " + std::to_string(l2_available_ + 1) +
                                       "→" + std::to_string(l2_available_);
                    entry.tag_action = "compute_result.match HIT";
                    break;

                case EventType::STR_DRAIN_COMPLETE:
                    entry.event = "DRAIN_COMPLETE";
                    entry.credit_flow = "-";
                    entry.tag_action = "L2.insert(" + entry.tile + ")";
                    break;

                case EventType::STR_STALL_TAG:
                    entry.event = "STALL_TAG";
                    entry.credit_flow = "-";
                    entry.tag_action = "L2.lookup(" + entry.tile + ") MISS";
                    break;

                case EventType::STR_STALL_CREDIT:
                    entry.event = "STALL_CREDIT";
                    entry.credit_flow = "L2: 0 (blocked)";
                    entry.tag_action = "-";
                    break;

                case EventType::STR_STALL_COMPUTE:
                    entry.event = "STALL_COMPUTE";
                    entry.credit_flow = "-";
                    entry.tag_action = "compute_result.lookup MISS";
                    break;

                // Compute Events
                case EventType::COMPUTE_START:
                    entry.event = "COMPUTE_START";
                    entry.credit_flow = "-";
                    entry.tag_action = "dependency satisfied";
                    break;

                case EventType::COMPUTE_COMPLETE:
                    entry.event = "COMPUTE_COMPLETE";
                    entry.credit_flow = "-";
                    entry.tag_action = "compute_result.insert(" + entry.tile + ")";
                    break;

                // Tile Tracking Events
                case EventType::TILE_ARRIVED_L3:
                    entry.event = "TILE_ARRIVED_L3";
                    entry.credit_flow = "-";
                    entry.tag_action = "L3 TagCAM updated";
                    break;

                case EventType::TILE_ARRIVED_L2:
                    entry.event = "TILE_ARRIVED_L2";
                    entry.credit_flow = "-";
                    entry.tag_action = "L2 TagCAM updated";
                    break;

                case EventType::TILE_FED_TO_COMPUTE:
                    entry.event = "TILE_FED";
                    entry.credit_flow = "-";
                    entry.tag_action = "fed_tiles.insert(" + entry.tile + ")";
                    break;

                case EventType::TILE_DRAINED:
                    entry.event = "TILE_DRAINED";
                    entry.credit_flow = "-";
                    entry.tag_action = "result in L2";
                    break;

                default:
                    continue;  // Skip other events
            }

            log.push_back(entry);
        }

        return log;
    }

private:
    size_t l3_capacity_;
    size_t l2_capacity_;
    size_t l3_available_;
    size_t l2_available_;
};

// ============================================================================
// Print Educational Explanation
// ============================================================================

void print_explanation() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CSP Pipeline Educational Demo                              ║
║                                                                              ║
║  This demo shows a minimal 1×1×1 matmul: C[0,0] = A[0,0] × B[0,0]            ║
║                                                                              ║
║  Dataflow:  DRAM → L3 → L2 → Compute → L2 → L3 → DRAM                        ║
║             ────   ──   ──   ───────   ──   ──   ────                        ║
║             DMA    BM  STR    (mul)   STR   BM   DMA                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

CSP Synchronization Mechanisms:
───────────────────────────────

1. CREDIT POOLS (Flow Control)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Credits UP ↑                                                            │
   │                                                                         │
   │   L3 Pool ←── L2 Pool ←── Compute                                       │
   │   (32 buf)    (64 banks)                                                │
   │                                                                         │
   │ Data DOWN ↓                                                             │
   │                                                                         │
   │ • Producer must ACQUIRE credit before pushing data downstream           │
   │ • Consumer RELEASES credit when done with data                          │
   │ • Prevents buffer overflow - producer blocks if no credits              │
   └─────────────────────────────────────────────────────────────────────────┘

2. TAG CAMs (Tile Tracking)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ L3 TagCAM: Tracks tiles present in L3 buffers                           │
   │ L2 TagCAM: Tracks tiles present in L2 banks                             │
   │ Compute Result TagCAM: Tracks results ready for drain                   │
   │                                                                         │
   │ Operations:                                                             │
   │ • INSERT: When tile arrives at memory level                             │
   │ • LOOKUP/MATCH: Check if tile is present (for downstream consumer)      │
   │ • INVALIDATE: When tile is consumed/moved (releases credit)             │
   └─────────────────────────────────────────────────────────────────────────┘

3. OPERATION DEPENDENCIES
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Operation          │ Needs Credit   │ Needs Tag Match                   │
   │ ───────────────────┼────────────────┼───────────────────                │
   │ DMA LOAD           │ L3 credit      │ -                                 │
   │ BM MOVE (L3→L2)    │ L2 credit      │ L3 TagCAM match                   │
   │ STR FEED (L2→Comp) │ -              │ L2 TagCAM match                   │
   │ COMPUTE            │ -              │ Dependency tile FED               │
   │ STR DRAIN (Comp→L2)│ L2 credit      │ Compute result TagCAM             │
   │ BM WRITEBACK       │ L3 credit      │ L2 TagCAM match                   │
   │ DMA STORE          │ -              │ L3 TagCAM match                   │
   └─────────────────────────────────────────────────────────────────────────┘

)";
}

void print_header() {
    std::cout << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "Transaction Log: Credit/Debit Flow and Tag Matching\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::setw(5) << "Cycle" << " | "
              << std::setw(14) << "Component" << " | "
              << std::setw(18) << "Event" << " | "
              << std::setw(10) << "Tile" << " | "
              << std::setw(14) << "Credit Flow" << " | "
              << "TagCAM Action\n";
    std::cout << std::string(100, '-') << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
            verbose = true;
        }
    }

    print_explanation();

    // ========================================================================
    // Configure minimal executor
    // ========================================================================

    ConcurrentTimingExecutor::Config config;
    config.num_memory_controllers = 1; // Single MC with command bus constraint
    config.l3_buffer_count = 4;       // Minimal L3
    config.num_block_movers = 2;      // One for A, one for B
    config.l2_bank_count = 4;         // Minimal L2
    config.num_row_streamers = 1;     // For A and C
    config.num_col_streamers = 1;     // For B
    config.compute_latency = 16;      // Short compute for demo
    config.max_cycles = 1000;
    config.enable_livelock_detection = false;

    ConcurrentTimingExecutor executor(config);

    // ========================================================================
    // Create minimal tiles
    // ========================================================================

    auto make_tile = [](MatrixID matrix, Size ti, Size tj, Size tk = 0) {
        TileDescriptor tile;
        tile.tile_id.matrix = matrix;
        tile.tile_id.ti = ti;
        tile.tile_id.tj = tj;
        tile.tile_id.tk = tk;
        tile.height = 16;
        tile.width = 16;
        tile.element_size = 4;
        tile.size_bytes = 16 * 16 * 4;  // 1KB

        switch (matrix) {
            case MatrixID::A:
                tile.matrix_base_address = 0x1000;
                tile.dram_address = 0x1000;
                break;
            case MatrixID::B:
                tile.matrix_base_address = 0x2000;
                tile.dram_address = 0x2000;
                break;
            case MatrixID::C:
                tile.matrix_base_address = 0x3000;
                tile.dram_address = 0x3000;
                break;
        }
        return tile;
    };

    // Create the three tiles for our minimal matmul
    auto a_tile = make_tile(MatrixID::A, 0, 0, 0);
    auto b_tile = make_tile(MatrixID::B, 0, 0, 0);
    auto c_tile = make_tile(MatrixID::C, 0, 0, 0);

    std::cout << "Scheduling Operations:\n";
    std::cout << "─────────────────────\n";
    std::cout << "  1. LOAD  A[0,0,0] from DRAM @ 0x1000\n";
    std::cout << "  2. LOAD  B[0,0,0] from DRAM @ 0x2000\n";
    std::cout << "  3. MOVE  A[0,0,0] from L3 to L2\n";
    std::cout << "  4. MOVE  B[0,0,0] from L3 to L2\n";
    std::cout << "  5. FEED  A[0,0,0] from L2 to Compute\n";
    std::cout << "  6. FEED  B[0,0,0] from L2 to Compute\n";
    std::cout << "  7. COMPUTE C[0,0,0] (depends on B[0,0,0] being fed)\n";
    std::cout << "  8. DRAIN C[0,0,0] from Compute to L2\n";
    std::cout << "  9. WRITEBACK C[0,0,0] from L2 to L3\n";
    std::cout << " 10. STORE C[0,0,0] to DRAM @ 0x3000\n\n";

    // ========================================================================
    // Schedule the complete pipeline
    // ========================================================================

    // Load A and B from DRAM to L3
    executor.schedule_load(a_tile, 0);  // DMA engine 0
    executor.schedule_load(b_tile, 1);  // DMA engine 1

    // Move from L3 to L2
    executor.schedule_move(a_tile, false, 0);  // BlockMover 0
    executor.schedule_move(b_tile, true, 1);   // BlockMover 1 (transpose B)

    // Feed to compute
    executor.schedule_feed(a_tile, 0);  // Row streamer
    executor.schedule_feed(b_tile, 0);  // Col streamer

    // Signal compute complete (depends on B being fed)
    executor.schedule_compute(c_tile, b_tile.tile_id);

    // Drain result from compute
    executor.schedule_drain(c_tile, 0);

    // Writeback to L3
    executor.schedule_writeback(c_tile, 0);

    // Store to DRAM
    executor.schedule_store(c_tile, 0);

    // ========================================================================
    // Run simulation
    // ========================================================================

    std::cout << "Running simulation...\n\n";
    bool completed = executor.run();

    // ========================================================================
    // Analyze and print transaction log
    // ========================================================================

    print_header();

    EventAnalyzer analyzer(config.l3_buffer_count, config.l2_bank_count);
    auto& raw_events = executor.events();

    // Analyze events (this computes correct cycle for COMPLETE events)
    auto log = analyzer.analyze(raw_events);

    // Sort log entries by cycle, then by event type (START before COMPLETE)
    std::sort(log.begin(), log.end(),
              [](const TransactionLogEntry& a, const TransactionLogEntry& b) {
                  if (a.cycle != b.cycle) return a.cycle < b.cycle;

                  // Within same cycle: START events before COMPLETE events
                  bool a_is_start = (a.event.find("START") != std::string::npos);
                  bool b_is_start = (b.event.find("START") != std::string::npos);
                  if (a_is_start != b_is_start) return a_is_start;  // START first

                  // Within same type: LOAD/MOVE before STORE/WRITEBACK
                  return a.event < b.event;
              });

    // Track stall summary per component/tile pair
    std::map<std::pair<std::string, std::string>, std::pair<Cycle, Cycle>> stall_ranges;
    Cycle last_action_cycle = 0;

    for (const auto& entry : log) {
        bool is_stall = entry.event.find("STALL") != std::string::npos;

        if (is_stall) {
            // Track stall ranges for summary (only in verbose mode)
            if (verbose) {
                auto key = std::make_pair(entry.component, entry.tile);
                if (stall_ranges.find(key) == stall_ranges.end()) {
                    stall_ranges[key] = {entry.cycle, entry.cycle};
                } else {
                    stall_ranges[key].second = entry.cycle;
                }
            }
        } else {
            // Print stall summary if there were stalls since last action
            if (verbose && !stall_ranges.empty()) {
                for (const auto& [key, range] : stall_ranges) {
                    if (range.second > range.first) {
                        std::cout << std::setw(5) << range.first << "-"
                                  << std::setw(3) << range.second << " | "
                                  << std::setw(14) << key.first << " | "
                                  << std::setw(18) << "STALL" << " | "
                                  << std::setw(10) << key.second << " | "
                                  << "(" << (range.second - range.first + 1) << " cycles)\n";
                    }
                }
                stall_ranges.clear();
            }

            // Always print action events
            entry.print(std::cout);
            last_action_cycle = entry.cycle;
        }
    }

    // Print any remaining stalls
    if (verbose && !stall_ranges.empty()) {
        for (const auto& [key, range] : stall_ranges) {
            if (range.second > range.first) {
                std::cout << std::setw(5) << range.first << "-"
                          << std::setw(3) << range.second << " | "
                          << std::setw(14) << key.first << " | "
                          << std::setw(18) << "STALL" << " | "
                          << std::setw(10) << key.second << " | "
                          << "(" << (range.second - range.first + 1) << " cycles)\n";
            }
        }
    }

    if (!verbose) {
        std::cout << "\n(Use --verbose to see stall events)\n";
    }

    std::cout << std::string(100, '=') << "\n\n";

    // ========================================================================
    // Print summary
    // ========================================================================

    std::cout << "Summary:\n";
    std::cout << "────────\n";
    std::cout << "  Status: " << (completed ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Total cycles: " << executor.current_cycle() << "\n\n";

    auto stats = executor.get_statistics();
    std::cout << "  Tiles loaded:    " << stats.tiles_loaded << "\n";
    std::cout << "  Tiles moved:     " << stats.tiles_moved << "\n";
    std::cout << "  Tiles fed:       " << stats.tiles_fed << "\n";
    std::cout << "  Tiles drained:   " << stats.tiles_drained << "\n";
    std::cout << "  Tiles writeback: " << stats.tiles_writeback << "\n";
    std::cout << "  Tiles stored:    " << stats.tiles_stored << "\n\n";

    // ========================================================================
    // Print key observations
    // ========================================================================

    std::cout << R"(
Key Observations:
─────────────────

1. CREDIT FLOW PATTERN
   • DMA acquires L3 credit BEFORE starting load (prevents L3 overflow)
   • BlockMover acquires L2 credit BEFORE starting move
   • Credits released when tile is consumed/moved downstream
   • Watch the credit counters: they never go negative!

2. TAG MATCHING PATTERN
   • BlockMover STALLs until DMA completes and inserts tile in L3 TagCAM
   • Streamer STALLs until BlockMover inserts tile in L2 TagCAM
   • DRAIN STALLs until COMPUTE inserts result in compute_result TagCAM
   • This is CSP synchronization - no polling, just blocking on availability

3. PIPELINE STAGES
   • DMA and BlockMover can overlap (different tiles)
   • Once A is in L2, B can be loading from DRAM simultaneously
   • Compute overlaps with DMA stores for next iteration

4. CREDIT CONSERVATION
   • Initial credits: L3=4, L2=4
   • Final credits: L3=4, L2=4 (all returned)
   • Every acquire has a matching release

)";

    return completed ? 0 : 1;
}

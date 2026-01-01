# NoC Benchmark Chrome Trace Design

## Overview

This document describes the Chrome Trace capture system for visualizing NoC benchmark data movement and routing behavior. The trace format is compatible with:
- Chrome's `chrome://tracing`
- [Perfetto UI](https://ui.perfetto.dev)

## Goals

1. **Visualize data movement patterns** - Show tile transfers across the mesh
2. **Analyze routing behavior** - Track flit-level routing through routers
3. **Identify bottlenecks** - Highlight congestion hotspots and backpressure
4. **Compare NoC implementations** - Side-by-side comparison of WormholeNoC vs DataflowNoC
5. **Debug benchmark issues** - Detailed event traces for investigation

---

## Chrome Trace Event Format

Chrome Trace uses a JSON array of event objects. Key event types:

| Phase | Description | Use Case |
|-------|-------------|----------|
| `X` | Complete event (has duration) | Tile transfers, compute |
| `B`/`E` | Begin/End pair | Long-running operations |
| `i` | Instant event | State changes, markers |
| `s`/`f` | Flow events (start/finish) | **Packet causality tracking** |
| `M` | Metadata | Process/thread names |
| `C` | Counter | Bandwidth utilization |

### Flow Events for Packet Tracking

The key innovation is using **flow events** to show packet causality:

```json
// Injection at source
{"name": "A[0,0]", "cat": "inject", "ph": "s", "id": 12345,
 "ts": 100, "pid": 1, "tid": 0}

// Delivery at destination
{"name": "A[0,0]", "cat": "deliver", "ph": "f", "id": 12345,
 "ts": 200, "pid": 1, "tid": 5, "bp": "e"}
```

In Perfetto, this renders as an **arrow** from injection to delivery.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      NoCBenchmarkTracer                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ PacketTracker   │  │ RouterTracker   │  │ LinkTracker     │     │
│  │                 │  │                 │  │                 │     │
│  │ - inject_cycle  │  │ - input_events  │  │ - occupancy     │     │
│  │ - deliver_cycle │  │ - output_events │  │ - contention    │     │
│  │ - hop_sequence  │  │ - backpressure  │  │ - bandwidth     │     │
│  │ - tile_info     │  │ - buffer_levels │  │ - utilization   │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ChromeTraceBuilder                        │   │
│  │                                                              │   │
│  │  - emit_metadata()      // Process/thread names             │   │
│  │  - emit_flow_events()   // Packet inject→deliver arrows     │   │
│  │  - emit_router_events() // Per-router activity             │   │
│  │  - emit_link_events()   // Link utilization                 │   │
│  │  - emit_counters()      // Bandwidth/backpressure counters  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Process Hierarchy

The trace uses a hierarchical process structure for clear visualization:

```
PID 1: "NoC Routers"
  ├── TID 0: "Router[0,0]"
  ├── TID 1: "Router[0,1]"
  ├── ...
  └── TID 15: "Router[3,3]"

PID 2: "East Links (→)"
  ├── TID 0: "[0,0]→[0,1]"
  ├── TID 1: "[0,1]→[0,2]"
  ├── ...
  └── TID 11: "[3,2]→[3,3]"

PID 3: "South Links (↓)"
  ├── TID 0: "[0,0]↓[1,0]"
  ├── ...
  └── TID 11: "[2,3]↓[3,3]"

PID 4: "Packets"
  ├── TID 0: "Injected Packets"
  └── TID 1: "Delivered Packets"

PID 5: "Counters"
  ├── TID 0: "Bandwidth (B/cycle)"
  ├── TID 1: "Backpressure (%)"
  └── TID 2: "Active Flits"
```

---

## Event Types

### 1. Packet Events (with Flow Arrows)

```cpp
struct PacketEvent {
    uint64_t packet_id;      // Unique ID for flow correlation
    TileDescriptor tile;     // A[m,k], B[k,n], C[m,n]
    uint8_t src_router;
    uint8_t dst_router;
    uint64_t inject_cycle;
    uint64_t deliver_cycle;
    std::vector<HopEvent> hops;  // Full path through mesh
};
```

**Chrome Trace Output:**
```json
// Flow start (injection)
{"name": "A[0,1]", "cat": "packet", "ph": "s", "id": 1001,
 "ts": 50, "pid": 1, "tid": 0,
 "args": {"src": "R[0,0]", "dst": "R[1,2]", "size": 4096}},

// Router hop (complete event showing time at router)
{"name": "HOP", "cat": "routing", "ph": "X",
 "ts": 55, "dur": 5, "pid": 1, "tid": 1,
 "args": {"packet": 1001, "in": "WEST", "out": "EAST"}},

// Flow end (delivery)
{"name": "A[0,1]", "cat": "packet", "ph": "f", "id": 1001,
 "ts": 100, "pid": 1, "tid": 6, "bp": "e",
 "args": {"latency": 50, "hops": 3}}
```

### 2. Router Activity Events

```cpp
struct RouterEvent {
    uint8_t router_id;
    uint64_t cycle;
    RouterState state;  // IDLE, ROUTING, BLOCKED
    uint8_t input_port;
    uint8_t output_port;
    uint8_t buffer_occupancy;
};
```

**Chrome Trace Output:**
```json
// Router busy routing packet
{"name": "ROUTING", "cat": "router", "ph": "X",
 "ts": 50, "dur": 10, "pid": 1, "tid": 5,
 "args": {"in": "WEST", "out": "EAST", "packet": 1001}},

// Router blocked (backpressure)
{"name": "BLOCKED", "cat": "backpressure", "ph": "X",
 "ts": 60, "dur": 5, "pid": 1, "tid": 5,
 "args": {"reason": "output_full", "port": "EAST"}}
```

### 3. Link Utilization Events

```cpp
struct LinkEvent {
    uint8_t src_router;
    uint8_t dst_router;
    Direction direction;  // EAST, SOUTH, etc.
    uint64_t start_cycle;
    uint64_t end_cycle;
    uint32_t bytes;
};
```

**Chrome Trace Output:**
```json
{"name": "TRANSFER", "cat": "link", "ph": "X",
 "ts": 50, "dur": 64, "pid": 2, "tid": 3,
 "args": {"bytes": 4096, "bw": 64.0}}
```

### 4. Counter Events (Time Series)

```json
// Bandwidth counter
{"name": "bandwidth", "cat": "counter", "ph": "C",
 "ts": 100, "pid": 5, "tid": 0,
 "args": {"bytes_per_cycle": 48.5}},

// Backpressure counter
{"name": "backpressure", "cat": "counter", "ph": "C",
 "ts": 100, "pid": 5, "tid": 1,
 "args": {"blocked_routers": 4, "percent": 25.0}},

// Active flits in network
{"name": "active_flits", "cat": "counter", "ph": "C",
 "ts": 100, "pid": 5, "tid": 2,
 "args": {"count": 128}}
```

---

## API Design

### NoCBenchmarkTracer

```cpp
namespace sw::benchmark {

/// Configuration for benchmark tracing
struct TracerConfig {
    bool enabled = true;
    bool trace_packets = true;        // Packet inject/deliver with flow arrows
    bool trace_hops = true;           // Per-hop routing events
    bool trace_router_state = true;   // Router activity
    bool trace_link_usage = true;     // Link occupancy
    bool trace_counters = true;       // Bandwidth/backpressure counters
    uint64_t counter_interval = 10;   // Cycles between counter samples
    size_t max_events = 10000000;     // Memory limit
};

/// Main tracer class for benchmark visualization
class NoCBenchmarkTracer {
public:
    explicit NoCBenchmarkTracer(const TracerConfig& config = {});

    // ========== Event Recording ==========

    /// Record packet injection (returns packet_id for correlation)
    uint64_t record_inject(uint8_t src_router, uint8_t dst_router,
                           const TileDescriptor& tile, uint64_t cycle);

    /// Record packet delivery (uses packet_id from inject)
    void record_deliver(uint64_t packet_id, uint64_t cycle);

    /// Record a single hop through a router
    void record_hop(uint64_t packet_id, uint8_t router_id,
                    PortDir in_port, PortDir out_port,
                    uint64_t enter_cycle, uint64_t exit_cycle);

    /// Record router state change
    void record_router_state(uint8_t router_id, RouterState state,
                             uint64_t cycle, const std::string& reason = "");

    /// Record link transfer
    void record_link_transfer(uint8_t src_router, uint8_t dst_router,
                              Direction dir, uint64_t start_cycle,
                              uint64_t end_cycle, uint32_t bytes);

    /// Record counter sample
    void record_counters(uint64_t cycle, double bandwidth,
                         uint32_t blocked_routers, uint32_t active_flits);

    // ========== Export ==========

    /// Export to Chrome Trace JSON format
    bool export_chrome_trace(const std::string& filename,
                             uint8_t mesh_rows, uint8_t mesh_cols) const;

    /// Export with mesh config
    bool export_chrome_trace(const std::string& filename,
                             const NoCConfig& config) const;

    // ========== Statistics ==========

    size_t num_packets() const;
    size_t num_events() const;
    uint64_t total_hops() const;

    void clear();
    void enable() { config_.enabled = true; }
    void disable() { config_.enabled = false; }

private:
    TracerConfig config_;

    struct PacketInfo {
        TileDescriptor tile;
        uint8_t src_router;
        uint8_t dst_router;
        uint64_t inject_cycle;
        uint64_t deliver_cycle = 0;
    };
    std::unordered_map<uint64_t, PacketInfo> packets_;

    struct HopInfo {
        uint64_t packet_id;
        uint8_t router_id;
        PortDir in_port;
        PortDir out_port;
        uint64_t enter_cycle;
        uint64_t exit_cycle;
    };
    std::vector<HopInfo> hops_;

    struct RouterStateInfo {
        uint8_t router_id;
        RouterState state;
        uint64_t cycle;
        std::string reason;
    };
    std::vector<RouterStateInfo> router_states_;

    struct LinkInfo {
        uint8_t src_router;
        uint8_t dst_router;
        Direction dir;
        uint64_t start_cycle;
        uint64_t end_cycle;
        uint32_t bytes;
    };
    std::vector<LinkInfo> link_transfers_;

    struct CounterSample {
        uint64_t cycle;
        double bandwidth;
        uint32_t blocked_routers;
        uint32_t active_flits;
    };
    std::vector<CounterSample> counter_samples_;

    uint64_t next_packet_id_ = 1;
};

} // namespace sw::benchmark
```

### Integration with INoC

```cpp
class INoC {
public:
    // ... existing methods ...

    /// Set tracer for benchmark visualization
    virtual void set_tracer(std::shared_ptr<NoCBenchmarkTracer> tracer) {
        tracer_ = tracer;
    }

protected:
    std::shared_ptr<NoCBenchmarkTracer> tracer_;
};
```

---

## CLI Integration

```bash
# Run benchmark with trace capture
kpu-noc-bench --compare --trace noc_trace.json

# Run specific benchmark with detailed trace
kpu-noc-bench --patterns --pattern systolic --trace systolic_trace.json

# Control trace detail level
kpu-noc-bench --compare --trace trace.json --trace-packets --no-trace-hops

# Limit trace size
kpu-noc-bench --compare --trace trace.json --max-trace-events 100000
```

---

## Visualization Examples

### 1. Packet Flow Visualization

In Perfetto, flow events render as arrows showing packet movement:

```
Router[0,0] ─────┐ inject A[0,0]
                 │
Router[0,1] ───  ├──► hop
                 │
Router[0,2] ───  ├──► hop
                 │
Router[1,2] ─────┘ deliver A[0,0]
                  ←─────────────────→
                    50 cycles latency
```

### 2. Congestion Hotspot

Counter events show bandwidth/backpressure over time:

```
Bandwidth (B/cycle)
    64 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    32 ┤                ▓▓▓▓▓▓▓▓ (congestion)
     0 ┼────────────────────────────→ time

Backpressure (%)
   100 ┤                ████████
    50 ┤
     0 ┼▓▓▓▓▓▓▓▓▓▓▓▓▓▓──────────→ time
```

### 3. Systolic Flow Pattern

Shows A tiles flowing East, B tiles flowing South:

```
East Links:
[0,0]→[0,1] ─── A[0,0] ───────────────────────
[0,1]→[0,2] ─────────── A[0,0] ───────────────
[0,2]→[0,3] ─────────────────── A[0,0] ───────

South Links:
[0,0]↓[1,0] ─── B[0,0] ───────────────────────
[1,0]↓[2,0] ─────────── B[0,0] ───────────────
[2,0]↓[3,0] ─────────────────── B[0,0] ───────
```

---

## Implementation Plan

### Phase 1: Core Tracer

1. Create `include/sw/benchmark/noc_benchmark_tracer.hpp`
2. Create `src/benchmark/noc_benchmark_tracer.cpp`
3. Implement packet and hop tracking
4. Implement Chrome Trace export with flow events

### Phase 2: NoC Integration

1. Add tracer hooks to WormholeNoCAdapter
2. Add tracer hooks to DataflowNoCAdapter
3. Hook into existing WormholeTracer/NoCTracer events

### Phase 3: Benchmark Integration

1. Add `--trace` option to kpu-noc-bench CLI
2. Enable tracing during benchmark runs
3. Export trace at end of benchmark

### Phase 4: Counter Events

1. Add periodic counter sampling during simulation
2. Track bandwidth utilization over time
3. Track backpressure/congestion metrics

---

## File Changes

| File | Action |
|------|--------|
| `include/sw/benchmark/noc_benchmark_tracer.hpp` | New - Tracer API |
| `src/benchmark/noc_benchmark_tracer.cpp` | New - Implementation |
| `src/benchmark/CMakeLists.txt` | Update - Add new source |
| `tools/benchmark/kpu-noc-bench/main.cpp` | Update - Add --trace option |
| `src/noc/noc_adapters.cpp` | Update - Hook tracer |

---

## Example Output

```json
{"traceEvents":[
  {"name":"process_name","ph":"M","pid":1,"args":{"name":"NoC Routers"}},
  {"name":"thread_name","ph":"M","pid":1,"tid":0,"args":{"name":"Router[0,0]"}},
  {"name":"thread_name","ph":"M","pid":1,"tid":1,"args":{"name":"Router[0,1]"}},

  {"name":"A[0,0]","cat":"packet","ph":"s","id":1,"ts":0,"pid":1,"tid":0,
   "args":{"src":"R[0,0]","dst":"R[1,2]","size":4096}},

  {"name":"HOP","cat":"routing","ph":"X","ts":5,"dur":5,"pid":1,"tid":1,
   "args":{"packet":1,"in":"WEST","out":"EAST"}},

  {"name":"A[0,0]","cat":"packet","ph":"f","id":1,"ts":50,"pid":1,"tid":6,
   "bp":"e","args":{"latency":50,"hops":3}},

  {"name":"bandwidth","cat":"counter","ph":"C","ts":10,"pid":5,"tid":0,
   "args":{"bytes_per_cycle":32.5}},

  {"name":"active_flits","cat":"counter","ph":"C","ts":10,"pid":5,"tid":2,
   "args":{"count":64}}
]}
```

---

## Next Steps

1. Implement `NoCBenchmarkTracer` class
2. Add hooks to NoC adapters
3. Add CLI options to kpu-noc-bench
4. Test with real benchmark runs
5. Validate visualization in Perfetto

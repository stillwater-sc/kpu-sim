# Visualization and Performance Engineering Tools Plan

## Overview

This document outlines the implementation plan for enhanced observability and performance engineering tools for the KPU simulator. The goal is to provide multiple views on resource utilization and efficiency to enable optimization of the systolic dataflow algorithms.

## Current State

### Existing Tools

| Tool | Format | Purpose | Location |
|------|--------|---------|----------|
| `tile_flow_gantt.py` | PNG (matplotlib) | Per-L3 Gantt timeline, mesh transfer patterns | `tools/visualization/` |
| `generate_noc_animation.py` | HTML+JS | Interactive topology animation with playback | `tools/visualization/` |
| `TileFlowTracer` | CSV, JSON, Chrome Trace | C++ tracer with multi-format export | `include/sw/kpu/dataflow/tile_flow_tracer.hpp` |
| `NoCTracer` | CSV (Chrome stub) | NoC-level event tracing | `include/sw/kpu/noc/noc.hpp` |

### Generated Files
- `tile_flow_trace.csv` - Tile movement events
- `tile_flow_trace.json` - JSON version with metadata
- `tile_flow_trace_chrome.json` - Full Chrome Trace format (works with Perfetto)
- `/tmp/noc_trace.csv` - NoC-level packet/FLIT events

### Key Insight: Offline Architecture Works Well
The offline trace-based approach provides:
- Manageable file sizes
- Ability to "reverse time" when abnormalities are found
- Multiple visualization tools can process the same trace
- No simulation re-run needed for different views

---

## New Tool Requirements

### 1. Resource Utilization & Efficiency Metrics

**Definition:**
- **Utilization** = busy_cycles / total_cycles (per resource, per period)
- **Efficiency** = useful_work / theoretical_maximum

**Resources to Track:**
| Resource Type | Utilization Metric | Efficiency Metric |
|--------------|-------------------|-------------------|
| NoC Links | Cycles with FLITs flowing | Bytes transferred / (bandwidth × time) |
| Routers | Cycles with packets routing | Packets routed / injection rate |
| L3 Caches | Cycles receiving/sending | Tiles served / capacity |
| DMA Channels | Cycles transferring | Bytes / (bandwidth × time) |
| Compute Units | Cycles computing | MACs executed / peak MACs |

**Period-Based Calculation:**
- User selects start_cycle and end_cycle
- Metrics calculated over that window
- Enable zooming into problem areas

### 2. Sliding Window Utilization Animation

**Concept:**
A new visualization that shows how utilization evolves over time, revealing:
- Pipeline startup/drain overhead
- Periods of low utilization (bubbles)
- Contention hotspots

**Display Options:**
- Time-series line chart of aggregate utilization
- Per-resource sparklines
- Heat map with time on X-axis, resources on Y-axis

### 3. Heat Map Visualization

**Purpose:** Scale to larger arrays (64×64 and beyond)

**Design:**
- Individual tile labels don't scale; use aggregate metrics
- Color intensity represents utilization/efficiency
- Can animate over time or show aggregate
- Supports different metrics via dropdown

### 4. Chrome Trace for NoC (Causality Analysis)

**Current Gap:** `NoCTracer::export_chrome_trace()` is a skeleton

**Enhancement:**
- Full Chrome Trace export for NoC events
- Flow events (`ph: "s"` and `ph: "f"`) linking packet inject→eject
- Enables causality analysis in Perfetto

### 5. Quick Terminal Gantt Chart

**Purpose:** Instant shape visualization without loading browser

**Output:**
```
Cycle:    0        1000      2000      3000      4000
L3[0,0]:  ████████                     ██████
L3[0,1]:           ████████████████████
L3[0,2]:                    ████████████████████
...
```

---

## Implementation Plan

### Phase 1: Metrics Infrastructure (Foundation)

**Files to Create:**
```
include/sw/kpu/analysis/
├── utilization_metrics.hpp    # Core metric calculation
└── efficiency_analyzer.hpp    # Efficiency analysis framework

src/analysis/
├── utilization_metrics.cpp
└── efficiency_analyzer.cpp
```

**Key Classes:**

```cpp
// utilization_metrics.hpp
namespace sw::kpu::analysis {

struct ResourceUtilization {
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;
    uint64_t total_bytes = 0;

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / total : 0.0;
    }
};

struct PeriodMetrics {
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;

    // Per-resource utilization
    std::array<ResourceUtilization, 16> l3_utilization;      // 16 L3s
    std::array<ResourceUtilization, 24> east_links;          // 4 rows × 3 links
    std::array<ResourceUtilization, 12> south_links;         // 3 rows × 4 links
    std::array<ResourceUtilization, 8> dma_channels;         // 4+4 channels

    // Aggregate metrics
    double aggregate_link_utilization() const;
    double aggregate_compute_utilization() const;
    double efficiency() const;  // bytes_transferred / (bandwidth × time)
};

class UtilizationCalculator {
public:
    // Calculate metrics for a time window from trace events
    static PeriodMetrics calculate(
        const std::vector<noc::NoCTraceEvent>& events,
        uint64_t start_cycle,
        uint64_t end_cycle);

    // Calculate sliding window metrics
    static std::vector<PeriodMetrics> sliding_window(
        const std::vector<noc::NoCTraceEvent>& events,
        uint64_t window_size,
        uint64_t step_size);
};

} // namespace sw::kpu::analysis
```

**Implementation Approach:**
1. Parse trace events to build busy/idle intervals per resource
2. Intersect with query window to compute utilization
3. Cache results for efficiency

---

### Phase 2: Enhanced NoC Animation with Metrics Panel

**File:** `tools/visualization/generate_noc_animation.py`

**New GUI Controls:**
```
┌─────────────────────────────────────────┐
│ Analysis Period                         │
│ ┌─────────────────────────────────────┐ │
│ │ Start: [    0    ] End: [   5000  ] │ │
│ │ [Apply] [Reset to Full]             │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Resource Utilization                    │
│ ┌─────────────────────────────────────┐ │
│ │ East Links:  ████████░░  82%        │ │
│ │ South Links: ██████░░░░  64%        │ │
│ │ L3 Caches:   ███████░░░  71%        │ │
│ │ Routers:     █████░░░░░  48%        │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Efficiency                              │
│ ┌─────────────────────────────────────┐ │
│ │ Link BW Efficiency:  67%            │ │
│ │ Pipeline Efficiency: 45%            │ │
│ │ Bytes Moved: 24.5 MB                │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**JavaScript Implementation:**

```javascript
// Metrics calculation
function calculateMetrics(startCycle, endCycle) {
    const metrics = {
        eastLinks: { busy: 0, total: 0 },
        southLinks: { busy: 0, total: 0 },
        l3Activity: Array(16).fill(0),
        routerActivity: Array(16).fill(0),
        bytesTransferred: 0
    };

    // Filter events in range
    const rangeEvents = events.filter(
        e => e.cycle >= startCycle && e.cycle <= endCycle
    );

    // Count busy cycles per resource from FLIT_SEND events
    // ... calculation logic ...

    return metrics;
}

// Update metrics panel when range changes
function updateMetricsPanel() {
    const start = parseInt(document.getElementById('metric-start').value);
    const end = parseInt(document.getElementById('metric-end').value);
    const metrics = calculateMetrics(start, end);

    // Update progress bars
    updateProgressBar('east-links-bar', metrics.eastLinkUtil);
    updateProgressBar('south-links-bar', metrics.southLinkUtil);
    // ... etc
}
```

---

### Phase 3: Heat Map Visualization

**File:** `tools/visualization/generate_utilization_heatmap.py`

**Features:**
- Supports arbitrary mesh sizes (4×4 to 128×128)
- Time-animated or aggregate view
- Multiple metrics: link utilization, cache occupancy, router activity
- Color scale: blue (cold/idle) → red (hot/busy)

**Output:** Standalone HTML with embedded data

**Design:**

```
┌──────────────────────────────────────────────────────┐
│          NoC Utilization Heat Map (64×64 mesh)       │
├──────────────────────────────────────────────────────┤
│ Metric: [Link Utilization ▼]  Time: [▶ Animation]   │
├──────────────────────────────────────────────────────┤
│                                                      │
│   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│   ░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░    │
│   ░░░░░▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒░░░░░    │
│   ░░░░▒▒▒▓▓▓▓▓█████████████████▓▓▓▓▒▒▒░░░░    │
│   ░░░▒▒▓▓▓███████████████████████▓▓▒▒░░░    │
│   ░░▒▒▓▓████████████████████████████▓▒▒░░    │
│   ░░▒▓███████████████████████████████▓▒░░    │
│   ...                                              │
│                                                      │
├──────────────────────────────────────────────────────┤
│ Scale: 0%  ░░░▒▒▒▓▓▓███  100%                       │
│ Cycle: 12,450 / 61,478    Avg Utilization: 67%      │
└──────────────────────────────────────────────────────┘
```

**Implementation:**

```python
def generate_heatmap_html(trace_file, mesh_size=64):
    """Generate heat map HTML for large mesh visualization."""

    # Load trace
    events = load_noc_trace(trace_file)

    # Calculate per-cell utilization
    utilization = calculate_cell_utilization(events, mesh_size)

    # Generate SVG heat map
    svg = generate_svg_heatmap(utilization, mesh_size)

    # Embed in HTML with controls
    html = f'''
    <html>
    <head>
        <title>NoC Utilization Heat Map</title>
        <style>/* ... */</style>
    </head>
    <body>
        <svg id="heatmap">{svg}</svg>
        <script>
            const utilizationData = {json.dumps(utilization)};
            // Animation and interaction logic
        </script>
    </body>
    </html>
    '''
    return html
```

---

### Phase 4: Chrome Trace for NoC

**File:** `src/noc/noc.cpp` - Implement `NoCTracer::export_chrome_trace()`

**Structure:**
```json
{
  "traceEvents": [
    // Process metadata
    {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "NoC Routers"}},
    {"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "East Links"}},
    {"name": "process_name", "ph": "M", "pid": 3, "args": {"name": "South Links"}},

    // Thread metadata (one per router/link)
    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "R[0,0]"}},

    // Duration events
    {"name": "A[0,0] R0→R1", "cat": "transfer", "ph": "B", "ts": 2, "pid": 2, "tid": 0},
    {"name": "A[0,0] R0→R1", "cat": "transfer", "ph": "E", "ts": 4098, "pid": 2, "tid": 0},

    // Flow events for causality (inject → eject)
    {"name": "Pkt0", "cat": "flow", "ph": "s", "ts": 2, "pid": 1, "tid": 0, "id": 0},
    {"name": "Pkt0", "cat": "flow", "ph": "f", "ts": 8200, "pid": 1, "tid": 3, "id": 0}
  ]
}
```

**Key Additions:**
1. **Flow Events**: `"ph": "s"` (flow start) at inject, `"ph": "f"` (flow finish) at eject
2. **Shared ID**: Links inject→hops→eject for same packet
3. **FLIT-level detail**: Optional high-detail mode showing individual FLITs

---

### Phase 5: Terminal ASCII Gantt

**File:** `tools/visualization/ascii_gantt.py`

**Usage:**
```bash
python3 ascii_gantt.py tile_flow_trace.csv --width 80
```

**Output:**
```
TILE FLOW ASCII GANTT
=====================
Time range: 0 - 61478 cycles (scale: 769 cycles/char)

         0         10000     20000     30000     40000     50000     60000
         |         |         |         |         |         |         |
L3[0,0]: ████░░░░░░░░░░░░░░░░████████░░░░░░████████░░░░░░████████░░
L3[0,1]: ░░░░████████░░░░░░░░░░░░████████░░░░░░████████░░░░░░████████
L3[0,2]: ░░░░░░░░████████░░░░░░░░░░░░████████░░░░░░████████░░░░░░████
L3[0,3]: ░░░░░░░░░░░░████████░░░░░░░░░░░░████████░░░░░░████████░░░░░░
L3[1,0]: ░░░░░░░░████████░░░░████████░░░░████████░░░░████████░░░░████
...

Legend: █ = Busy (L3 transfer)  ▒ = Compute  ░ = Idle

Summary:
  Total cycles: 61478
  L3 avg utilization: 42%
  Peak activity window: 4000-8000 (67% utilization)
```

**Features:**
- Auto-scales to terminal width
- Shows multiple resource types with different characters
- Prints summary statistics
- Works over SSH, in CI logs, etc.

---

### Phase 6: Unified Metrics Export

**File:** `tools/visualization/export_metrics.py`

Generates a JSON metrics file that any visualization can consume:

```json
{
  "metadata": {
    "trace_file": "noc_trace.csv",
    "total_cycles": 61478,
    "mesh_size": [4, 4],
    "generated": "2025-12-30T10:00:00Z"
  },
  "aggregate": {
    "east_link_utilization": 0.67,
    "south_link_utilization": 0.58,
    "router_utilization": 0.45,
    "efficiency": 0.72,
    "bytes_transferred": 25690112,
    "packets_delivered": 96
  },
  "time_series": {
    "window_size": 1000,
    "utilization": [0.12, 0.45, 0.78, 0.82, ...],
    "efficiency": [0.10, 0.42, 0.71, 0.75, ...]
  },
  "per_resource": {
    "east_links": [
      {"id": "[0,0]→[0,1]", "utilization": 0.71, "bytes": 3145728},
      ...
    ],
    "south_links": [...],
    "routers": [...]
  },
  "hotspots": [
    {"cycle_range": [4000, 4500], "resource": "R[0,0]", "utilization": 0.95},
    ...
  ]
}
```

---

## Implementation Priority

| Priority | Phase | Tool | Effort | Value |
|----------|-------|------|--------|-------|
| 1 | 5 | ASCII Gantt | Low | High (instant feedback) |
| 2 | 2 | Animation Metrics Panel | Medium | High (integrated) |
| 3 | 4 | Chrome Trace for NoC | Medium | High (causality) |
| 4 | 1 | Metrics Infrastructure | Medium | Foundation |
| 5 | 3 | Heat Map Visualization | Medium | Medium (scale) |
| 6 | 6 | Unified Metrics Export | Low | Medium (integration) |

---

## File Structure After Implementation

```
tools/visualization/
├── generate_noc_animation.py      # Enhanced with metrics panel
├── generate_utilization_heatmap.py # New: heat map visualization
├── generate_ascii_gantt.py        # New: terminal Gantt
├── tile_flow_gantt.py             # Existing matplotlib Gantt
├── export_metrics.py              # New: unified metrics export
└── README.md                      # Tool documentation

include/sw/kpu/analysis/
├── utilization_metrics.hpp        # Core metrics calculation
└── efficiency_analyzer.hpp        # Efficiency framework

src/analysis/
├── utilization_metrics.cpp
├── efficiency_analyzer.cpp
└── CMakeLists.txt
```

---

## Success Criteria

1. **Instant Shape Inspection**: ASCII Gantt shows pipeline shape in < 1 second
2. **Period Analysis**: Can select time window and see utilization metrics
3. **Causality Tracing**: Chrome Trace shows packet flow with flow events
4. **Scalability**: Heat map works for 64×64 mesh without performance issues
5. **Integration**: All tools work from same trace files

---

## Questions for User

1. What mesh sizes do you anticipate testing? (impacts heat map design)
2. Priority: Do you want causality analysis (Chrome Trace) or utilization metrics first?
3. Should efficiency include compute utilization, or focus on data movement?
4. Terminal ASCII: preferred width (80, 120, or auto-detect)?

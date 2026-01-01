# Chrome Trace Capture for NoC Benchmarks

Files Created/Modified:

  1. docs/design/noc_benchmark_chrome_trace.md - Design document
  2. include/sw/benchmark/noc_benchmark_tracer.hpp - Tracer API (~350 lines)
  3. tools/benchmark/kpu-noc-bench/main.cpp - Added trace CLI options

Features:

  1. Packet Flow Events (with flow arrows in Perfetto)
    - ph:"s" - Flow start at injection
    - ph:"f" - Flow end at delivery
    - Shows A[m,k] and B[k,n] tile names
  2. Hop Events - Router-level routing activity
    - Input/output port directions (L=LOCAL, E=EAST, S=SOUTH, etc.)
  3. Link Events - Per-link bandwidth utilization
    - East Links (A tiles →)
    - South Links (B tiles ↓)
  4. Counter Events - Time-series metrics
    - Bandwidth (bytes/cycle)
    - Blocked routers
    - Active flits in network

CLI Usage:

```bash
  # Export trace during benchmark
  kpu-noc-bench --compare --trace noc_trace.json

  # With 4x4 mesh
  kpu-noc-bench --compare --rows 4 --cols 4 --trace trace.json

  # Disable specific trace categories
  kpu-noc-bench --compare --trace trace.json --no-trace-hops
```

Viewing:

  - Open trace file in https://ui.perfetto.dev
  - Or use chrome://tracing

The trace shows:
  - Routers arranged by ID (Router[row,col])
  - East/South links showing data flow direction
  - Flow arrows connecting packet inject to deliver
  - Bandwidth/backpressure counters over time

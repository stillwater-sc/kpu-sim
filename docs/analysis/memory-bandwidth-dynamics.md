# Memory Bandwidth Dynamics

All bandwidth patterns are complete. Here's a summary of the results:

Bandwidth Patterns Created

  | Pattern                    | Banks | Peak Throughput  | Page Hit Ratio |
  |----------------------------|-------|------------------|----------------|
  | LPDDR5 max_bandwidth       | 8     | 2.80 bytes/cycle | 87.5%          |
  | GDDR6 eight_bank_bandwidth | 8     | 2.56 bytes/cycle | 87.5%          |
  | GDDR6 max_bandwidth        | 16    | 2.56 bytes/cycle | 75.0%          |

Key Findings

GPU vs Accelerator Comparison (GDDR6):
  - GPU-style (16 banks active): 2.56 bytes/cycle (1601 cycles)
  - Accelerator-style (4 banks): 1.72 bytes/cycle (2381 cycles)
  - GPU has 1.5x bandwidth advantage due to better bank concurrency utilization

Bandwidth Scaling (GDDR6):
  - 1 bank: 1.59 bytes/cycle (baseline)
  - 2 banks: 1.99 bytes/cycle (1.3x)
  - 4 banks: 2.28 bytes/cycle (1.4x)
  - 8 banks: 2.46 bytes/cycle (1.5x)
  - 16 banks: 2.56 bytes/cycle (1.6x)

This confirms your hypothesis - GPU-style concurrent thread access gets more bandwidth than tile-based accelerator access because it can keep more banks active simultaneously. The tile-based approach tends to concentrate accesses on fewer banks, causing page conflicts and reducing effective bandwidth.

Generated traces:
  - traces/memory/lpddr5/bandwidth/max_bandwidth_trace.json
  - traces/memory/gddr6/bandwidth/eight_bank_bandwidth_trace.json
  - traces/memory/gddr6/bandwidth/max_bandwidth_trace.json


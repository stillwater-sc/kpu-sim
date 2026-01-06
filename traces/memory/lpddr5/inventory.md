# Trace file inventory

All 15 trace files generated across all levels:

  | Level | Pattern      | Trace File                                                                 |
  |-------|--------------|----------------------------------------------------------------------------|
  | 1     | Single Bank  | page_hits_trace.json, page_conflicts_trace.json, mixed_rw_trace.json       |
  | 2     | Two Bank     | same_group_trace.json, diff_groups_trace.json                              |
  | 3     | Three Bank   | same_group_trace.json, mixed_groups_trace.json                             |
  | 4     | Four Bank    | full_group_trace.json, across_groups_trace.json, page_hit_burst_trace.json |
  | 5     | Dual Channel | independent_trace.json, interleaved_trace.json                             |
  | 6     | Complex      | strided_trace.json, random_trace.json, tile_load_trace.json                |

Traces are in traces/memory/lpddr5/<level>/.

## Directory structure

traces/memory/lpddr5/
├── complex
│   ├── random_trace.json
│   ├── strided_trace.json
│   └── tile_load_trace.json
├── dual-channel
│   ├── independent_trace.json
│   └── interleaved_trace.json
├── four-bank
│   ├── across_groups_trace.json
│   ├── full_group_trace.json
│   └── page_hit_burst_trace.json
├── inventory.md
├── single-bank
│   ├── mixed_rw_trace.json
│   ├── page_conflicts_trace.json
│   └── page_hits_trace.json
├── three-bank
│   ├── mixed_groups_trace.json
│   └── same_group_trace.json
└── two-bank
    ├── diff_groups_trace.json
    └── same_group_trace.json

7 directories, 16 files


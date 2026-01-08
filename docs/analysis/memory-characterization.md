# Memory Controller Latency and Bandwidth Characterization

This document characterizes the latency and bandwidth behavior of the LPDDR5, GDDR6, HBM2, and HBM3 memory controllers in the KPU simulator. It serves as a reference for understanding architectural tradeoffs between these DRAM technologies.

## Technology Summary

| Metric | LPDDR5-6400 | GDDR6-16000 | HBM2-2000 | HBM3-5600 |
|--------|-------------|-------------|-----------|-----------|
| **Architecture** |
| Clock Frequency | 3.2 GHz | 2.0 GHz | 1.0 GHz | 2.8 GHz |
| Data Rate | 6.4 Gbps | 16 Gbps | 2.0 Gbps | 5.6 Gbps |
| Bus Width | 32-bit | 32-bit | 1024-bit | 1024-bit |
| Channels/Stack | 1-2 | 2 | 8 | 16 |
| Pseudo-Channels | N/A | N/A | 2/channel | 2/channel |
| Banks per Channel | 8 | 16 | 32 (16/PC) | 32 (16/PC) |
| Bank Groups | 2 | 4 | 4/PC | 4/PC |
| **Performance** |
| Peak BW (per stack) | 25.6 GB/s | 64 GB/s | 256 GB/s | 716.8 GB/s |
| Burst Length | BL16/32 | BL16 | BL4 | BL8 |
| Page Size | 8 KB | 8 KB | 2 KB | 1 KB |
| **Power** |
| TDP per Device | 1-2 W | 10-15 W | 10-15 W | 15-20 W |
| Energy per Bit (pJ/bit) | 5-7 | 8-12 | 3-4 | 2.5-3.5 |
| Power per GB/s (mW) | 60-80 | 150-200 | 40-60 | 25-35 |
| **Power Efficiency** |
| Typical Use | Mobile/Edge | Graphics | HPC/AI | HPC/AI |
| BW Efficiency | Low | Medium | High | **Highest** |

---

## Power Efficiency Analysis

Understanding power efficiency in memory systems requires distinguishing between three key metrics:

### Key Power Metrics

| Metric | Definition | Units | What It Measures |
|--------|------------|-------|------------------|
| **TDP** | Total Device Power | Watts (W) | Total power consumed by memory device |
| **Energy per Bit** | Energy to transfer one bit | pJ/bit | Interface efficiency |
| **Power per BW** | Power to sustain 1 GB/s | mW/(GB/s) | System-level efficiency |

### Why HBM Has the Highest Bandwidth Efficiency

HBM achieves superior energy efficiency per bit despite higher total device power due to fundamental physics advantages:

```
Signal Path Comparison (approximate)
─────────────────────────────────────────────────────────────────────
LPDDR5/GDDR6: DRAM Die → Package → PCB traces (5-15 cm) → SoC Package → Logic Die
              └── Long wires require high-swing signaling, termination, ESD protection

HBM:          DRAM Die → TSV (<50 μm) → Interposer → TSV → Logic Die
              └── Ultra-short paths enable low-swing CMOS, no termination needed
─────────────────────────────────────────────────────────────────────
```

| Factor | LPDDR5/GDDR6 | HBM | Energy Impact |
|--------|--------------|-----|---------------|
| **Signal Path** | 5-15 cm PCB traces | <100 μm TSV | 50-100x shorter |
| **Signal Swing** | 1.1-1.5V differential | 0.4-0.6V CMOS | 3-5x lower voltage |
| **Termination** | Required (ODT) | Not required | Saves ~30% power |
| **I/O Capacitance** | 2-5 pF (package + PCB) | 0.1-0.3 pF (TSV) | 10-20x lower |
| **Bus Width** | 32-bit | 1024-bit | Parallel = lower frequency |
| **PHY Complexity** | SerDes, CDR, DFE | Simple CMOS drivers | Much simpler |

### Energy per Bit Breakdown

| Technology | Interface Energy | DRAM Core Energy | Total pJ/bit |
|------------|-----------------|------------------|--------------|
| **LPDDR5** | 3-4 pJ/bit | 2-3 pJ/bit | **5-7 pJ/bit** |
| **GDDR6** | 5-8 pJ/bit | 3-4 pJ/bit | **8-12 pJ/bit** |
| **HBM2** | 1-2 pJ/bit | 2-2.5 pJ/bit | **3-4 pJ/bit** |
| **HBM3** | 0.8-1.5 pJ/bit | 1.5-2 pJ/bit | **2.5-3.5 pJ/bit** |

*Note: Interface energy dominates for off-chip memory (LPDDR5, GDDR6). HBM's TSV interface dramatically reduces this component.*

### System Power Comparison for 1 TB/s Bandwidth

| Technology | Devices Needed | Total TDP | Energy/bit | System Power |
|------------|---------------|-----------|------------|--------------|
| **LPDDR5-6400** | 40 channels | 60-80 W | 5-7 pJ/bit | **625-875 W** |
| **GDDR6-16000** | 16 devices | 160-240 W | 8-12 pJ/bit | **1000-1500 W** |
| **HBM2-2000** | 4 stacks | 40-60 W | 3-4 pJ/bit | **375-500 W** |
| **HBM3-5600** | 2 stacks | 30-40 W | 2.5-3.5 pJ/bit | **312-437 W** |

**Key Insight**: For bandwidth-intensive AI/HPC workloads, HBM delivers the same bandwidth at **40-60% lower system power** than alternatives. This is why all modern AI accelerators (H100, MI300, TPUv5) use HBM.

### Power Efficiency Visualization

```
Energy per Bit (pJ/bit) - Lower is Better
────────────────────────────────────────────────────────────────

GDDR6-16000   ████████████████████████████████████  8-12 pJ/bit
LPDDR5-6400   ██████████████████████████            5-7 pJ/bit
HBM2-2000     ████████████████                      3-4 pJ/bit
HBM3-5600     ██████████████                        2.5-3.5 pJ/bit

              └──────────────────────────────────────────────────┘
              0          5          10         15
```

```
Power per GB/s Bandwidth (mW) - Lower is Better
────────────────────────────────────────────────────────────────

GDDR6-16000   ██████████████████████████████████████  150-200 mW/GB/s
LPDDR5-6400   ████████████████                        60-80 mW/GB/s
HBM2-2000     ████████████                            40-60 mW/GB/s
HBM3-5600     ████████                                25-35 mW/GB/s

              └──────────────────────────────────────────────────┘
              0         50        100       150       200
```

---

## Technology Use Cases

| Technology | Primary Use Case | Key Advantage | Key Limitation |
|------------|------------------|---------------|----------------|
| **LPDDR5** | Mobile AI, Edge inference | Low TDP, SoC integration | Lower bandwidth |
| **GDDR6** | Graphics, gaming AI | Good bandwidth/cost ratio | Higher pJ/bit |
| **HBM2** | Data center AI, HPC | High BW, best energy/bit | Cost, interposer required |
| **HBM3** | Cutting-edge AI training | Highest BW, best efficiency | Cost, thermal management |

---

## LPDDR5-6400 Characterization

### Timing Parameters

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCD | 14 | Row to Column Delay |
| tCL | 16 | CAS Latency |
| tRP | 14 | Row Precharge |
| tRAS | 28 | Row Active Time |
| tRC | 42 | Row Cycle Time |
| tBurst | 8 | Burst Duration (BL16) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 4 | ACT-to-ACT (different group) |
| tCCD_L | 4 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |

### Latency Characterization

| Access Pattern | Avg Latency (cycles) | Description |
|---------------|---------------------|-------------|
| Page Hit | ~243 | Row already open, CAS only |
| Page Empty | ~305 | Bank idle, ACT + CAS |
| Page Conflict | ~305 | Wrong row open, PRE + ACT + CAS |

**Latency Breakdown:**
- **Page Hit**: tCL + tBurst = 16 + 8 = 24 cycles (minimum)
- **Page Empty**: tRCD + tCL + tBurst = 14 + 16 + 8 = 38 cycles (minimum)
- **Page Conflict**: tRP + tRCD + tCL + tBurst = 14 + 14 + 16 + 8 = 52 cycles (minimum)

*Note: Measured latencies include queue delays and scheduling overhead.*

### Bandwidth Characterization

#### Single Bank Performance

| Pattern | Throughput (bytes/cycle) | Page Hit Rate |
|---------|-------------------------|---------------|
| Sequential Page Hits | 5.59 | 93.8% |
| Page Conflicts | 0.96 | 0% |

#### Multi-Bank Scaling

| Banks | Throughput (bytes/cycle) | Speedup |
|-------|-------------------------|---------|
| 1 | 2.20 | 1.0x |
| 2 | 2.50 | 1.1x |
| 4 | 2.69 | 1.2x |
| 8 | 2.80 | 1.3x |

#### STREAM Benchmark Performance

| Kernel | Operations | Total Bytes | Throughput (bytes/cycle) |
|--------|-----------|-------------|-------------------------|
| Copy | 1R + 1W | 32 KB | 19.04 |
| Scale | 1R + 1W | 32 KB | 19.04 |
| Add | 2R + 1W | 48 KB | 26.67 |
| Triad | 2R + 1W | 48 KB | 26.67 |

#### Multi-DMA Tile Loading

| DMA Engines | Total Bytes | Throughput (bytes/cycle) | Speedup |
|-------------|-------------|-------------------------|---------|
| 4 | 64 KB | 44.73 | 1.0x |
| 8 | 128 KB | 89.47 | 2.0x |
| 16 | 128 KB | 89.47 | 2.0x |

#### Page Utilization Impact

| Accesses/Page | Page Hit Rate | Throughput (bytes/cycle) |
|---------------|---------------|-------------------------|
| 8 | 87.5% | 2.80 |
| 32 | ~97% | ~4.0 |
| 128 (full page) | 99.2% | 5.59 |

**Key Insight**: Maximizing page hits (128 cache lines per 8KB page) yields 2.5x bandwidth improvement over minimal page utilization.

---

## GDDR6-16000 Characterization

### Timing Parameters

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCDRD | 18 | Row to Column Delay (Read) |
| tRCDWR | 18 | Row to Column Delay (Write) |
| tRL | 20 | CAS Read Latency |
| tWL | 8 | CAS Write Latency |
| tRP | 18 | Row Precharge |
| tRAS | 28 | Row Active Time |
| tRC | 46 | Row Cycle Time |
| tBurst | 4 | Burst Duration (BL16) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 4 | ACT-to-ACT (different group) |
| tCCD_L | 3 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |
| tFAW | 16 | Four Activate Window |

### Latency Characterization

| Access Pattern | Avg Latency (cycles) | Description |
|---------------|---------------------|-------------|
| Page Hit | ~172-184 | Row already open, CAS only |
| Page Empty | ~88 | Bank idle, ACT + CAS |
| Page Conflict | ~298-328 | Wrong row open, PRE + ACT + CAS |

**Latency Breakdown:**
- **Page Hit Read**: tRL + tBurst = 20 + 4 = 24 cycles (minimum)
- **Page Empty Read**: tRCDRD + tRL + tBurst = 18 + 20 + 4 = 42 cycles (minimum)
- **Page Conflict Read**: tRP + tRCDRD + tRL + tBurst = 18 + 18 + 20 + 4 = 60 cycles (minimum)

### Bandwidth Characterization

#### Single Bank Performance

| Pattern | Throughput (bytes/cycle) | Page Hit Rate |
|---------|-------------------------|---------------|
| Sequential Page Hits | 5.12 | 93.8% |
| Page Conflicts | ~1.0 | 0% |

#### Multi-Bank Scaling (16 Banks)

| Banks | Throughput (bytes/cycle) | Speedup |
|-------|-------------------------|---------|
| 1 | 1.59 | 1.0x |
| 2 | 1.99 | 1.3x |
| 4 | 2.28 | 1.4x |
| 8 | 2.46 | 1.5x |
| 16 | 2.56 | 1.6x |

#### STREAM Benchmark Performance

| Kernel | Operations | Total Bytes | Throughput (bytes/cycle) |
|--------|-----------|-------------|-------------------------|
| Copy | 1R + 1W | 64 KB | 38.24 |
| Add | 2R + 1W | 96 KB | 59.08 |
| Triad | 2R + 1W | 96 KB | 59.08 |

#### Multi-DMA Tile Loading (16 Banks)

| DMA Engines | Total Bytes | Throughput (bytes/cycle) | Speedup |
|-------------|-------------|-------------------------|---------|
| 4 | 64 KB | 40.93 | 1.0x |
| 8 | 128 KB | 81.87 | 2.0x |
| 16 | 128 KB | 81.87 | 2.0x |
| 32 | 128 KB | 81.87 | 2.0x |

#### Access Pattern Comparison

| Pattern | Throughput (bytes/cycle) | Notes |
|---------|-------------------------|-------|
| GPU-style (64 threads) | 2.56 | Round-robin across 16 banks |
| Accelerator-style (tiles) | 1.72 | Sequential tile loads |
| Sustained Max | 10.23 | 256 accesses, optimal scheduling |

**Key Insight**: GDDR6's 16 banks provide 1.5-2x advantage over 8-bank configurations when fully utilized.

---

## HBM2-2000 Characterization

HBM2 (High Bandwidth Memory Gen 2) uses a 3D-stacked architecture with a wide interface (1024-bit) to achieve high bandwidth in a compact footprint.

### Architecture Overview

```
HBM2 Stack (1024-bit interface)
├── 8 Channels (128-bit each)
│   └── 2 Pseudo-Channels per Channel (64-bit each)
│       └── 16 Banks per Pseudo-Channel
│           └── 4 Bank Groups (4 banks each)
└── Total: 8 × 2 × 16 = 256 banks per stack
```

### Timing Parameters (HBM2-2000 @ 1.0 GHz CK)

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCDRD | 12 | Row to Column Delay (Read) |
| tRCDWR | 6 | Row to Column Delay (Write) |
| tRL | 18 | CAS Read Latency |
| tWL | 7 | CAS Write Latency |
| tRP | 14 | Row Precharge |
| tRAS | 28 | Row Active Time |
| tRC | 42 | Row Cycle Time |
| tWR | 16 | Write Recovery |
| tRTP | 6 | Read to Precharge |
| tBurst | 2 | Burst Duration (BL4) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 3 | ACT-to-ACT (different group) |
| tCCD_L | 4 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |
| tFAW | 16 | Four Activate Window |

### Latency Characterization

| Access Pattern | Latency (cycles) | Description |
|---------------|------------------|-------------|
| Page Hit Read | 20 | tRL + tBurst = 18 + 2 |
| Page Empty Read | 32 | tRCDRD + tRL + tBurst = 12 + 18 + 2 |
| Page Conflict Read | 46 | tRP + tRCDRD + tRL + tBurst = 14 + 12 + 18 + 2 |
| Page Hit Write | 9 | tWL + tBurst = 7 + 2 |
| Page Empty Write | 15 | tRCDWR + tWL + tBurst = 6 + 7 + 2 |

### Bandwidth Characteristics

| Configuration | Peak Bandwidth | Notes |
|---------------|---------------|-------|
| Per Pseudo-Channel | 16 GB/s | 64-bit × 2.0 Gbps |
| Per Channel | 32 GB/s | 2 pseudo-channels |
| Per Stack (8 ch) | 256 GB/s | Full HBM2-2000 stack |
| HBM2E-3200 | 409.6 GB/s | Enhanced variant |

### Power Efficiency

| Metric | Value | Notes |
|--------|-------|-------|
| **TDP per Stack** | 10-15 W | 4-8 Hi stack |
| **Energy per Bit** | 3-4 pJ/bit | Interface + core |
| **Interface Energy** | 1-2 pJ/bit | TSV efficiency |
| **Power per GB/s** | 40-60 mW | System-level |
| **Bandwidth/Watt** | 17-25 GB/s/W | Best-in-class |

**HBM2 Power Advantages:**

1. **TSV Interface**: 50-100x shorter signal paths than PCB traces (~50 μm vs 5-15 cm)
2. **Low Swing**: 0.4-0.6V CMOS signaling vs 1.1-1.5V for GDDR6
3. **No Termination**: On-die termination (ODT) not required, saves ~30% I/O power
4. **Parallel Width**: 1024-bit bus allows lower per-pin frequency
5. **Simple PHY**: No SerDes, CDR, or equalization needed

### Key HBM2 Characteristics

1. **Pseudo-Channel Independence**: Each pseudo-channel operates independently with its own bank array
2. **Asymmetric Read/Write**: tRCDWR (6) is half of tRCDRD (12) - writes have faster row access
3. **Through-Silicon Vias (TSVs)**: Vertical connections enable 1024-bit interface in small footprint
4. **Stacked DRAM Dies**: 4-8 DRAM dies per stack, connected to logic die
5. **Energy Efficiency**: Best-in-class pJ/bit enables sustained high bandwidth in power-constrained systems

---

## HBM3-5600 Characterization

HBM3 doubles the channel count (16 vs 8) and increases data rates to achieve nearly 3x the bandwidth of HBM2.

### Architecture Overview

```
HBM3 Stack (1024-bit interface)
├── 16 Channels (64-bit each)
│   └── 2 Pseudo-Channels per Channel (32-bit each)
│       └── 16 Banks per Pseudo-Channel
│           └── 4 Bank Groups (4 banks each)
└── Total: 16 × 2 × 16 = 512 banks per stack
```

### Timing Parameters (HBM3-5600 @ 2.8 GHz CK)

| Parameter | Value (cycles) | Description |
|-----------|---------------|-------------|
| tRCD | 8 | Row to Column Delay |
| tRL | 8 | CAS Read Latency |
| tWL | 4 | CAS Write Latency |
| tRP | 8 | Row Precharge |
| tRAS | 16 | Row Active Time |
| tRC | 24 | Row Cycle Time |
| tWR | 12 | Write Recovery |
| tRTP | 4 | Read to Precharge |
| tBurst | 4 | Burst Duration (BL8) |
| tRRD_L | 4 | ACT-to-ACT (same group) |
| tRRD_S | 2 | ACT-to-ACT (different group) |
| tCCD_L | 4 | CAS-to-CAS (same group) |
| tCCD_S | 2 | CAS-to-CAS (different group) |
| tFAW | 16 | Four Activate Window |

### Latency Characterization

| Access Pattern | Latency (cycles) | Description |
|---------------|------------------|-------------|
| Page Hit Read | 12 | tRL + tBurst = 8 + 4 |
| Page Empty Read | 20 | tRCD + tRL + tBurst = 8 + 8 + 4 |
| Page Conflict Read | 28 | tRP + tRCD + tRL + tBurst = 8 + 8 + 8 + 4 |
| Page Hit Write | 8 | tWL + tBurst = 4 + 4 |
| Page Empty Write | 16 | tRCD + tWL + tBurst = 8 + 4 + 4 |

### Bandwidth Characteristics

| Configuration | Peak Bandwidth | Notes |
|---------------|---------------|-------|
| Per Pseudo-Channel | 22.4 GB/s | 32-bit × 5.6 Gbps |
| Per Channel | 44.8 GB/s | 2 pseudo-channels |
| Per Stack (16 ch) | 716.8 GB/s | Full HBM3-5600 stack |
| HBM3E-9200 | 1.18 TB/s | Enhanced variant |

### Power Efficiency

| Metric | Value | Notes |
|--------|-------|-------|
| **TDP per Stack** | 15-20 W | 8-16 Hi stack |
| **Energy per Bit** | 2.5-3.5 pJ/bit | Best-in-class |
| **Interface Energy** | 0.8-1.5 pJ/bit | Improved TSV |
| **Power per GB/s** | 25-35 mW | Industry leading |
| **Bandwidth/Watt** | 30-40 GB/s/W | 2x better than HBM2 |

**HBM3 Power Improvements over HBM2:**

| Improvement | HBM2 | HBM3 | Benefit |
|-------------|------|------|---------|
| Interface pJ/bit | 1-2 | 0.8-1.5 | 25-30% reduction |
| Core efficiency | 2-2.5 | 1.5-2 | 20-25% reduction |
| Total pJ/bit | 3-4 | 2.5-3.5 | ~15-25% improvement |
| BW/Watt | 17-25 | 30-40 | 60-80% improvement |

**Why HBM3 is More Efficient:**

1. **Higher Data Rate per Pin**: More bits per clock cycle = better amortization of static power
2. **Improved TSV Technology**: Lower capacitance, tighter pitch
3. **Advanced Process Node**: 1x nm vs 20nm DRAM reduces leakage
4. **Per-Bank Refresh**: Reduces refresh overhead vs distributed refresh
5. **Smaller Page Size**: 1 KB pages reduce activation energy for random access

### Key HBM3 Characteristics

1. **Doubled Channels**: 16 channels vs HBM2's 8 channels
2. **Narrower Per-Channel Width**: 64-bit vs 128-bit, but more parallelism
3. **Higher Data Rate**: 5.6 Gbps (HBM3) to 9.2 Gbps (HBM3E)
4. **1 KB Pages**: Smaller pages than HBM2 (2 KB) improve random access
5. **Best Energy Efficiency**: 2.5-3.5 pJ/bit enables sustained multi-TB/s systems

---

## HBM Evolution: HBM2 → HBM3 → HBM4

### Generation Comparison

| Feature | HBM2 | HBM2E | HBM3 | HBM3E | HBM4 (Planned) |
|---------|------|-------|------|-------|----------------|
| **Data Rate** | 2.0 Gbps | 3.6 Gbps | 5.6 Gbps | 9.2 Gbps | 12+ Gbps |
| **Peak BW/Stack** | 256 GB/s | 460 GB/s | 716 GB/s | 1.18 TB/s | 1.5+ TB/s |
| **Channels** | 8 | 8 | 16 | 16 | 16+ |
| **Channel Width** | 128-bit | 128-bit | 64-bit | 64-bit | TBD |
| **Capacity/Stack** | 8 GB | 16 GB | 16 GB | 24 GB | 32+ GB |
| **Die Stacking** | 4-8 Hi | 8-12 Hi | 8-12 Hi | 12-16 Hi | 16+ Hi |
| **Process Node** | 20nm | 1x nm | 1x nm | 1y nm | 1z nm |
| **Energy/Bit (pJ)** | 3-4 | 3-3.5 | 2.5-3.5 | 2-3 | <2 |
| **BW/Watt (GB/s/W)** | 17-25 | 25-30 | 30-40 | 40-50 | 50+ |

### Key Architectural Improvements

#### HBM2 → HBM3

| Improvement | HBM2 | HBM3 | Benefit |
|------------|------|------|---------|
| Channels | 8 | 16 | 2x parallelism |
| Data Rate | 2.0-3.6 Gbps | 5.6-9.2 Gbps | 1.5-2.5x speed |
| Page Size | 2 KB | 1 KB | Better random access |
| ECC | Optional | Inline | Always-on reliability |
| Refresh | Distributed | Per-bank | Lower latency impact |

#### HBM3 → HBM4 (Expected)

| Improvement | HBM3E | HBM4 (Est.) | Benefit |
|------------|-------|-------------|---------|
| Data Rate | 9.2 Gbps | 12+ Gbps | 30%+ speed |
| Capacity | 24 GB | 32+ GB | More model weights |
| Die Count | 12-16 | 16+ | Higher density |
| Energy/bit | 2-3 pJ/bit | <2 pJ/bit | Better efficiency |
| BW/Watt | 40-50 GB/s/W | 50+ GB/s/W | Lower TCO |

### Bandwidth Scaling Across Generations

```
Bandwidth per Stack (TB/s)
                                                          ┌───────┐
                                                          │ HBM4  │
                                                          │ 1.5+  │
                                            ┌───────┐     └───────┘
                                            │ HBM3E │
                                            │ 1.18  │
                              ┌───────┐     └───────┘
                              │ HBM3  │
                              │ 0.72  │
              ┌───────┐       └───────┘
              │ HBM2E │
              │ 0.46  │
┌───────┐     └───────┘
│ HBM2  │
│ 0.26  │
└───────┘
  2016      2018      2021      2023      2025+
```

### Use Cases by Generation

| Generation | Typical Products | Target Workloads |
|------------|-----------------|------------------|
| **HBM2** | V100, MI25 | FP32/FP16 training |
| **HBM2E** | A100, MI100 | Large model training |
| **HBM3** | H100, MI300 | LLM training/inference |
| **HBM3E** | H200, B100 | Trillion-param models |
| **HBM4** | Future | Multi-modal, AGI research |

---

## Comparative Analysis

### LPDDR5 vs GDDR6

| Metric | LPDDR5-6400 | GDDR6-16000 | Winner |
|--------|-------------|-------------|--------|
| Banks | 8 | 16 | GDDR6 |
| Bank Groups | 2 | 4 | GDDR6 |
| Page Hit Latency | 24 cycles | 24 cycles | Tie |
| Page Empty Latency | 38 cycles | 42 cycles | LPDDR5 |
| Page Conflict Latency | 52 cycles | 60 cycles | LPDDR5 |
| STREAM Copy | 19.04 B/cyc | 38.24 B/cyc | GDDR6 |
| STREAM Triad | 26.67 B/cyc | 59.08 B/cyc | GDDR6 |
| Multi-DMA (8 engines) | 89.47 B/cyc | 81.87 B/cyc | LPDDR5 |

### LPDDR5 vs HBM2

| Metric | LPDDR5-6400 | HBM2-2000 | Winner |
|--------|-------------|-----------|--------|
| Peak Bandwidth | 25.6 GB/s | 256 GB/s | HBM2 (10x) |
| Banks | 8 | 256 | HBM2 (32x) |
| Page Hit Latency | 24 cycles | 20 cycles | HBM2 |
| Page Conflict Latency | 52 cycles | 46 cycles | HBM2 |
| Energy per Bit | 5-7 pJ/bit | 3-4 pJ/bit | **HBM2** |
| Total Device Power | 1-2 W | 10-15 W | LPDDR5 |
| Cost | $ | $$$$$ | LPDDR5 |
| Form Factor | SoC package | Interposer | LPDDR5 |

*Note: HBM2 wins on energy/bit but LPDDR5 wins on TDP. For bandwidth-intensive workloads, HBM2 delivers more bandwidth per watt.*

### HBM2 vs HBM3

| Metric | HBM2-2000 | HBM3-5600 | Winner |
|--------|-----------|-----------|--------|
| Peak Bandwidth | 256 GB/s | 716.8 GB/s | HBM3 (2.8x) |
| Channels | 8 | 16 | HBM3 |
| Banks per Stack | 256 | 512 | HBM3 (2x) |
| Page Hit Latency | 20 cycles @ 1 GHz | 12 cycles @ 2.8 GHz | Similar (ns) |
| Page Size | 2 KB | 1 KB | HBM3 (random) |
| Capacity/Stack | 8-16 GB | 16-24 GB | HBM3 |
| Energy per Bit | 3-4 pJ/bit | 2.5-3.5 pJ/bit | HBM3 |
| BW/Watt | 17-25 GB/s/W | 30-40 GB/s/W | HBM3 (60%+ better) |

### All Technologies Comparison

| Metric | LPDDR5 | GDDR6 | HBM2 | HBM3 |
|--------|--------|-------|------|------|
| **Bandwidth** | 25.6 GB/s | 64 GB/s | 256 GB/s | 716.8 GB/s |
| **Banks** | 8 | 16 | 256 | 512 |
| **Cost (relative)** | 1x | 2x | 10x | 15x |
| **TDP (W)** | 1-2 | 10-15 | 10-15 | 15-20 |
| **Energy (pJ/bit)** | 5-7 | 8-12 | **3-4** | **2.5-3.5** |
| **BW/Watt (GB/s/W)** | 13-25 | 4-6 | **17-25** | **30-40** |
| **Latency (ns)** | 7.5 | 12 | 20 | 4.3 |
| **Integration** | Easy | Easy | Hard | Hard |
| **Best For** | Edge AI | Graphics | AI Training | LLM Training |

**Key Insight**: HBM has the **best energy efficiency per bit** (lowest pJ/bit) and **best bandwidth per watt**. This is the fundamental reason all HPC and AI accelerators use HBM - it delivers the highest bandwidth at the lowest energy cost.

### Technology Selection Guide

| Workload | Recommended | Rationale |
|----------|-------------|-----------|
| **Mobile AI** | LPDDR5 | Power efficiency, SoC integration |
| **Gaming GPU** | GDDR6 | Good BW/cost, simple integration |
| **Cloud AI Inference** | HBM2E | High bandwidth, proven reliability |
| **LLM Training** | HBM3 | Maximum bandwidth for weight updates |
| **Edge Vision** | LPDDR5 | Power budget constraints |
| **HPC Simulation** | HBM3 | Bandwidth for large-scale stencils |

### Design Recommendations

1. **For Edge/Mobile (< 5W)**: LPDDR5 with page-hit optimized access patterns - lowest TDP
2. **For Desktop Graphics**: GDDR6 with full 16-bank utilization - good cost/BW tradeoff
3. **For Data Center AI**: HBM2E/HBM3 - best pJ/bit enables sustained high bandwidth
4. **For Maximum Throughput**: HBM3 with interleaved channel access - highest BW/Watt
5. **For Power-Constrained AI**: HBM3 delivers 3x more bandwidth per watt than GDDR6
6. **For Latency-Critical Inference**: Optimize page locality regardless of technology

### Why AI Accelerators Use HBM

Modern AI accelerators (NVIDIA H100, AMD MI300, Google TPUv5) exclusively use HBM because:

1. **Energy Efficiency**: HBM's 2.5-4 pJ/bit is 2-4x better than alternatives
2. **Power Budget**: A 700W accelerator can sustain 3+ TB/s with HBM vs <500 GB/s with GDDR6
3. **Memory Bandwidth Wall**: AI models are memory-bound; HBM maximizes FLOP utilization
4. **TCO**: Lower energy cost per inference dominates operating expenses

```
System Power Budget Example (400W for memory)
─────────────────────────────────────────────────────────────────
GDDR6 (8-12 pJ/bit):  400W / 10 pJ/bit = 40 GB/s × 8 bits = 320 GB/s
HBM3 (2.5-3.5 pJ/bit): 400W / 3 pJ/bit = 133 GB/s × 8 bits = 1066 GB/s
─────────────────────────────────────────────────────────────────
Result: HBM3 delivers 3.3x more bandwidth within same power envelope
```

---

## Pattern Categories

### Level 1: Single Bank
- `page_hits` - Sequential access to same row
- `page_conflicts` - Alternating rows (worst case)
- `mixed_rw` - Interleaved reads and writes

### Level 2: Two Banks
- `same_group` - Banks in same bank group
- `diff_groups` - Banks in different bank groups

### Level 3: Three Banks
- `same_group` - All banks in one group
- `mixed_groups` - Banks across multiple groups

### Level 4: Four Banks
- `full_group` - Complete bank group
- `across_groups` - One bank per group
- `page_hit_burst` - Sustained page hits

### Level 5: Dual Channel (GDDR6)
- `independent` - Separate channel access
- `interleaved` - Alternating channel access

### Level 6: Complex Patterns
- `stream` - STREAM benchmark (Copy, Scale, Add, Triad)
- `tile_load` - ML accelerator tile loading
- `multi_dma` - Concurrent DMA engine simulation
- `random` - Random access pattern
- `strided` - Regular strided access

### Level 7: Bandwidth Patterns
- `page_burst` - Maximum page hits (128/page)
- `max_bandwidth` - Peak achievable bandwidth

---

## Trace Files

All patterns generate Chrome Trace Format JSON files viewable in:
- Perfetto UI: https://ui.perfetto.dev
- Chrome: chrome://tracing
- Custom viewers in `traces/memory/{lpddr5,gddr6}/tools/`

See `traces/README.md` for detailed visualization instructions.

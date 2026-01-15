# SoC Architecture

High-level comparison of the cache hierarchy across three major ARM-based vendors—Apple (M3 series), Qualcomm (Snapdragon 8 Gen 3), and Ampere (Altra Max)—highlighting how each targets different performance and power domains.

---

## Cache Hierarchy Comparison: Apple vs Qualcomm vs Ampere

| Vendor     | Architecture / SoC         | L1 Cache (per core)           | L2 Cache                     | L3 / System-Level Cache     | Notes |
|------------|-----------------------------|-------------------------------|------------------------------|-----------------------------|-------|
| **Apple**  | M3 / M3 Pro / M3 Max / Ultra | Perf: 192+128 KiB<br>Eff: 128+64 KiB | Perf: 16–64 MiB shared<br>Eff: 4 MiB shared | Unified system-wide L3 (UMA) | Unified memory architecture; L3 acts as last-level cache for all cores |
| **Qualcomm** | Snapdragon 8 Gen 3           | Not explicitly disclosed (likely ~64 KiB I/D) | Not disclosed | 12 MB shared L3 cache | Mobile-optimized; L3 shared across all cores |
| **Ampere** | Altra Max (128-core server)   | 64 KiB I + 64 KiB D per core | 1 MB private per core       | 16 MB system-level cache (SLC) | No shared L2 or L3; fully private L2 per core for predictability |

---

### 🔍 Architectural Insights

- **Apple M3 Series**:
  - Uses a **unified memory architecture (UMA)** where L3 cache is tightly integrated with LPDDR5 memory controllers.
  - L2 cache is **shared among performance cores**, with larger capacities in Pro/Max/Ultra variants.
  - Emphasizes **dynamic caching** for GPU workloads, optimizing memory allocation in real time.

- **Qualcomm Snapdragon 8 Gen 3**:
  - Designed for smartphones, with **shared L3 cache** across all cores.
  - Likely uses **private L2 per core**, though not explicitly disclosed.
  - Prioritizes **energy efficiency and burst performance** for mobile tasks.

- **Ampere Altra Max**:
  - Built for cloud-native workloads with **predictable performance**.
  - Each core has **private L1 and L2 caches**, avoiding noisy neighbor effects.
  - A modest **16 MB system-level cache** supports coherence across 128 cores via a mesh interconnect.

---

### Design Philosophy Contrast

- **Apple**: Unified, high-bandwidth cache system tuned for mixed workloads (CPU + GPU + NPU).
- **Qualcomm**: Balanced mobile cache hierarchy with shared L3 for latency-sensitive tasks.
- **Ampere**: Server-grade, deterministic performance via fully private caching and massive core counts.

## Private L2 Architecture

To make an **L2 cache private to a core** in an ARM (or any) architecture, you need to design the cache hierarchy and interconnect such that each core has **exclusive access** to its own L2 slice. Here's how it's typically done:

---

## Architectural Requirements for Private L2

### 1. **Dedicated L2 Cache per Core**
- Each core is physically paired with its own L2 cache.
- The L2 is not accessible by other cores—no shared bus or crossbar to other L2s.
- This avoids coherence traffic between L2s and simplifies latency paths.

### 2. **Core-to-L2 Coupling**
- The L2 is tightly coupled to the core's pipeline and memory interface.
- Often implemented as a **core-local cache controller** that handles L1 misses and forwards requests to memory if L2 misses.

### 3. **No Shared L2 Interconnect**
- Unlike shared L2 designs (e.g., cluster-level L2 with a crossbar), private L2s avoid interconnect arbitration.
- Each L2 connects directly to the core and to the next level (e.g., L3 or memory controller).

### 4. **Coherence via L3 or Snoop Filters**
- Coherence is maintained at the L3 level or via a **snoop filter** or **directory-based protocol**.
- This ensures that private L2s remain coherent without direct peer-to-peer communication.

---

## Implementation Example: Ampere Altra Max

- **128 cores**, each with:
  - **64 KiB L1 I-cache**
  - **64 KiB L1 D-cache**
  - **1 MiB private L2 cache**
- No shared L2 or L3; instead, a **16 MB system-level cache (SLC)** handles coherence and memory access.
- This design favors **predictable latency** and **scalability** for cloud workloads.

---

## Benefits of Private L2

- **Lower latency**: No arbitration or contention.
- **Simplified design**: Easier to scale core count.
- **Predictable performance**: No noisy neighbor effects.
- **Energy efficiency**: Reduced interconnect power.

---

## Tradeoffs

- **Higher area cost**: Duplicated cache logic per core.
- **Lower cache utilization**: No sharing means unused capacity can't be leveraged.
- **More complex coherence**: Requires robust L3 or snoop filters.

---


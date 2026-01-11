# Why Overfetch is NOT Symmetric: Root Cause Analysis

## Your Excellent Question

> "I would expect that a tall-skinny and a wide-deep matmul we would simply flip the stationary tensor and thus they would have the same scheduling behavior."

**You are absolutely correct!** This SHOULD be symmetric, but it's not in our current implementation.

## Terminology Clarification

Given `C = A × B` where A is M×K, B is K×N, C is M×N:

| Aspect Ratio | Definition | Example | Tensor Sizes |
|-------------|------------|---------|-------------|
| **TALL** | M >> N, K | 16384×512×512 | A=32MB, B=1MB, C=32MB |
| **WIDE** | N >> M, K | 512×16384×2048 | A=4MB, B=128MB, C=32MB |
| **DEEP** | K >> M, N | 1024×1024×8192 | A=32MB, B=32MB, C=4MB |
| **TALL-WIDE** | M >> K, N >> K | 32768×7168×7168 | A=896MB, B=196MB, C=896MB |

**TALL-WIDE** (your 32k×7k case): Both M and N are large, but K is also large. 
- This is NOT the transpose of TALL or WIDE
- It's more like "large batch × large hidden dimension"
- Think: transformer inference with 32k tokens and 7k model dimension

## The Asymmetry: Current Results

| Aspect | Avg Overfetch | Dominant Tensor |
|--------|--------------|-----------------|
| Tall | 1.02× | None (all ~1×) |
| Wide | 4.91× | **B: 5.97×** |
| Deep | 7.80× | **B: 16.45×** |
| Tall-Wide | 23.29× | **B: 228×** |

**Key observation**: Tensor B dominates for everything except Tall!

## Root Cause: Hard-Coded Output-Stationary Tile Ordering

Looking at `l2_tile_scheduler.cpp:354-370`:

```cpp
std::vector<std::tuple<Size, Size, Size>> L2TileScheduler::generate_compute_order(
    const L2Schedule& schedule) const
{
    std::vector<std::tuple<Size, Size, Size>> order;

    // Output-stationary: iterate over C tiles (ti, tj)
    // For each C tile, accumulate across K dimension (tk)
    for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
        for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
            for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                order.push_back(std::make_tuple(ti, tj, tk));
            }
        }
    }

    return order;
}
```

**This is ALWAYS output-stationary ordering, regardless of the `strategy` parameter!**

The loop nest order is **FIXED**:
```
for ti (iterate over M):
  for tj (iterate over N):
    for tk (iterate over K):
      Compute C[ti,tj] += A[ti,tk] × B[tk,tj]
```

## Why This Breaks Symmetry

### Output-Stationary Behavior

With the current `ti → tj → tk` loop order:

**For each output tile C[ti,tj]:**
- C tile stays resident (loaded once per output tile)
- A[ti, tk] is loaded once per K tile (reused across N)
- B[tk, tj] is loaded once per K tile (reused across M)

**Tile access patterns:**
- C accessed: `M_tiles × N_tiles` times (once per output tile)
- A accessed: `M_tiles × K_tiles` times (loaded M_tiles times across K)
- B accessed: `N_tiles × K_tiles` times (loaded N_tiles times across K)

### Why Tall Works Well

For **TALL** (M=16384, N=512, K=512):
- Many M tiles, few N tiles, few K tiles
- B is small (1MB) → fits in L3
- A is reused poorly, but doesn't matter because it fits in L3
- C is large but output-stationary keeps it resident
- **Result**: ~1× overfetch on all tensors

### Why Wide Fails

For **WIDE** (M=512, N=16384, K=2048):
- Few M tiles, many N tiles, some K tiles
- B is large (128MB) >> L3 (64MB)
- B accessed `N_tiles × K_tiles` times
- Each access to B column stripe requires scanning across K
- B thrashes in L3!
- **Result**: B gets 9.6× overfetch

### Why Tall-Wide is Catastrophic

For **TALL-WIDE** (M=32768, N=7168, K=7168):
- Many M tiles, many N tiles, many K tiles
- B is huge (196MB) >> L3
- B accessed `M_tiles × N_tiles` times (huge!)
- For each of the M_tiles output rows, we scan through all of B
- **Result**: B gets **342× overfetch!**

## What SHOULD Happen with Proper Strategy Selection

With strategy-aware tile ordering, we should have:

### Weight-Stationary (WS): Keep B resident
```cpp
for tk (iterate over K):      // Outer loop on K
  for ti (iterate over M):
    for tj (iterate over N):
      Compute C[ti,tj] += A[ti,tk] × B[tk,tj]
```
- B[tk, :] stays in L3 across all output tiles
- Best for: **WIDE** and **DEEP** (large B)

### Input-Stationary (IS): Keep A resident
```cpp
for tk (iterate over K):      // Outer loop on K
  for tj (iterate over N):
    for ti (iterate over M):
      Compute C[ti,tj] += A[ti,tk] × B[tk,tj]
```
- A[ti, :] stays in L3
- Best for: **TALL** (large A)

### Output-Stationary (OS): Keep C resident
```cpp
for ti (iterate over M):      // Current implementation
  for tj (iterate over N):
    for tk (iterate over K):
      Compute C[ti,tj] += A[ti,tk] × B[tk,tj]
```
- C[ti,tj] accumulates in place
- Best for: **Small C**, or when both A and B fit in L3

## Why All Strategies Show Same Overfetch

Looking at `l2_tile_scheduler.cpp:26-30`:

```cpp
L2TileScheduler::L2Schedule L2TileScheduler::generate_schedule(
    Size M, Size N, Size K,
    const TileOptimizer::TileConfig& config,
    ReplacementPolicy policy,
    SchedulingStrategy strategy)  // <-- This parameter is IGNORED!
```

The `strategy` parameter is stored (line 65) but **never used** in `generate_compute_order()`.

## The Fix

To make the analysis symmetric and strategy-dependent:

```cpp
std::vector<std::tuple<Size, Size, Size>> L2TileScheduler::generate_compute_order(
    const L2Schedule& schedule) const
{
    std::vector<std::tuple<Size, Size, Size>> order;

    switch (strategy_) {
        case SchedulingStrategy::WEIGHT_STATIONARY:
            // tk → ti → tj (keep B resident)
            for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                    for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;

        case SchedulingStrategy::INPUT_STATIONARY:
            // tk → tj → ti (keep A resident)
            for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                    for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;

        case SchedulingStrategy::OUTPUT_STATIONARY:
        default:
            // ti → tj → tk (keep C resident) - CURRENT IMPLEMENTATION
            for (Size ti = 0; ti < schedule.num_tile_rows_C; ++ti) {
                for (Size tj = 0; tj < schedule.num_tile_cols_C; ++tj) {
                    for (Size tk = 0; tk < schedule.num_tile_cols_A; ++tk) {
                        order.push_back(std::make_tuple(ti, tj, tk));
                    }
                }
            }
            break;
    }

    return order;
}
```

## Expected Results After Fix

With proper strategy selection:

| Shape | Best Strategy | Why |
|-------|--------------|-----|
| Tall (large M) | OS or IS | C or A fits in L3 |
| Wide (large N) | WS | B column stripe reuse |
| Deep (large K) | WS | B stays resident across K |
| Tall-Wide | WS | B is critical bottleneck |

After the fix, we should see:
- **WS** giving best results for Wide/Deep/Tall-Wide
- **IS/OS** giving best results for Tall
- Symmetry between transposed shapes with appropriate strategy

## Summary

**Your intuition is correct!** The overfetch SHOULD be symmetric with proper strategy selection.

**Current bug:** The L2 tile scheduler ignores the strategy parameter and always uses output-stationary ordering. This is why:
- Tall matrices do well (output-stationary is a good match)
- Wide/Deep matrices do poorly (wrong strategy for the shape)
- All three strategies (WS/IS/OS) show identical overfetch

**Fix:** Make `generate_compute_order()` respect the `strategy_` member variable.

**Result:** Overfetch will become strategy-dependent, and you'll be able to choose the right strategy for each workload shape.

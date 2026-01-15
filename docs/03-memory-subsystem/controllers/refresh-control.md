# Refresh Control API Design

This document analyzes the requirements for controllable refresh in KPU-SIM and proposes a refined API that balances flexibility with simplicity.

## Requirements Analysis

### Use Cases

| Use Case | Requirements | Current Support |
|----------|--------------|-----------------|
| **Short pattern tests** | No refresh interference | Need DISABLED mode |
| **Sequence analysis** | Inject refresh at known points | Need explicit injection |
| **DNN simulation** | Realistic refresh impact, no user intervention | Need automatic with proper timing |
| **Batch boundary study** | Inject refresh between inference batches | Need cycle/event triggering |
| **Worst-case analysis** | Force maximum refresh impact | Need aggressive refresh mode |

### Real Controller Behavior

Real memory controllers implement a two-tier refresh scheduling:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Refresh Scheduler                            │
│                                                                 │
│   Tier 1: Opportunistic                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ • Monitor bus idle cycles                               │   │
│   │ • If pending refresh AND bus idle → issue REF           │   │
│   │ • Minimizes impact on active workloads                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   Tier 2: Deadline Enforcement                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ • Track refresh debt per bank                           │   │
│   │ • If debt × tREFI > deadline → FORCE REF                │   │
│   │ • Cannot be deferred (data integrity)                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The deadline is typically 8-9 × tREFI (JEDEC allows postponing up to 8 refresh commands).

### Original Proposal Problems

The original three-mode design had issues:

```cpp
enum class RefreshMode : uint8_t {
    AUTOMATIC,  // OK - default behavior
    MANUAL,     // Problem: only transaction boundaries
    DISABLED    // OK - for short tests
};
```

**Problem with MANUAL:** Requires user to know transaction boundaries. For DNN simulation with millions of operations, this is impractical. Users need cycle-based control without tracking individual transactions.

## Proposed Design

### Core Insight: Separate Scheduling Policy from Safety

The key insight is that refresh control has two orthogonal dimensions:

1. **Scheduling policy:** When/how to trigger refresh
2. **Safety guarantee:** Whether deadline enforcement is active

### Refined Enum

```cpp
/// Refresh scheduling mode
enum class RefreshMode : uint8_t {
    AUTOMATIC,      // Default: opportunistic + deadline enforcement
    INTERVAL,       // Fixed cycle interval (user-specified)
    OPPORTUNISTIC,  // Opportunistic only, no proactive scheduling
    EXPLICIT,       // Only when inject_refresh() called
    DISABLED        // No refresh (for short tests only)
};
```

**Mode Descriptions:**

| Mode | Scheduling | Deadline | Use Case |
|------|-----------|----------|----------|
| AUTOMATIC | Opportunistic + proactive | Enforced | Production, realistic simulation |
| INTERVAL | Fixed cycle count | Optional | DNN simulation with controlled timing |
| OPPORTUNISTIC | Idle periods only | Enforced | Study opportunistic scheduling |
| EXPLICIT | User calls inject_refresh() | Optional | Pattern tests, sequence analysis |
| DISABLED | None | None | Short deterministic tests |

### Complete Interface

```cpp
// ========================================================================
// Refresh Control (IMemoryController additions)
// ========================================================================

/// Refresh scheduling mode
enum class RefreshMode : uint8_t {
    AUTOMATIC,      // Opportunistic + deadline (default)
    INTERVAL,       // Fixed interval with optional deadline
    OPPORTUNISTIC,  // Opportunistic only + deadline
    EXPLICIT,       // Manual injection + optional deadline
    DISABLED        // No refresh (test mode)
};

// --- Mode Configuration ---

/// Set refresh scheduling mode
/// @param mode The scheduling mode to use
virtual void set_refresh_mode(RefreshMode mode) = 0;

/// Get current refresh mode
virtual RefreshMode refresh_mode() const = 0;

// --- Interval Mode Configuration ---

/// Set refresh interval for INTERVAL mode
/// @param cycles Number of cycles between refresh commands
/// @note Only effective when mode is INTERVAL
virtual void set_refresh_interval(uint64_t cycles) = 0;

/// Get configured refresh interval
virtual uint64_t refresh_interval() const = 0;

// --- Safety Configuration ---

/// Enable/disable deadline enforcement
/// @param enforce If true, force refresh at deadline regardless of mode
/// @note Default is true; set to false only for controlled tests
/// @warning Disabling deadline enforcement risks data loss in real hardware
virtual void set_deadline_enforcement(bool enforce) = 0;

/// Check if deadline enforcement is active
virtual bool deadline_enforced() const = 0;

// --- Query Interface ---

/// Get cycles until refresh deadline for a bank
/// @param channel Memory channel
/// @param bank Bank number
/// @return Cycles until deadline (0 if deadline passed)
virtual uint64_t cycles_until_deadline(uint8_t channel, uint8_t bank) const = 0;

/// Check if refresh is pending for a bank
/// @param channel Memory channel
/// @param bank Bank number
/// @return true if bank has pending refresh
virtual bool refresh_pending(uint8_t channel, uint8_t bank) const = 0;

/// Get accumulated refresh debt for a bank
/// @param channel Memory channel
/// @param bank Bank number
/// @return Number of deferred refresh operations
virtual uint32_t refresh_debt(uint8_t channel, uint8_t bank) const = 0;

// --- Manual Injection ---

/// Inject a refresh command
/// @param channel Target channel
/// @param bank Target bank (-1 for all-bank refresh)
/// @return true if refresh was issued, false if blocked
virtual bool inject_refresh(uint8_t channel, int8_t bank = -1) = 0;
```

### Usage Examples

**1. Pattern Test (no refresh interference):**
```cpp
DMAHarness harness(dma_cfg, mc_cfg);
auto& mc = harness.memory_controller();

// Disable refresh completely for deterministic test
mc.set_refresh_mode(RefreshMode::DISABLED);

harness.submit_tile_read(...);
harness.run_until_complete();
harness.export_trace("pattern_no_refresh.json");
```

**2. Pattern Test with Controlled Refresh:**
```cpp
// Use explicit mode with deadline off for full control
mc.set_refresh_mode(RefreshMode::EXPLICIT);
mc.set_deadline_enforcement(false);

// Run first phase
for (int i = 0; i < 6; i++) {
    harness.submit_tile_read(i, ...);
}
harness.run_cycles(500);

// Inject refresh at tile boundary
mc.inject_refresh(0, 0);  // Refresh channel 0, bank 0
harness.run_cycles(mc_cfg.timing.tRFC);  // Wait for refresh

// Run second phase
for (int i = 6; i < 12; i++) {
    harness.submit_tile_read(i, ...);
}
harness.run_until_complete();
```

**3. DNN Simulation with Interval Refresh:**
```cpp
// Set interval mode for predictable refresh behavior
mc.set_refresh_mode(RefreshMode::INTERVAL);
mc.set_refresh_interval(7800);  // Every 7800 cycles (~tREFI at 2GHz)
mc.set_deadline_enforcement(true);  // Safety on

// Run full inference workload - refresh happens automatically every 7800 cycles
run_inference(model, inputs);
```

**4. Study Opportunistic Scheduling:**
```cpp
// Opportunistic only - no proactive scheduling
mc.set_refresh_mode(RefreshMode::OPPORTUNISTIC);

// Run workload and observe when refresh actually occurs
run_workload();

// Analyze trace to see how refresh was scheduled around active transfers
```

**5. Worst-Case Latency Analysis:**
```cpp
// Force frequent refresh to study worst case
mc.set_refresh_mode(RefreshMode::INTERVAL);
mc.set_refresh_interval(1000);  // Very aggressive refresh
mc.set_deadline_enforcement(true);

run_latency_sensitive_workload();
analyze_tail_latency();
```

## Implementation Notes

### State Additions

Each memory controller implementation needs:

```cpp
// Refresh control state
RefreshMode refresh_mode_ = RefreshMode::AUTOMATIC;
uint64_t refresh_interval_ = 0;  // 0 = use tREFI
bool deadline_enforcement_ = true;
uint64_t last_interval_refresh_ = 0;  // For INTERVAL mode tracking
```

### Modified handle_refresh()

```cpp
void handle_refresh() {
    // DISABLED: no refresh at all
    if (refresh_mode_ == RefreshMode::DISABLED) {
        return;
    }

    // Check deadline enforcement (applies to all modes except DISABLED)
    if (deadline_enforcement_) {
        for (each bank) {
            if (cycles_until_deadline(ch, b) == 0) {
                force_refresh(ch, b);  // Cannot be deferred
                return;
            }
        }
    }

    // Mode-specific scheduling
    switch (refresh_mode_) {
        case RefreshMode::AUTOMATIC:
            // Opportunistic + proactive (current behavior)
            do_opportunistic_refresh();
            do_proactive_refresh();
            break;

        case RefreshMode::INTERVAL:
            // Fixed interval
            if (current_cycle_ >= last_interval_refresh_ + refresh_interval_) {
                do_round_robin_refresh();
                last_interval_refresh_ = current_cycle_;
            }
            break;

        case RefreshMode::OPPORTUNISTIC:
            // Only during idle
            if (bus_is_idle()) {
                do_opportunistic_refresh();
            }
            break;

        case RefreshMode::EXPLICIT:
            // Nothing - wait for inject_refresh() call
            break;
    }
}
```

## Design Rationale

### Why Not Transaction Boundaries?

Transaction boundary triggering was considered but rejected:

1. **Complexity:** Requires tracking transaction completion events
2. **Ambiguity:** What counts as a "transaction"? A single read? A tile transfer? A layer?
3. **Coupling:** Ties refresh logic to DMA/request semantics
4. **Limited use:** Only useful for very specific test scenarios

Cycle-based INTERVAL mode is more general and easier to reason about.

### Why Separate Deadline Enforcement?

Deadline enforcement is a safety mechanism, orthogonal to scheduling:

- EXPLICIT mode users might want deadline protection (safe experimentation)
- INTERVAL mode users might want to disable deadline (study pure interval behavior)
- Making it separate allows all combinations

### Why Not Callbacks?

A callback-based design was considered:

```cpp
// Rejected approach
using RefreshDecider = std::function<bool(uint64_t cycle, uint8_t ch, uint8_t bank)>;
void set_refresh_callback(RefreshDecider cb);
```

Rejected because:
1. **Performance:** Function call overhead every cycle
2. **Complexity:** Callback state management
3. **Debugging:** Harder to trace refresh decisions
4. **Serialization:** Cannot save/restore callback state

The enum-based approach is simpler, faster, and sufficient for all identified use cases.

## Migration Path

Existing code using automatic refresh continues to work unchanged.

New code can opt into controlled refresh:

```cpp
// Old code (still works)
auto mc = create_memory_controller(config);
mc->tick();  // Automatic refresh as before

// New code (controlled refresh)
auto mc = create_memory_controller(config);
mc->set_refresh_mode(RefreshMode::INTERVAL);
mc->set_refresh_interval(5000);
mc->tick();  // Refresh every 5000 cycles
```

## Summary

The refined API provides:

| Feature | Mechanism |
|---------|-----------|
| No refresh (tests) | `DISABLED` mode |
| User-controlled timing | `EXPLICIT` mode + `inject_refresh()` |
| Fixed interval (DNN) | `INTERVAL` mode + `set_refresh_interval()` |
| Realistic behavior | `AUTOMATIC` mode (default) |
| Opportunistic study | `OPPORTUNISTIC` mode |
| Safety guarantee | `set_deadline_enforcement()` |
| Visibility | `cycles_until_deadline()`, `refresh_debt()` |

This design avoids "messing up" the API by:
1. Keeping the enum simple and orthogonal
2. Using separate methods for configuration
3. Maintaining backward compatibility
4. Providing clear semantics for each mode

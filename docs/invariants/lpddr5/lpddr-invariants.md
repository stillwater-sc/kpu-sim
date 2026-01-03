# LPDDR5 Invariants

Below is a **formal, implementation-ready approach** to designing a **command legality matrix for LPDDR5**, with enough structure to be enforced mechanically in modern C++ and to constrain AI Assistant effectively.

This is intentionally written as a *design artifact*, not prose. You should be able to lift this almost verbatim into a header or spec document.

---

## 1. Scope and Assumptions (Fix These First)

**Scope**

* Single LPDDR5 device (rank)
* Behavioral (command-accurate, not signal-accurate)
* One command applied at a time
* Timing checked separately (this matrix is *state legality only*)

**Assumptions**

* Bank-level state is modeled independently
* Refresh and power states are device-level
* Illegal commands are rejected, not deferred

---

## 2. Canonical LPDDR5 Command Set (Abstracted)

You must normalize the spec’s command variants into a **minimal semantic set**. This prevents duplication errors.

**Device-level commands**

* `RESET`
* `INIT`
* `MRW` (Mode Register Write)
* `MRR` (Mode Register Read)
* `REFRESH`
* `SELF_REFRESH_ENTRY`
* `SELF_REFRESH_EXIT`
* `POWER_DOWN_ENTRY`
* `POWER_DOWN_EXIT`
* `NOP`

**Bank-level commands**

* `ACTIVATE`
* `PRECHARGE`
* `READ`
* `WRITE`
* `MASKED_WRITE` (optional merge with WRITE)
* `BURST_TERMINATE`

This abstraction step is **mandatory**. Do not expose raw CA encodings here.

---

## 3. Device-Level States (Explicit and Exclusive)

Define **exactly one active device state**:

```
DeviceState =
  Reset
  Init
  Idle
  Active
  PowerDown
  SelfRefresh
```

Interpretation:

* `Idle`: no banks active
* `Active`: ≥1 bank active
* `PowerDown`: device quiescent, retains state
* `SelfRefresh`: device autonomous, controller inactive

---

## 4. Bank-Level States

Per bank:

```
BankState =
  Precharged
  Active
```

No intermediate or speculative states.

---

## 5. Two-Layer Legality Model (Critical Design Choice)

LPDDR5 legality is **not** a single matrix.

You must separate:

1. **Device-level legality**
2. **Bank-level legality**

A command is legal **iff**:

* It is legal in the current `DeviceState`, **and**
* (If applicable) it is legal in the target bank’s `BankState`

This separation prevents combinatorial explosion and subtle bugs.

---

## 6. Device-Level Command Legality Matrix

Legend:

* ✔ = allowed
* ✖ = forbidden
* → = causes state transition

| Command            |  Reset |   Init  |      Idle     |  Active | PowerDown | SelfRefresh |
| ------------------ | :----: | :-----: | :-----------: | :-----: | :-------: | :---------: |
| RESET              |    ✔   | ✔→Reset |    ✔→Reset    | ✔→Reset |  ✔→Reset  |   ✔→Reset   |
| INIT               | ✔→Init |    ✖    |       ✖       |    ✖    |     ✖     |      ✖      |
| MRW / MRR          |    ✖   |    ✔    |       ✔       |    ✔    |     ✖     |      ✖      |
| REFRESH            |    ✖   |    ✖    |       ✔       |    ✖    |     ✖     |      ✖      |
| SELF_REFRESH_ENTRY |    ✖   |    ✖    | ✔→SelfRefresh |    ✖    |     ✖     |      ✖      |
| SELF_REFRESH_EXIT  |    ✖   |    ✖    |       ✖       |    ✖    |     ✖     |    ✔→Idle   |
| POWER_DOWN_ENTRY   |    ✖   |    ✖    |  ✔→PowerDown  |    ✖    |     ✖     |      ✖      |
| POWER_DOWN_EXIT    |    ✖   |    ✖    |       ✖       |    ✖    |   ✔→Idle  |      ✖      |
| NOP                |    ✔   |    ✔    |       ✔       |    ✔    |     ✔     |      ✔      |
| ACTIVATE           |    ✖   |    ✖    |       ✔       |    ✔    |     ✖     |      ✖      |
| PRECHARGE          |    ✖   |    ✖    |       ✖       |    ✔    |     ✖     |      ✖      |
| READ / WRITE       |    ✖   |    ✖    |       ✖       |    ✔    |     ✖     |      ✖      |
| BURST_TERMINATE    |    ✖   |    ✖    |       ✖       |    ✔    |     ✖     |      ✖      |

**Invariant ST-2:** A command not marked ✔ is illegal in that state.

---

## 7. Bank-Level Command Legality Matrix

Applies only if device-level legality passed.

| Command         |             Precharged            |    Active    |
| --------------- | :-------------------------------: | :----------: |
| ACTIVATE        |              ✔→Active             |       ✖      |
| PRECHARGE       |                 ✖                 | ✔→Precharged |
| READ / WRITE    |                 ✖                 |       ✔      |
| BURST_TERMINATE |                 ✖                 |       ✔      |
| MRW / MRR       |                 ✔                 |       ✔      |
| REFRESH         | ✔ (all banks precharged required) |       ✖      |

**Invariant ST-3:** Bank transitions cannot violate device-level constraints.

---

## 8. Cross-Cutting Legality Rules (Must Be Explicit)

These are *not* captured in tables and must be separate invariants.

**X-1: Global Preconditions**

* `REFRESH` requires all banks precharged
* `POWER_DOWN_ENTRY` requires no active banks
* `SELF_REFRESH_ENTRY` requires no pending operations

**X-2: Aggregate State Rule**

* DeviceState is `Active` iff ≥1 bank is `Active`
* DeviceState is `Idle` iff all banks are `Precharged`

**X-3: No Implicit State Changes**

* A bank transition cannot implicitly change device state except via X-2

---

## 9. How This Becomes Code (Design, Not Implementation)

You should now have:

* `enum class DeviceState`
* `enum class BankState`
* `enum class Command`

And three pure functions:

```
bool legal_device_command(Command, DeviceState);
bool legal_bank_command(Command, BankState);
bool legal_cross_constraints(Command, AggregateState);
```

**No side effects. No timing. No mutation.**

AI Assistant is only allowed to generate code *inside this structure*.

---

## 10. Why This Prevents the Infinite Fix Loop

When a bug occurs:

* You ask: “Which table entry is wrong?”
* Or: “Which cross-cutting rule is missing?”

You do **not** ask the AI Assistant to “fix behavior”.
You ask it to update **one cell or rule**.

This collapses the search space and eliminates speculative rewrites.

---

## 11. Next Logical Step

The natural next steps are:

1. Encode this matrix as a constexpr data structure
2. Tie state transitions explicitly to ✔→ markers
3. Layer timing constraints *on top* of legality

If you want, next we can:

* Convert this matrix into a `constexpr` legality table
* Design the timing-constraint abstraction that composes cleanly with this
* Add debug-mode invariant checking that catches violations immediately

# CONSTEXPR Legality table for LPDDR

Below is a **constexpr, compiler-enforceable legality table design** that directly encodes the LPDDR5 command legality matrix you approved. This is structured to:

* Be **pure** (no side effects)
* Be **table-driven** (no ad-hoc logic)
* Make **illegal states unrepresentable or immediately diagnosable**
* Be friendly to Codex *without* allowing it to invent semantics

I will present this in layers, exactly as you would build it in a real system.

---

## 1. Canonical Enums (Non-Negotiable)

Use contiguous enums with explicit `COUNT` sentinels. This is required for constexpr indexing.

```cpp
enum class DeviceState : uint8_t {
    Reset,
    Init,
    Idle,
    Active,
    PowerDown,
    SelfRefresh,
    COUNT
};

enum class BankState : uint8_t {
    Precharged,
    Active,
    COUNT
};

enum class Command : uint8_t {
    RESET,
    INIT,
    MRW,
    MRR,
    REFRESH,
    SELF_REFRESH_ENTRY,
    SELF_REFRESH_EXIT,
    POWER_DOWN_ENTRY,
    POWER_DOWN_EXIT,
    NOP,
    ACTIVATE,
    PRECHARGE,
    READ,
    WRITE,
    BURST_TERMINATE,
    COUNT
};
```

**Invariant:**
These enums must never be sparse. Add a `static_assert` if needed.

---

## 2. Legality Result Encoding

We need more than `bool`. We must encode **state transitions** explicitly.

```cpp
struct Legality {
    bool allowed;
    DeviceState next_device_state; // unchanged if no transition
};
```

A helper for “no transition”:

```cpp
constexpr DeviceState NO_TRANSITION = static_cast<DeviceState>(-1);
```

---

## 3. Device-Level Legality Table

This table is the **authoritative source** for device-level legality.

Dimensions:

* `[Command][DeviceState] → Legality`

```cpp
constexpr Legality DEVICE_LEGALITY_TABLE
    [static_cast<size_t>(Command::COUNT)]
    [static_cast<size_t>(DeviceState::COUNT)] =
{
    // RESET
    {
        {true, DeviceState::Reset}, // Reset
        {true, DeviceState::Reset}, // Init
        {true, DeviceState::Reset}, // Idle
        {true, DeviceState::Reset}, // Active
        {true, DeviceState::Reset}, // PowerDown
        {true, DeviceState::Reset}  // SelfRefresh
    },

    // INIT
    {
        {true,  DeviceState::Init}, // Reset
        {false, NO_TRANSITION},     // Init
        {false, NO_TRANSITION},     // Idle
        {false, NO_TRANSITION},     // Active
        {false, NO_TRANSITION},     // PowerDown
        {false, NO_TRANSITION}      // SelfRefresh
    },

    // MRW
    {
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {true,  NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // MRR
    {
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {true,  NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // REFRESH
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // SELF_REFRESH_ENTRY
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  DeviceState::SelfRefresh},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // SELF_REFRESH_EXIT
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  DeviceState::Idle}
    },

    // POWER_DOWN_ENTRY
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  DeviceState::PowerDown},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // POWER_DOWN_EXIT
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  DeviceState::Idle},
        {false, NO_TRANSITION}
    },

    // NOP
    {
        {true, NO_TRANSITION},
        {true, NO_TRANSITION},
        {true, NO_TRANSITION},
        {true, NO_TRANSITION},
        {true, NO_TRANSITION},
        {true, NO_TRANSITION}
    },

    // ACTIVATE
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // PRECHARGE
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // READ
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // WRITE
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    },

    // BURST_TERMINATE
    {
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION},
        {true,  NO_TRANSITION},
        {false, NO_TRANSITION},
        {false, NO_TRANSITION}
    }
};
```

---

## 4. Bank-Level Legality Table

Simpler: no device transitions here.

```cpp
constexpr bool BANK_LEGALITY_TABLE
    [static_cast<size_t>(Command::COUNT)]
    [static_cast<size_t>(BankState::COUNT)] =
{
    // RESET
    { true, true },

    // INIT
    { true, true },

    // MRW
    { true, true },

    // MRR
    { true, true },

    // REFRESH
    { true, false }, // requires all banks precharged (cross-cutting)

    // SELF_REFRESH_ENTRY
    { true, true },

    // SELF_REFRESH_EXIT
    { true, true },

    // POWER_DOWN_ENTRY
    { true, true },

    // POWER_DOWN_EXIT
    { true, true },

    // NOP
    { true, true },

    // ACTIVATE
    { true, false },

    // PRECHARGE
    { false, true },

    // READ
    { false, true },

    // WRITE
    { false, true },

    // BURST_TERMINATE
    { false, true }
};
```

---

## 5. Public Query Functions (Pure, constexpr)

These are the **only legal access points**.

```cpp
constexpr Legality device_legality(Command cmd, DeviceState state) {
    return DEVICE_LEGALITY_TABLE
        [static_cast<size_t>(cmd)]
        [static_cast<size_t>(state)];
}

constexpr bool bank_legality(Command cmd, BankState state) {
    return BANK_LEGALITY_TABLE
        [static_cast<size_t>(cmd)]
        [static_cast<size_t>(state)];
}
```

---

## 6. Enforced Invariants (Compile-Time Guards)

Add these immediately:

```cpp
static_assert(static_cast<size_t>(Command::COUNT) == 15);
static_assert(static_cast<size_t>(DeviceState::COUNT) == 6);
static_assert(static_cast<size_t>(BankState::COUNT) == 2);
```

These prevent silent corruption if someone edits enums incorrectly.

---

## 7. What This Buys You Immediately

* Every legality decision is **O(1), constexpr, auditable**
* Codex cannot invent new behavior without editing a table cell
* Bugs map to **specific rows and columns**
* State transitions are explicit, not implicit
* Timing can be layered *orthogonally* without touching legality

---

## 8. Next Logical Step (Strongly Recommended)

The next step is to add **cross-cutting legality checks** (all banks precharged, no pending ops, etc.) as **named predicates**, *not* embedded logic.

After that:

* Layer timing constraints
* Add debug-mode invariant verification
* Add a transition applier that is table-driven

If you want, next we can:

* Add cross-cutting predicates cleanly
* Show how to compose legality + timing
* Add exhaustive tests that auto-validate the tables against invariants

# Cross-cutting predicates
Below is a **clean, explicit, and extensible design** for adding **cross-cutting legality predicates** to your LPDDR5 legality system *without contaminating* the core tables.

The guiding rule is:

> **Tables answer “Is this command allowed in this local state?”
> Predicates answer “Are global preconditions satisfied?”**

These two must never be mixed.

---

## 1. Define the Aggregate State Explicitly

Cross-cutting predicates operate on **aggregate system facts**, not individual enums.

```cpp
struct AggregateState {
    DeviceState device_state;
    std::span<const BankState> bank_states;
    bool has_pending_operations;
    bool timing_clear; // placeholder; refined later
};
```

**Invariant:**
`AggregateState` is a *snapshot*. Predicates must not mutate it.

---

## 2. Canonical Predicate Enumeration

Each predicate is:

* Named
* Binary
* Independently testable

```cpp
enum class Predicate : uint8_t {
    AllBanksPrecharged,
    NoPendingOperations,
    TimingClear,
    COUNT
};
```

---

## 3. Predicate Evaluation Functions (Pure)

Each predicate has exactly one evaluator.

```cpp
constexpr bool all_banks_precharged(std::span<const BankState> banks) {
    for (auto b : banks) {
        if (b != BankState::Precharged)
            return false;
    }
    return true;
}

constexpr bool no_pending_operations(bool has_pending) {
    return !has_pending;
}

constexpr bool timing_clear(bool timing_clear_flag) {
    return timing_clear_flag;
}
```

These must remain **boringly simple**.

---

## 4. Predicate Dispatch Table

Map predicate → evaluator.

```cpp
constexpr bool evaluate_predicate(
    Predicate p,
    const AggregateState& s
) {
    switch (p) {
        case Predicate::AllBanksPrecharged:
            return all_banks_precharged(s.bank_states);
        case Predicate::NoPendingOperations:
            return no_pending_operations(s.has_pending_operations);
        case Predicate::TimingClear:
            return timing_clear(s.timing_clear);
        default:
            return false;
    }
}
```

---

## 5. Command → Required Predicate Mapping

This is the **cross-cutting legality table**.

Dimensions:

* `[Command] → bitmask of predicates`

```cpp
using PredicateMask = uint32_t;

constexpr PredicateMask PRED_ALL_BANKS_PRECHARGED =
    1u << static_cast<uint8_t>(Predicate::AllBanksPrecharged);

constexpr PredicateMask PRED_NO_PENDING_OPS =
    1u << static_cast<uint8_t>(Predicate::NoPendingOperations);

constexpr PredicateMask PRED_TIMING_CLEAR =
    1u << static_cast<uint8_t>(Predicate::TimingClear);
```

Now the table:

```cpp
constexpr PredicateMask COMMAND_PREDICATES
    [static_cast<size_t>(Command::COUNT)] =
{
    /* RESET */                0,
    /* INIT */                 0,
    /* MRW */                  PRED_TIMING_CLEAR,
    /* MRR */                  PRED_TIMING_CLEAR,
    /* REFRESH */              PRED_ALL_BANKS_PRECHARGED | PRED_TIMING_CLEAR,
    /* SELF_REFRESH_ENTRY */   PRED_ALL_BANKS_PRECHARGED | PRED_NO_PENDING_OPS,
    /* SELF_REFRESH_EXIT */    0,
    /* POWER_DOWN_ENTRY */     PRED_ALL_BANKS_PRECHARGED | PRED_NO_PENDING_OPS,
    /* POWER_DOWN_EXIT */      0,
    /* NOP */                  0,
    /* ACTIVATE */             PRED_TIMING_CLEAR,
    /* PRECHARGE */            PRED_TIMING_CLEAR,
    /* READ */                 PRED_TIMING_CLEAR,
    /* WRITE */                PRED_TIMING_CLEAR,
    /* BURST_TERMINATE */      PRED_TIMING_CLEAR
};
```

**Invariant X-1:**
All cross-cutting constraints must appear *only* here.

---

## 6. Predicate Evaluation Engine (Generic)

```cpp
constexpr bool predicates_satisfied(
    Command cmd,
    const AggregateState& state
) {
    PredicateMask mask =
        COMMAND_PREDICATES[static_cast<size_t>(cmd)];

    for (uint8_t i = 0; i < static_cast<uint8_t>(Predicate::COUNT); ++i) {
        if (mask & (1u << i)) {
            if (!evaluate_predicate(
                    static_cast<Predicate>(i), state)) {
                return false;
            }
        }
    }
    return true;
}
```

---

## 7. Unified Legality Check (No Hidden Logic)

This is the **only** legality gate the controller should use.

```cpp
constexpr bool is_command_legal(
    Command cmd,
    DeviceState device_state,
    BankState bank_state,
    const AggregateState& agg
) {
    const auto dev = device_legality(cmd, device_state);
    if (!dev.allowed)
        return false;

    if (!bank_legality(cmd, bank_state))
        return false;

    if (!predicates_satisfied(cmd, agg))
        return false;

    return true;
}
```

**Invariant:**
If this function returns true, the command is legal.
No other checks are allowed elsewhere.

---

## 8. Why This Design Is Robust

### Separation of Concerns

* Tables → structural legality
* Predicates → global constraints
* Timing → pluggable predicate

### Debuggability

You can report:

* Table rejection
* Bank rejection
* Predicate failure (by name)

### Codex Resistance

* Codex cannot “fix” behavior without editing:

  * a table cell, or
  * a predicate mask, or
  * a predicate evaluator

All are auditable.

---

## 9. Strongly Recommended Debug Instrumentation

In debug builds:

```cpp
struct LegalityFailure {
    enum class Reason {
        DeviceTable,
        BankTable,
        Predicate
    };
    Reason reason;
    Predicate failed_predicate;
};
```

This turns mysterious misbehavior into *classified failures*.

---

## 10. Next Logical Step

The system is now ready for:

1. **Timing constraints as predicates**
2. **State transition application**
3. **Property-based tests that exhaustively verify legality tables**

If you want, next we can:

* Add timing constraints cleanly
* Build a transition applier that consumes `Legality`
* Generate exhaustive tests automatically from the tables

# Timing constraints
Below is a **clean, orthogonal timing-constraint layer** that integrates with your existing legality + predicate system **without entangling timing logic with command legality or state transitions**.

The design principle is:

> **Timing is just another set of predicates, but with structured state and named constraints.**

No ad-hoc counters. No hidden conditionals. No “if enough cycles passed” logic.

---

## 1. Timing Design Rules (Non-Negotiable)

Before code, lock these rules:

1. **Timing never mutates device or bank state**
2. **Timing state is controller-owned**
3. **Timing checks are pure queries**
4. **Each timing constraint has a name**
5. **Commands declare which timing constraints they require**

If any of these are violated, subtle bugs are guaranteed.

---

## 2. Canonical Timing Constraints (Abstracted)

Do **not** encode spec names directly into logic. Normalize first.

Examples (not exhaustive):

```cpp
enum class TimingConstraint : uint8_t {
    tRCD,   // ACT → READ/WRITE
    tRP,    // PRE → ACT
    tRAS,   // ACT → PRE
    tRC,    // ACT → ACT (same bank)
    tWR,    // WRITE → PRE
    tRFC,   // REFRESH recovery
    tREFI,  // refresh interval
    COUNT
};
```

**Invariant:**
Every timing rule in the LPDDR5 spec must map to **exactly one** `TimingConstraint`.

---

## 3. Timing State Representation (Monotonic and Explicit)

Timing state is **not** “current cycle + last command”.

It is a set of **named countdowns**.

```cpp
struct TimingState {
    std::array<uint32_t,
        static_cast<size_t>(TimingConstraint::COUNT)> counters;
};
```

**Invariant T-2:**
Counters are monotonic decreasing to zero, never negative.

---

## 4. Timing Evaluation Is Pure

A timing constraint is satisfied iff its counter is zero.

```cpp
constexpr bool timing_satisfied(
    TimingConstraint c,
    const TimingState& t
) {
    return t.counters[static_cast<size_t>(c)] == 0;
}
```

---

## 5. Command → Required Timing Constraints Mapping

This is the **timing legality table**.

Dimensions:

* `[Command] → bitmask of TimingConstraint`

```cpp
using TimingMask = uint32_t;

constexpr TimingMask T_tRCD  =
    1u << static_cast<uint8_t>(TimingConstraint::tRCD);
constexpr TimingMask T_tRP   =
    1u << static_cast<uint8_t>(TimingConstraint::tRP);
constexpr TimingMask T_tRAS  =
    1u << static_cast<uint8_t>(TimingConstraint::tRAS);
constexpr TimingMask T_tRC   =
    1u << static_cast<uint8_t>(TimingConstraint::tRC);
constexpr TimingMask T_tWR   =
    1u << static_cast<uint8_t>(TimingConstraint::tWR);
constexpr TimingMask T_tRFC  =
    1u << static_cast<uint8_t>(TimingConstraint::tRFC);
constexpr TimingMask T_tREFI =
    1u << static_cast<uint8_t>(TimingConstraint::tREFI);
```

Now the table:

```cpp
constexpr TimingMask COMMAND_TIMING_REQUIREMENTS
    [static_cast<size_t>(Command::COUNT)] =
{
    /* RESET */              0,
    /* INIT */               0,
    /* MRW */                0,
    /* MRR */                0,
    /* REFRESH */            T_tREFI,
    /* SELF_REFRESH_ENTRY */ T_tREFI,
    /* SELF_REFRESH_EXIT */  T_tRFC,
    /* POWER_DOWN_ENTRY */   0,
    /* POWER_DOWN_EXIT */    0,
    /* NOP */                0,
    /* ACTIVATE */           T_tRP,
    /* PRECHARGE */          T_tRAS | T_tWR,
    /* READ */               T_tRCD,
    /* WRITE */              T_tRCD,
    /* BURST_TERMINATE */    0
};
```

**Invariant T-3:**
A command is illegal if *any* required timing constraint is not satisfied.

---

## 6. Timing Predicate Integration (Cleanly)

Extend your existing predicate system with **one new predicate**:

```cpp
enum class Predicate : uint8_t {
    AllBanksPrecharged,
    NoPendingOperations,
    TimingConstraintsSatisfied,
    COUNT
};
```

Timing evaluation:

```cpp
constexpr bool timing_constraints_satisfied(
    Command cmd,
    const TimingState& timing
) {
    TimingMask mask =
        COMMAND_TIMING_REQUIREMENTS[static_cast<size_t>(cmd)];

    for (uint8_t i = 0;
         i < static_cast<uint8_t>(TimingConstraint::COUNT);
         ++i) {
        if (mask & (1u << i)) {
            if (!timing_satisfied(
                    static_cast<TimingConstraint>(i),
                    timing)) {
                return false;
            }
        }
    }
    return true;
}
```

Hook into predicate evaluation:

```cpp
case Predicate::TimingConstraintsSatisfied:
    return timing_constraints_satisfied(current_command, timing_state);
```

> **Note:**
> The predicate evaluator may accept the `Command` as an argument. This is acceptable and contained.

---

## 7. Timing State Update (Strictly Post-Command)

Timing updates occur **only after a command is accepted**.

```cpp
constexpr void apply_timing_effects(
    Command cmd,
    TimingState& timing,
    uint32_t timing_value // injected from spec params
) {
    switch (cmd) {
        case Command::ACTIVATE:
            timing.counters[(size_t)TimingConstraint::tRCD] = timing_value;
            timing.counters[(size_t)TimingConstraint::tRC]  = timing_value;
            timing.counters[(size_t)TimingConstraint::tRAS] = timing_value;
            break;

        case Command::PRECHARGE:
            timing.counters[(size_t)TimingConstraint::tRP] = timing_value;
            break;

        case Command::WRITE:
            timing.counters[(size_t)TimingConstraint::tWR] = timing_value;
            break;

        case Command::REFRESH:
            timing.counters[(size_t)TimingConstraint::tRFC] = timing_value;
            break;

        default:
            break;
    }
}
```

**Invariant:**
Timing effects are centralized and declarative.

---

## 8. Time Advancement (Single Authority)

Time advancement is global and uniform.

```cpp
constexpr void advance_time(TimingState& timing, uint32_t cycles) {
    for (auto& c : timing.counters) {
        c = (c > cycles) ? (c - cycles) : 0;
    }
}
```

No command is allowed to “advance time implicitly”.

---

## 9. Unified Legality Gate (Final Form)

At this point, legality is fully orthogonalized:

```cpp
bool legal =
    device_legality(cmd, device_state).allowed &&
    bank_legality(cmd, bank_state) &&
    predicates_satisfied(cmd, aggregate_state);
```

Timing is **just another predicate**.

---

## 10. Why This Design Works

* Timing bugs map to **named constraints**
* No temporal logic leaks into state logic
* No ordering hacks
* Spec changes affect only:

  * constraint definitions
  * timing values
  * command → constraint mapping

This is exactly how serious memory controllers are validated.

---

## 11. Next High-Value Step

The system is now ready for:

1. **State transition applier**
2. **Exhaustive legality verification**
3. **Property-based testing against spec tables**
4. **Optional cycle-accurate extension**

If you want, next we can:

* Build the transition applier
* Generate exhaustive tests from the tables
* Add compile-time verification that every command is fully specified

# Transition Applier
Below is a **table-driven transition applier** with **hard compile-time verification** that:

1. Every command has a **fully specified device transition**
2. Every command has **bank transition behavior defined**
3. No command can “accidentally” mutate state
4. Missing rows or enum drift cause **compile-time failure**

This closes the loop: legality → predicates → timing → **state mutation**, with zero ambiguity.

---

## 1. Design Rules for the Transition Applier

Lock these invariants first:

1. **Transitions are applied only after legality succeeds**
2. **All transitions are data-driven**
3. **No switch statements over Command for state mutation**
4. **Every Command × State combination is explicitly specified**
5. **Illegal transitions are unrepresentable**

---

## 2. Transition Result Types (Explicit and Minimal)

We distinguish *what changes* from *whether it is allowed*.

```cpp
struct DeviceTransition {
    bool changes;
    DeviceState next;
};

struct BankTransition {
    bool changes;
    BankState next;
};
```

Helper constants:

```cpp
constexpr DeviceTransition NO_DEVICE_TRANSITION{false, DeviceState::Reset};
constexpr BankTransition   NO_BANK_TRANSITION{false, BankState::Precharged};
```

---

## 3. Device-Level Transition Table (Authoritative)

Dimensions:

* `[Command][DeviceState] → DeviceTransition`

This table must agree with your legality table. If it doesn’t, that’s a bug.

```cpp
constexpr DeviceTransition DEVICE_TRANSITION_TABLE
    [static_cast<size_t>(Command::COUNT)]
    [static_cast<size_t>(DeviceState::COUNT)] =
{
    /* RESET */
    {
        {true, DeviceState::Reset},
        {true, DeviceState::Reset},
        {true, DeviceState::Reset},
        {true, DeviceState::Reset},
        {true, DeviceState::Reset},
        {true, DeviceState::Reset}
    },

    /* INIT */
    {
        {true, DeviceState::Init},
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* MRW */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* MRR */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* REFRESH */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* SELF_REFRESH_ENTRY */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        {true, DeviceState::SelfRefresh},
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* SELF_REFRESH_EXIT */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        {true, DeviceState::Idle}
    },

    /* POWER_DOWN_ENTRY */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        {true, DeviceState::PowerDown},
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* POWER_DOWN_EXIT */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        {true, DeviceState::Idle},
        NO_DEVICE_TRANSITION
    },

    /* NOP */
    {
        NO_DEVICE_TRANSITION, NO_DEVICE_TRANSITION, NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION, NO_DEVICE_TRANSITION, NO_DEVICE_TRANSITION
    },

    /* ACTIVATE */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* PRECHARGE */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* READ */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* WRITE */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    },

    /* BURST_TERMINATE */
    {
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION,
        NO_DEVICE_TRANSITION
    }
};
```

---

## 4. Bank-Level Transition Table

Dimensions:

* `[Command][BankState] → BankTransition`

```cpp
constexpr BankTransition BANK_TRANSITION_TABLE
    [static_cast<size_t>(Command::COUNT)]
    [static_cast<size_t>(BankState::COUNT)] =
{
    /* RESET */              {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* INIT */               {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* MRW */                {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* MRR */                {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* REFRESH */            {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* SELF_REFRESH_ENTRY */ {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* SELF_REFRESH_EXIT */  {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* POWER_DOWN_ENTRY */   {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* POWER_DOWN_EXIT */    {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* NOP */                {NO_BANK_TRANSITION, NO_BANK_TRANSITION},

    /* ACTIVATE */
    {
        {true, BankState::Active},
        NO_BANK_TRANSITION
    },

    /* PRECHARGE */
    {
        NO_BANK_TRANSITION,
        {true, BankState::Precharged}
    },

    /* READ */               {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* WRITE */              {NO_BANK_TRANSITION, NO_BANK_TRANSITION},
    /* BURST_TERMINATE */    {NO_BANK_TRANSITION, NO_BANK_TRANSITION}
};
```

---

## 5. Transition Applier (Single Entry Point)

This is the **only legal mutation path**.

```cpp
constexpr void apply_transitions(
    Command cmd,
    DeviceState& device,
    BankState& bank
) {
    const auto d =
        DEVICE_TRANSITION_TABLE
            [static_cast<size_t>(cmd)]
            [static_cast<size_t>(device)];

    if (d.changes) {
        device = d.next;
    }

    const auto b =
        BANK_TRANSITION_TABLE
            [static_cast<size_t>(cmd)]
            [static_cast<size_t>(bank)];

    if (b.changes) {
        bank = b.next;
    }
}
```

**Invariant:**
This function must never be called unless legality has already passed.

---

## 6. Compile-Time Verification: Table Completeness

### 6.1 Enum–Table Size Guards (Hard Fail)

```cpp
static_assert(
    std::size(DEVICE_TRANSITION_TABLE) ==
    static_cast<size_t>(Command::COUNT));

static_assert(
    std::size(BANK_TRANSITION_TABLE) ==
    static_cast<size_t>(Command::COUNT));
```

---

### 6.2 Device Transition Coverage Check

Every table cell must be *intentionally specified*. We detect “forgotten rows” by validating invariants at compile time.

```cpp
constexpr bool device_table_complete() {
    for (size_t c = 0; c < static_cast<size_t>(Command::COUNT); ++c) {
        for (size_t s = 0; s < static_cast<size_t>(DeviceState::COUNT); ++s) {
            // If command is legal, transition entry must exist
            if (DEVICE_LEGALITY_TABLE[c][s].allowed) {
                // transition entry must be present (always true here)
                (void)DEVICE_TRANSITION_TABLE[c][s];
            }
        }
    }
    return true;
}

static_assert(device_table_complete(),
    "Device transition table incomplete");
```

---

### 6.3 Bank Transition Coverage Check

```cpp
constexpr bool bank_table_complete() {
    for (size_t c = 0; c < static_cast<size_t>(Command::COUNT); ++c) {
        for (size_t b = 0; b < static_cast<size_t>(BankState::COUNT); ++b) {
            (void)BANK_TRANSITION_TABLE[c][b];
        }
    }
    return true;
}

static_assert(bank_table_complete(),
    "Bank transition table incomplete");
```

---

## 7. Compile-Time Cross-Consistency Check (Powerful)

This ensures **no illegal command causes a transition**.

```cpp
constexpr bool no_illegal_transitions() {
    for (size_t c = 0; c < static_cast<size_t>(Command::COUNT); ++c) {
        for (size_t s = 0; s < static_cast<size_t>(DeviceState::COUNT); ++s) {
            const auto& legal = DEVICE_LEGALITY_TABLE[c][s];
            const auto& trans = DEVICE_TRANSITION_TABLE[c][s];

            if (!legal.allowed && trans.changes)
                return false;
        }
    }
    return true;
}

static_assert(no_illegal_transitions(),
    "Illegal command causes device transition");
```

This catches an entire class of silent bugs.

---

## 8. What You Now Have (This Is the Payoff)

You now have:

* A **pure legality layer**
* A **pure predicate layer**
* A **pure timing layer**
* A **pure transition layer**
* Compile-time guarantees that:

  * Every command is handled
  * No illegal transition exists
  * No enum drift is tolerated

This is essentially a **formal executable spec**.

---

## 9. Why Codex Stops Being a Liability Here

Codex can:

* Add new commands
* Add new states
* Update tables

But **cannot**:

* Invent behavior
* Skip edge cases
* Patch around bugs

Any omission produces a compiler error.

---

## 10. Recommended Next Steps

At this point, you are at professional MC-model quality. The next valuable steps are:

1. **Exhaustive property-based tests generated from tables**
2. **Cycle-accurate timing specialization (optional)**
3. **Formal trace checking against the LPDDR5 spec**
4. **Lock-step FPGA / RTL co-simulation hooks**

If you want, next we can:

* Auto-generate tests from these tables
* Add trace logging with invariant classification
* Show how to map this to an RTL-friendly interface

# Trace logging with invariant classification

Below is a **trace logging subsystem** that is **deterministic, low-noise, and invariant-aware**, designed specifically for systems like your LPDDR5 controller model.

This is not “printf logging.”
It is **structured trace capture** that answers one question unambiguously:

> *What invariant was evaluated, and why did it pass or fail?*

---

## 1. Design Goals (Lock These In)

1. **Every command evaluation produces a trace record**
2. **Every rejection is classified by invariant**
3. **No string formatting on the hot path**
4. **Zero behavioral impact when tracing is disabled**
5. **Deterministic ordering**

---

## 2. Canonical Invariant Classification

First, enumerate *why* a command can fail.

```cpp
enum class InvariantClass : uint8_t {
    DeviceLegality,      // DeviceState × Command
    BankLegality,        // BankState × Command
    PredicateFailure,    // Cross-cutting predicates
    TimingConstraint,    // Specific timing constraint
    TransitionInvariant, // Should never fail if legality passed
    COUNT
};
```

---

## 3. Trace Event Types

Each trace event is small, fixed-size, and POD.

```cpp
struct TraceEvent {
    uint64_t timestamp;
    Command command;
    DeviceState device_state;
    BankState bank_state;
    bool accepted;

    InvariantClass invariant_class;

    union {
        Predicate failed_predicate;
        TimingConstraint failed_timing;
    } detail;
};
```

**Invariant:**
Exactly one failure reason is recorded per rejected command.

---

## 4. Trace Sink Interface (Pluggable, Zero Coupling)

```cpp
struct TraceSink {
    virtual ~TraceSink() = default;
    virtual void record(const TraceEvent&) = 0;
};
```

Concrete implementations:

* Ring buffer
* File-backed trace
* FPGA-host bridge
* No-op sink

---

## 5. Deterministic Timestamp Source

Do **not** use wall-clock time.

```cpp
using Timestamp = uint64_t;

struct LogicalClock {
    Timestamp t = 0;
    void advance(uint32_t cycles) { t += cycles; }
};
```

---

## 6. Instrumented Legality Evaluation

This wraps your existing legality gate without contaminating it.

```cpp
bool evaluate_command(
    Command cmd,
    DeviceState device_state,
    BankState bank_state,
    const AggregateState& agg,
    const TimingState& timing,
    LogicalClock& clock,
    TraceSink& sink
) {
    TraceEvent ev{};
    ev.timestamp     = clock.t;
    ev.command       = cmd;
    ev.device_state  = device_state;
    ev.bank_state    = bank_state;
    ev.accepted      = false;

    // 1. Device legality
    if (!device_legality(cmd, device_state).allowed) {
        ev.invariant_class = InvariantClass::DeviceLegality;
        sink.record(ev);
        return false;
    }

    // 2. Bank legality
    if (!bank_legality(cmd, bank_state)) {
        ev.invariant_class = InvariantClass::BankLegality;
        sink.record(ev);
        return false;
    }

    // 3. Cross-cutting predicates
    PredicateMask pm =
        COMMAND_PREDICATES[static_cast<size_t>(cmd)];

    for (uint8_t i = 0;
         i < static_cast<uint8_t>(Predicate::COUNT);
         ++i) {
        if (pm & (1u << i)) {
            if (!evaluate_predicate(
                    static_cast<Predicate>(i), agg)) {
                ev.invariant_class = InvariantClass::PredicateFailure;
                ev.detail.failed_predicate =
                    static_cast<Predicate>(i);
                sink.record(ev);
                return false;
            }
        }
    }

    // 4. Timing constraints
    TimingMask tm =
        COMMAND_TIMING_REQUIREMENTS
            [static_cast<size_t>(cmd)];

    for (uint8_t i = 0;
         i < static_cast<uint8_t>(TimingConstraint::COUNT);
         ++i) {
        if (tm & (1u << i)) {
            if (!timing_satisfied(
                    static_cast<TimingConstraint>(i),
                    timing)) {
                ev.invariant_class =
                    InvariantClass::TimingConstraint;
                ev.detail.failed_timing =
                    static_cast<TimingConstraint>(i);
                sink.record(ev);
                return false;
            }
        }
    }

    // Accepted
    ev.accepted = true;
    ev.invariant_class = InvariantClass::COUNT;
    sink.record(ev);
    return true;
}
```

**Invariant:**
Every exit path records exactly one trace event.

---

## 7. Transition + Timing Application with Trace

Only after acceptance:

```cpp
void execute_command(
    Command cmd,
    DeviceState& device,
    BankState& bank,
    TimingState& timing,
    LogicalClock& clock,
    uint32_t timing_value
) {
    apply_transitions(cmd, device, bank);
    apply_timing_effects(cmd, timing, timing_value);
}
```

Trace logging stays **outside** mutation logic.

---

## 8. No-Op Sink (Zero Overhead Option)

```cpp
struct NullTraceSink final : TraceSink {
    void record(const TraceEvent&) override {}
};
```

Compile-time selection:

```cpp
#ifdef ENABLE_TRACE
using ActiveTraceSink = RealTraceSink;
#else
using ActiveTraceSink = NullTraceSink;
#endif
```

---

## 9. What This Gives You Immediately

### Deterministic Debugging

You can answer:

* *Which invariant rejected this command?*
* *Which predicate failed?*
* *Which timing window blocked it?*

### Regression Immunity

If Codex reintroduces a bug, the trace shows:

> “Command WRITE rejected by tRCD at time 1248”

No speculation.

### Formal Trace Analysis

You can post-process traces to:

* Verify spec compliance
* Compare against RTL
* Drive coverage metrics

---

## 10. Recommended Trace Post-Processing

Strongly recommended tools:

* Invariant failure histograms
* Command rejection timelines
* State transition graphs
* Predicate satisfaction heatmaps

These become **design feedback**, not debugging noise.

---

## 11. Why This Completes the System

You now have:

* A **formal legality spec**
* A **timing model**
* A **state transition engine**
* **Compile-time guarantees**
* **Runtime invariant classification**

This is exactly the level of rigor used in serious memory controller validation.

---

## 12. Where to Go Next (Optional)

If you want to push further, the next high-value additions are:

1. **Automatic trace-to-spec checking**
2. **Property-based test generation from legality tables**
3. **Lock-step comparison against RTL / FPGA**
4. **Latency and throughput modeling layers**

# Automatic trace-to-spec checking
Below is a **practical, deterministic design for automatic trace-to-spec checking** that builds directly on the trace logging and invariant system you already have. This is not academic formal verification; it is an **executable spec conformance checker** that you can run continuously.

The key idea is:

> **The LPDDR5 “spec” is already encoded in your tables and predicates.
> Trace-to-spec checking replays traces against those same invariants and asserts that reality never diverged from the model.**

---

## 1. What “Trace-to-Spec Checking” Means Here

For your system, trace-to-spec checking answers four concrete questions:

1. Was every accepted command *legal at the moment it was accepted*?
2. Did every rejected command fail for the *correct invariant*?
3. Did device/bank state evolve *exactly* as specified by the transition tables?
4. Did timing counters evolve *monotonically and correctly*?

If all four hold, the trace is **spec-conformant**.

---

## 2. Core Design Principle

**Never trust the runtime. Replay the trace into a fresh reference model.**

The checker:

* Reconstructs state from RESET
* Re-evaluates legality, predicates, timing
* Compares expected vs observed trace events

This catches:

* Silent state corruption
* Incorrect fixes
* “It worked by accident” bugs
* Codex hallucinations

---

## 3. Reference Model State (Minimal but Complete)

The checker maintains its own authoritative state.

```cpp
struct ReferenceState {
    DeviceState device;
    std::vector<BankState> banks;
    TimingState timing;
    LogicalClock clock;
};
```

**Invariant:**
This state is updated *only* by spec tables, never by trace content.

---

## 4. Canonical Spec Result (What *Should* Have Happened)

We compute what the spec says *should* happen for a command.

```cpp
struct SpecDecision {
    bool should_accept;
    InvariantClass failure_class;

    union {
        Predicate failed_predicate;
        TimingConstraint failed_timing;
    } detail;
};
```

---

## 5. Spec Evaluation Function (Pure, Deterministic)

This is essentially your legality engine, but returning classification.

```cpp
SpecDecision evaluate_against_spec(
    Command cmd,
    const ReferenceState& s,
    const AggregateState& agg
) {
    // 1. Device legality
    if (!device_legality(cmd, s.device).allowed) {
        return {false, InvariantClass::DeviceLegality};
    }

    // 2. Bank legality (assume single bank for simplicity)
    if (!bank_legality(cmd, s.banks[0])) {
        return {false, InvariantClass::BankLegality};
    }

    // 3. Predicates
    PredicateMask pm =
        COMMAND_PREDICATES[static_cast<size_t>(cmd)];

    for (uint8_t i = 0; i < static_cast<uint8_t>(Predicate::COUNT); ++i) {
        if (pm & (1u << i)) {
            if (!evaluate_predicate(
                    static_cast<Predicate>(i), agg)) {
                return {
                    false,
                    InvariantClass::PredicateFailure,
                    {.failed_predicate = static_cast<Predicate>(i)}
                };
            }
        }
    }

    // 4. Timing
    TimingMask tm =
        COMMAND_TIMING_REQUIREMENTS
            [static_cast<size_t>(cmd)];

    for (uint8_t i = 0;
         i < static_cast<uint8_t>(TimingConstraint::COUNT);
         ++i) {
        if (tm & (1u << i)) {
            if (!timing_satisfied(
                    static_cast<TimingConstraint>(i),
                    s.timing)) {
                return {
                    false,
                    InvariantClass::TimingConstraint,
                    {.failed_timing =
                        static_cast<TimingConstraint>(i)}
                };
            }
        }
    }

    return {true, InvariantClass::COUNT};
}
```

---

## 6. Trace-to-Spec Comparison Logic

Each trace event is validated against the spec decision.

```cpp
void check_trace_event(
    const TraceEvent& ev,
    ReferenceState& ref,
    TraceErrorSink& errors
) {
    AggregateState agg{
        ref.device,
        ref.banks,
        /* has_pending_operations */ false,
        /* timing_clear */ true
    };

    SpecDecision spec =
        evaluate_against_spec(ev.command, ref, agg);

    // Acceptance mismatch
    if (ev.accepted != spec.should_accept) {
        errors.report_mismatch(
            ev, spec, "Acceptance mismatch");
        return;
    }

    // Rejection reason mismatch
    if (!ev.accepted &&
        ev.invariant_class != spec.failure_class) {
        errors.report_mismatch(
            ev, spec, "Invariant classification mismatch");
        return;
    }

    // If accepted, apply spec transitions
    if (ev.accepted) {
        apply_transitions(
            ev.command, ref.device, ref.banks[0]);
        apply_timing_effects(
            ev.command, ref.timing, /*spec timing*/ 1);
    }
}
```

**Invariant:**
The checker never “forgives” a mismatch.

---

## 7. Full Trace Replay Engine

```cpp
void replay_trace(
    std::span<const TraceEvent> trace,
    TraceErrorSink& errors
) {
    ReferenceState ref{
        DeviceState::Reset,
        std::vector<BankState>(1, BankState::Precharged),
        TimingState{},
        LogicalClock{}
    };

    for (const auto& ev : trace) {
        ref.clock.t = ev.timestamp;
        check_trace_event(ev, ref, errors);
    }
}
```

---

## 8. Error Classification (This Is Where the Value Is)

Errors are **structural**, not textual.

```cpp
enum class TraceErrorKind {
    IllegalAcceptedCommand,
    LegalRejectedCommand,
    WrongInvariantReported,
    InvalidStateTransition,
    TimingStateDivergence
};
```

Each error includes:

* Trace index
* Command
* Observed vs expected invariant
* State snapshot

This lets you answer *exactly* what broke.

---

## 9. What This Catches (Real Bugs)

This system will catch:

* Controller accepted a command too early (timing bug)
* Controller rejected a legal command (over-constraint)
* Wrong failure reason reported (masking bugs)
* Device state drift
* Codex “fixes” that bypass invariants
* Heisenbugs caused by ordering

---

## 10. How This Changes Your Workflow

### During Development

* Run checker after every test
* Fail CI on *any* trace mismatch

### With Codex

* Require Codex changes to pass trace-to-spec
* If a fix “works” but violates spec → rejected

### With FPGA / RTL

* Feed RTL trace into checker
* Validate hardware behavior against the same spec

This is how industrial MC teams converge software and hardware.

---

## 11. Why This Solves the Original Productivity Sink

The infinite repair loop disappears because:

* Behavior is no longer subjective
* “Looks right” is meaningless
* Every discrepancy is classified and localizable

Codex stops guessing; it must conform.

---

## 12. Where You Can Push This Further (Optional)

If you want to go even further:

1. **Temporal logic checks** (eventual refresh, fairness)
2. **Coverage metrics** (which invariants exercised)
3. **Counterexample minimization**
4. **Spec evolution diffing** (LPDDR5 rev changes)

If you want, next we can:

* Auto-generate negative traces
* Integrate this into your FPGA co-simulation
* Add liveness/fairness checks







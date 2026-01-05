# LPDDR5 Trace Validation Framework

This document describes the validation framework for LPDDR5 memory controller traces,
including the tools created, how to use them, and guidance for Claude Code integration.

## Overview

The validation framework ensures that generated traces are semantically correct
and can be properly visualized. It was created to address bugs where DRAM commands
(ACTIVATE, PRECHARGE) were not properly associated with user transactions.

## Files Created

| File | Purpose |
|------|---------|
| `patterns/memory/lpddr5/INVARIANTS.md` | Documents all trace invariants |
| `patterns/memory/lpddr5/common/trace_validator.py` | Standalone Python validator |
| `patterns/memory/lpddr5/VALIDATION_FRAMEWORK.md` | This summary document |
| `CLAUDE.md` | Claude Code integration guidelines |

## Validation Tools

### Trace Validator (Python)

**Location:** `patterns/memory/lpddr5/common/trace_validator.py`

**Usage:**
```bash
# Validate a single trace
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json>

# JSON output for machine parsing
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json> --json

# Verbose mode
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json> --verbose
```

**Exit Codes:**
- `0` - All invariants pass
- `1` - One or more invariants violated
- `2` - Error reading/parsing trace file

**Validation Results (January 5, 2026):**
| Trace | Status | Notes |
|-------|--------|-------|
| page_hits_trace.json | PASSED | All READs correctly associated |
| mixed_rw_trace.json | PASSED | READs and WRITEs correctly associated |
| page_conflicts_trace.json | PASSED | PRECHARGE now has correct txn_id |

## Key Invariants

### INV-001: Valid Transaction ID Semantics
Every `txn_id` must represent exactly ONE user request (READ or WRITE).
DRAM commands (ACTIVATE, PRECHARGE) generated to satisfy that request must share the same `txn_id`.

### INV-002: Command Ownership
ACTIVATE and PRECHARGE commands must be associated with the transaction that triggered them,
not assigned arbitrary or sentinel transaction IDs.

### INV-003: Temporal Ordering
Events within a transaction must be temporally ordered correctly according to DRAM protocol.

See `INVARIANTS.md` for the complete list.

## Bug Fixed

### Root Cause
PRECHARGE commands were traced with `txn_id=0` (sentinel value) instead of the
transaction ID of the request that opened the page.

**Location of bug:** `src/components/memory/lpddr5_memory_controller.cpp`

**Original code (line 681):**
```cpp
trace_command(channel, bank, "PRECHARGE", timing.tRP, 0);  // BUG: hardcoded 0
```

### Fix Applied

1. **Added tracking field to Bank structure:**
   ```cpp
   // In include/sw/kpu/components/lpddr5_memory_controller.hpp
   struct Bank {
       // ...
       uint64_t page_opener_request_id = 0;  // Track which request opened page
   };
   ```

2. **Store request_id when activating:**
   ```cpp
   // In do_activate()
   b.page_opener_request_id = request_id;
   ```

3. **Use stored request_id when precharging:**
   ```cpp
   // In do_precharge()
   uint64_t request_id = b.page_opener_request_id;
   trace_command(channel, bank, "PRECHARGE", timing.tRP, request_id);
   ```

### Before Fix
```json
{"name": "ACTIVATE",   "args": {"txn_id": 1, ...}}
{"name": "BURST_READ", "args": {"txn_id": 1, ...}}
{"name": "PRECHARGE",  "args": {"txn_id": 0, ...}}  // Wrong!
```

### After Fix
```json
{"name": "ACTIVATE",   "args": {"txn_id": 1, ...}}
{"name": "BURST_READ", "args": {"txn_id": 1, ...}}
{"name": "PRECHARGE",  "args": {"txn_id": 1, ...}}  // Correct!
```

## Visualization Fix

The timing diagram (`traces/lpddr5_timing.html`) was also fixed to properly handle
transactions without data operations:

**Original code:**
```javascript
const isRead = evts.some(e => e.name.includes('READ'));
type: isRead ? 'READ' : 'WRITE',  // Defaulted to WRITE!
```

**Fixed code:**
```javascript
const hasRead = evts.some(e => e.name.includes('READ'));
const hasWrite = evts.some(e => e.name.includes('WRITE'));
if (!hasRead && !hasWrite) {
    continue; // Skip - not a user request
}
type: hasRead ? 'READ' : 'WRITE',
```

## Development Workflow

### For Claude Code

1. **Before generating trace-related code:**
   - Read `INVARIANTS.md` to understand constraints
   - Check `CLAUDE.md` for integration guidelines

2. **After generating code:**
   - Build: `cmake --build build --target <pattern>`
   - Run pattern to generate trace
   - Validate: `python3 patterns/memory/lpddr5/common/trace_validator.py <trace.json>`
   - If validation fails, parse output and fix issues

3. **Never declare complete without validation passing**

### Validation Loop
```
Generate Code → Build → Run Pattern → Validate Trace
     ↑                                      │
     │                                      │
     └──── Fix Issues ←──── Parse Errors ←──┘
```

## Adding New Invariants

When a new bug is discovered:

1. **Document the invariant** in `INVARIANTS.md`:
   - Unique ID (INV-XXX)
   - Description and rationale
   - Validation logic
   - Failure example
   - Fix hint

2. **Implement in validator** (`trace_validator.py`):
   ```python
   def _check_inv_xxx_name(self):
       """INV-XXX: Description."""
       if violation_detected:
           self.violations.append(Violation(
               invariant='INV-XXX',
               severity=Severity.ERROR,
               message="...",
               fix_hint="..."
           ))
   ```

3. **Add to validate() method**

4. **Run validator on all traces** to ensure no regressions

## Key Insights for Claude Code

### Why Validation Matters

The trace generator produces output that visualization tools consume. Without validation:
- Bugs in trace generation silently produce incorrect visualizations
- Timing analysis is wrong because commands are orphaned
- Debugging becomes difficult because the causal chain is broken

### Actionable Error Messages

The validator provides **fix hints**, not just error messages:

```
[ERROR] INV-001: txn_id=0 has no data operation (only PRECHARGE events)
        txn_id: 0
        FIX: ACTIVATE/PRECHARGE should have same txn_id as READ/WRITE that triggered them
```

This allows Claude Code to:
1. Identify exactly what's wrong
2. Understand why it's wrong
3. Know how to fix it

### Invariants as Constraints

Invariants serve as **constraints** for code generation. When Claude Code reads `INVARIANTS.md`,
it gains understanding of what makes a trace "correct", enabling it to:
- Generate correct code from the start
- Recognize when generated code might violate invariants
- Self-correct before validation

## References

- `INVARIANTS.md` - Complete invariant documentation
- `CLAUDE.md` - Claude Code integration guidelines
- `README.md` - Pattern organization and timing parameters
- `traces/lpddr5_timing.html` - Timing diagram visualization
- `traces/lpddr5_blockdiagram.html` - Block diagram animation

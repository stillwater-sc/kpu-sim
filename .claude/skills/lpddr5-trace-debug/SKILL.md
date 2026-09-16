---
name: lpddr5-trace-debug
description: Known LPDDR5 trace-generation bug classes and their fixes — wrong PRECHARGE txn_id, request type defaulting to WRITE, and tRCD/tRP/tRRD timing violations. Use when the trace validator reports INV-001/INV-002 violations or timing-constraint failures.
---

# LPDDR5 Trace Bugs and Fixes

Migrated out of CLAUDE.md. Each entry is a bug class that has already bitten this repo.

## Bug: PRECHARGE has wrong txn_id

**Symptom:** Validator reports INV-001/INV-002 violation for `txn_id=0`

**Root Cause:** PRECHARGE assigned a sentinel txn_id instead of the original request's txn_id

**Fix:** Track which request opened the page, and use that txn_id for the PRECHARGE

## Bug: Request type defaults to WRITE

**Symptom:** Requests labeled WRITE when they should be READ

**Root Cause:** Type detection logic: `type = isRead ? 'READ' : 'WRITE'`

**Fix:** Explicit check:
```javascript
if (!hasRead && !hasWrite) continue; // Skip - not a request
type = hasRead ? 'READ' : 'WRITE';
```

## Bug: Timing constraint violation

**Symptom:** tRCD, tRP, tRRD violations

**Root Cause:** Commands issued without respecting timing parameters

**Fix:** Add timing checks before issuing commands; respect the parameters in TIMING

## Where to look

- Invariant definitions: `patterns/memory/lpddr5/INVARIANTS.md`
- Validator: `patterns/memory/lpddr5/common/trace_validator.py`
- C++ harness: `patterns/memory/lpddr5/common/lpddr5_harness.hpp`

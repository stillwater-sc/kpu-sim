---
name: lpddr5-trace-debug
description: Known LPDDR5 recording and controller bug classes and their fixes — a PRECHARGE credited to the wrong request, a refresh treated as a request, a request's direction defaulting to write, and DRAM timing violations. Use when mflow_check.py reports M9/M10/M11 ownership or order violations or M5 timing failures on a .mflow record.
---

# LPDDR5 Record Bugs and Fixes

Migrated out of CLAUDE.md. Each entry is a bug class that has already bitten this repo. The
checks are `tools/trace/mflow_check.py`'s; `patterns/memory/lpddr5/INVARIANTS.md` maps the old
INV IDs to them.

## Bug: PRECHARGE credited to the wrong request

**Symptom:** M9 reports a PRE that "serves no burst", or closes a row its burst is not in
(formerly INV-001/INV-002 on `txn_id=0`)

**Root Cause:** the PRECHARGE was given a sentinel id instead of the request that opened the page

**Fix:** track which request opened each bank's page, and name that request on its PRECHARGE

## Bug: Refresh treated as a request

**Symptom:** M11 reports a REF that serves a burst (formerly INV-006)

**Root Cause:** a refresh given a request id, so it is grouped with the requests

**Fix:** a REF serves no burst (`NONE`): it is maintenance

## Bug: Request direction defaults to WRITE

**Symptom:** a view or summary counts PRECHARGE-only groups as writes (formerly INV-200)

**Root Cause:** direction inferred by the absence of a READ: `type = isRead ? 'READ' : 'WRITE'`

**Fix:** read the record's own direction (`bursts.is_load`), never infer it from commands

## Bug: Timing constraint violation

**Symptom:** M5 reports tRCD, tRP, tRAS, tRC, tRFCpb, tRRD, tCCD, tFAW, tWTR or tRTW

**Root Cause:** a command issued without respecting a timing parameter

**Fix:** check the parameter before issuing; M5 checks against the record's declared table
(`dram_timing.params`), in controller ticks

## Where to look

- Invariant definitions and their checks: `patterns/memory/lpddr5/INVARIANTS.md`
- Checker: `tools/trace/mflow_check.py` (its docstring lists every check)
- C++ harness: `patterns/memory/lpddr5/common/lpddr5_harness.hpp` (`export_mflow`)
- Controller: `src/models/temporal/memory/controllers/lpddr5_controller.cpp`

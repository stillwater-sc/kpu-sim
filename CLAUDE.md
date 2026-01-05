# Claude Code Integration Guidelines

This document provides guidance for Claude Code when working on the KPU-SIM project.
It defines validation requirements, invariant locations, and the development workflow
that ensures correct code generation.

## Core Principle: Validate Before Declaring Complete

**Never declare code generation complete without validation.**

Claude Code must follow this workflow for any code that produces artifacts:

```
1. Generate code
2. Run validation tools
3. If validation fails:
   a. Parse error output
   b. Identify root cause
   c. Fix the issue
   d. Return to step 2
4. Only declare complete when validation passes
```

## Project Structure for Validation

```
kpu-sim/
├── CLAUDE.md                              # This file - read first!
├── patterns/
│   └── memory/
│       └── lpddr5/
│           ├── INVARIANTS.md              # Trace invariants (MUST READ)
│           └── common/
│               ├── trace_validator.py     # Standalone trace validator
│               └── lpddr5_harness.hpp     # C++ test harness
├── traces/
│   └── memory/
│       └── lpddr5/                        # Generated trace files
└── docs/
    └── sessions/                          # Session logs and changelogs
```

## Validation Tools

### 1. Trace Validator (Python)

**Location:** `patterns/memory/lpddr5/common/trace_validator.py`

**Usage:**
```bash
python3 patterns/memory/lpddr5/common/trace_validator.py <trace_file.json>
```

**Exit Codes:**
- `0` - All invariants pass
- `1` - One or more invariants violated
- `2` - Error reading/parsing trace file

**When to Run:**
- After generating any trace file
- After modifying trace generation code
- After modifying memory controller behavior

**Output Parsing:**
The validator produces structured output that Claude Code should parse:

```json
{
  "status": "FAILED",
  "violations": [
    {
      "invariant": "INV-001",
      "message": "txn_id=0 has no data operation",
      "fix_hint": "PRECHARGE should have same txn_id as READ/WRITE"
    }
  ]
}
```

Use `--json` flag for machine-readable output:
```bash
python3 trace_validator.py trace.json --json
```

### 2. C++ Test Harness

**Location:** `patterns/memory/lpddr5/common/lpddr5_harness.hpp`

The harness provides:
- `verify_no_violations()` - Check for invariant violations during simulation
- `verify_stats()` - Verify expected statistics
- Runtime invariant checking in the memory controller

### 3. Build Verification

Always run builds and tests after code changes:
```bash
cmake --preset release && cmake --build --preset release
ctest --preset release
```

## Invariant Documentation

**Primary Location:** `patterns/memory/lpddr5/INVARIANTS.md`

### Key Invariants to Enforce

| ID | Description | Severity |
|----|-------------|----------|
| INV-001 | Every txn_id must have exactly ONE data operation | ERROR |
| INV-002 | ACTIVATE/PRECHARGE must belong to valid transactions | ERROR |
| INV-003 | Commands must be temporally ordered correctly | ERROR |
| INV-100 | tRCD constraint (ACT to READ/WRITE) | WARNING |
| INV-101 | tRP constraint (PRE to ACT) | ERROR |

### Adding New Invariants

When adding new invariants:

1. Document in `INVARIANTS.md` with:
   - Unique ID (INV-XXX)
   - Description
   - Rationale
   - Validation logic
   - Failure example
   - Fix hint

2. Implement in `trace_validator.py`:
   ```python
   def _check_inv_xxx_name(self):
       """INV-XXX: Description."""
       # Validation logic
       if violation_detected:
           self.violations.append(Violation(
               invariant='INV-XXX',
               severity=Severity.ERROR,
               message="...",
               fix_hint="..."
           ))
   ```

3. Add to validator's `validate()` method

## Development Workflow

### For Trace Generation Code

```
1. Modify C++ trace generation code
2. Rebuild: cmake --build --preset release
3. Run pattern to generate trace:
   ./build/patterns/memory/lpddr5/single-bank/page-conflicts
4. Validate trace:
   python3 patterns/memory/lpddr5/common/trace_validator.py \
     traces/memory/lpddr5/single-bank/page_conflicts_trace.json
5. If failed:
   - Read violation messages
   - Trace to C++ code causing issue
   - Fix and repeat from step 2
6. When passed: commit changes
```

### For Visualization Code

```
1. Modify HTML visualization code
2. Run validator on all traces:
   for f in traces/memory/lpddr5/single-bank/*.json; do
     python3 patterns/memory/lpddr5/common/trace_validator.py "$f"
   done
3. Test visualization in browser
4. When passed: commit changes
```

## Common Bugs and Fixes

### Bug: PRECHARGE has wrong txn_id

**Symptom:** Validator reports INV-001/INV-002 violation for txn_id=0

**Root Cause:** PRECHARGE assigned sentinel txn_id instead of original request's txn_id

**Fix:** Track which request opened the page, use that txn_id for PRECHARGE

### Bug: Request type defaults to WRITE

**Symptom:** Requests labeled as WRITE when they should be READ

**Root Cause:** Type detection logic: `type = isRead ? 'READ' : 'WRITE'`

**Fix:** Explicit check:
```javascript
if (!hasRead && !hasWrite) continue; // Skip - not a request
type = hasRead ? 'READ' : 'WRITE';
```

### Bug: Timing constraint violation

**Symptom:** tRCD, tRP, tRRD violations

**Root Cause:** Commands issued without respecting timing parameters

**Fix:** Add timing checks before issuing commands, respect parameters in TIMING

## Session Logging

After significant work, create a session log:

**Location:** `docs/sessions/YYYY-MM-DD_description.md`

Include:
- What was done
- What bugs were found
- What invariants were added/modified
- Validation results

## Questions to Ask

Before generating code, ask:

1. "What invariants apply to this code?"
2. "What validation tools should I run?"
3. "What are the expected outputs?"

After generating code, ask:

1. "Did all validations pass?"
2. "Are there any warnings I should address?"
3. "Is the code covered by existing invariants?"

## Remember

**Correct code is more valuable than fast code.**

Take time to:
- Read INVARIANTS.md before generating trace-related code
- Run validators after every change
- Parse and act on validation failures
- Add new invariants when bugs are discovered
- Document changes in session logs

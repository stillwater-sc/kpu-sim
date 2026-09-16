---
name: add-invariant
description: Add a new trace invariant (INV-XXX) to the LPDDR5 validation suite — documents it in INVARIANTS.md, implements the check in trace_validator.py, and wires it into validate(). Use when a new trace bug class is discovered and needs a permanent guard.
---

# Adding a New Invariant

Migrated out of CLAUDE.md so it loads only when an invariant is actually being added.

## Existing invariants to enforce

| ID | Description | Severity |
|----|-------------|----------|
| INV-001 | Every txn_id must have exactly ONE data operation | ERROR |
| INV-002 | ACTIVATE/PRECHARGE must belong to valid transactions | ERROR |
| INV-003 | Commands must be temporally ordered correctly | ERROR |
| INV-100 | tRCD constraint (ACT to READ/WRITE) | WARNING |
| INV-101 | tRP constraint (PRE to ACT) | ERROR |

`patterns/memory/lpddr5/INVARIANTS.md` is authoritative — read it before adding an ID.

## Steps

1. Document in `patterns/memory/lpddr5/INVARIANTS.md` with:
   - Unique ID (INV-XXX)
   - Description
   - Rationale
   - Validation logic
   - Failure example
   - Fix hint

2. Implement in `patterns/memory/lpddr5/common/trace_validator.py`:
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

3. Add the new check to the validator's `validate()` method.

4. Re-run the validator over every existing trace (`/trace-check`) to confirm the new
   invariant doesn't false-positive on known-good traces.

---
name: add-invariant
description: Add a new trace invariant (INV-XXX) to the LPDDR5 validation suite — documents it in INVARIANTS.md, implements the check in trace_validator.py, and wires it into validate(). Use when a new trace bug class is discovered and needs a permanent guard.
---

# Adding a New Invariant

Migrated out of CLAUDE.md so it loads only when an invariant is actually being added.

## Existing invariants

**Do not keep a copy of the list here.** `patterns/memory/lpddr5/INVARIANTS.md` is
authoritative for definitions, and `trace_validator.py`'s `validate()` is authoritative for
what is actually enforced. The two already differ — the validator registers 8 IDs while
INVARIANTS.md documents 13 — so a third copy in this skill would only add a way to pick a
duplicate ID.

Before choosing an ID, read both:

```bash
grep -oE 'INV-[0-9]+' patterns/memory/lpddr5/INVARIANTS.md | sort -u          # documented
grep -oE "invariant='INV-[0-9]+'" patterns/memory/lpddr5/common/trace_validator.py | sort -u  # enforced
```

Ranges in use, per INVARIANTS.md's own sections: `INV-0xx` trace structure, `INV-1xx`
timing constraints, `INV-2xx` visualization. Pick the next unused number in the right range.

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

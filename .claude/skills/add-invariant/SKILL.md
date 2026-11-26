---
name: add-invariant
description: Add a new memory-side invariant — documents it in patterns/memory/lpddr5/INVARIANTS.md, implements it as an M-check in tools/trace/mflow_check.py, and adds a self-test in test_mflow_check.py that breaks a clean record and sees the check fail. Use when a new DRAM or memory-side bug class is discovered and needs a permanent guard.
---

# Adding a New Invariant

Migrated out of CLAUDE.md so it loads only when an invariant is actually being added.

Invariants are checked on the `.mflow` record by `tools/trace/mflow_check.py`. (The Chrome-trace
`trace_validator.py` and its INV checks are retired; INVARIANTS.md keeps the INV definitions,
each mapped to the M-check that enforces it.)

## Existing invariants

**Do not keep a copy of the list here.** `mflow_check.py`'s module docstring is authoritative
for what is checked, and `patterns/memory/lpddr5/INVARIANTS.md` for the definitions and their
mapping. Before choosing an ID, read both:

```bash
grep -oE 'r\.ran\("[A-Z]+[0-9]+"\)' tools/trace/mflow_check.py | sort -u     # enforced
grep -n '^| INV' patterns/memory/lpddr5/INVARIANTS.md                         # mapped
```

A new check takes the next unused `M` number.

## Steps

1. **Define it** in `patterns/memory/lpddr5/INVARIANTS.md`: description, rationale, the
   failure it catches, and a fix hint. Add its row to the "Where each invariant is enforced"
   table.

2. **Check it** in `tools/trace/mflow_check.py`'s `check(rec, run)`: declare it with
   `r.ran("Mn")` and report each violation with `r.fail("Mn", message)`. The message names
   the command, burst or request (`where(i)`, `cmd(i)` for a command), so a reader can find it.
   Check DRAM timing on the controller's ticks (the `k_*` columns), never on rounded cycles.
   An index the check follows that is out of range is unreadable input: reject it in `load()`
   (exit 2), not as a violation. Document the check in the module docstring's list.

3. **Self-test it** in `tools/trace/test_mflow_check.py`: copy the clean bundle (`Bundle`),
   break exactly what the invariant forbids (`set(table, column, row, value)`), and
   `assertFails("Mn", needle)`. Add `"Mn"` to `test_clean_bundle_passes`'s checked list. If the
   clean run does not exercise the case, extend `test_the_run_exercises_what_is_checked`.

4. **Run it** over every record the build writes:
   ```bash
   cd build && ctest -R 'mflow' --output-on-failure
   ```
   The LPDDR5 patterns' records (`lpddr5_*_mflow_check`), `kpu-memsim`'s and the cycle core's
   must still pass: a new check must not fire on a known-good run.

Run trace validators on all generated trace files.

1. Find all trace files:
   `find traces/ -name "*.json" 2>/dev/null`

2. For each trace file, run:
   `python3 patterns/memory/lpddr5/common/trace_validator.py <file> --json`

3. Report results:
   | Trace File | Status | Violations |
   |------------|--------|------------|

4. For any FAILED traces:
   - Show the invariant ID and violation message
   - Cross-reference with patterns/memory/lpddr5/INVARIANTS.md
   - Identify the C++ code likely causing the violation

5. If no trace files exist, report that and suggest running a pattern
   to generate traces first.

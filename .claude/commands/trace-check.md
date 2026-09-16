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

## Trace generation fix loop

When trace-generation C++ code changes (migrated from CLAUDE.md):

1. Modify the C++ trace generation code
2. Rebuild: `cmake --build --preset release`
3. Run the pattern to generate a trace:
   `./build/patterns/memory/lpddr5/single-bank/page-conflicts`
4. Validate:
   `python3 patterns/memory/lpddr5/common/trace_validator.py traces/memory/lpddr5/single-bank/page_conflicts_trace.json`
5. If failed: read the violation messages, trace back to the C++ code causing it,
   fix, and repeat from step 2. See the `lpddr5-trace-debug` skill for known bug classes.
6. When passed: commit

## Visualization code loop

1. Modify the HTML visualization code
2. Run the validator over all traces:
   ```bash
   for f in traces/memory/lpddr5/single-bank/*.json; do
     python3 patterns/memory/lpddr5/common/trace_validator.py "$f"
   done
   ```
3. Test the visualization in a browser
4. When passed: commit

Check every generated memory-flow record (`.mflow`) with `tools/trace/mflow_check.py`.

1. Find the records (bundles are directories):
   `find build traces runs -type d -name "*.mflow" 2>/dev/null`

2. For each, run:
   `python3 tools/trace/mflow_check.py <bundle> --json`
   Exit 0: every invariant holds. 1: violations, in `violations`. 2: the record cannot be
   read; there are no violations to parse — report the error instead.

3. Report results:
   | Record | Status | Violations |
   |--------|--------|------------|

4. For any FAILED record:
   - Show the check ID (TF10, TF11, M1–M11) and the message
   - Cross-reference with `patterns/memory/lpddr5/INVARIANTS.md` (its table maps each INV to
     its check) and `mflow_check.py`'s docstring
   - Identify the C++ code likely causing the violation

5. If no records exist, report that and suggest generating them:
   `cd build && ctest -R 'lpddr5_|mflow'` (the LPDDR5 patterns write
   `build/patterns/memory/lpddr5/<name>.mflow`).

## Record generation fix loop

When recording or controller code changes:

1. Modify the C++ code
2. Rebuild: `cmake --build --preset release`
3. Run a pattern to write its record:
   `./build/patterns/memory/lpddr5/lpddr5_page_conflicts --trace /tmp/page_conflicts.mflow`
4. Check it:
   `python3 tools/trace/mflow_check.py /tmp/page_conflicts.mflow --summary`
5. If failed: read the violations, trace back to the C++ code causing them, fix, and repeat
   from step 2. See the `lpddr5-trace-debug` skill for known bug classes.
6. When passed: run `ctest -R 'lpddr5_|mflow'` and commit

## Visualization code loop

1. Modify a view (`tools/visualization/dram/` or `tools/visualization/memflow/`)
2. Run the views' tests over real records:
   `cd build && ctest -R 'dram_views|memflow|lpddr5_interleaved_viewer' --output-on-failure`
3. Open the view on a record in a browser (the debug server, or `pack.py`)
4. When passed: commit

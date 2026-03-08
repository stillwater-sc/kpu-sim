Build the project and run all timing tests. Report results clearly.

Steps:
1. Build: `cmake --build --preset release 2>&1`
   - If build fails, show ONLY the first error and fix it
   - Rebuild after fix

2. Run timing tests: `cd build && ctest -L timing --output-on-failure 2>&1`

3. Report results as a table:
   | Test | Status | Details |
   |------|--------|---------|

4. If any tests FAIL:
   - Show the specific assertion that failed
   - Read the test file and the source code it tests
   - Propose a fix (but don't apply without asking)

5. If all tests PASS: report "All N tests passing" with total assertion count.

Do NOT run tests individually — use ctest for consistent reporting.

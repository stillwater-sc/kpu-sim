Find and fix all failing tests. $ARGUMENTS

1. Build the project first. If build fails, fix build errors.

2. Run all timing tests with verbose output:
   `cd build && ctest -L timing --output-on-failure 2>&1`

3. For each failing test:
   a. Parse the failure output to get file:line and assertion
   b. Read the test source to understand intent
   c. Read the implementation being tested
   d. Determine if the bug is in the TEST or the IMPLEMENTATION:
      - If test expectations are wrong (API changed): fix the test
      - If implementation is wrong (regression): fix the implementation
   e. Apply the fix

4. After all fixes, rebuild and rerun ALL tests to verify no regressions.

5. Report what was fixed:
   | Test | Failure | Root Cause | Fix Applied |
   |------|---------|------------|-------------|

IMPORTANT: Never skip a failing test. Never mark it as "to fix later."
Debug root causes, don't mask symptoms.

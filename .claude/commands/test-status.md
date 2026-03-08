Quick health check of the project. Run these in parallel:

1. `cmake --build --preset release 2>&1 | tail -5` (build status)
2. `cd build && ctest -L timing --output-on-failure 2>&1` (test status)
3. `git status --short` (uncommitted changes)
4. `git log --oneline -5` (recent commits)

Report as:

## Project Health
- **Build:** PASS/FAIL
- **Tests:** X/Y passing (list failures if any)
- **Uncommitted:** N files
- **Last commit:** hash message

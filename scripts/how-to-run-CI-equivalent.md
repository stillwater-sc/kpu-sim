# How to run CI equivalent work

## 1. Main CI tests (excludes benchmarks):
```bash
  # Build
  cmake --preset release
  cmake --build --preset release

  # Run tests like CI does (excludes performance/benchmark tests)
  cd build && ctest --build-config Release --output-on-failure --timeout 300 -LE "performance|benchmark"
```

## 2. Benchmark regression tests (what the benchmark CI runs):
```bash
  # Build benchmark tool
  cmake --build --preset release --target kpu-benchmark

  # Run benchmark regression script
  python3 scripts/benchmark_regression.py check
```

## 3. Full benchmark tests (the slow expansive matmul sweep that takes 5min and 12GB of memory):
```bash
  # This runs ALL tests including the benchmark_matmul Catch2 tests
  cd build && ctest --output-on-failure

  # Or run just the benchmark tests:
  cd build && ctest -L benchmark --output-on-failure
```

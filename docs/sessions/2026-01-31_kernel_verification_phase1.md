# Kernel Verification Phase 1 Decision Log

**Date:** 2026-01-31
**Version:** v0.8.x
**Status:** Complete
**Tests:** 74/74 passing

## 1. Summary

Implemented kernel-level verification harnesses for Class 0 (Elementwise) and
Class 1 (Dense Linear) under `verification/kernels/`. These test individual
kernels in isolation with parameterized sweeps against NumPy references at
BEHAVIORAL and TRANSACTIONAL fidelity levels.

## 2. Scope

| Feature | Status | Tests |
|---------|--------|-------|
| Elementwise ops (12 ops x 4 shapes) | DONE | 48/48 PASS |
| Matmul BEHAVIORAL (10 size configs) | DONE | 10/10 PASS |
| Matmul TRANSACTIONAL (4 size configs + FLOP validation) | DONE | 4/4 PASS |
| Fused ops (4 patterns x 3 sizes) | DONE | 12/12 PASS |
| TAXONOMY.md Phase 1 update | DONE | N/A |

## 3. Technical Decisions

**Decision 1: Roofline check is informational, not a gate**
- **Choice:** Roofline GFLOPS comparison reports values but does not cause FAIL
- **Alternatives Considered:** Strict 10% roofline gate as originally planned
- **Rationale:** Simulator throughput is fundamentally different from hardware
  throughput. The roofline model predicts hardware performance; the simulator
  executes functionally. FLOP count accuracy (exact match) is the meaningful
  metric. Roofline validation belongs in hardware-calibrated testing.
- **Files Modified:** `verification/kernels/class1_dense_linear/verify_matmul.py`

**Decision 2: Fidelity + clock frequency ordering**
- **Choice:** Set fidelity before clock frequency when switching to TRANSACTIONAL
- **Rationale:** `_init_native_sim()` creates the C++ runtime with the current
  fidelity level. If clock frequency is set while fidelity is still BEHAVIORAL,
  it configures a BEHAVIORAL native runtime. Switching fidelity later doesn't
  propagate the clock setting to the new TRANSACTIONAL runtime.
- **Files Modified:** `verification/kernels/class1_dense_linear/verify_matmul.py`

**Decision 3: XUE counter reset per transactional test**
- **Choice:** Call `kpu.reset_xue_counters()` before each transactional test case
- **Rationale:** Without reset, FLOP counts accumulate across test cases,
  causing false FLOP validation failures.
- **Files Modified:** `verification/kernels/class1_dense_linear/verify_matmul.py`

## 4. Issues Encountered

**Issue 1: `@kpu.compile` ignores `fidelity` kwarg for clock frequency**
- **Symptom:** `RuntimeError: Clock frequency not set for transactional mode`
  even after calling `kpu.set_clock_frequency(1.0)`
- **Root cause:** The native runtime singleton is created with the fidelity
  level active at the time of first `_init_native_sim()` call. Setting clock
  frequency while in BEHAVIORAL mode configures a BEHAVIORAL native runtime.
  Switching to TRANSACTIONAL later creates/uses a different runtime path.
- **Fix:** Set fidelity to TRANSACTIONAL before setting clock frequency.

## 5. Wrong Decisions

**Wrong Decision 1: Initially passed `fidelity=kpu.TRANSACTIONAL` to `@kpu.compile`**
- **Why wrong:** The decorator's fidelity parameter doesn't override the global
  runtime's native sim initialization order. The clock frequency must be set
  on the correct native runtime instance.
- **Correction:** Use global `kpu.set_fidelity()` + `kpu.set_clock_frequency()`
  instead of per-decorator fidelity.
- **Lesson:** The KPU runtime is a singleton with order-dependent initialization.
  Always configure fidelity globally before clock frequency.

## 6. Verification

```bash
PYTHONPATH=python:$PYTHONPATH python verification/kernels/class0_elementwise/verify_elementwise.py
# Total: 48  PASS: 48  FAIL: 0  ERROR: 0

PYTHONPATH=python:$PYTHONPATH python verification/kernels/class1_dense_linear/verify_matmul.py
# Total: 14  PASS: 14  FAIL: 0  ERROR: 0

PYTHONPATH=python:$PYTHONPATH python verification/kernels/class1_dense_linear/verify_fused_ops.py
# Total: 12  PASS: 12  FAIL: 0  ERROR: 0
```

## 7. Files Modified

| File | Action |
|------|--------|
| `verification/kernels/class0_elementwise/verify_elementwise.py` | CREATE |
| `verification/kernels/class1_dense_linear/verify_matmul.py` | CREATE |
| `verification/kernels/class1_dense_linear/verify_fused_ops.py` | CREATE |
| `verification/kernels/TAXONOMY.md` | UPDATE |
| `CHANGELOG.md` | UPDATE |

## 8. Next Steps

- Add NumPy reference checks to `python/examples/minimal_mlp.py`
- Add NumPy reference checks to `python/examples/xue_validation.py`
- Implement Class 2 (Spatial Convolution) verification harness
- Investigate runtime singleton fidelity-switching to make it more robust

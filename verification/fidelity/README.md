# Fidelity Verification

**Axis:** Simulation accuracy across BEHAVIORAL, TRANSACTIONAL, and TEMPORAL tiers

This directory verifies that the three simulation fidelity levels produce
consistent and accurate results, and that the lower-cost models (BEHAVIORAL,
TRANSACTIONAL) predict the higher-cost models (TEMPORAL) within calibrated
error bounds.

## The Three Fidelity Levels

| Level | Name | What It Computes | Speed | Verification Target |
|-------|------|-----------------|-------|-------------------|
| 0 | **BEHAVIORAL** | Actual tensor values | 100-1000x | Bit-accurate vs NumPy/PyTorch reference |
| 1 | **TRANSACTIONAL** | Statistical timing estimates | 10-100x | Within 20% of TEMPORAL predictions |
| 2 | **TEMPORAL** | Cycle-accurate timing | 1x baseline | Within 10% of target hardware |

## Verification Strategy

### Level 0: BEHAVIORAL Correctness

**Question:** Does the simulator produce the right answer?

The BEHAVIORAL tier computes actual tensor values through the C++
BehavioralComputeFabric. Verification compares simulator output against
a golden reference (NumPy or PyTorch) for each supported operation.

| Test Category | Tolerance | Reference |
|--------------|-----------|-----------|
| MatMul (FP32) | max_diff < 1e-5 | NumPy `@` operator |
| Conv2D (FP32) | max_diff < 1e-4 | NumPy manual convolution |
| Elementwise (ReLU, Add, Mul, Neg) | exact (0.0) | NumPy equivalent |
| Transcendentals (GELU, SiLU, Sigmoid, Tanh) | max_diff < 1e-6 | NumPy/SciPy |
| Softmax | max_diff < 1e-6 | NumPy (numerically stable) |
| LayerNorm | max_diff < 1e-5 | NumPy manual |
| BatchNorm2d | max_diff < 1e-5 | NumPy manual |
| Pooling (Max, Avg) | exact for Max; < 1e-6 for Avg | NumPy manual |
| Fused ops (MatMul+Bias+Act) | Same as unfused composition | Unfused reference |
| End-to-end model | Model-specific tolerance | PyTorch inference |

**Verification approach:**
1. Deterministic weight initialization (fixed seed)
2. Known input data (fixed seed or canonical dataset)
3. Element-wise comparison against reference
4. Report max absolute difference and location of worst element
5. PASS/FAIL against tolerance threshold

### Level 1: TRANSACTIONAL Performance Estimation

**Question:** Does the timing model predict performance correctly?

The TRANSACTIONAL tier uses throughput-based models (operations/cycle,
bytes/cycle) to estimate execution time without cycle-level simulation.
Verification compares TRANSACTIONAL predictions against TEMPORAL results.

| Metric | Tolerance | Measured By |
|--------|-----------|-------------|
| Total cycles | within 20% of TEMPORAL | Cross-fidelity comparison |
| Compute cycles | within 15% of TEMPORAL | Matmul/Conv cycle breakdown |
| Memory cycles | within 25% of TEMPORAL | DRAM access cycle estimate |
| FLOP count | exact match | XUE event counter audit |
| Memory bytes moved | within 10% of TEMPORAL | XUE memory hierarchy |
| Arithmetic intensity | within 5% of analytical | Theoretical calculation |
| Roofline GFLOPS | within 10% of achieved | Roofline model validation |

**Verification approach:**
1. Run same workload at both TRANSACTIONAL and TEMPORAL
2. Compare timing predictions
3. Flag any metric outside tolerance
4. Track prediction accuracy trend across model sizes

**Key XUE metrics for TRANSACTIONAL validation:**
- `total_flops`: Must match 2*M*N*K for matmul (exact)
- `total_bytes_moved`: Must match analytical model within tolerance
- `arithmetic_intensity`: total_flops / total_bytes_moved
- `achieved_gflops`: total_flops / (total_cycles / clock_freq)
- `predicted_gflops`: From roofline model (min of compute roof, memory roof)

### Level 2: TEMPORAL Cycle-Accurate Timing

**Question:** Does the cycle-level simulation match hardware behavior?

The TEMPORAL tier models the full pipeline: DMA, BlockMover, Streamer,
Compute, with credit-based flow control, bank conflicts, and protocol
timing constraints. Verification uses formal invariants and (eventually)
comparison against FPGA or silicon measurements.

| Invariant Category | Description | Severity |
|-------------------|-------------|----------|
| Protocol timing | tRCD, tRP, tRRD, tCCD constraints | ERROR |
| Credit flow | No credit underflow/overflow | ERROR |
| Pipeline ordering | Commands temporally ordered | ERROR |
| Resource conflicts | No double-allocation of buffers | ERROR |
| Data integrity | Same functional result as BEHAVIORAL | ERROR |
| Performance bounds | Cycles within 10% of analytical model | WARNING |

**Verification approach:**
1. Run workload at TEMPORAL fidelity
2. Validate all timing invariants (no violations)
3. Compare functional output against BEHAVIORAL (must match)
4. Compare cycle count against analytical prediction
5. Eventually: compare against hardware measurements

## Cross-Fidelity Consistency

The fundamental invariant across fidelity levels:

```
BEHAVIORAL output == TRANSACTIONAL output == TEMPORAL output
```

The functional result must be **identical** regardless of fidelity level.
Only the timing/performance predictions differ. This is verified by running
the same model at all available fidelity levels and comparing tensor outputs.

```
┌─────────────────────────────────────────────────────────┐
│                    Same Input                            │
│                       │                                  │
│          ┌────────────┼────────────┐                     │
│          ▼            ▼            ▼                     │
│    BEHAVIORAL   TRANSACTIONAL   TEMPORAL                │
│    (values)     (values+timing) (values+cycles)         │
│          │            │            │                     │
│          └────────────┼────────────┘                     │
│                       ▼                                  │
│              Values must match                           │
│              Timing: TRANS ≈ TEMPORAL (within 20%)       │
│              Cycles: TEMPORAL ≈ Hardware (within 10%)    │
└─────────────────────────────────────────────────────────┘
```

## Current Fidelity Verification Status

| Component | BEHAVIORAL | TRANSACTIONAL | TEMPORAL |
|-----------|-----------|---------------|----------|
| MatMul | Verified (FP32) | Timing model active | Systolic array defined, not wired |
| Conv2D | Verified (im2col) | Timing model defined, not active | Not implemented |
| Elementwise | Verified | Timing model active | Not implemented |
| Softmax | Verified | Timing model active | Not implemented |
| LayerNorm | Verified | Timing model active | Not implemented |
| BatchNorm2d | Verified | Timing model active | Not implemented |
| Pool2D | Verified | Timing model defined, not active | Not implemented |
| Memory Controller | Fixed latency | Queue model defined | Full DRAM FSM exists |
| DMA Engine | Instant transfer | Bandwidth model defined | Not implemented |
| NoC | Zero latency | Hop count model defined | Not implemented |

## Implementation Roadmap

### Phase F1: BEHAVIORAL Complete (v0.8.x) - CURRENT

All kernels produce correct functional results. This is the foundation
that all other fidelity levels build on.

| Task | Status |
|------|--------|
| MatMul correctness | DONE |
| Conv2D correctness | DONE |
| Fused op correctness | DONE (bug fixed 2026-01-31) |
| All activations | DONE |
| Softmax, LayerNorm, BatchNorm | DONE |
| Pooling ops | DONE |
| Concat, Reshape, Transpose | Partial (Python-only for some) |
| Cross-reference all ops vs NumPy | TODO (systematic harness) |

### Phase F2: TRANSACTIONAL Calibrated (v0.9.x)

TRANSACTIONAL timing predictions validated against TEMPORAL for core
kernels. All XUE metrics producing meaningful values.

| Task | Status |
|------|--------|
| MatMul throughput model | Active, producing stats |
| Conv2D timing model | Defined, not wired |
| Pool2D timing model | Defined, not wired |
| Memory controller queue model | Defined |
| XUE FLOP validation | Partial (matmul only) |
| XUE memory traffic validation | Partial |
| Roofline prediction accuracy | Partial (10% target) |
| Cross-validate TRANS vs TEMPORAL | Blocked on TEMPORAL implementation |

### Phase F3: TEMPORAL Validated (v1.0.x)

Cycle-accurate simulation with formal invariant checking. Performance
predictions within 10% of analytical models.

| Task | Status |
|------|--------|
| Systolic array pipeline model | Defined in temporal/compute/ |
| Full DRAM FSM (LPDDR5) | Implemented for memory patterns |
| Credit-based flow simulation | Framework exists |
| Timing invariant checking | Framework exists (INVARIANTS.md) |
| Cross-validate TEMPORAL vs hardware | Future (FPGA/silicon) |

## Relationship to Other Verification Axes

- **Kernels** (`../kernels/`): Fidelity tests exercise specific kernels;
  kernel correctness is a prerequisite for fidelity validation
- **DNN** (`../dnn/`): DNN models are the integration test for fidelity;
  a model passing at all three levels proves the fidelity stack works

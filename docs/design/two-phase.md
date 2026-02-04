# Two phase design


We just encountered a bug where the three Communicating Sequential Processes (CSPs) did not properly synchronize, but the functional check passed. 
Without adhering to the CSP dependency, it would be impossible to generate the correct results of the matmul, and thus our assumption on what to
test to check that we are functionally correct was wrong. 

## Root Cause Analysis: Decoupled Two-Phase Architecture

The answer is on lines 177-181:

  // Phase 1: Execute behaviorally (moves real data, computes real values)
  // All behavioral operations are instant (memcpy), so we can run to completion first.
  bool halted = behavioral_.run();

  // Phase 2: Compute timing overlay

## The Architecture

  The TransactionalProgramExecutor uses a two-phase decoupled design:
  ┌─────────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────────┐
  │  Phase  │        What It Does         │          Data Movement          │         Timing          │
  ├─────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
  │ Phase 1 │ behavioral_.run()           │ YES (actual memcpy, matmul)     │ NO (instant)            │
  ├─────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
  │ Phase 2 │ dispatch_with_timing() loop │ NO (just iterates instructions) │ YES (calculates cycles) │
  └─────────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────────┘

## Why Functional Tests Passed

The behavioral executor runs to completion first, moving all data correctly with instant operations (memcpy). 
The timing overlay is computed after the fact by re-iterating through the instructions without touching any data.

The timing model is purely analytical - it doesn't actually move data or affect the simulation state. It just calculates "when would this have happened?"

## The Hidden Bug

This decoupling means:
  - Correct results: Always (behavioral handles it)
  - Correct timing: Only if the timing model is correct (it wasn't)

The bug was invisible to functional tests because they only verify numerical correctness. The only way to catch it was visual inspection of the Chrome trace.

## Implications

This is a design tradeoff:

  - Pro: Simpler, faster - behavioral runs once, timing is just arithmetic
  - Con: Timing bugs are silent - no validation that timing matches physical constraints

To catch timing bugs automatically, we'd need either:

  1. Timing invariant checks - validate that no operation starts before its dependencies complete
  2. Interleaved execution - compute timing as operations execute, stall if dependencies aren't ready

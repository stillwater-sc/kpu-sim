# Functional Simulation Gap Analysis: XOR MLP on KPU

## Executive Summary

The KPU simulator is currently a **timing and performance model**, not a functional simulator.
It accurately models:
- Instruction scheduling and cycle counts
- Memory hierarchy latencies
- Data movement through the system
- Resource contention

It does **NOT** compute actual numerical results during kernel execution. This document
analyzes the gap between timing simulation and functional simulation, using the XOR MLP
as a concrete example.

## The XOR MLP Problem

### Network Architecture

```
Input Layer     Hidden Layer      Output Layer
   [2]      →      [4]        →      [1]

   x1 ─┬─→ h0 ─┬
       │       │
       ├─→ h1 ─┼──→ output
       │       │
       ├─→ h2 ─┤
       │       │
   x2 ─┴─→ h3 ─┘
```

### Computational Requirements

| Layer | Operation | Dimensions | MACs | Parameters |
|-------|-----------|------------|------|------------|
| Hidden | MatMul + Bias + ReLU | [batch, 2] × [2, 4] + [4] | 8 × batch | 12 |
| Output | MatMul + Bias | [batch, 4] × [4, 1] + [1] | 4 × batch | 5 |
| **Total** | | | **12 × batch** | **17** |

For batch=4 (all XOR cases): 48 MAC operations, 17 parameters.

## The Mismatch: Tiny Operators on Large Hardware

### KPU Configuration in XOR Example

```cpp
KPUSimulator::Config config;
config.processor_array_rows = 16;
config.processor_array_cols = 16;
config.use_systolic_array_mode = true;
// Total: 256 processing elements (PEs)
```

### The Problem

| Aspect | XOR MLP | KPU Array | Utilization |
|--------|---------|-----------|-------------|
| Hidden layer matmul | 2×4 = 8 MACs | 16×16 = 256 PEs | 3.1% |
| Output layer matmul | 4×1 = 4 MACs | 16×16 = 256 PEs | 1.6% |

The XOR MLP uses less than 4% of the compute array capacity. This is a valid
scenario (small models exist), but the simulator should model how such
small operations actually execute on large arrays.

### Missing: Small Operator Mapping Strategy

The simulator doesn't explain or implement how small operators execute on large arrays.
Real hardware has several strategies:

1. **Operator Padding**: Pad 2×4 to 16×16 with zeros
   - Wastes compute but simple to implement
   - Timing: Same as full 16×16 matmul

2. **Multi-Operator Batching**: Execute multiple small ops simultaneously
   - Pack several 2×4 matmuls into one 16×16 cycle
   - More complex scheduling

3. **Array Partitioning**: Use subset of array
   - 2×4 only activates 8 PEs
   - Power savings but timing unchanged

**Gap**: None of these strategies are implemented or documented.

## Current Architecture: Timing Model Only

### What the Simulator Actually Does

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIMING MODEL PATH                             │
│                                                                  │
│  Kernel::create_mlp()                                           │
│         │                                                        │
│         ▼                                                        │
│  Program (ISA instructions)                                      │
│         │                                                        │
│         ▼                                                        │
│  ConcurrentExecutor::execute()                                   │
│         │                                                        │
│         ├─→ Schedule instructions to hardware units              │
│         ├─→ Model resource contention                            │
│         ├─→ Track cycle counts                                   │
│         │                                                        │
│         ▼                                                        │
│  LaunchResult { success=true, cycles=1500 }                      │
│                                                                  │
│  ⚠️  NO ACTUAL COMPUTATION PERFORMED                            │
│  ⚠️  NO MATRIX VALUES COMPUTED                                  │
│  ⚠️  OUTPUT TENSORS CONTAIN UNINITIALIZED DATA                  │
└─────────────────────────────────────────────────────────────────┘
```

### Evidence from Code

**ConcurrentExecutor** (src/isa/concurrent_executor.cpp):
```cpp
Cycle ConcurrentExecutor::execute(const Program& program) {
    // Schedules instructions, returns cycle count
    // Does NOT call compute fabric with actual data
    // Does NOT load matrices from memory
    // Does NOT compute results
    return total_cycles;
}
```

**GraphRunner::run()** (src/runtime/graph.cpp):
```cpp
GraphExecutionResult GraphRunner::run() {
    for (size_t node_idx : graph_->execution_order()) {
        // Launches kernel for timing
        auto result = runtime_->launch(kernel, addresses);
        // Only captures cycle count, not computed values
        total_cycles += result.cycles;
    }
    // Output tensors are NOT populated with computed values
    return GraphExecutionResult{true, total_cycles, ...};
}
```

### What the XOR Example Actually Does

The XOR example works around this limitation:

```cpp
// Run KPU simulation (timing only)
auto result = runner.run();  // Returns cycle count

// Use SEPARATE reference implementation for actual values
float reference = reference_xor_forward(x1, x2);  // Actually computes
```

The reference implementation computes correct XOR outputs; the KPU simulator
only provides timing estimates.

## Gap Analysis: Missing Functional Behaviors

### 1. Compute Fabric Integration Gap

| Component | Exists | Functional | Gap |
|-----------|--------|------------|-----|
| SystolicArray class | ✓ | ✓ | Not invoked during execution |
| ProcessingElement::cycle() | ✓ | ✓ | Has MAC but never called |
| ComputeFabric::execute_matmul() | ✓ | ✓ | Has impl but not in main path |
| BehavioralComputeFabric | ✓ | ✓ | Alternative impl, not integrated |

**The compute fabric exists and works, but is not invoked during kernel execution.**

### 2. Data Flow Gap

```
CURRENT (Timing Only):
┌────────────────────────────────────────────────────────────────┐
│ Host Memory ──DMA──▶ L3 ──BM──▶ L2 ──BM──▶ L1 ──▶ ???         │
│                                                                │
│ Data is tracked for timing but values are not used            │
│ Compute results are never written back                        │
└────────────────────────────────────────────────────────────────┘

REQUIRED (Functional):
┌────────────────────────────────────────────────────────────────┐
│ Host Memory ──DMA──▶ L3 ──BM──▶ L2 ──BM──▶ L1 ──▶ Page Buffers │
│                                           │                    │
│                                           ▼                    │
│                                    Compute Fabric              │
│                                    (Actual MatMul)             │
│                                           │                    │
│                                           ▼                    │
│ Host Memory ◀──DMA── L3 ◀──BM── L2 ◀──BM── L1 ◀── Result      │
└────────────────────────────────────────────────────────────────┘
```

### 3. Instruction Execution Gap

| ISA Instruction | Timing Modeled | Functionally Executed |
|-----------------|----------------|----------------------|
| MATMUL | ✓ Cycles counted | ✗ No actual computation |
| LOAD/STORE | ✓ Memory latency | ✗ Data not transferred |
| BIAS_ADD | ✓ Cycles counted | ✗ No addition performed |
| RELU/GELU/etc | ✓ Cycles counted | ✗ No activation applied |

### 4. Memory Consistency Gap

When `runtime_->memcpy_h2d()` is called to load weights:
- Data IS written to simulated memory
- But this data is NEVER read during execution
- Computation uses no actual values

When `runner.get_output()` is called:
- Memory IS read from simulated memory
- But values are uninitialized (never computed)

### 5. Activation Functions Gap

The XOR MLP requires ReLU activation. Current state:

```cpp
// Kernel specifies ReLU
auto hidden_layer = Kernel::create_mlp(
    BATCH_SIZE, HIDDEN_DIM, INPUT_DIM,
    ActivationType::RELU, true);  // ReLU requested
```

```cpp
// But execution path does NOT apply it
// ConcurrentExecutor schedules RELU instruction
// Cycle cost is counted
// But max(0, x) is never computed
```

## Detailed Requirements for Functional Simulation

### Requirement 1: Execute Compute Instructions

**Current**: `ConcurrentExecutor::execute()` returns cycle count only.

**Required Changes**:

```cpp
// In concurrent_executor.cpp - execute()
case Opcode::MATMUL: {
    // Current: just schedule and count cycles
    // Required: actually perform computation

    // 1. Load A matrix from simulated memory
    auto* a = memory_->read<float>(instr.src1_addr, m * k);

    // 2. Load B matrix from simulated memory
    auto* b = memory_->read<float>(instr.src2_addr, k * n);

    // 3. Call compute fabric
    auto* c = compute_fabric_->execute_matmul(a, b, m, k, n);

    // 4. Write result to simulated memory
    memory_->write(instr.dst_addr, c, m * n);

    // 5. Still return cycle count
    cycles += compute_fabric_->matmul_cycles(m, k, n);
    break;
}

case Opcode::RELU: {
    // Current: just count cycles
    // Required: apply activation

    auto* data = memory_->read<float>(instr.src_addr, size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
    memory_->write(instr.dst_addr, data, size);

    cycles += vector_engine_->activation_cycles(size);
    break;
}
```

### Requirement 2: Small Operator Handling

For XOR's 2×4 matmul on 16×16 array:

**Option A: Transparent Padding**
```cpp
// In systolic_array.hpp
template<uint32_t ROWS, uint32_t COLS>
class SystolicArray {
    void execute(const float* a, const float* b, float* c,
                 uint32_t m, uint32_t k, uint32_t n) {
        // Pad to array dimensions
        uint32_t padded_m = std::max(m, ROWS);
        uint32_t padded_n = std::max(n, COLS);

        // Execute full array (unused PEs get zeros)
        // ... systolic computation ...

        // Extract m×n result from padded output
    }
};
```

**Option B: Partial Array Activation**
```cpp
// Only use 2×4 subset of 16×16 array
void execute_partial(uint32_t m, uint32_t k, uint32_t n) {
    // Same cycle count as full array (control overhead)
    // But model power savings
    active_pes = m * n;  // 8 instead of 256
    power_factor = float(active_pes) / float(ROWS * COLS);
}
```

### Requirement 3: Layer-to-Layer Data Flow

For multi-layer MLP:

```cpp
// GraphRunner::run() - functional version
GraphExecutionResult GraphRunner::run() {
    for (size_t node_idx : graph_->execution_order()) {
        const auto& node = graph_->node(node_idx);

        // Get input tensor (from previous layer or graph input)
        Address input_addr = get_tensor_address(node.inputs[0]);

        // Get weight tensors
        Address weight_addr = get_tensor_address(node.name + "_B");
        Address bias_addr = get_tensor_address(node.name + "_bias");

        // Get output tensor address
        Address output_addr = get_tensor_address(node.outputs[0]);

        // Execute with FUNCTIONAL computation
        auto result = runtime_->launch_functional(
            node.kernel,
            {input_addr, weight_addr, bias_addr, output_addr}
        );

        // Output tensor NOW contains computed values
        // Next layer will use these values as input
    }
}
```

### Requirement 4: Verification Infrastructure

```cpp
// Verification helper for XOR example
bool verify_xor_functional(GraphRunner& runner) {
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float expected[4] = {0, 1, 1, 0};

    for (int i = 0; i < 4; i++) {
        runner.set_input("input", inputs[i], 2 * sizeof(float));
        runner.run();  // Functional execution

        float output;
        runner.get_output("output", &output);

        // Output should match expected (within tolerance)
        bool correct = (output > 0.5f) == (expected[i] > 0.5f);
        if (!correct) return false;
    }
    return true;
}
```

## Implementation Roadmap

### Phase 1: Functional Compute Fabric (Core)

| Task | Files | Effort |
|------|-------|--------|
| Add functional mode flag to executor | concurrent_executor.hpp/cpp | 1 |
| Implement MATMUL functional execution | concurrent_executor.cpp | 2 |
| Implement BIAS_ADD functional execution | concurrent_executor.cpp | 1 |
| Implement activation functions | concurrent_executor.cpp | 2 |
| Add memory read/write during execution | memory components | 2 |

### Phase 2: Small Operator Handling

| Task | Files | Effort |
|------|-------|--------|
| Document mapping strategies | docs/ | 1 |
| Implement padding for undersized ops | systolic_array.hpp | 2 |
| Add utilization metrics | compute_fabric.cpp | 1 |

### Phase 3: Graph Execution Integration

| Task | Files | Effort |
|------|-------|--------|
| Modify GraphRunner for functional mode | graph.cpp | 2 |
| Add tensor value verification | graph.cpp | 1 |
| Update examples for functional mode | examples/mlp/*.cpp | 2 |

### Phase 4: Validation

| Task | Files | Effort |
|------|-------|--------|
| XOR MLP functional test | tests/ | 1 |
| Sine approximation test | tests/ | 1 |
| MNIST digit verification | tests/ | 2 |
| Reference comparison framework | tests/ | 2 |

## Conclusion

The KPU simulator has the foundational components for functional simulation:
- SystolicArray with actual MAC operations
- ComputeFabric with matmul implementation
- BehavioralComputeFabric as alternative

**The gap is in integration**: These functional components are not invoked during
the normal execution path. The ConcurrentExecutor only schedules instructions for
timing; it doesn't execute them functionally.

### Summary of Gaps

| Category | Gap | Impact | Priority |
|----------|-----|--------|----------|
| Compute | MACs not executed | No actual results | Critical |
| Memory | Data not read during exec | Weights ignored | Critical |
| Activation | Functions not applied | Wrong outputs | Critical |
| Small ops | No mapping strategy | Unclear behavior | High |
| Verification | No functional tests | Can't validate | High |
| Documentation | Timing-only not clear | User confusion | Medium |

### Recommended Next Steps

1. **Immediate**: Add explicit documentation that this is a timing-only model
2. **Short-term**: Add functional execution mode flag
3. **Medium-term**: Implement functional MATMUL/BIAS/activation
4. **Long-term**: Full functional simulation with verification suite

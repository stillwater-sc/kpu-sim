# True Kernel Fusion Implementation Plan

## Overview

Implement true kernel fusion in kpu-sim to eliminate external memory round-trips for intermediate tensors. Expected result: 2-4× memory traffic reduction for fused matmul chains.

## Current State

**Working:**
- Fusion detection: `find_fusible_pairs()`, `can_fuse()` (kernel_graph.cpp:413-458)
- FusionStrategy enum: NONE, PRODUCER_CONSUMER, HORIZONTAL, PIPELINE
- Metadata tracking: `is_fused`, `fused_with` fields on KernelNode

**Stubbed (TODO):**
- `compile_fused_pair()` at kernel_graph.cpp:610-619 - just appends sequentially with barrier
- `compile()` always falls back to `compile_sequential()` when fusion requested

## Key Insight

For producer-consumer matmul fusion (C1 = A1 @ B1, then C2 = C1 @ B2):

**Sequential Flow:**
```
Producer: DMA_LOAD A1, DMA_LOAD B1, compute, STR_DRAIN C1, DMA_STORE C1
          ↓ (External Memory Round-Trip)
Consumer: DMA_LOAD C1(as A2), DMA_LOAD B2, compute, STR_DRAIN C2, DMA_STORE C2
```

**Fused Flow:**
```
Producer: DMA_LOAD A1, DMA_LOAD B1, compute, STR_DRAIN C1 (stays in L3)
          ↓ (L3 Cache, no External Memory)
Consumer: (skip A2 load - use C1 in L3), DMA_LOAD B2, compute, STR_DRAIN C2, DMA_STORE C2
```

**Savings:** 2 × M × N × sizeof(dtype) bytes per fused pair

## Implementation Plan

### Phase 1: Instruction Analysis Helpers

**File:** `include/sw/kpu/kernel_graph.hpp`

Add to private section:
```cpp
struct FusionAnalysis {
    std::vector<size_t> c_store_indices;    // DMA_STORE_TILE for C
    std::vector<size_t> a_load_indices;     // DMA_LOAD_TILE for A
    Address c_l3_offset = 0;                // Where C resides in L3
};

FusionAnalysis analyze_producer_for_fusion(const Kernel& producer) const;
FusionAnalysis analyze_consumer_for_fusion(const Kernel& consumer) const;
void remap_instruction_a_to_c(isa::DMInstruction& instr, Address c_l3_offset) const;

KernelGraphCompileResult compile_with_fusion(
    const std::vector<std::pair<size_t, size_t>>& fused_pairs,
    const KernelGraphCompileOptions& options) const;

void update_fused_estimates(
    KernelGraphCompileResult& result,
    const std::vector<std::pair<size_t, size_t>>& fused_pairs) const;
```

### Phase 2: Producer Analysis

**File:** `src/simulator/kernel_graph.cpp`

```cpp
KernelGraph::FusionAnalysis
KernelGraph::analyze_producer_for_fusion(const Kernel& producer) const {
    FusionAnalysis analysis;
    const auto& instrs = producer.program().instructions;

    for (size_t i = 0; i < instrs.size(); ++i) {
        if (instrs[i].opcode == isa::DMOpcode::DMA_STORE_TILE) {
            const auto& ops = std::get<isa::DMAOperands>(instrs[i].operands);
            if (ops.matrix == isa::MatrixID::C) {
                analysis.c_store_indices.push_back(i);
                analysis.c_l3_offset = ops.l3_offset;
            }
        }
    }
    return analysis;
}
```

### Phase 3: Consumer Analysis

```cpp
KernelGraph::FusionAnalysis
KernelGraph::analyze_consumer_for_fusion(const Kernel& consumer) const {
    FusionAnalysis analysis;
    const auto& instrs = consumer.program().instructions;

    for (size_t i = 0; i < instrs.size(); ++i) {
        if (instrs[i].opcode == isa::DMOpcode::DMA_LOAD_TILE) {
            const auto& ops = std::get<isa::DMAOperands>(instrs[i].operands);
            if (ops.matrix == isa::MatrixID::A) {
                analysis.a_load_indices.push_back(i);
            }
        }
    }
    return analysis;
}
```

### Phase 4: True compile_fused_pair()

Replace stub at kernel_graph.cpp:610-619:

```cpp
void KernelGraph::compile_fused_pair(isa::DMProgram& target,
                                     const KernelNode& producer,
                                     const KernelNode& consumer,
                                     Address base_offset) const {
    auto prod_analysis = analyze_producer_for_fusion(*producer.kernel);
    auto cons_analysis = analyze_consumer_for_fusion(*consumer.kernel);

    // Build skip sets
    std::set<size_t> prod_skip(prod_analysis.c_store_indices.begin(),
                                prod_analysis.c_store_indices.end());
    std::set<size_t> cons_skip(cons_analysis.a_load_indices.begin(),
                                cons_analysis.a_load_indices.end());

    // Emit producer (skip C stores and HALT)
    for (size_t i = 0; i < producer.kernel->program().instructions.size(); ++i) {
        const auto& instr = producer.kernel->program().instructions[i];
        if (prod_skip.count(i) || instr.opcode == isa::DMOpcode::HALT) continue;
        target.instructions.push_back(instr);
    }

    // Minimal sync barrier
    target.instructions.push_back(isa::DMInstruction::barrier());

    // Emit consumer (skip A loads, remap BM_MOVE for A)
    for (size_t i = 0; i < consumer.kernel->program().instructions.size(); ++i) {
        if (cons_skip.count(i)) continue;

        auto instr = consumer.kernel->program().instructions[i];

        // Remap A's L3 source to producer's C location
        if (instr.opcode == isa::DMOpcode::BM_MOVE_TILE) {
            auto& ops = std::get<isa::BlockMoverOperands>(instr.operands);
            if (ops.matrix == isa::MatrixID::A) {
                ops.src_offset = prod_analysis.c_l3_offset;
            }
        }

        target.instructions.push_back(instr);
    }
}
```

### Phase 5: Update compile() Method

Modify compile() to use fusion when requested:

```cpp
KernelGraphCompileResult KernelGraph::compile(
    const KernelGraphCompileOptions& options) const {
    // ... validation ...

    if (options.fusion_strategy == FusionStrategy::NONE) {
        return compile_sequential();
    }

    auto fusible = find_fusible_pairs();
    if (fusible.empty()) {
        return compile_sequential();
    }

    if (options.fusion_strategy == FusionStrategy::PRODUCER_CONSUMER) {
        return compile_with_fusion(fusible, options);
    }

    return compile_sequential();
}
```

### Phase 6: compile_with_fusion()

```cpp
KernelGraphCompileResult KernelGraph::compile_with_fusion(
    const std::vector<std::pair<size_t, size_t>>& fused_pairs,
    const KernelGraphCompileOptions& options) const {

    KernelGraphCompileResult result;
    result.execution_order = get_execution_order();
    result.fused_pairs = fused_pairs;
    result.program.name = name_ + "_fused";
    result.program.dataflow = isa::DMProgram::Dataflow::OUTPUT_STATIONARY;

    std::unordered_set<size_t> processed;

    for (size_t node_id : result.execution_order) {
        if (processed.count(node_id)) continue;

        // Check if producer of a fused pair
        for (const auto& [prod, cons] : fused_pairs) {
            if (prod == node_id) {
                compile_fused_pair(result.program, get_node(prod), get_node(cons), 0);
                processed.insert(prod);
                processed.insert(cons);
                break;
            }
        }

        // If not fused, compile normally
        if (!processed.count(node_id)) {
            append_kernel_program(result.program, *get_node(node_id).kernel, 0);
            if (node_id != result.execution_order.back()) {
                result.program.instructions.push_back(isa::DMInstruction::barrier());
            }
            processed.insert(node_id);
        }
    }

    update_fused_estimates(result, fused_pairs);
    result.success = true;
    return result;
}
```

### Phase 7: Update Memory Estimates

```cpp
void KernelGraph::update_fused_estimates(
    KernelGraphCompileResult& result,
    const std::vector<std::pair<size_t, size_t>>& fused_pairs) const {

    // Base estimates from all kernels
    Size total_external = 0;
    Size total_flops = 0;
    for (size_t node_id : result.execution_order) {
        const auto& kernel = *get_node(node_id).kernel;
        result.program.estimates.total_cycles += kernel.program().estimates.total_cycles;
        total_external += kernel.program().estimates.external_mem_bytes;
        total_flops += kernel.total_flops();
    }

    // Calculate savings: 2 * intermediate_size per fused pair
    Size bytes_saved = 0;
    for (const auto& [prod_id, cons_id] : fused_pairs) {
        const auto& producer = *get_node(prod_id).kernel;
        bytes_saved += 2 * producer.M() * producer.N() * dtype_size(producer.dtype());
    }

    result.program.estimates.external_mem_bytes = total_external - bytes_saved;
    result.program.estimates.arithmetic_intensity =
        static_cast<double>(total_flops) / result.program.estimates.external_mem_bytes;
}
```

## Files to Modify

1. **`include/sw/kpu/kernel_graph.hpp`** - Add FusionAnalysis struct and method declarations
2. **`src/simulator/kernel_graph.cpp`** - Implement fusion logic (~200 lines)
3. **`tests/driver/test_kernel_graph.cpp`** - Add fusion verification tests

## Test Strategy

**File:** `tests/driver/test_kernel_graph.cpp`

```cpp
TEST_CASE("True kernel fusion", "[kernel_graph][fusion]") {
    KernelGraph graph;
    size_t k1 = graph.add_kernel(Kernel::create_matmul(64, 128, 64), "layer1");
    size_t k2 = graph.add_kernel(Kernel::create_matmul(64, 256, 128), "layer2");
    graph.add_edge(k1, k2, "C", "A");

    SECTION("Fused has fewer instructions") {
        auto seq = graph.compile_sequential();
        KernelGraphCompileOptions opts;
        opts.fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
        auto fused = graph.compile(opts);

        REQUIRE(fused.program.instructions.size() < seq.program.instructions.size());
    }

    SECTION("Fused has reduced memory traffic") {
        auto seq = graph.compile_sequential();
        KernelGraphCompileOptions opts;
        opts.fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
        auto fused = graph.compile(opts);

        Size expected_savings = 2 * 64 * 128 * 4;  // 65536 bytes
        Size actual_savings = seq.program.estimates.external_mem_bytes -
                             fused.program.estimates.external_mem_bytes;
        REQUIRE(actual_savings >= expected_savings * 0.8);
    }

    SECTION("No DMA_STORE for producer C") {
        KernelGraphCompileOptions opts;
        opts.fusion_strategy = FusionStrategy::PRODUCER_CONSUMER;
        auto fused = graph.compile(opts);

        // Count should be reduced
        size_t c_stores = 0;
        for (const auto& instr : fused.program.instructions) {
            if (instr.opcode == isa::DMOpcode::DMA_STORE_TILE) {
                const auto& ops = std::get<isa::DMAOperands>(instr.operands);
                if (ops.matrix == isa::MatrixID::C) c_stores++;
            }
        }
        // Only consumer's C should be stored
        auto seq = graph.compile_sequential();
        size_t seq_c_stores = 0;
        for (const auto& instr : seq.program.instructions) {
            if (instr.opcode == isa::DMOpcode::DMA_STORE_TILE) {
                const auto& ops = std::get<isa::DMAOperands>(instr.operands);
                if (ops.matrix == isa::MatrixID::C) seq_c_stores++;
            }
        }
        REQUIRE(c_stores < seq_c_stores);
    }
}
```

## Verification

1. **Build:** `cmake --build --preset release`
2. **Run tests:** `ctest --preset default -R kernel_graph`
3. **Verify metrics:**
   - Fused instruction count < sequential
   - External memory bytes reduced by ~2×M×N×dtype_size per pair
   - Arithmetic intensity increased

## Expected Results

| Fused Pair Size | Memory Savings |
|-----------------|----------------|
| 64×128 FP32 | 64 KB per pair |
| 256×256 FP32 | 512 KB per pair |
| 1024×1024 FP32 | 8 MB per pair |

Arithmetic intensity improvement: 1.5-4× depending on chain length.

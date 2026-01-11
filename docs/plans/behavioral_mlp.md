# Behavioral MLP Execution Plan

## Overview

Implement a behavioral (functional) simulation path that executes actual matrix multiplication and MLP inference on the KPU simulator. This enables validation of data flow and correctness before temporal (cycle-accurate) simulation.

**Key Insight from User:** The behavioral model skips L1 (L1 transforms rows/columns into temporal streams - that's temporal model territory). Focus on: Host → L3 → L2 → Compute → L2 → L3 → Host.

## Architecture

```
Host Memory (input matrices A, B, weights, bias)
        │
       DMA  ──────────────────┐
        │                     │
        ▼                     ▼
    L3 Tile[0]            L3 Tile[1]     (on-chip SRAM cache)
        │                     │
   BlockMover            BlockMover
        │                     │
        ▼                     ▼
    L2 Bank[0]            L2 Bank[1]     (operand buffers)
        │                     │
        └────────┬────────────┘
                 │
                 ▼
         Compute Fabric                   (matmul: C = A @ B)
                 │
                 ▼
         Vector Engine                    (bias + activation)
                 │
                 ▼
            L2 Bank[2]                    (result buffer)
                 │
            BlockMover
                 │
                 ▼
            L3 Tile[2]
                 │
                DMA
                 │
                 ▼
           Host Memory                    (output C)
```

## Existing Components (Ready to Use)

| Component | Location | Status |
|-----------|----------|--------|
| BehavioralMemoryModel | `include/sw/kpu/behavioral/memory_model.hpp` | Ready |
| BehavioralL3Tile | `include/sw/kpu/behavioral/l3_tile.hpp` | Ready |
| BehavioralDMAEngine | `include/sw/kpu/behavioral/dma_engine.hpp` | Ready |
| BehavioralComputeFabric | `include/sw/kpu/behavioral/compute_fabric.hpp` | Ready |
| SFU (activations) | `include/sw/kpu/components/sfu.hpp` | Ready |

## New Components to Implement

### Phase 1: BehavioralBlockMover

**Purpose:** Transfer tile data between L3 and L2 with optional transpose.

**File:** `include/sw/kpu/behavioral/block_mover.hpp`

```cpp
class BehavioralBlockMover {
public:
    struct TransferDescriptor {
        uint8_t src_region_type;   // L3 or L2
        uint8_t src_region_id;     // tile_id or bank_id
        uint64_t src_offset;
        uint8_t dst_region_type;
        uint8_t dst_region_id;
        uint64_t dst_offset;
        uint32_t height, width;    // tile dimensions
        uint32_t element_size;
        uint32_t src_stride;       // 0 = contiguous
        uint32_t dst_stride;
        bool transpose;
    };

    void transfer(const TransferDescriptor& desc, BehavioralMemoryModel* memory);

    // Convenience methods
    void l3_to_l2(uint8_t l3_tile, uint64_t l3_off,
                  uint8_t l2_bank, uint64_t l2_off,
                  uint32_t h, uint32_t w, uint32_t elem_size,
                  BehavioralMemoryModel* memory, bool transpose = false);

    void l2_to_l3(uint8_t l2_bank, uint64_t l2_off,
                  uint8_t l3_tile, uint64_t l3_off,
                  uint32_t h, uint32_t w, uint32_t elem_size,
                  BehavioralMemoryModel* memory);

    // Statistics
    struct Stats { uint64_t transfers, bytes_moved; };
    const Stats& stats() const;
};
```

**Implementation:** `src/models/behavioral/datamovement/block_mover.cpp`
- Use `BehavioralMemoryModel::copy()` for identity transfers
- Implement transpose as element-by-element with index remapping
- Track statistics

---

### Phase 2: BehavioralVectorEngine

**Purpose:** Apply bias addition and activation functions.

**File:** `include/sw/kpu/behavioral/vector_engine.hpp`

```cpp
class BehavioralVectorEngine {
public:
    // Process tile in L2: output[i,j] = activation(input[i,j] + bias[j])
    void apply_bias_activation(
        BehavioralMemoryModel* memory,
        uint64_t data_addr,         // L2 address of [height, width] tile
        uint32_t height, uint32_t width,
        const float* bias,          // [width] or nullptr
        ActivationType activation);

    // Direct in-place processing (for testing)
    void process_inplace(float* data, uint32_t count,
                         const float* bias, ActivationType activation);

private:
    SFU sfu_;  // Reuse existing activation implementation
};
```

**Implementation:** `src/models/behavioral/compute/vector_engine.cpp`
- Reuse `SFU::evaluate()` for activation functions
- Bias is row-broadcast: each row gets the same bias vector added

---

### Phase 3: BehavioralOrchestrator

**Purpose:** Coordinate the complete data flow pipeline.

**File:** `include/sw/kpu/behavioral/orchestrator.hpp`

```cpp
class BehavioralOrchestrator {
public:
    struct Config {
        BehavioralMemoryModel* memory;
        BehavioralComputeFabric* compute;
        BehavioralDMAEngine* dma;
        BehavioralBlockMover* block_mover;
        BehavioralVectorEngine* vector_engine;

        // Resource counts
        uint8_t num_l3_tiles = 4;
        uint8_t num_l2_banks = 8;
    };

    explicit BehavioralOrchestrator(const Config& config);

    // Execute matmul: C[M,N] = A[M,K] @ B[K,N]
    void execute_matmul(
        uint64_t host_a, uint32_t m, uint32_t k,
        uint64_t host_b, uint32_t n,
        uint64_t host_c,
        bool accumulate = false);

    // Execute MLP layer: C = activation(A @ B + bias)
    void execute_mlp_layer(
        uint64_t host_input, uint32_t batch, uint32_t in_dim,
        uint64_t host_weights, uint32_t out_dim,
        uint64_t host_bias,     // 0 = no bias
        uint64_t host_output,
        ActivationType activation);

    // Tiled execution for large matrices
    void execute_tiled_matmul(
        uint64_t host_a, uint32_t m, uint32_t k,
        uint64_t host_b, uint32_t n,
        uint64_t host_c,
        uint32_t tile_m, uint32_t tile_n, uint32_t tile_k);

    struct Stats {
        uint64_t dma_bytes;
        uint64_t block_mover_bytes;
        uint64_t compute_flops;
        uint64_t matmul_count;
    };
    const Stats& stats() const;

private:
    Config config_;
    Stats stats_;

    // L3/L2 allocation tracking
    uint64_t l3_next_offset_[16] = {};
    uint64_t l2_next_offset_[8] = {};

    uint64_t alloc_l3(uint8_t tile_id, uint32_t bytes);
    uint64_t alloc_l2(uint8_t bank_id, uint32_t bytes);
    void reset_allocations();
};
```

**Implementation:** `src/models/behavioral/orchestrator.cpp`

Key execution flow:
```cpp
void BehavioralOrchestrator::execute_matmul(...) {
    reset_allocations();

    // 1. DMA: Host -> L3
    uint64_t l3_a = alloc_l3(0, m * k * sizeof(float));
    uint64_t l3_b = alloc_l3(1, k * n * sizeof(float));
    dma_->submit({HOST, 0, host_a, L3, 0, l3_a, m * k * sizeof(float)});
    dma_->submit({HOST, 0, host_b, L3, 1, l3_b, k * n * sizeof(float)});
    dma_->drain();

    // 2. BlockMover: L3 -> L2
    uint64_t l2_a = alloc_l2(0, m * k * sizeof(float));
    uint64_t l2_b = alloc_l2(1, k * n * sizeof(float));
    block_mover_->l3_to_l2(0, l3_a, 0, l2_a, m, k, sizeof(float), memory_);
    block_mover_->l3_to_l2(1, l3_b, 1, l2_b, k, n, sizeof(float), memory_);

    // 3. Compute: matmul
    uint64_t l2_c = alloc_l2(2, m * n * sizeof(float));
    float* a_ptr = memory_->get_ptr<float>(l2_a);
    float* b_ptr = memory_->get_ptr<float>(l2_b);
    float* c_ptr = memory_->get_ptr<float>(l2_c);

    MatMulDescriptor desc{m, n, k, l2_a, l2_b, l2_c, sizeof(float), accumulate};
    compute_->submit_matmul(desc, a_ptr, b_ptr, c_ptr);
    compute_->drain();

    // 4. BlockMover: L2 -> L3
    uint64_t l3_c = alloc_l3(2, m * n * sizeof(float));
    block_mover_->l2_to_l3(2, l2_c, 2, l3_c, m, n, sizeof(float), memory_);

    // 5. DMA: L3 -> Host
    dma_->submit({L3, 2, l3_c, HOST, 0, host_c, m * n * sizeof(float)});
    dma_->drain();

    stats_.compute_flops += 2ULL * m * n * k;
    stats_.matmul_count++;
}
```

---

### Phase 4: BehavioralMLPExecutor

**Purpose:** Execute complete multi-layer MLP networks.

**File:** `include/sw/kpu/behavioral/mlp_executor.hpp`

```cpp
class BehavioralMLPExecutor {
public:
    struct Layer {
        uint32_t input_dim;
        uint32_t output_dim;
        ActivationType activation;
        bool use_bias;
        std::vector<float> weights;  // [input_dim, output_dim]
        std::vector<float> bias;     // [output_dim]
    };

    struct Network {
        std::vector<Layer> layers;
        uint32_t batch_size;
    };

    explicit BehavioralMLPExecutor(BehavioralOrchestrator* orchestrator);

    // Load network weights into host memory region
    void load_network(const Network& network);

    // Forward pass
    void forward(const std::vector<float>& input, std::vector<float>& output);

    // Forward pass with pre-allocated host addresses
    void forward(uint64_t input_addr, uint64_t output_addr);

    struct Stats {
        uint64_t total_flops;
        uint64_t total_bytes;
        std::vector<uint64_t> layer_flops;
    };
    Stats get_stats() const;

private:
    BehavioralOrchestrator* orchestrator_;
    Network network_;

    // Host memory addresses for weights/biases
    std::vector<uint64_t> weight_addrs_;
    std::vector<uint64_t> bias_addrs_;

    // Intermediate activation buffers
    std::vector<uint64_t> activation_addrs_;
};
```

**Implementation:** `src/models/behavioral/mlp_executor.cpp`

```cpp
void BehavioralMLPExecutor::forward(uint64_t input_addr, uint64_t output_addr) {
    uint64_t current_input = input_addr;

    for (size_t i = 0; i < network_.layers.size(); ++i) {
        const auto& layer = network_.layers[i];

        // Output goes to final buffer or intermediate
        uint64_t current_output = (i == network_.layers.size() - 1)
            ? output_addr
            : activation_addrs_[i];

        // Execute: output = activation(input @ weights + bias)
        orchestrator_->execute_mlp_layer(
            current_input, network_.batch_size, layer.input_dim,
            weight_addrs_[i], layer.output_dim,
            layer.use_bias ? bias_addrs_[i] : 0,
            current_output,
            layer.activation);

        current_input = current_output;
    }
}
```

---

## Example Programs

### examples/behavioral/xor_behavioral.cpp

```cpp
// XOR classifier: 2 -> 4 (ReLU) -> 1 (linear)
// Validates behavioral execution produces correct results

int main() {
    // Setup behavioral components
    BehavioralMemoryModel memory(default_config());
    BehavioralComputeFabric compute({});
    BehavioralDMAEngine dma({});
    BehavioralBlockMover block_mover;
    BehavioralVectorEngine vector_engine;

    BehavioralOrchestrator orchestrator({
        &memory, &compute, &dma, &block_mover, &vector_engine
    });

    // XOR network with pre-trained weights
    BehavioralMLPExecutor::Network network;
    network.batch_size = 4;

    // Hidden: 2 -> 4, ReLU
    network.layers.push_back({
        .input_dim = 2, .output_dim = 4,
        .activation = ActivationType::RELU, .use_bias = true,
        .weights = {1,1, 1,1, -1,-1, -1,-1},  // XOR pattern
        .bias = {0, -1, 0, -1}
    });

    // Output: 4 -> 1, linear
    network.layers.push_back({
        .input_dim = 4, .output_dim = 1,
        .activation = ActivationType::NONE, .use_bias = true,
        .weights = {1, -2, -2, 1},
        .bias = {0}
    });

    BehavioralMLPExecutor executor(&orchestrator);
    executor.load_network(network);

    // Input: all 4 XOR cases
    std::vector<float> input = {0,0, 0,1, 1,0, 1,1};
    std::vector<float> output(4);

    executor.forward(input, output);

    // Verify: expected outputs are 0, 1, 1, 0
    float expected[] = {0, 1, 1, 0};
    bool all_correct = true;
    for (int i = 0; i < 4; ++i) {
        bool pred = output[i] > 0.5f;
        bool exp = expected[i] > 0.5f;
        std::cout << "XOR(" << input[i*2] << "," << input[i*2+1]
                  << ") = " << output[i] << (pred == exp ? " OK" : " FAIL") << "\n";
        all_correct &= (pred == exp);
    }

    return all_correct ? 0 : 1;
}
```

### examples/behavioral/matmul_behavioral.cpp

```cpp
// Simple matmul test: C[64,64] = A[64,128] @ B[128,64]

int main() {
    // Setup
    BehavioralMemoryModel memory(default_config());
    // ... create components ...
    BehavioralOrchestrator orchestrator({...});

    const uint32_t M = 64, K = 128, N = 64;

    // Allocate host memory
    auto a_alloc = memory.allocate_host(M * K * sizeof(float), "A");
    auto b_alloc = memory.allocate_host(K * N * sizeof(float), "B");
    auto c_alloc = memory.allocate_host(M * N * sizeof(float), "C");

    // Initialize: A=1, B=1 => C should be K (128)
    std::vector<float> ones_a(M * K, 1.0f);
    std::vector<float> ones_b(K * N, 1.0f);
    memory.write_floats(a_alloc.address, ones_a);
    memory.write_floats(b_alloc.address, ones_b);

    // Execute
    orchestrator.execute_matmul(
        a_alloc.address, M, K,
        b_alloc.address, N,
        c_alloc.address);

    // Verify: all elements should be K
    auto c_data = memory.read_floats(c_alloc.address, M * N);
    bool correct = true;
    for (size_t i = 0; i < M * N; ++i) {
        if (std::abs(c_data[i] - float(K)) > 1e-5f) {
            std::cout << "Error at C[" << i << "]: " << c_data[i]
                      << " != " << K << "\n";
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "PASS" : "FAIL") << "\n";
    return correct ? 0 : 1;
}
```

---

## File Structure

```
include/sw/kpu/behavioral/
├── memory_model.hpp          # (exists)
├── l3_tile.hpp               # (exists)
├── dma_engine.hpp            # (exists)
├── compute_fabric.hpp        # (exists)
├── block_mover.hpp           # NEW
├── vector_engine.hpp         # NEW
├── orchestrator.hpp          # NEW
└── mlp_executor.hpp          # NEW

src/models/behavioral/
├── memory/
│   └── memory_model.cpp      # (exists)
├── datamovement/
│   ├── dma_engine.cpp        # (exists)
│   └── block_mover.cpp       # NEW
├── compute/
│   ├── compute_fabric.cpp    # (exists)
│   └── vector_engine.cpp     # NEW
├── orchestrator.cpp          # NEW
└── mlp_executor.cpp          # NEW

src/models/behavioral/CMakeLists.txt  # UPDATE

examples/behavioral/
├── CMakeLists.txt            # NEW
├── matmul_behavioral.cpp     # NEW
├── xor_behavioral.cpp        # NEW
└── mlp_mnist.cpp             # NEW (optional, larger example)
```

---

## CMakeLists.txt Updates

### src/models/behavioral/CMakeLists.txt

```cmake
set(BEHAVIORAL_SOURCES
    memory/memory_model.cpp
    memory/l3_tile.cpp
    memory/memory_controller.cpp
    datamovement/dma_engine.cpp
    datamovement/block_mover.cpp      # NEW
    compute/compute_fabric.cpp
    compute/vector_engine.cpp         # NEW
    noc/noc.cpp
    orchestrator.cpp                  # NEW
    mlp_executor.cpp                  # NEW
)

add_library(kpu_behavioral ${BEHAVIORAL_SOURCES})
# ... existing configuration ...
```

### examples/behavioral/CMakeLists.txt

```cmake
add_executable(matmul_behavioral matmul_behavioral.cpp)
target_link_libraries(matmul_behavioral PRIVATE kpu_behavioral)

add_executable(xor_behavioral xor_behavioral.cpp)
target_link_libraries(xor_behavioral PRIVATE kpu_behavioral)

# Tests
add_test(NAME behavioral_matmul COMMAND matmul_behavioral)
add_test(NAME behavioral_xor COMMAND xor_behavioral)
```

---

## Implementation Order

| Phase | Component | Dependencies | Effort |
|-------|-----------|--------------|--------|
| 1 | BehavioralBlockMover | BehavioralMemoryModel | Small |
| 2 | BehavioralVectorEngine | SFU | Small |
| 3 | BehavioralOrchestrator | Phases 1-2, existing components | Medium |
| 4 | BehavioralMLPExecutor | Phase 3 | Small |
| 5 | Examples & Tests | Phase 4 | Small |

---

## Verification Plan

1. **Unit Tests:**
   - BlockMover: verify copy and transpose
   - VectorEngine: verify bias+activation for each activation type
   - Orchestrator: verify single matmul produces correct results

2. **Integration Tests:**
   - `matmul_behavioral`: C = A @ B where A=1, B=1, verify C=K
   - `xor_behavioral`: verify XOR network produces 0,1,1,0

3. **Build:**
   ```bash
   cmake --build --preset release
   ctest --preset release -R behavioral
   ```

4. **Run Examples:**
   ```bash
   ./build/examples/behavioral/matmul_behavioral
   ./build/examples/behavioral/xor_behavioral
   ```

---

## Key Design Decisions

1. **Skip L1:** Behavioral model operates on L2 data directly. L1's role is temporal stream generation (systolic array feeding) - that's for the temporal model.

2. **Reuse SFU:** The existing `SFU` class provides LUT-based activation evaluation. BehavioralVectorEngine wraps it with bias broadcast.

3. **Stateless Orchestrator:** Each execution is self-contained. Allocations are reset between calls.

4. **Direct Pointer Access:** BehavioralComputeFabric uses `get_ptr<float>()` for actual computation, not addresses.

5. **Layer-by-Layer MLP:** Sequential execution enables per-layer profiling and matches inference behavior.

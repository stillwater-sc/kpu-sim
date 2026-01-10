# KPU Software API: Gaps Analysis and DNN Execution Roadmap

## Executive Summary

This document provides a detailed analysis of gaps in the KPU software stack required to execute full DNNs on the kpu-sim. The goal is to progress from:

1. **Single layer MLP** (current capability)
2. **Multi-layer MLP** (immediate target)
3. **Full torchvision DNN** (e.g., SqueezeNet 1.0, MobileNetV2)

> **Important: Timing Model vs Functional Simulation**
>
> The KPU simulator is currently a **timing/performance model**, not a functional simulator.
> It accurately models cycle counts, memory latencies, and resource contention, but does NOT
> compute actual matrix multiplication results. For a detailed analysis of what's needed for
> functional simulation, see [FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md](FUNCTIONAL_SIMULATION_GAP_ANALYSIS.md).

## Current API Status

### Production-Ready Components ✅

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **Runtime API** | `include/sw/runtime/runtime.hpp` | ✅ Complete | CUDA-like API: malloc/free, memcpy, launch, streams, events |
| **GraphExecutor** | `include/sw/runtime/executor.hpp` | ✅ Complete | High-level tensor I/O with automatic memory management |
| **Resource Manager** | `include/sw/kpu/resource_api.hpp` | ✅ Complete | Memory allocation, buffer management, resource tracking |
| **Kernel Compiler** | `include/sw/compiler/kernel_compiler.hpp` | ✅ Complete | Tile optimization, program generation |
| **ConcurrentExecutor** | `include/sw/kpu/isa/concurrent_executor.hpp` | ✅ Complete | Cycle-accurate multi-resource execution model |
| **Data Movement ISA** | `include/sw/kpu/isa/data_movement_isa.hpp` | ✅ Complete | 20+ opcodes for DMA, BlockMover, Streamer operations |
| **C API** | `src/bindings/c/kpu_c_api.h` | ✅ Complete | Full C bindings for all major functionality |
| **SFU (Activations)** | `include/sw/kpu/components/sfu.hpp` | ✅ Complete | LUT-based activations: ReLU, GELU, Sigmoid, Tanh, SiLU, Softplus, LeakyReLU |

### Supported Kernel Types

| Kernel | Status | Details |
|--------|--------|---------|
| `MATMUL` | ✅ Complete | C = A × B with automatic tiling, tile caching |
| `MLP` | ✅ Complete | C = activation(A × B + bias), fused operation |
| `BATCH_MATMUL` | ⚠️ Defined | Enum exists, no implementation |
| `CONV2D` | ⚠️ Defined | Enum exists, no implementation |
| `ELEMENTWISE` | ⚠️ Defined | Enum exists, no implementation |

### Supported Data Types

| Type | Size | Status |
|------|------|--------|
| FLOAT32 | 4B | ✅ Full support |
| FLOAT16 | 2B | ✅ Defined, compute support pending |
| BFLOAT16 | 2B | ✅ Defined, compute support pending |
| INT32 | 4B | ✅ Accumulator support |
| INT8 | 1B | ✅ Defined, quantized inference pending |
| UINT8 | 1B | ✅ Defined |
| INT4 | 0.5B | ✅ Defined, packed format |

---

## Gap Analysis: What's Missing

### 1. Multi-Layer Graph Execution 🔴 Critical

**Current limitation:** GraphExecutor can only run a single kernel at a time. There's no graph scheduling for multi-layer networks.

**Required additions:**

```cpp
// include/sw/runtime/graph.hpp (NEW FILE)

namespace sw::runtime {

/**
 * @brief Computational graph node
 */
struct GraphNode {
    std::string name;
    Kernel kernel;
    std::vector<std::string> inputs;   // Names of input tensors
    std::vector<std::string> outputs;  // Names of output tensors
    std::vector<size_t> dependencies;  // Node indices this depends on
};

/**
 * @brief Multi-layer computational graph
 */
class ComputeGraph {
public:
    // Graph construction
    size_t add_node(const std::string& name, const Kernel& kernel,
                    const std::vector<std::string>& inputs,
                    const std::vector<std::string>& outputs);

    // Tensor management
    void set_input(const std::string& name, const std::vector<Size>& shape);
    void set_output(const std::string& name);

    // Analysis
    std::vector<size_t> topological_order() const;
    Size total_memory_bytes() const;
    Size peak_memory_bytes() const;

    // Serialization
    void save(const std::string& path) const;
    static ComputeGraph load(const std::string& path);
};

/**
 * @brief Executor for multi-layer graphs
 */
class GraphRunner {
public:
    explicit GraphRunner(KPURuntime* runtime);

    void set_graph(const ComputeGraph& graph);

    // Automatic memory planning
    void allocate_memory();

    // Execution
    ExecutionResult run();

    // Per-layer profiling
    struct LayerProfile {
        std::string name;
        Cycle cycles;
        double time_ms;
    };
    std::vector<LayerProfile> get_layer_profiles() const;
};

} // namespace sw::runtime
```

**Files to create:**
- `include/sw/runtime/graph.hpp` - Graph representation
- `src/runtime/graph.cpp` - Graph implementation

**Estimated effort:** 2-3 days

---

### 2. Conv2D Kernel 🔴 Critical

**Current limitation:** Conv2D is declared in `KernelOpType` but not implemented.

**Required additions:**

```cpp
// In include/sw/kpu/kernel.hpp

/**
 * @brief Create a 2D convolution kernel
 * @param batch_size N - batch dimension
 * @param in_height H - input height
 * @param in_width W - input width
 * @param in_channels C_in - input channels
 * @param out_channels C_out - output channels (number of filters)
 * @param kernel_h Filter height
 * @param kernel_w Filter width
 * @param stride_h Vertical stride
 * @param stride_w Horizontal stride
 * @param pad_h Vertical padding
 * @param pad_w Horizontal padding
 * @param activation Fused activation function
 * @param has_bias Whether to apply bias
 * @param dtype Data type
 */
static Kernel create_conv2d(
    Size batch_size, Size in_height, Size in_width, Size in_channels,
    Size out_channels, Size kernel_h, Size kernel_w,
    Size stride_h = 1, Size stride_w = 1,
    Size pad_h = 0, Size pad_w = 0,
    ActivationType activation = ActivationType::NONE,
    bool has_bias = true,
    DataType dtype = DataType::FLOAT32);
```

**Implementation strategy:**

Conv2D on systolic array typically uses im2col transformation:
1. Transform input [N, H, W, C_in] → [N*H_out*W_out, C_in*K_h*K_w]
2. Perform GEMM: [N*H_out*W_out, C_in*K_h*K_w] × [C_in*K_h*K_w, C_out]
3. Apply bias + activation
4. Reshape output to [N, H_out, W_out, C_out]

**Files to modify/create:**
- `include/sw/kpu/kernel.hpp` - Add factory method
- `src/kpu/kernel.cpp` - Add implementation
- `include/sw/compiler/kernel_compiler.hpp` - Add compile_conv2d()
- `src/compiler/kernel_compiler.cpp` - Conv2D compilation logic

**Alternative: Direct convolution schedule**

For small kernels (1×1, 3×3), direct convolution may be more efficient:
```cpp
// In data_movement_isa.hpp
DMOpcode::CONV_SLIDE,        // Sliding window data movement
DMOpcode::CONV_ACCUMULATE,   // Accumulate partial products
```

**Estimated effort:** 5-7 days

---

### 3. Pooling Operations 🟡 Important

**Current limitation:** No pooling (MaxPool, AvgPool) operations.

**Required additions:**

```cpp
// In include/sw/kpu/kernel.hpp

enum class PoolType : uint8_t {
    MAX_POOL = 0,
    AVG_POOL = 1,
    GLOBAL_AVG_POOL = 2,
    ADAPTIVE_AVG_POOL = 3
};

/**
 * @brief Create a 2D pooling kernel
 */
static Kernel create_pool2d(
    Size batch_size, Size height, Size width, Size channels,
    PoolType pool_type,
    Size kernel_h, Size kernel_w,
    Size stride_h = 1, Size stride_w = 1,
    DataType dtype = DataType::FLOAT32);
```

**Implementation notes:**

Pooling is typically memory-bound and can be fused with adjacent convolutions. The Vector Engine could handle this inline during drain operations.

**Files to modify:**
- `include/sw/kpu/kernel.hpp` - Add factory method
- `include/sw/kpu/components/sfu.hpp` - Add pooling to Vector Engine
- ISA may need new opcodes for strided reads

**Estimated effort:** 2-3 days

---

### 4. Batch Normalization 🟡 Important

**Current limitation:** No batch normalization support.

**Required additions:**

```cpp
// Can be fused into activation path
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Pre-compute: scale = gamma / sqrt(var + eps), shift = beta - mean * scale
// Runtime: y = x * scale + shift (simple affine transform)

/**
 * @brief Create a batch normalization kernel
 * For inference, BN is a simple affine transform: y = x * scale + bias
 */
static Kernel create_batch_norm(
    Size batch_size, Size channels, Size height, Size width,
    DataType dtype = DataType::FLOAT32);
```

**Implementation strategy:**

For inference, batch norm can be folded into the preceding convolution:
- `W' = W * scale`
- `b' = b * scale + shift`

This is a compiler optimization, not a runtime operation.

**Files to create:**
- `include/sw/compiler/bn_folding.hpp` - BN folding utility

**Estimated effort:** 1 day

---

### 5. Elementwise Operations 🟡 Important

**Current limitation:** No standalone elementwise operations (add, multiply, residual connections).

**Required additions:**

```cpp
// In include/sw/kpu/kernel.hpp

enum class ElementwiseOp : uint8_t {
    ADD = 0,        // C = A + B
    SUB = 1,        // C = A - B
    MUL = 2,        // C = A * B
    DIV = 3,        // C = A / B
    SCALE = 4,      // C = A * scalar
    ADD_SCALAR = 5, // C = A + scalar
    RESIDUAL = 6    // C = A + B (same as ADD, semantically for skip connections)
};

/**
 * @brief Create an elementwise kernel
 */
static Kernel create_elementwise(
    ElementwiseOp op,
    const std::vector<Size>& shape,
    DataType dtype = DataType::FLOAT32);
```

**Files to modify:**
- `include/sw/kpu/kernel.hpp` - Add factory method
- Vector Engine can handle simple element-wise ops inline

**Estimated effort:** 2 days

---

### 6. Softmax Operation 🟡 Important

**Current limitation:** No softmax (required for classification output).

**Required additions:**

```cpp
// In include/sw/kpu/components/sfu.hpp

enum class ActivationType : uint8_t {
    // ... existing ...
    SOFTMAX = 8,        // exp(x) / sum(exp(x)) - needs reduction
};

// Or as standalone kernel:
static Kernel create_softmax(
    Size batch_size, Size num_classes,
    DataType dtype = DataType::FLOAT32);
```

**Implementation notes:**

Softmax requires a reduction operation across the class dimension:
1. Find max(x) for numerical stability
2. Compute exp(x - max)
3. Compute sum of exp values
4. Divide each element by sum

This is more complex than other activations and may need dedicated ISA support.

**Files to modify:**
- `include/sw/kpu/components/sfu.hpp` - Add softmax
- Possibly new ISA opcodes for reduction

**Estimated effort:** 2-3 days

---

### 7. Reshape/Flatten Operations 🟢 Nice to Have

**Current limitation:** No tensor reshape operations.

**Required additions:**

```cpp
/**
 * @brief Create a reshape kernel (logical reshape, no data movement)
 * For Flatten before FC layers: [N, H, W, C] → [N, H*W*C]
 */
static Kernel create_reshape(
    const std::vector<Size>& input_shape,
    const std::vector<Size>& output_shape,
    DataType dtype = DataType::FLOAT32);
```

**Implementation notes:**

Reshape is typically a no-op in terms of data movement if tensors are contiguous. The runtime just needs to track the logical shape.

**Estimated effort:** 0.5 days

---

### 8. Concatenation 🟢 Nice to Have

**Current limitation:** No tensor concatenation (needed for some architectures).

```cpp
/**
 * @brief Create a concatenation kernel
 */
static Kernel create_concat(
    const std::vector<std::vector<Size>>& input_shapes,
    Size concat_dim,
    DataType dtype = DataType::FLOAT32);
```

**Estimated effort:** 1 day

---

### 9. ONNX/PyTorch Model Loading 🔴 Critical for DNN

**Current limitation:** No way to load trained models.

**Required additions:**

```cpp
// include/sw/compiler/model_loader.hpp (NEW FILE)

namespace sw::compiler {

/**
 * @brief Load a computational graph from ONNX format
 */
class ONNXLoader {
public:
    /**
     * @brief Load ONNX model and create compute graph
     * @param path Path to .onnx file
     * @return ComputeGraph ready for execution
     */
    static runtime::ComputeGraph load(const std::string& path);

    /**
     * @brief Load ONNX model with weight extraction
     * @param path Path to .onnx file
     * @param weights Output map of weight name → data
     */
    static runtime::ComputeGraph load_with_weights(
        const std::string& path,
        std::unordered_map<std::string, std::vector<float>>& weights);
};

/**
 * @brief Load from PyTorch TorchScript format
 */
class TorchScriptLoader {
public:
    static runtime::ComputeGraph load(const std::string& path);
};

} // namespace sw::compiler
```

**Dependencies:**
- ONNX Runtime or protobuf for ONNX parsing
- LibTorch for TorchScript (optional)

**Alternative (simpler):**

Create a custom JSON-based model format:

```json
{
  "format": "kpu-model-v1",
  "inputs": [{"name": "input", "shape": [1, 3, 224, 224]}],
  "outputs": [{"name": "output", "shape": [1, 1000]}],
  "nodes": [
    {
      "name": "conv1",
      "op": "conv2d",
      "params": {"out_channels": 64, "kernel": [7, 7], "stride": 2, "padding": 3},
      "inputs": ["input"],
      "outputs": ["conv1_out"]
    }
  ],
  "weights": "weights.bin"  // Separate binary file
}
```

**Estimated effort:**
- Custom JSON format: 3-4 days
- ONNX loader: 5-7 days

---

### 10. Text Assembler Format 🟢 Nice to Have

**Current limitation:** Programs are built programmatically; no human-readable assembly.

**Required additions:**

```asm
; example.kpu_asm
.program matmul_1024x1024x1024_os
.version 1
.dimensions M=1024, N=1024, K=1024
.tiles Ti=64, Tj=64, Tk=128
.dataflow output_stationary

.section data_movement
    ; Load A[0,0] tile from external memory to L3
    DMA_LOAD_TILE   A, tile(0,0), 0x10000, l3_tile=0, buf=0

    ; Move from L3 to L2
    BM_MOVE_TILE    A, tile(0,0), src_l3=0, dst_l2=0, transform=IDENTITY

    ; Stream to systolic array
    STR_FEED_ROWS   A, tile(0,0), l2_bank=0, l1_buf=0

    BARRIER
    HALT
```

**Files to create:**
- `include/sw/isa/text_assembler.hpp` - Parser
- `src/isa/text_assembler.cpp` - Implementation

**Estimated effort:** 3-4 days

---

## Roadmap: From MLP to Full DNN

### Phase 1: Multi-Layer MLP (Week 1)

**Goal:** Execute 2+ layer MLP with proper memory management.

**Tasks:**
1. ✅ Single MLP layer works (current state)
2. Implement `ComputeGraph` for layer sequencing
3. Implement `GraphRunner` for multi-kernel execution
4. Add memory planning to avoid re-allocation between layers
5. Add per-layer profiling

**Test case:**
```cpp
// 3-layer MLP: 784 → 256 → 128 → 10
auto layer1 = Kernel::create_mlp(batch, 256, 784, ActivationType::RELU, true);
auto layer2 = Kernel::create_mlp(batch, 128, 256, ActivationType::RELU, true);
auto layer3 = Kernel::create_mlp(batch, 10, 128, ActivationType::NONE, true);

ComputeGraph graph;
graph.add_node("fc1", layer1, {"input"}, {"h1"});
graph.add_node("fc2", layer2, {"h1"}, {"h2"});
graph.add_node("fc3", layer3, {"h2"}, {"output"});

GraphRunner runner(&runtime);
runner.set_graph(graph);
runner.allocate_memory();
runner.set_input("input", mnist_batch.data(), {64, 784});
runner.run();
runner.get_output("output", predictions.data());
```

**Success criteria:**
- 3-layer MLP executes correctly
- Memory is properly reused between layers
- Per-layer timing is accurate

---

### Phase 2: Add Conv2D (Week 2)

**Goal:** Execute simple CNN architectures.

**Tasks:**
1. Implement `Kernel::create_conv2d()` using im2col + GEMM
2. Add to `KernelCompiler::compile_conv2d()`
3. Implement basic pooling (MaxPool2D)
4. Add batch norm folding optimization

**Test case:**
```cpp
// Simple CNN: Conv → ReLU → Pool → Flatten → FC
auto conv1 = Kernel::create_conv2d(
    1, 28, 28, 1,    // MNIST: 1×28×28
    32, 3, 3,        // 32 filters, 3×3 kernel
    1, 1, 1, 1,      // stride=1, pad=1
    ActivationType::RELU, true);

auto pool1 = Kernel::create_pool2d(
    1, 28, 28, 32,
    PoolType::MAX_POOL,
    2, 2, 2, 2);     // 2×2 pool, stride 2 → 14×14

auto fc1 = Kernel::create_mlp(1, 10, 14*14*32, ActivationType::NONE, true);
```

**Success criteria:**
- Conv2D produces correct output (validate against PyTorch)
- End-to-end CNN works on MNIST

---

### Phase 3: Residual Connections & More Ops (Week 3)

**Goal:** Support residual architectures like ResNet.

**Tasks:**
1. Implement elementwise add kernel
2. Add skip connection support in `ComputeGraph`
3. Implement global average pooling
4. Add softmax activation

**Test case:**
```cpp
// ResNet-style block
// x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
graph.add_node("conv1", conv1_kernel, {"input"}, {"conv1_out"});
graph.add_node("conv2", conv2_kernel, {"conv1_out"}, {"conv2_out"});
graph.add_node("add", add_kernel, {"conv2_out", "input"}, {"block_out"});
```

**Success criteria:**
- Residual blocks execute correctly
- Memory layout handles skip connections

---

### Phase 4: SqueezeNet 1.0 (Week 4)

**Goal:** Execute first real torchvision model.

**Why SqueezeNet:**
- Small model (~5MB parameters)
- Simple architecture: Conv + Fire modules
- No complex ops (just conv, pool, concat, activation)
- 1000-class ImageNet output

**Architecture summary:**
```
Input [3, 224, 224]
  ↓
Conv1 (96 filters, 7×7, stride 2) → ReLU → MaxPool
  ↓
Fire2 (squeeze: 16, expand: 64+64)
Fire3 (squeeze: 16, expand: 64+64)
Fire4 (squeeze: 32, expand: 128+128) → MaxPool
  ↓
... more Fire modules ...
  ↓
Conv10 (1000 filters, 1×1) → GlobalAvgPool → Softmax
```

**Fire module:**
- Squeeze: 1×1 conv (channel reduction)
- Expand: 1×1 conv + 3×3 conv (parallel), then concatenate

**Tasks:**
1. Implement model loader (JSON or ONNX)
2. Export SqueezeNet weights from PyTorch
3. Implement concat operation
4. End-to-end validation

**Validation:**
```python
# PyTorch reference
model = torchvision.models.squeezenet1_0(pretrained=True)
model.eval()
with torch.no_grad():
    ref_output = model(input_tensor)

# KPU execution
kpu_output = run_on_kpu(input_tensor, "squeezenet1_0.json")

# Compare
assert torch.allclose(ref_output, kpu_output, rtol=1e-3, atol=1e-5)
```

**Success criteria:**
- SqueezeNet executes on kpu-sim
- Output matches PyTorch reference within tolerance
- Performance metrics collected

---

### Phase 5: MobileNetV2 (Week 5+)

**Goal:** Execute more efficient architecture with depthwise separable convolutions.

**Why MobileNetV2:**
- Lightweight (~3.4M parameters)
- Uses depthwise separable convolutions (efficient)
- Inverted residual blocks
- Better accuracy than SqueezeNet

**Additional ops needed:**
- Depthwise convolution (special case of grouped conv where groups = channels)
- Inverted residual (expand → depthwise → project)

**Tasks:**
1. Implement depthwise convolution kernel
2. Handle channel expansion/projection
3. Export MobileNetV2 weights
4. End-to-end validation

---

## Priority Summary

| Priority | Component | Effort | Impact |
|----------|-----------|--------|--------|
| 🔴 P0 | Multi-layer graph execution | 3 days | Enables any multi-layer network |
| 🔴 P0 | Conv2D kernel | 5 days | Enables CNNs |
| 🔴 P0 | Model loader (JSON) | 3 days | Enables loading pretrained models |
| 🟡 P1 | Pooling operations | 2 days | Required for most CNNs |
| 🟡 P1 | Softmax | 2 days | Required for classification |
| 🟡 P1 | Elementwise add | 1 day | Required for residual networks |
| 🟡 P1 | BN folding | 1 day | Optimization for inference |
| 🟢 P2 | Concat | 1 day | Fire modules, Inception |
| 🟢 P2 | Reshape/Flatten | 0.5 days | CNN → FC transition |
| 🟢 P2 | Text assembler | 3 days | Developer convenience |

**Total estimated effort to SqueezeNet:** ~3-4 weeks

---

## Appendix A: Smallest Torchvision Models

| Model | Params | Top-1 Acc | Complexity | Notes |
|-------|--------|-----------|------------|-------|
| **SqueezeNet 1.0** | 1.2M | 58.1% | Simple | Fire modules only |
| **SqueezeNet 1.1** | 1.2M | 58.2% | Simple | Optimized SqueezeNet |
| **MobileNet V2** | 3.4M | 72.0% | Medium | Depthwise conv needed |
| **MobileNet V3 Small** | 2.5M | 67.7% | Medium | Squeeze-excite blocks |
| **ShuffleNet V2 x0.5** | 1.4M | 60.6% | Medium | Channel shuffle |
| **EfficientNet B0** | 5.3M | 77.7% | Complex | Many ops |

**Recommendation:** Start with SqueezeNet 1.0, then MobileNet V2.

---

## Appendix B: Example Multi-Layer Graph JSON

```json
{
  "name": "simple_cnn",
  "version": 1,
  "inputs": {
    "input": {"shape": [1, 1, 28, 28], "dtype": "float32"}
  },
  "outputs": ["output"],
  "nodes": [
    {
      "name": "conv1",
      "op": "conv2d",
      "inputs": ["input"],
      "outputs": ["conv1_out"],
      "params": {
        "out_channels": 32,
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1],
        "activation": "relu",
        "bias": true
      },
      "weights": ["conv1.weight", "conv1.bias"]
    },
    {
      "name": "pool1",
      "op": "max_pool2d",
      "inputs": ["conv1_out"],
      "outputs": ["pool1_out"],
      "params": {
        "kernel_size": [2, 2],
        "stride": [2, 2]
      }
    },
    {
      "name": "flatten",
      "op": "flatten",
      "inputs": ["pool1_out"],
      "outputs": ["flat_out"],
      "params": {"start_dim": 1}
    },
    {
      "name": "fc1",
      "op": "linear",
      "inputs": ["flat_out"],
      "outputs": ["output"],
      "params": {
        "out_features": 10,
        "bias": true
      },
      "weights": ["fc1.weight", "fc1.bias"]
    }
  ]
}
```

---

## Appendix C: Interface Summary

### New Headers to Create

```
include/sw/runtime/
├── graph.hpp           # ComputeGraph, GraphRunner
└── graph_memory.hpp    # Memory planning utilities

include/sw/compiler/
├── model_loader.hpp    # JSON/ONNX model loading
├── bn_folding.hpp      # BatchNorm folding optimization
└── graph_optimizer.hpp # Layer fusion, memory planning

include/sw/isa/
└── text_assembler.hpp  # Human-readable assembly format
```

### New Source Files

```
src/runtime/
├── graph.cpp           # Graph execution
└── graph_memory.cpp    # Memory allocation planning

src/compiler/
├── model_loader.cpp    # Model loading
├── conv2d_compiler.cpp # Conv2D program generation
└── pool_compiler.cpp   # Pooling program generation
```

### Modified Files

| File | Changes |
|------|---------|
| `include/sw/kpu/kernel.hpp` | Add create_conv2d, create_pool2d, create_elementwise |
| `include/sw/compiler/kernel_compiler.hpp` | Add compile_conv2d, compile_pool2d |
| `include/sw/kpu/components/sfu.hpp` | Add SOFTMAX activation |
| `include/sw/kpu/isa/data_movement_isa.hpp` | Add reduction opcodes if needed |

---

*Document generated: 2026-01-10*
*Target: Execute SqueezeNet 1.0 on kpu-sim*

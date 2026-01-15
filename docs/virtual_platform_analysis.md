# KPU Virtual Platform: Gap Analysis and Development Plan

## Executive Summary

This document analyzes the evolution of kpu-sim from a performance timing model into a **Virtual Platform** capable of executing compiled C++ programs targeting the KPU architecture. Unlike traditional CPU virtualization (QEMU), the KPU is a **dataflow accelerator** requiring a fundamentally different approach.

### The Key Insight

The KPU is NOT a stored-program processor. You cannot "compile C++ to KPU machine code" in the way you compile to x86 or ARM. Instead:

| Traditional CPU | KPU Dataflow Accelerator |
|----------------|-------------------------|
| Compile C → machine instructions | Compile tensor ops → data movement programs |
| Execute instructions sequentially | Execute via credit-based dataflow |
| CPU fetches instructions | Hardware reacts to data availability |
| General-purpose computation | Domain-specific (matrix/tensor operations) |

The "virtual platform" for KPU is conceptually closer to:
- **CUDA**: C++ with `__device__` → PTX → GPU execution
- **TVM/XLA**: High-level operators → IR → accelerator code
- **NVDLA**: User mode driver compiles DNN layers to loadable programs

---

## Current State Assessment

### What Exists (Strengths)

| Component | Location | Status | Capability |
|-----------|----------|--------|------------|
| **Multi-Fidelity Simulator** | `include/sw/kpu/` | Complete | BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE |
| **Memory Hierarchy** | `components/memory/` | Complete | L1/L2/L3 buffers, LPDDR5/DDR5/HBM controllers |
| **Compute Fabric** | `components/compute_fabric.hpp` | Complete | Systolic array with actual computation |
| **Data Movement ISA** | `isa/data_movement_isa.hpp` | Complete | 20+ opcodes for DMA/BlockMover/Streamer |
| **Runtime API** | `runtime/runtime.hpp` | Complete | CUDA-like malloc/memcpy/launch/streams |
| **Kernel Compiler** | `compiler/kernel_compiler.hpp` | Complete | Tile optimization, program generation |
| **DFX IR** | `compiler/dfx/dfx.hpp` | Complete | PTX-equivalent intermediate representation |
| **Graph Executor** | `runtime/executor.hpp` | Complete | Single-kernel execution with tensor I/O |
| **Trace Generation** | Various | Complete | Chrome Trace, JSON traces |

### What's Missing (Gaps)

#### GAP 1: No C++ Programming Model
- **Current**: Programs must be constructed as explicit kernel calls or computational graphs
- **Needed**: A way for users to write C++ code that targets the KPU

```cpp
// CURRENT: Explicit kernel construction
auto kernel = Kernel::create_matmul(M, N, K);
runtime.launch(kernel, {A_addr, B_addr, C_addr});

// DESIRED: C++ code that compiles to KPU
void my_neural_network(kpu::Tensor& input, kpu::Tensor& output) {
    auto h1 = kpu::matmul(input, weights1);
    auto h2 = kpu::relu(h1);
    output = kpu::matmul(h2, weights2);
}
```

#### GAP 2: No Compiler Frontend
- **Current**: No way to parse C++ source and extract tensor operations
- **Needed**: Source-to-IR compiler that transforms C++ tensor expressions to DFX

#### GAP 3: No Device Driver Model
- **Current**: Direct API calls to simulator
- **Needed**: Proper driver abstraction (User Mode Driver + optional Kernel Mode Driver)

```text
Application
    ↓
User Mode Driver (UMD)
    - Compiles graphs/kernels to DFX
    - Memory management
    - Work submission
    ↓
[Optional] Kernel Mode Driver (KMD)
    - Hardware register access (simulation only)
    - Interrupt handling
    ↓
Hardware Model (Simulator)
```

#### GAP 4: Incomplete Operator Coverage
**Current**: Only MATMUL and MLP kernels are implemented
**Needed**: Full DNN operator coverage

| Operator | Status | Priority |
|----------|--------|----------|
| MatMul | Complete | - |
| MLP (fused) | Complete | - |
| Conv2D | Declared, not implemented | P0 |
| Pooling | Not implemented | P1 |
| Softmax | Not implemented | P1 |
| BatchNorm | Not implemented (can fold) | P1 |
| Elementwise | Not implemented | P1 |
| Reduce | Not implemented | P2 |
| Concat | Not implemented | P2 |

#### GAP 5: No Model Loading
**Current**: No way to load pretrained models (PyTorch, ONNX)
**Needed**: Model importers to create executable graphs

#### GAP 6: No Multi-Kernel Graph Execution
**Current**: GraphExecutor runs single kernels
**Needed**: Full graph scheduling with memory planning

#### GAP 7: No Debug/Profile Integration
**Current**: Trace files are generated, but no integrated debug workflow
**Needed**: Source-level debugging, breakpoints, variable inspection

---

## Reference Architecture: NVDLA Virtual Platform

NVDLA's approach provides a useful reference:

```
┌─────────────────────────────────────────────────────────┐
│                    Application                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               User Mode Driver (UMD)                    │
│  - Compiler: ONNX → NVDLA loadable                     │
│  - Runtime: Submit jobs, manage memory                  │
│  - API: nvdla_load(), nvdla_submit()                   │
└─────────────────────────────────────────────────────────┘
                          ↓ ioctl()
┌─────────────────────────────────────────────────────────┐
│              Kernel Mode Driver (KMD)                   │
│  - Scheduler: Queue and dispatch work                   │
│  - Register access: Configure hardware                  │
│  - Interrupts: Handle completion                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                Hardware (or VP)                         │
│  - SystemC model or actual silicon                      │
└─────────────────────────────────────────────────────────┘
```

For KPU, we can simplify (no actual kernel driver needed in simulation):

```
┌─────────────────────────────────────────────────────────┐
│                    Application                          │
│  - C++ with tensor operations                          │
│  - Or: Python with PyTorch-style API                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               KPU SDK / Compiler                        │
│  - Frontend: Parse C++ tensor expressions              │
│  - Optimizer: Graph optimization, operator fusion      │
│  - Backend: Generate DFX programs                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               KPU Runtime Library                       │
│  - Memory manager: Device memory allocation            │
│  - Loader: DFX → hardware configuration               │
│  - Executor: Schedule and run programs                 │
│  - Profiler: Collect timing/trace data                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│            KPU Virtual Platform (Simulator)             │
│  - Hardware models at configurable fidelity            │
│  - Cycle-accurate or behavioral simulation             │
│  - Trace generation                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Development Plan: Incremental Milestones

### Phase 0: Foundation (Current + Fixes)
**Goal**: Stabilize current infrastructure and fill immediate gaps

**Deliverables**:
1. Complete multi-layer graph execution (`ComputeGraph`, `GraphRunner`)
2. Implement Conv2D kernel using im2col + GEMM
3. Implement pooling operations
4. Implement elementwise operations
5. JSON model format with loader

**Outcome**: Can execute simple CNNs (e.g., LeNet) via explicit graph construction

---

### Phase 1: SDK Core
**Goal**: Create the foundation for a programmable virtual platform

#### Milestone 1.1: Tensor Library
Create a C++ tensor library that looks like NumPy/PyTorch but compiles to KPU:

```cpp
// include/sw/sdk/tensor.hpp
namespace kpu {

class Tensor {
public:
    // Construction
    static Tensor zeros(std::initializer_list<size_t> shape);
    static Tensor from_data(const float* data, std::initializer_list<size_t> shape);

    // Properties
    std::vector<size_t> shape() const;
    size_t numel() const;
    DataType dtype() const;

    // Data access
    float* data();
    const float* data() const;

    // Device memory (internal)
    Address device_address() const;
    void to_device(KPURuntime* runtime);
    void to_host();
};

// Operations (return lazy computation graph nodes)
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor relu(const Tensor& x);
Tensor conv2d(const Tensor& input, const Tensor& weight,
              int stride = 1, int padding = 0);
Tensor max_pool2d(const Tensor& input, int kernel_size, int stride);
Tensor softmax(const Tensor& x, int dim = -1);
Tensor operator+(const Tensor& a, const Tensor& b);

} // namespace kpu
```

**Key Design**: Operations build a **lazy computation graph**. Actual execution happens when:
- User calls `result.eval()` or `result.to_host()`
- Or implicitly at synchronization points

#### Milestone 1.2: Graph Builder
Internal infrastructure to capture tensor operations as a graph:

```cpp
// include/sw/sdk/graph_builder.hpp
namespace kpu::internal {

class GraphBuilder {
public:
    static GraphBuilder& instance();  // Thread-local singleton

    // Record operations
    NodeId add_input(const std::string& name,
                     const std::vector<size_t>& shape);
    NodeId add_op(OpType op, const std::vector<NodeId>& inputs,
                  const OpParams& params);
    NodeId add_constant(const std::string& name, const float* data,
                        const std::vector<size_t>& shape);

    // Build
    ComputeGraph build();
    void reset();

    // Execution context
    void set_runtime(KPURuntime* runtime);
    KPURuntime* runtime() const;
};

} // namespace kpu::internal
```

#### Milestone 1.3: Eager vs Lazy Execution
Support both execution modes:

```cpp
// Lazy mode (default) - builds graph, executes on sync
kpu::Tensor a = kpu::Tensor::from_data(host_a, {M, K});
kpu::Tensor b = kpu::Tensor::from_data(host_b, {K, N});
kpu::Tensor c = kpu::matmul(a, b);  // No execution yet
kpu::Tensor d = kpu::relu(c);        // Still building graph
d.to_host();                          // Execute entire graph, copy result

// Eager mode - immediate execution
kpu::set_execution_mode(kpu::ExecutionMode::EAGER);
kpu::Tensor c = kpu::matmul(a, b);  // Executes immediately
```

---

### Phase 2: Compiler Infrastructure
**Goal**: Transform computation graphs to executable DFX programs

#### Milestone 2.1: Graph Optimizer
Optimize computation graphs before code generation:

```cpp
// include/sw/compiler/graph_optimizer.hpp
namespace kpu::compiler {

class GraphOptimizer {
public:
    // Optimization passes
    void fuse_matmul_bias();           // C = A @ B + bias
    void fuse_matmul_activation();     // C = relu(A @ B)
    void fold_batch_norm();            // Fold BN into preceding conv
    void eliminate_dead_nodes();       // Remove unused computations
    void plan_memory();                // Compute memory allocation plan

    // Run all optimizations
    ComputeGraph optimize(const ComputeGraph& input);
};

} // namespace kpu::compiler
```

#### Milestone 2.2: DFX Code Generator
Generate DFX programs from optimized graphs:

```cpp
// include/sw/compiler/dfx_codegen.hpp
namespace kpu::compiler {

class DFXCodeGenerator {
public:
    DFXCodeGenerator(const HardwareConfig& hw);

    // Generate DFX for a kernel
    dfx::Program generate(const ComputeGraph::Node& node);

    // Generate DFX for entire graph
    std::vector<dfx::Program> generate_all(const ComputeGraph& graph);

    // Linking: combine multiple programs with proper memory layout
    dfx::ExecutableBundle link(const std::vector<dfx::Program>& programs);
};

} // namespace kpu::compiler
```

#### Milestone 2.3: Loadable Binary Format
Create a serializable binary format for compiled programs:

```cpp
// include/sw/compiler/dfx/dfx_object_file.hpp (extend existing)
namespace kpu::compiler::dfx {

// Structure of a .kpx (KPU Executable) file:
// +----------------+
// | Header         |  Magic, version, metadata
// +----------------+
// | Tensor Section |  Tensor descriptors (shapes, dtypes)
// +----------------+
// | Weight Section |  Constant data (weights, biases)
// +----------------+
// | Code Section   |  DFX operation stream
// +----------------+
// | Relocation     |  Address fixups for loader
// +----------------+

class ObjectFile {
public:
    void write(const std::string& path);
    static ObjectFile read(const std::string& path);

    // Contents
    std::vector<TensorDescriptor> tensors;
    std::vector<uint8_t> weight_data;
    std::vector<Program> programs;
    RelocationTable relocations;
};

} // namespace kpu::compiler::dfx
```

---

### Phase 3: Runtime System
**Goal**: Execute compiled programs on the virtual platform

#### Milestone 3.1: Loader
Load .kpx files and prepare for execution:

```cpp
// include/sw/runtime/loader.hpp
namespace kpu::runtime {

class Loader {
public:
    explicit Loader(KPURuntime* runtime);

    // Load compiled program
    ExecutionContext load(const std::string& kpx_path);
    ExecutionContext load(const dfx::ObjectFile& object);

    // Resolve memory addresses
    void bind_tensor(ExecutionContext& ctx,
                     const std::string& name,
                     Address device_addr);

    // Allocate all tensors
    void allocate(ExecutionContext& ctx);

private:
    // Relocate addresses based on actual allocation
    void apply_relocations(ExecutionContext& ctx);
};

struct ExecutionContext {
    std::vector<dfx::Program> programs;
    std::unordered_map<std::string, Address> tensor_addresses;
    Address weight_base;
    bool ready = false;
};

} // namespace kpu::runtime
```

#### Milestone 3.2: Scheduler
Schedule programs for execution with resource management:

```cpp
// include/sw/runtime/scheduler.hpp
namespace kpu::runtime {

class Scheduler {
public:
    explicit Scheduler(KPURuntime* runtime);

    // Submit work
    void submit(const ExecutionContext& ctx);

    // Execution control
    void run();                    // Run until completion
    void step();                   // Execute one operation
    void run_until(Cycle target);  // Run until cycle

    // Status
    bool is_complete() const;
    Cycle current_cycle() const;

    // Profiling
    std::vector<LayerProfile> get_profiles() const;
};

} // namespace kpu::runtime
```

#### Milestone 3.3: Memory Manager
Sophisticated device memory management:

```cpp
// include/sw/runtime/memory_manager.hpp
namespace kpu::runtime {

class MemoryManager {
public:
    explicit MemoryManager(KPURuntime* runtime);

    // Allocation strategies
    enum class Strategy {
        SIMPLE_BUMP,      // Fast but no reuse
        POOL,             // Fixed-size pools
        BUDDY,            // Power-of-two buddy allocator
        MEMORY_PLANNING   // Optimal lifetime-based planning
    };

    void set_strategy(Strategy s);

    // Graph-aware allocation
    void plan_for_graph(const ComputeGraph& graph);

    // Statistics
    size_t peak_usage() const;
    size_t current_usage() const;
};

} // namespace kpu::runtime
```

---

### Phase 4: Programming Model
**Goal**: Enable natural C++ programming for KPU

#### Milestone 4.1: Module System
Allow users to define reusable modules (like PyTorch nn.Module):

```cpp
// include/sw/sdk/module.hpp
namespace kpu {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;

    // Parameter management
    std::vector<Tensor*> parameters();
    void load_state_dict(const StateDict& dict);
    StateDict state_dict() const;

protected:
    // Register a parameter
    Tensor& register_parameter(const std::string& name, Tensor param);
    // Register a submodule
    template<typename M>
    M& register_module(const std::string& name, M module);
};

// Built-in modules
class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Tensor forward(const Tensor& input) override;
private:
    Tensor weight_, bias_;
};

class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels,
           size_t kernel_size, size_t stride = 1, size_t padding = 0);
    Tensor forward(const Tensor& input) override;
private:
    Tensor weight_, bias_;
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

class Sequential : public Module {
public:
    template<typename... Modules>
    Sequential(Modules&&... modules);
    Tensor forward(const Tensor& input) override;
};

} // namespace kpu
```

#### Milestone 4.2: Model Definition API
Users can define models in C++:

```cpp
// Example: User-defined model
class MyModel : public kpu::Module {
public:
    MyModel() {
        conv1 = register_module("conv1",
            kpu::Conv2d(1, 32, 3, 1, 1));
        conv2 = register_module("conv2",
            kpu::Conv2d(32, 64, 3, 1, 1));
        fc = register_module("fc",
            kpu::Linear(64 * 7 * 7, 10));
    }

    kpu::Tensor forward(const kpu::Tensor& x) override {
        auto h = kpu::relu(conv1.forward(x));
        h = kpu::max_pool2d(h, 2, 2);
        h = kpu::relu(conv2.forward(h));
        h = kpu::max_pool2d(h, 2, 2);
        h = h.flatten(1);
        return fc.forward(h);
    }

private:
    kpu::Conv2d conv1, conv2;
    kpu::Linear fc;
};

// Usage
int main() {
    kpu::Device device;
    MyModel model;
    model.load_state_dict(kpu::load("model.pth"));
    model.to(device);

    kpu::Tensor input = kpu::Tensor::from_file("input.bin", {1, 1, 28, 28});
    kpu::Tensor output = model.forward(input);

    output.to_host();
    // ... use output.data() ...
}
```

#### Milestone 4.3: JIT Compilation
Compile models at runtime (like PyTorch TorchScript):

```cpp
// include/sw/sdk/jit.hpp
namespace kpu::jit {

// Trace-based JIT
class TracedModule {
public:
    // Trace a module by running it with example input
    static TracedModule trace(Module& module, const Tensor& example_input);

    // Execute traced module
    Tensor forward(const Tensor& input);

    // Serialize
    void save(const std::string& path);
    static TracedModule load(const std::string& path);

private:
    dfx::ObjectFile compiled_;
};

// Ahead-of-time compilation
dfx::ObjectFile compile(Module& module,
                        const std::vector<TensorSpec>& input_specs);

} // namespace kpu::jit
```

---

### Phase 5: Model Import
**Goal**: Load models from existing frameworks

#### Milestone 5.1: ONNX Importer

```cpp
// include/sw/compiler/onnx_importer.hpp
namespace kpu::compiler {

class ONNXImporter {
public:
    // Load ONNX model
    static ComputeGraph load(const std::string& path);

    // Load with weights
    static std::pair<ComputeGraph, WeightDict> load_with_weights(
        const std::string& path);

    // Supported operators
    static std::vector<std::string> supported_ops();

    // Validation
    static bool validate(const std::string& path);
};

} // namespace kpu::compiler
```

#### Milestone 5.2: PyTorch Importer
Python tool to import PyTorch models into KPU format:

```python
# tools/kpu_import_pytorch.py
import torch
import kpu_import

# Load PyTorch model
model = torch.load("model.pth")
model.eval()

# Import to KPU format
kpu_import.from_pytorch(model,
                        example_input=torch.randn(1, 3, 224, 224),
                        output_path="model.kpx")
```

---

### Phase 6: Debug and Profile
**Goal**: Provide a complete development experience

#### Milestone 6.1: Debugger Integration

```cpp
// include/sw/debug/debugger.hpp
namespace kpu::debug {

class Debugger {
public:
    explicit Debugger(KPURuntime* runtime);

    // Breakpoints
    void set_breakpoint(BreakpointType type, const BreakpointCondition& cond);
    void clear_breakpoint(int id);

    // Execution control
    void run();
    void step_cycle();
    void step_operation();
    void continue_to(Cycle target);

    // Inspection
    std::vector<TileInfo> inspect_l3_buffers() const;
    std::vector<TileInfo> inspect_l2_banks() const;
    std::vector<float> read_memory(Address addr, size_t count) const;

    // State
    Cycle current_cycle() const;
    std::string current_operation() const;
};

enum class BreakpointType {
    CYCLE,           // Break at specific cycle
    OPERATION,       // Break before specific operation
    MEMORY_ACCESS,   // Break on memory access
    TILE_ARRIVAL,    // Break when tile arrives at buffer
    ERROR            // Break on any error
};

} // namespace kpu::debug
```

#### Milestone 6.2: Profiler

```cpp
// include/sw/profile/profiler.hpp
namespace kpu::profile {

class Profiler {
public:
    explicit Profiler(KPURuntime* runtime);

    // Enable/disable profiling
    void enable();
    void disable();

    // Get results
    ProfileReport report() const;

    // Export
    void export_chrome_trace(const std::string& path);
    void export_tensorboard(const std::string& log_dir);
};

struct ProfileReport {
    // Per-layer breakdown
    struct LayerStats {
        std::string name;
        Cycle cycles;
        double time_ms;
        size_t dram_bytes;
        double arithmetic_intensity;
        double compute_utilization;
    };
    std::vector<LayerStats> layers;

    // Aggregate statistics
    Cycle total_cycles;
    double total_time_ms;
    size_t total_dram_bytes;
    double avg_compute_utilization;

    // Bottleneck analysis
    std::string bottleneck;  // "COMPUTE_BOUND" or "MEMORY_BOUND"
    std::vector<std::string> recommendations;
};

} // namespace kpu::profile
```

---

### Phase 7: Full Virtual Platform
**Goal**: Complete platform for software development

#### Milestone 7.1: CLI Tools

```bash
# Compile a model
kpu-compile model.onnx -o model.kpx

# Run a model
kpu-run model.kpx --input input.bin --output output.bin

# Profile a model
kpu-profile model.kpx --input input.bin --trace profile.json

# Debug interactively
kpu-debug model.kpx --input input.bin
(kpu-dbg) break cycle 1000
(kpu-dbg) run
(kpu-dbg) inspect l3
(kpu-dbg) continue
```

#### Milestone 7.2: IDE Integration
VS Code extension for KPU development:

- Syntax highlighting for assembly/config files
- Debugger protocol (DAP) support
- Trace visualization
- Memory inspector
- Performance dashboard

#### Milestone 7.3: Continuous Integration
Testing infrastructure:

```yaml
# .github/workflows/kpu-tests.yml
- name: Build KPU Simulator
  run: cmake --preset release && cmake --build --preset release

- name: Run Unit Tests
  run: ctest --preset release

- name: Run Model Tests
  run: |
    kpu-compile tests/models/squeezenet.onnx -o squeezenet.kpx
    kpu-run squeezenet.kpx --input tests/data/image.bin \
                           --expected tests/data/expected.bin \
                           --tolerance 1e-3
```

---

## Summary: Milestone Dependencies

```
Phase 0: Foundation
    └── Multi-layer graph execution
    └── Conv2D, Pooling, Elementwise
    └── JSON model loader

Phase 1: SDK Core
    ├── 1.1 Tensor Library
    ├── 1.2 Graph Builder
    └── 1.3 Eager/Lazy Execution

Phase 2: Compiler Infrastructure
    ├── 2.1 Graph Optimizer
    ├── 2.2 DFX Code Generator
    └── 2.3 Loadable Binary Format (.kpx)

Phase 3: Runtime System
    ├── 3.1 Loader
    ├── 3.2 Scheduler
    └── 3.3 Memory Manager

Phase 4: Programming Model
    ├── 4.1 Module System
    ├── 4.2 Model Definition API
    └── 4.3 JIT Compilation

Phase 5: Model Import
    ├── 5.1 ONNX Importer
    └── 5.2 PyTorch Importer

Phase 6: Debug and Profile
    ├── 6.1 Debugger Integration
    └── 6.2 Profiler

Phase 7: Full Virtual Platform
    ├── 7.1 CLI Tools
    ├── 7.2 IDE Integration
    └── 7.3 CI/CD Infrastructure
```

---

## Comparison with Alternatives

| Approach | Pros | Cons | Fit for KPU |
|----------|------|------|-------------|
| **SystemC + QEMU (NVDLA)** | Full OS support, industry standard | Heavy dependencies, complex setup | Overkill |
| **gem5 + SALAM** | Modular, research-friendly | Complex, GPU/CPU focused | Could work |
| **Custom Event-Driven (Current)** | Lightweight, flexible, pure C++ | More development effort | Good fit |
| **Verilator** | RTL-accurate, fast | Need HDL first | Future option |

**Recommendation**: Continue with the current event-driven C++ approach, adding the SDK/compiler/runtime layers incrementally.

---

## What "Execute a C++ Program" Means for KPU

Unlike a traditional CPU virtual platform where you literally compile C to machine code and execute instructions, for KPU "executing a C++ program" means:

1. **User writes C++ using the kpu::Tensor API**
   ```cpp
   kpu::Tensor a = kpu::Tensor::from_file("weights.bin", {M, K});
   kpu::Tensor b = kpu::Tensor::from_file("input.bin", {K, N});
   kpu::Tensor c = kpu::matmul(a, b);
   c.to_host();
   ```

2. **Compiler captures tensor operations as a graph**
   - At compile time (AOT) or runtime (JIT)
   - Graph represents dataflow, not control flow

3. **Graph is optimized and compiled to DFX programs**
   - Tile scheduling
   - Memory planning
   - DMA/BlockMover/Streamer operation generation

4. **Runtime loads and executes DFX programs**
   - Maps to virtual hardware resources
   - Simulates at chosen fidelity level
   - Collects traces and profiles

5. **Results are returned to host C++ code**
   - Tensor data copied back to host memory
   - Profiling data available

This is analogous to how CUDA programs work: the user writes C++ with CUDA extensions, nvcc compiles device code to PTX, the driver JITs to SASS, and the GPU executes. The key difference is that KPU is a dataflow accelerator, not a SIMT processor.

---

*Document created: 2026-01-15*
*Purpose: Guide evolution of kpu-sim into a virtual platform for software development*

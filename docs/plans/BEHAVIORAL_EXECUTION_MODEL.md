# Behavioral Execution Model for KPU DNN Execution

## Overview

This document derives a behavioral algorithm for functionally modeling the KPU
executing a simple DNN (single and multi-layer perceptron), and proposes an
implementation path.

## KPU Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST MEMORY                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Graph Desc  │  │   Weights   │  │   Inputs    │  │   Outputs   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ DMA (Host↔KPU)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              L3 TILES (On-Chip)                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
│  │ Tile 0  │  │ Tile 1  │  │ Tile 2  │  │ Tile 3  │  ...                   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                        │
└───────┼────────────┼────────────┼────────────┼──────────────────────────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           │ BlockMover (L3↔L2, L3↔L3)
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              L2 BANKS                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
│  │ Bank 0  │  │ Bank 1  │  │ Bank 2  │  │ Bank 3  │  ...                   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                        │
└───────┼────────────┼────────────┼────────────┼──────────────────────────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           │ Streamers (L2→L1, push-only)
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L1 STREAMING BUFFERS                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ Row Ingress  │  │ Col Ingress  │  │   Egress     │                      │
│  │   Buffers    │  │   Buffers    │  │   Buffers    │                      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                      │
└─────────┼─────────────────┼─────────────────┼───────────────────────────────┘
          │                 │                 ▲
          ▼                 ▼                 │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPUTE TILE                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     16×16 Systolic Array                             │   │
│  │   Row →  ┌───┬───┬───┬───┐                                          │   │
│  │   Data   │PE │PE │...│PE │ → Accumulated                            │   │
│  │          ├───┼───┼───┼───┤    Results                               │   │
│  │          │PE │PE │...│PE │                                          │   │
│  │   Col ↓  ├───┼───┼───┼───┤                                          │   │
│  │   Data   │...│...│...│...│                                          │   │
│  │          ├───┼───┼───┼───┤                                          │   │
│  │          │PE │PE │...│PE │                                          │   │
│  │          └───┴───┴───┴───┘                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Execution Characteristics

### 1. Push-Only Compute Tiles
- Compute tiles do NOT pull data
- Streamers push data into L1 buffers
- L1 buffers autonomously stream into compute tile ingress ports
- Results pushed from compute tile → L1 egress → L2

### 2. Data Must Be Pre-Positioned
- Compute cannot start until RIGHT data is in L2
- BlockMover cannot write to L2 unless space is allocated
- Streamers run fixed FSM - cannot bubble (stall)

### 3. Decomposition is Central
- Compiler decomposes operators into tiles fitting L3/L2/L1
- This decomposition drives BOTH functional AND performance behavior
- Runtime orchestrates based on compiler's tile schedule

## Behavioral Algorithm

### Phase 0: Graph Compilation (Offline or JIT)

```
COMPILE(graph):
    for each operator in topological_order(graph):
        input_shapes = get_input_shapes(operator)
        output_shape = infer_output_shape(operator)

        # Decompose into tiles that fit memory hierarchy
        tile_schedule = compute_tiling(
            operator,
            input_shapes,
            l3_capacity,
            l2_capacity,
            l1_capacity,
            compute_tile_size  # e.g., 16×16
        )

        # Generate programs for each hardware unit
        dma_commands = generate_dma_commands(tile_schedule, input_shapes)
        blockmover_commands = generate_blockmover_commands(tile_schedule)
        streamer_programs = generate_streamer_programs(tile_schedule)
        compute_kernel = generate_compute_kernel(operator, tile_schedule)

        emit(operator_program: {
            dma_commands,
            blockmover_commands,
            streamer_programs,
            compute_kernel,
            tile_schedule
        })
```

### Phase 1: Graph Initialization (Runtime)

```
INITIALIZE_GRAPH(compiled_graph, input_data, weights):
    # Allocate host memory for graph structures
    host_graph = allocate_host_memory(compiled_graph)

    # Copy weights to host memory region
    copy_weights_to_host(host_graph, weights)

    # Copy input data to host memory region
    copy_inputs_to_host(host_graph, input_data)

    # Allocate output buffer in host memory
    allocate_outputs(host_graph)

    return host_graph
```

### Phase 2: Orchestrator Loop (Runtime - Host CPU)

```
EXECUTE_GRAPH(host_graph):
    resource_manager = get_resource_manager()

    for each operator_program in host_graph.operators:
        EXECUTE_OPERATOR(operator_program, resource_manager)

    # Final results now in host memory output buffer
    return host_graph.outputs
```

### Phase 3: Operator Execution (Per-Operator)

```
EXECUTE_OPERATOR(op_program, resource_mgr):
    tile_schedule = op_program.tile_schedule

    # Iterate through tile schedule (may be multi-dimensional)
    for tile_idx in tile_schedule.output_tiles:

        # Determine which input tiles are needed for this output tile
        input_tile_coords = compute_input_dependencies(tile_idx, tile_schedule)

        #─────────────────────────────────────────────────────────────────
        # PHASE 3A: Resource Allocation
        #─────────────────────────────────────────────────────────────────
        l3_alloc = resource_mgr.allocate_l3(
            input_tiles = input_tile_coords,
            output_tile = tile_idx
        )

        l2_alloc = resource_mgr.allocate_l2(
            a_tile_size = tile_schedule.a_tile_size,
            b_tile_size = tile_schedule.b_tile_size,
            c_tile_size = tile_schedule.c_tile_size
        )

        l1_alloc = resource_mgr.allocate_l1(
            row_buffers = tile_schedule.row_streams,
            col_buffers = tile_schedule.col_streams,
            out_buffers = tile_schedule.out_streams
        )

        compute_alloc = resource_mgr.allocate_compute_tile()

        #─────────────────────────────────────────────────────────────────
        # PHASE 3B: DMA - Host Memory to L3
        #─────────────────────────────────────────────────────────────────
        for input_tile in input_tile_coords:
            if not is_in_l3(input_tile):
                dma_cmd = create_dma_gather_command(
                    src = host_address(input_tile),
                    dst = l3_alloc.address(input_tile),
                    size = tile_size(input_tile)
                )
                resource_mgr.submit_dma(dma_cmd)

        resource_mgr.wait_dma_complete()

        #─────────────────────────────────────────────────────────────────
        # PHASE 3C: BlockMover - L3 to L2
        #─────────────────────────────────────────────────────────────────
        # A matrix tile: L3 → L2
        bm_cmd_a = create_blockmover_command(
            src = l3_alloc.a_tile_address,
            dst = l2_alloc.a_address,
            shape = tile_schedule.a_tile_size,
            stride = tile_schedule.a_stride
        )
        resource_mgr.submit_blockmover(bm_cmd_a)

        # B matrix tile: L3 → L2
        bm_cmd_b = create_blockmover_command(
            src = l3_alloc.b_tile_address,
            dst = l2_alloc.b_address,
            shape = tile_schedule.b_tile_size,
            stride = tile_schedule.b_stride
        )
        resource_mgr.submit_blockmover(bm_cmd_b)

        resource_mgr.wait_blockmover_complete()

        #─────────────────────────────────────────────────────────────────
        # PHASE 3D: Streamer Programming
        #─────────────────────────────────────────────────────────────────
        # Program row streamers (A matrix rows)
        for row in range(tile_schedule.m_tile):
            streamer_program = create_streamer_program(
                src = l2_alloc.a_address + row * tile_schedule.k_tile,
                dst_l1_buffer = l1_alloc.row_buffer[row],
                length = tile_schedule.k_tile,
                element_size = sizeof(float)
            )
            resource_mgr.program_streamer(row, streamer_program)

        # Program column streamers (B matrix columns)
        for col in range(tile_schedule.n_tile):
            streamer_program = create_streamer_program(
                src = l2_alloc.b_address + col,
                dst_l1_buffer = l1_alloc.col_buffer[col],
                length = tile_schedule.k_tile,
                stride = tile_schedule.n_tile,  # Column-major stride
                element_size = sizeof(float)
            )
            resource_mgr.program_streamer(col, streamer_program)

        #─────────────────────────────────────────────────────────────────
        # PHASE 3E: Compute Execution
        #─────────────────────────────────────────────────────────────────
        # Start streamers (they run autonomously)
        resource_mgr.start_streamers()

        # Compute tile executes as data streams in
        # In behavioral model: we compute the actual matmul
        C_tile = BEHAVIORAL_MATMUL(
            A = read_l2(l2_alloc.a_address, tile_schedule.a_tile_size),
            B = read_l2(l2_alloc.b_address, tile_schedule.b_tile_size)
        )

        # Apply activation if needed (e.g., ReLU)
        if op_program.activation != NONE:
            C_tile = APPLY_ACTIVATION(C_tile, op_program.activation)

        # Write result to L1 egress (behaviorally: to L2)
        write_l2(l2_alloc.c_address, C_tile)

        resource_mgr.wait_compute_complete()

        #─────────────────────────────────────────────────────────────────
        # PHASE 3F: Results Movement - L2 to L3 to Host
        #─────────────────────────────────────────────────────────────────
        # BlockMover: L2 → L3
        bm_cmd_c = create_blockmover_command(
            src = l2_alloc.c_address,
            dst = l3_alloc.c_tile_address,
            shape = tile_schedule.c_tile_size
        )
        resource_mgr.submit_blockmover(bm_cmd_c)
        resource_mgr.wait_blockmover_complete()

        # DMA: L3 → Host (if final output or needed by next layer)
        if is_graph_output(tile_idx) or needs_host_copy(tile_idx):
            dma_cmd = create_dma_scatter_command(
                src = l3_alloc.c_tile_address,
                dst = host_output_address(tile_idx),
                size = tile_size(tile_idx)
            )
            resource_mgr.submit_dma(dma_cmd)
            resource_mgr.wait_dma_complete()

        #─────────────────────────────────────────────────────────────────
        # PHASE 3G: Resource Deallocation
        #─────────────────────────────────────────────────────────────────
        resource_mgr.free_l1(l1_alloc)
        resource_mgr.free_l2(l2_alloc)
        # L3 may be retained for next operator's input
        if not needed_by_next_operator(tile_idx):
            resource_mgr.free_l3(l3_alloc)
```

### Behavioral Compute Functions

```
BEHAVIORAL_MATMUL(A[M,K], B[K,N]) -> C[M,N]:
    C = zeros(M, N)
    for i in range(M):
        for j in range(N):
            sum = 0
            for k in range(K):
                sum += A[i,k] * B[k,j]
            C[i,j] = sum
    return C

APPLY_ACTIVATION(X, activation_type):
    match activation_type:
        case RELU:
            return max(0, X)  # element-wise
        case GELU:
            return X * Φ(X)   # Φ is standard normal CDF
        case SIGMOID:
            return 1 / (1 + exp(-X))
        case TANH:
            return tanh(X)
        case NONE:
            return X
```

## Data Structure Definitions

### Tile Schedule

```cpp
struct TileSchedule {
    // Output tile iteration space
    Size output_tile_count_m;  // Number of output tiles in M dimension
    Size output_tile_count_n;  // Number of output tiles in N dimension

    // Tile dimensions (must fit in compute array)
    Size m_tile;  // Rows per tile (≤ array_rows)
    Size n_tile;  // Cols per tile (≤ array_cols)
    Size k_tile;  // Reduction dimension per tile

    // Memory requirements per tile
    Size a_tile_bytes;  // M_tile × K_tile × element_size
    Size b_tile_bytes;  // K_tile × N_tile × element_size
    Size c_tile_bytes;  // M_tile × N_tile × element_size

    // Strides for sub-tile extraction from larger matrices
    Size a_row_stride;  // Stride between rows of A in host memory
    Size b_row_stride;  // Stride between rows of B in host memory
    Size c_row_stride;  // Stride between rows of C in host memory
};
```

### DMA Command

```cpp
struct DMACommand {
    enum Type { GATHER, SCATTER };
    Type type;

    Address host_address;    // Host memory address
    Address device_address;  // L3 tile address
    Size size_bytes;

    // For strided transfers (sub-matrix gather)
    Size row_count;
    Size row_size_bytes;
    Size host_stride;
    Size device_stride;
};
```

### BlockMover Command

```cpp
struct BlockMoverCommand {
    Address src_address;     // L3 or L2 address
    Address dst_address;     // L3 or L2 address

    Size rows;
    Size cols;
    Size element_size;

    Size src_stride;         // Bytes between rows at source
    Size dst_stride;         // Bytes between rows at destination
};
```

### Streamer Program

```cpp
struct StreamerProgram {
    Address l2_base_address;
    Size element_count;
    Size element_size;
    Size stride;             // For non-contiguous access (column major)

    L1BufferID target_buffer;
    StreamDirection direction;  // L2_TO_L1 or L1_TO_L2
};
```

## Implementation Plan

### Phase 1: Core Infrastructure

**1.1 Behavioral Memory Model**

Create a unified memory model that tracks data location and content:

```cpp
// include/sw/runtime/behavioral/memory_model.hpp

class BehavioralMemoryModel {
public:
    // Memory regions
    void allocate_host_region(const std::string& name, Size bytes);
    void allocate_l3_region(TileID tile, Size bytes);
    void allocate_l2_region(BankID bank, Size bytes);
    void allocate_l1_region(BufferID buffer, Size bytes);

    // Data operations (actual values)
    void write(Address addr, const void* data, Size bytes);
    void read(Address addr, void* data, Size bytes);
    void copy(Address dst, Address src, Size bytes);

    // Track where data lives
    DataLocation locate(const std::string& tensor_name);
    bool is_valid(Address addr, Size bytes);
};
```

**Files to create:**
- `include/sw/runtime/behavioral/memory_model.hpp`
- `src/runtime/behavioral/memory_model.cpp`

**1.2 Resource Manager Enhancement**

Extend ResourceManager to handle allocations across memory hierarchy:

```cpp
// include/sw/runtime/behavioral/resource_manager.hpp

class BehavioralResourceManager {
public:
    // Allocations
    L3Allocation allocate_l3(Size bytes, const std::vector<TileID>& preferred_tiles);
    L2Allocation allocate_l2(Size bytes, const std::vector<BankID>& preferred_banks);
    L1Allocation allocate_l1(Size bytes, BufferType type);
    ComputeAllocation allocate_compute();

    // Command submission
    void submit_dma(const DMACommand& cmd);
    void submit_blockmover(const BlockMoverCommand& cmd);
    void program_streamer(StreamerID id, const StreamerProgram& prog);

    // Synchronization
    void wait_dma_complete();
    void wait_blockmover_complete();
    void start_streamers();
    void wait_compute_complete();

    // Deallocation
    void free(const Allocation& alloc);

private:
    BehavioralMemoryModel memory_;
    // ... command queues, state tracking
};
```

**Files to create:**
- `include/sw/runtime/behavioral/resource_manager.hpp`
- `src/runtime/behavioral/resource_manager.cpp`

### Phase 2: Tiling and Decomposition

**2.1 Tile Schedule Generator**

Create tiling logic that decomposes operators:

```cpp
// include/sw/compiler/tiler.hpp

class Tiler {
public:
    Tiler(const HardwareConfig& hw_config);

    // Generate tile schedule for matmul
    TileSchedule tile_matmul(
        Size M, Size K, Size N,
        Size l3_capacity,
        Size l2_capacity,
        Size l1_capacity,
        Size array_rows,
        Size array_cols
    );

    // Generate tile schedule for MLP layer
    TileSchedule tile_mlp(
        Size batch, Size output_dim, Size input_dim,
        ActivationType activation,
        const HardwareConfig& hw
    );

private:
    HardwareConfig hw_config_;

    // Find tile sizes that maximize utilization
    std::tuple<Size, Size, Size> find_optimal_tile_size(
        Size M, Size K, Size N,
        Size memory_budget
    );
};
```

**Files to create:**
- `include/sw/compiler/tiler.hpp`
- `src/compiler/tiler.cpp`

**2.2 Command Buffer Generator**

Generate hardware commands from tile schedule:

```cpp
// include/sw/compiler/command_generator.hpp

class CommandGenerator {
public:
    // Generate all commands for a tile
    struct TileCommands {
        std::vector<DMACommand> dma_commands;
        std::vector<BlockMoverCommand> blockmover_commands;
        std::vector<StreamerProgram> streamer_programs;
        ComputeKernel compute_kernel;
    };

    TileCommands generate_matmul_commands(
        const TileSchedule& schedule,
        Size tile_m_idx, Size tile_n_idx,
        Address host_a_base, Address host_b_base, Address host_c_base
    );
};
```

**Files to create:**
- `include/sw/compiler/command_generator.hpp`
- `src/compiler/command_generator.cpp`

### Phase 3: Behavioral Compute

**3.1 Behavioral Compute Engine**

Implement actual computation:

```cpp
// include/sw/runtime/behavioral/compute_engine.hpp

class BehavioralComputeEngine {
public:
    // Matrix multiplication (the core operation)
    void matmul(
        const float* A, Size M, Size K, Size lda,
        const float* B, Size K2, Size N, Size ldb,
        float* C, Size ldc,
        bool accumulate = false
    );

    // Bias addition
    void add_bias(float* C, const float* bias, Size M, Size N);

    // Activations
    void apply_activation(float* data, Size count, ActivationType type);

    // Fused MLP operation
    void mlp_forward(
        const float* input, Size batch, Size input_dim,
        const float* weights, Size output_dim,
        const float* bias,
        float* output,
        ActivationType activation
    );
};
```

**Files to create:**
- `include/sw/runtime/behavioral/compute_engine.hpp`
- `src/runtime/behavioral/compute_engine.cpp`

### Phase 4: Behavioral Graph Executor

**4.1 Behavioral Graph Runner**

Integrate all components:

```cpp
// include/sw/runtime/behavioral/graph_executor.hpp

class BehavioralGraphExecutor {
public:
    BehavioralGraphExecutor(const HardwareConfig& hw_config);

    // Set the computational graph
    void set_graph(const ComputeGraph& graph);

    // Allocate all memory
    void allocate();

    // Set inputs (copies to host memory model)
    void set_input(const std::string& name, const void* data, Size bytes);

    // Set weights
    void set_weights(const std::string& layer_name,
                     const void* weights, Size weight_bytes,
                     const void* bias, Size bias_bytes);

    // Execute graph (behavioral - computes actual values)
    ExecutionResult run();

    // Get outputs (actual computed values)
    void get_output(const std::string& name, void* data);

private:
    HardwareConfig hw_config_;
    ComputeGraph graph_;
    BehavioralResourceManager resource_mgr_;
    BehavioralComputeEngine compute_;
    Tiler tiler_;
    CommandGenerator cmd_gen_;

    // Execute single operator
    void execute_operator(const GraphNode& node);

    // Execute tiled matmul
    void execute_tiled_matmul(
        const TileSchedule& schedule,
        Address a_host, Address b_host, Address c_host,
        ActivationType activation
    );
};
```

**Files to create:**
- `include/sw/runtime/behavioral/graph_executor.hpp`
- `src/runtime/behavioral/graph_executor.cpp`

### Phase 5: Example and Validation

**5.1 Behavioral XOR Example**

```cpp
// examples/behavioral/xor_behavioral.cpp

int main() {
    // Configure hardware
    HardwareConfig hw;
    hw.l3_tile_count = 4;
    hw.l3_tile_capacity_kb = 128;
    hw.l2_bank_count = 8;
    hw.l2_bank_capacity_kb = 64;
    hw.l1_buffer_count = 4;
    hw.l1_buffer_capacity_kb = 16;
    hw.array_rows = 16;
    hw.array_cols = 16;

    // Build graph
    ComputeGraph graph("xor_classifier");
    graph.add_input("input", {4, 2});
    graph.add_node("hidden", Kernel::create_mlp(4, 4, 2, RELU, true),
                   {"input"}, {"h1"});
    graph.add_node("output", Kernel::create_mlp(4, 1, 4, NONE, true),
                   {"h1"}, {"output"});
    graph.add_output("output");
    graph.finalize();

    // Create behavioral executor
    BehavioralGraphExecutor executor(hw);
    executor.set_graph(graph);
    executor.allocate();

    // Set weights
    executor.set_weights("hidden", HIDDEN_WEIGHTS, HIDDEN_BIAS);
    executor.set_weights("output", OUTPUT_WEIGHTS, OUTPUT_BIAS);

    // Set inputs
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    executor.set_input("input", inputs, sizeof(inputs));

    // EXECUTE - this actually computes!
    auto result = executor.run();

    // Get outputs - these are real computed values
    float outputs[4];
    executor.get_output("output", outputs);

    // Verify
    for (int i = 0; i < 4; i++) {
        bool predicted = outputs[i] > 0.0f;
        bool expected = (i == 1 || i == 2);  // XOR truth table
        assert(predicted == expected);
    }

    std::cout << "XOR Behavioral Execution: PASSED\n";
    return 0;
}
```

## File Summary

| File | Purpose | Phase |
|------|---------|-------|
| `include/sw/runtime/behavioral/memory_model.hpp` | Unified memory with actual data | 1.1 |
| `src/runtime/behavioral/memory_model.cpp` | Memory model implementation | 1.1 |
| `include/sw/runtime/behavioral/resource_manager.hpp` | Resource allocation & commands | 1.2 |
| `src/runtime/behavioral/resource_manager.cpp` | Resource manager implementation | 1.2 |
| `include/sw/compiler/tiler.hpp` | Operator decomposition | 2.1 |
| `src/compiler/tiler.cpp` | Tiler implementation | 2.1 |
| `include/sw/compiler/command_generator.hpp` | Command buffer generation | 2.2 |
| `src/compiler/command_generator.cpp` | Command generator implementation | 2.2 |
| `include/sw/runtime/behavioral/compute_engine.hpp` | Actual computation | 3.1 |
| `src/runtime/behavioral/compute_engine.cpp` | Compute engine implementation | 3.1 |
| `include/sw/runtime/behavioral/graph_executor.hpp` | Full graph execution | 4.1 |
| `src/runtime/behavioral/graph_executor.cpp` | Graph executor implementation | 4.1 |
| `examples/behavioral/xor_behavioral.cpp` | XOR validation example | 5.1 |
| `tests/behavioral/test_behavioral_mlp.cpp` | Unit tests | 5.1 |

## Key Design Decisions

### 1. Memory Model Tracks Actual Values

Unlike the timing model, the behavioral model maintains actual tensor data
and propagates computed values through the execution.

### 2. Commands Are Executed, Not Just Timed

DMA, BlockMover, and Streamer commands cause actual data movement in the
memory model, ensuring correct data is available for compute.

### 3. Tiling is Explicit

The tiler generates explicit tile schedules that are then executed. This
matches how the real hardware operates and enables future performance
model integration.

### 4. Resource Manager Enforces Constraints

The resource manager ensures L2 is allocated before BlockMover writes,
L1 is ready before Streamers start, etc. - matching hardware constraints.

### 5. Compute is Last Step in Pipeline

Computation only happens after data movement is complete, matching the
push-only nature of the compute tiles.

## Validation Strategy

1. **Unit Tests**: Test each component in isolation
   - Memory model: write/read/copy correctness
   - Tiler: tile sizes fit constraints
   - Compute: matmul correctness against reference

2. **Integration Tests**: End-to-end execution
   - XOR MLP: 4 test cases, known answers
   - Sine approximation: function estimation accuracy
   - MNIST subset: digit classification

3. **Cross-Validation**: Compare behavioral vs reference
   - Run same inputs through behavioral executor
   - Compare outputs to numpy/torch reference
   - Verify bit-exact (or within tolerance)

# Stillwater Knowledge Processing Unit (KPU) Architecture

## 1. Domain Flow Architecture (DFA) Foundation

### 1.1 Architectural Philosophy

Domain Flow Architectures are push-based dataflow pipeline machines designed for direct execution of Systems of Uniform Recurrence Equations (SUREs). The architecture maintains spatial structural relationships throughout computation to amortize concurrency control overhead across the entirety of linear algebra operators.

The key innovation of DFA is the spatial distribution of the Content Addressable Memory (CAM) structure—traditionally centralized in dataflow machines—across an array of compute elements. This distribution makes both cycle time and capacity scalable while preserving the dataflow execution model.

### 1.2 Core Architectural Components

#### 1.2.1 Memory Hierarchy

**Main Memory**
- External DRAM, typically off-SoC or off-Chiplet
- Stores input operands and final computational results
- Accessed through the Memory Controller

**Memory Controller**
- Interfaces between the SoC/Chiplet and external main memory
- Manages address generation and memory access protocols
- Coordinates with DMA engines for bulk data transfers

#### 1.2.2 Data Movement Subsystem

**DMA Engine Bank**
- Multiple DMA engines operating in parallel
- Programmed with the system-level domain schedule of the target operator
- Implements block-level schedules (e.g., tiled matrix-matrix multiply schedules)
- Pushes data blocks from main memory into distributed scratchpad memory
- Retrieves result blocks from scratchpad back to main memory
- Exploits the well-defined parallel structure of linear algebra operators

**Distributed Scratchpad Memory**
- Staging memory distributed across the chip
- Organized as cache-line addressable storage
- Buffers data between DMA engines and Streamers
- Provides low-latency access for the streaming subsystem

#### 1.2.3 Streaming Subsystem

**Streamers**
- Transform scratchpad cache lines into compute fabric I/O patterns
- Generate row, column, and potentially diagonal data streams
- Interface between memory-oriented representation (cache lines) and compute-oriented representation (token streams)
- Perform inverse transformation on output: convert result token streams back into vectors for scratchpad storage
- Positioned at the edge of the compute fabric

#### 1.2.4 Compute Fabric

**Systolic Array Structure**
- 2D or 3D array of Processing Elements (PEs)
- Lock-step operation across all PEs
- Spatial arrangement maps directly to the structure of SURE computations

**Processing Elements (PEs)**
- Execute operations when operand tokens arrive at their physical location
- Operand matching based on data token signatures
- Local instruction execution without global synchronization
- Route result tokens to next spatial location in the computation

**Data Token Structure**
- Payload: computational data value
- Signature: matching tag consisting of:
  - Spatial embedding tag (Euclidean index point in the problem space)
  - Recurrence variable identifier
- Token matching: execution occurs when all required operands with matching signatures arrive at a PE

**Distributed CAM Mechanism**
- Each PE maintains local matching logic for its spatial position
- Token signatures encode both spatial coordinates and temporal (recurrence) information
- Eliminates need for centralized CAM, enabling scalability
- Physical location in array corresponds to logical position in computation

### 1.3 Data Flow Path

The complete data flow through a DFA machine follows this pipeline:

```
Main Memory 
    ↓
Memory Controller
    ↓
DMA Engines (system-level block schedule)
    ↓
Distributed Scratchpad Memory
    ↓
Streamers (cache lines → token streams)
    ↓
Compute Fabric Edge
    ↓
Systolic Array (spatial computation)
    ↓
Compute Fabric Edge (result tokens)
    ↓
Streamers (token streams → vectors)
    ↓
Distributed Scratchpad Memory
    ↓
DMA Engines (block retrieval)
    ↓
Memory Controller
    ↓
Main Memory
```

### 1.4 Execution Model

DFA execution proceeds in three major phases:

1. **Input Phase**: DMA engines push input data blocks according to the domain schedule, filling the scratchpad and streaming data into the compute fabric.

2. **Computation Phase**: The systolic array executes the SURE in lock-step, with data tokens flowing through the spatial structure and matching at appropriate PEs to trigger local operations.

3. **Output Phase**: Result tokens flow back to fabric edges, are transformed by Streamers into memory-efficient representations, and are moved by DMAs back to main memory.

The push-based nature eliminates the need for global synchronization or centralized scheduling during computation, as the spatial structure and token signatures encode all necessary coordination information.

---

## 2. KPU Specialization

### 2.1 Micro-Architecture Philosophy

The Knowledge Processing Unit (KPU) represents a practical micro-architectural implementation of the abstract DFA concept, analogous to how an NVIDIA GPU is a specific implementation of the Single Instruction Multiple Thread (SIMT) data-parallel model. The KPU introduces hierarchical memory structures, sophisticated data movement engines, and optimization features required for high-performance linear algebra execution on real silicon.

### 2.2 Hierarchical Scratchpad Memory

On-chip memory performance degrades with size, necessitating a cache-like hierarchy to bridge the latency gap between external DRAM and the high-bandwidth compute fabric.

#### 2.2.1 Three-Level Scratchpad Hierarchy

**L3 Scratchpad (Distributed Large Buffers)**
- Composed of 1-8MB tiles distributed across the die
- Serves as the primary on-chip staging area for DMA transfers
- Provides locality for multiple operators in a computation graph
- Large capacity enables substantial working set retention

**L2 Scratchpad (Concurrent Access Buffers)**
- Organized as 16-64KB tiles
- Aggregated into 128-512KB blocks
- Designed for high concurrency to support:
  - Double and triple buffering schemes
  - Simultaneous ingress and egress port access
  - Overlapped computation and data movement
- Intermediate staging between bulk storage (L3) and streaming subsystem

**L1 Scratchpad (Stream Buffers)**
- Smallest, fastest tier
- Holds streams ready for immediate injection into compute fabric rows/columns/diagonals
- Sized to match fabric consumption rate
- Minimal latency to PE array

#### 2.2.2 Memory Hierarchy Rationale

Performance constraints drive the hierarchy design. A small 16x16 systolic fabric operating at 2GHz requires 32 data elements every 500 picoseconds. While this ingress bandwidth is achievable with small local buffers, larger tile sizes improve data movement efficiency by amortizing transfer overhead across bigger payloads. The three-level hierarchy resolves this tension by enabling efficient bulk transfers at outer levels while maintaining high throughput at the compute fabric interface.

### 2.3 Data Movement Control Engines

#### 2.3.1 Engine Hierarchy

**DMA Engines**
- Operate between external DRAM and distributed L3 scratchpad
- Programmed with system-level domain schedule
- Handle block-level data transfers with minimal CPU intervention

**BlockMover Engines**
- Transfer data between L3 and L2 scratchpad layers
- Manage intermediate block staging and buffering
- Coordinate with Streamers to maintain pipeline flow
- Enable concurrent read/write operations to L2 buffers

**Streamers**
- Transfer between L2 scratchpad and L1 stream buffers
- Transform memory-oriented vectors into fabric-oriented token streams
- Generate row/column/diagonal injection patterns
- Perform on-the-fly data layout transformations (e.g., transposition)
- Reverse transformation on egress: collect result tokens and pack into vectors

#### 2.3.2 Bandwidth-Matched Data Path

For compute-bound operators like matrix-matrix multiplication, the entire data path must be bandwidth-matched to the compute fabric's ingress/egress requirements:

```
DRAM ↔ Memory Controller ↔ DMA ↔ L3 ↔ BlockMover ↔ L2 ↔ Streamer ↔ L1 ↔ Compute Fabric
```

Each stage's transfer rate, bus width, and clock frequency are tuned to match the systolic array's consumption rate. The ingress bandwidth to the compute fabric is the most demanding requirement and effectively determines the micro-architectural family characteristics for a given KPU implementation.

### 2.4 Page Cache

**Integration with Memory Controller**
- On-chip memory buffer for DRAM request coalescing
- Aggregates multiple small requests into large memory chunks
- Enables efficient DRAM access up to full page size transfers
- Maximizes memory bandwidth utilization
- Amortizes address energy across maximum data payload

**Decoupling Benefits**
- Isolates DMA engines from DRAM timing constraints
- Provides fast write path for result eviction
- Reduces contention between read and write paths
- Improves overall system throughput

### 2.5 Programming Model

#### 2.5.1 Domain Flow Programs

**SURE Transformation**
- Systems of Uniform Recurrence Equations (SUREs) are transformed into recurrence update instructions
- Each instruction represents the equation computed by processing elements
- Domain constraints specify routing behavior:
  - Internal routing: send results to next computation in array
  - External routing: identify result tokens for egress from array

**Program Distribution**
- All processing elements receive the same SURE program
- The collective set of PE programs constitutes the Domain Flow Program
- Domain Flow Program defines the computational "kernel" for the array
- Program loading occurs as a wavefront ahead of first data tokens
- Programming latency nearly hidden from pipeline execution

**Reprogrammability**
- Different computations require loading new Domain Flow Programs
- Enables single hardware to execute multiple operator types
- Key differentiator from fixed-function systolic arrays (e.g., TPU)
- Supports trend toward smaller, specialized transformer models

#### 2.5.2 Processing Element Flexibility

**Arithmetic Precision Options**
- Integer: INT4, INT8, INT16, INT32
- Floating-point: FP4, FP8, FP16, FP32
- Brain float: BF16

**PE Architectural Variants**
- Scalar ALUs: single FMA per cycle
- Vector units: multiple elements per cycle
- Packed SIMD units: data-level parallelism within PE
- TensorCores: 4x4, 8x8, or 16x16 micro-matrix units

**Domain Flow Compatibility**
- Domain flow execution model remains unchanged across PE types
- Only granularity changes: scalar vs. vector vs. block oriented
- Tensor algebra naturally maps to TensorCore implementations

### 2.6 Streamer Enhancements

**Primary Functions**
- Extract vectors from L2 scratchpad
- Rearrange elements into fabric-appropriate token streams
- Inject streams into compute fabric rows/columns/diagonals

**On-the-Fly Transposition**
- Transforms data layout when storage format differs from required computation format
- Handles row-major to column-major conversions
- Eliminates need for separate transposition passes
- Zero additional memory traffic for layout transformation

### 2.7 Operator Fusion and Special Function Units

#### 2.7.1 Push-Based Fusion

**Architecture Advantage**
- Results emerge from array in rows/columns/diagonals
- Natural insertion points for fused operations in data path
- Eliminates memory round-trips for intermediate results
- Avoids data movement energy waste

**Fusible Operations**
- Bias addition
- Activation functions
- Arbitrary complex activations via SFU banks

**Contrast with GPU Architecture**
- No resource contention management required
- No thread divergence issues
- Clean, simple data flow maintained
- SFU banks operate on push-based streams without scheduling complexity

#### 2.7.2 Special Function Unit Banks

- Layered alongside data egress paths
- Support complex activation functions (sigmoid, tanh, GELU, etc.)
- Operate on streaming data without stalls
- Multiple SFU instances eliminate bottlenecks

### 2.8 Quantization Support

**Just-In-Time Quantization**
- Leverage lower-precision representations to reduce data movement energy
- Expand to computation precision only when needed
- Minimize memory bandwidth and storage requirements

**Multi-Stage Quantization Points**
- DMA engines: quantize during DRAM transfers
- BlockMovers: quantize during inter-scratchpad transfers
- Streamers: quantize during stream generation
- Processing Elements: quantize during computation

**Bidirectional Capability**
- Input stream quantization: compress incoming data
- Output stream quantization: compress results
- Transparent to programming model
- Trivial integration into push-based data path

### 2.9 Sparsity Support

**Sparse Representation**
- Block-level sparsity encoding during data movement
- Reduces bandwidth requirements for sparse matrices
- Exploits structure in transformer attention patterns and weight matrices

**Dynamic Gating**
- Zero-value detection in data path
- Clock/power gating when zeros propagate through pipeline
- Improves energy efficiency without algorithm changes
- Applicable at multiple pipeline stages

**Implementation Simplicity**
- Natural fit for push-based architecture
- No complex scheduling or resource allocation
- Transparent to domain flow program

### 2.10 KPU Differentiation

The KPU's programmability distinguishes it from fixed-function systolic arrays:

- Domain Flow Programs enable operator flexibility on single hardware
- Supports diverse neural network architectures
- Accommodates evolving model designs (e.g., smaller specialized transformers)
- Reduces need for multiple specialized accelerators
- Maintains efficiency of spatial computing while adding adaptability

The combination of hierarchical memory, bandwidth-matched data paths, flexible processing elements, and in-stream optimization (fusion, quantization, sparsity) creates a micro-architecture that balances performance, efficiency, and programmability for knowledge processing workloads.

### 2.11 Knowledge Processing: Operational Scope

The term "Knowledge Processing Unit" reflects the KPU's ability to execute the complete spectrum of linear algebra algorithms that underpin modern computational intelligence and scientific computing. Knowledge processing encompasses any and all linear algebra operators, executed in an energy-efficient, parallel, Multiple Instruction Multiple Data (MIMD) fashion through the Domain Flow Architecture.

#### 2.11.1 Algorithmic Coverage

**Digital Signal Processing**
- Filtering operations (FIR, IIR)
- Fourier transforms (FFT, DFT)
- Convolution and correlation
- Spectral analysis

**Basic Linear Algebra (BLAS)**
- Level 1: Vector operations (dot product, norms, scaling)
- Level 2: Matrix-vector operations (GEMV, rank updates)
- Level 3: Matrix-matrix operations (GEMM, TRMM, SYRK)

**Advanced Linear Algebra**
- Tensor contractions and reshaping
- Kronecker products
- Batched operations

**Statistical and Probabilistic Computing**
- Sensor fusion algorithms
- Bayesian inference and filtering
- Kalman filtering (extended, unscented variants)
- Particle filters

**Numerical Methods**
- Constraint solving (linear and quadratic programming)
- Direct solvers (LU, Cholesky, QR decompositions)
- Iterative solvers (conjugate gradient, GMRES, Jacobi, Gauss-Seidel)
- Optimization algorithms (gradient descent, Newton methods, L-BFGS)

**Neural Network Operations**
- Dense layer computations
- Convolutional operations
- Attention mechanisms (scaled dot-product, multi-head)
- Activation functions and normalizations
- Recurrent computations (LSTM, GRU)

#### 2.11.2 Execution Paradigm

The DFA concept enables any algorithm expressible as a System of Uniform Recurrence Equations to execute in domain flow fashion. The KPU's programmability allows it to adapt to the specific dataflow pattern of each operator, rather than forcing operators to conform to a fixed hardware structure. This architectural flexibility is fundamental to the KPU's ability to efficiently execute the diverse computational patterns found across the linear algebra algorithm space.

The MIMD execution model—where each processing element can potentially execute different operations based on its spatial position and the recurrence equations—contrasts with SIMD architectures that require uniform operations across data elements. This flexibility is essential for algorithms with varying computational requirements across the problem domain, such as sparse matrix operations, adaptive filtering, or spatially-varying convolutions.



## 3. Energy-Consuming Operations

This section catalogs the primary energy-consuming events within a KPU system, organized by architectural subsystem. Each operation is characterized by key parameters that influence energy consumption. This enumeration facilitates energy modeling and analysis of Domain Flow Architecture configurations.

### 3.1 Memory Subsystem Operations

#### 3.1.1 External DRAM Operations

**DRAM Read**
- Parameters: access size (bytes), burst length, row buffer hit/miss
- Energy components: row activation, column access, I/O driver, refresh overhead
- Access pattern: sequential bursts preferred for energy efficiency
- Triggered by: DMA read requests via memory controller

**DRAM Write**
- Parameters: access size (bytes), burst length, row buffer hit/miss
- Energy components: row activation, column access, I/O driver, write buffer
- Access pattern: sequential bursts preferred for energy efficiency
- Triggered by: DMA write requests via memory controller

**DRAM Refresh**
- Parameters: refresh rate, memory size
- Energy components: distributed background refresh cycles
- Continuous overhead independent of computation

#### 3.1.2 Page Cache Operations

**Page Cache Read**
- Parameters: cache line size, hit/miss rate
- Energy components: SRAM read, tag comparison, multiplexing
- Triggered by: DMA read coalescing

**Page Cache Write**
- Parameters: cache line size, write-through/write-back policy
- Energy components: SRAM write, tag update, write buffer management
- Triggered by: DMA write coalescing

**Page Cache Eviction**
- Parameters: eviction policy (LRU, etc.), writeback required
- Energy components: SRAM read, DRAM write (if dirty)
- Triggered by: capacity misses, explicit flushes

#### 3.1.3 L3 Scratchpad Operations

**L3 Read**
- Parameters: access size (cache line), tile location, bank conflicts
- Energy components: SRAM read, address decode, data multiplexing
- Access pattern: distributed across die, variable distance to requestor
- Triggered by: BlockMover read requests

**L3 Write**
- Parameters: access size (cache line), tile location, bank conflicts
- Energy components: SRAM write, address decode, data routing
- Access pattern: distributed across die, variable distance from source
- Triggered by: DMA writes from DRAM, BlockMover writebacks

#### 3.1.4 L2 Scratchpad Operations

**L2 Read**
- Parameters: access size, buffer index, concurrent port access
- Energy components: SRAM read, port arbitration, output multiplexing
- Concurrency: multiple simultaneous accesses supported
- Triggered by: Streamer read requests

**L2 Write**
- Parameters: access size, buffer index, concurrent port access
- Energy components: SRAM write, port arbitration, input demultiplexing
- Concurrency: overlapped with read operations
- Triggered by: BlockMover writes, Streamer writebacks

**L2 Buffer Swap**
- Parameters: buffer configuration update
- Energy components: control register update, minimal data movement
- Triggered by: double/triple buffer rotation

#### 3.1.5 L1 Scratchpad Operations

**L1 Read**
- Parameters: stream element size, read port count
- Energy components: SRAM read, minimal decode logic
- Latency: single-cycle to fabric
- Triggered by: Streamer push into fabric

**L1 Write**
- Parameters: stream element size, write port count
- Energy components: SRAM write, write buffer management
- Latency: single-cycle from fabric
- Triggered by: result token collection from fabric edge

### 3.2 Data Movement Operations

#### 3.2.1 DMA Engine Operations

**DMA Transfer Setup**
- Parameters: descriptor size, transfer parameters
- Energy components: configuration register writes, address calculation
- Frequency: per transfer block
- Triggered by: system schedule programming

**DMA Read Transfer (DRAM → L3)**
- Parameters: transfer size (bytes), number of bursts, physical distance
- Energy components: DRAM read, on-chip interconnect traversal, L3 write
- Energy scales with: transfer size, routing distance, burst efficiency
- Triggered by: scheduled input data loading

**DMA Write Transfer (L3 → DRAM)**
- Parameters: transfer size (bytes), number of bursts, physical distance
- Energy components: L3 read, on-chip interconnect traversal, DRAM write
- Energy scales with: transfer size, routing distance, burst efficiency
- Triggered by: scheduled result data writeback

**DMA Synchronization**
- Parameters: completion signaling mechanism
- Energy components: status register update, potential interrupt
- Frequency: per DMA transfer completion

#### 3.2.2 BlockMover Operations

**BlockMover Transfer Setup**
- Parameters: source/destination addresses, block dimensions
- Energy components: configuration register writes, address calculation
- Frequency: per block transfer
- Triggered by: L3 to L2 movement scheduling

**BlockMover Read (L3 → L2)**
- Parameters: block size (KB), tile-to-block distance
- Energy components: L3 read, on-chip interconnect, L2 write
- Energy scales with: block size, physical routing distance
- Triggered by: prefetching for upcoming computation

**BlockMover Write (L2 → L3)**
- Parameters: block size (KB), block-to-tile distance
- Energy components: L2 read, on-chip interconnect, L3 write
- Energy scales with: block size, physical routing distance
- Triggered by: result buffer eviction

#### 3.2.3 Streamer Operations

**Streamer Vector Read (L2 → L1)**
- Parameters: vector length, element size
- Energy components: L2 read, format conversion logic, L1 write
- Frequency: per vector streamed into fabric
- Triggered by: fabric input demand

**Streamer Vector Write (L1 → L2)**
- Parameters: vector length, element size
- Energy components: L1 read, format conversion logic, L2 write
- Frequency: per result vector collected from fabric
- Triggered by: fabric output availability

**Streamer Transposition**
- Parameters: matrix dimensions, element size
- Energy components: read buffer, transpose logic, write buffer
- Additional energy: intermediate buffering, address recalculation
- Triggered by: layout mismatch between storage and computation

**Streamer Stream Generation**
- Parameters: stream length, injection pattern (row/column/diagonal)
- Energy components: sequencing logic, token formatting, routing control
- Frequency: continuous during computation phase
- Triggered by: fabric ready signals

### 3.3 Compute Fabric Operations

#### 3.3.1 Token Routing

**Inter-PE Token Transfer**
- Parameters: token size (data + signature), routing distance
- Energy components: wire capacitance, repeaters, router logic
- Pattern: nearest-neighbor for most transfers
- Frequency: every cycle during computation
- Triggered by: PE result production

**Fabric Edge Injection**
- Parameters: token count, injection bandwidth
- Energy components: edge router logic, initial routing decision
- Frequency: per input token
- Triggered by: Streamer push operations

**Fabric Edge Collection**
- Parameters: token count, collection bandwidth
- Energy components: edge collection logic, token buffering
- Frequency: per output token
- Triggered by: result token arrival at fabric boundary

#### 3.3.2 Processing Element Operations

**Token Signature Matching**
- Parameters: signature width (spatial + recurrence tags), match complexity
- Energy components: comparator logic, CAM-like structures
- Frequency: every token arrival at PE
- Energy optimization: distributed matching reduces global CAM energy
- Triggered by: token arrival at PE input ports

**PE Computation (Arithmetic)**
- Parameters: operation type (FMA, etc.), data precision, operand count
- Energy components: arithmetic units (multiplier, adder), register file access
- Variants by precision:
  - INT4/INT8/INT16/INT32: integer arithmetic energy
  - FP8/FP16/FP32: floating-point arithmetic energy
  - BF16: brain float arithmetic energy
- Variants by PE type:
  - Scalar ALU: single operation energy
  - Vector unit: multiple parallel operations
  - TensorCore (4x4, 8x8, 16x16): block operation energy
- Frequency: per matched operand set
- Triggered by: signature match completion

**PE Result Token Formation**
- Parameters: token size, signature computation
- Energy components: result packaging, signature update, routing tag generation
- Frequency: per PE computation completion
- Triggered by: arithmetic operation completion

**PE Instruction Decode**
- Parameters: instruction width, decode complexity
- Energy components: instruction fetch from local memory, decode logic
- Frequency: per computation cycle (shared across systolic lock-step)
- Triggered by: clock cycle in active computation phase

#### 3.3.3 Domain Flow Program Loading

**Program Broadcast**
- Parameters: program size, number of PEs, fabric dimensions
- Energy components: program memory writes across all PEs, broadcast network
- Frequency: per kernel launch
- Energy amortization: hidden in pipeline latency, infrequent relative to computation
- Triggered by: operator change requiring new SURE program

### 3.4 Control and Coordination

**System Schedule Programming**
- Parameters: schedule complexity, DMA descriptor count
- Energy components: configuration register writes across DMA engines
- Frequency: per operator launch
- Triggered by: operator dispatch

**Synchronization and Barriers**
- Parameters: synchronization scope, barrier count
- Energy components: status checking, signal propagation
- Frequency: between pipeline phases, after operator completion
- Triggered by: phase transitions (input → compute → output)

**Clock Distribution**
- Parameters: clock frequency, distribution network area
- Energy components: clock tree traversal, flip-flop clocking
- Continuous overhead: every cycle across all active subsystems
- Major contributor: high-frequency fabric operation

**Power Management Transitions**
- Parameters: domain size, voltage/frequency change magnitude
- Energy components: domain state transitions, voltage regulator updates
- Frequency: workload-dependent, typically coarse-grained
- Triggered by: idle detection, performance scaling decisions

### 3.5 Optimization and Special Operations

#### 3.5.1 Operator Fusion

**Bias Addition**
- Parameters: vector length, element precision
- Energy components: adder logic in data path, bias value storage/fetch
- Location: fabric edge, between PE output and Streamer input
- Frequency: per result vector if bias present
- Energy savings: eliminates L2 write → read → add sequence

**Activation Function (SFU)**
- Parameters: activation type (ReLU, sigmoid, tanh, GELU, etc.), precision
- Energy components: SFU arithmetic (lookup tables, polynomial approximation, etc.)
- Location: fabric edge, post-computation data path
- Frequency: per result element if activation present
- Energy comparison: SFU energy vs. avoided memory round-trip

#### 3.5.2 Quantization

**Quantization (Higher → Lower Precision)**
- Parameters: input precision, output precision, quantization method
- Energy components: scaling, rounding, clipping logic
- Locations: DMA engines, BlockMovers, Streamers, PEs
- Frequency: per element in data stream
- Energy trade-off: quantization logic vs. reduced data movement and storage

**Dequantization (Lower → Higher Precision)**
- Parameters: input precision, output precision, scale factors
- Energy components: scaling logic, precision expansion
- Locations: DMA engines, BlockMovers, Streamers, PEs
- Frequency: per element in data stream (just-in-time expansion)
- Energy trade-off: dequantization logic vs. reduced upstream data movement

#### 3.5.3 Sparsity Handling

**Sparse Encoding/Decoding**
- Parameters: sparsity format (CSR, COO, block sparse, etc.), sparsity ratio
- Energy components: compression/decompression logic, metadata handling
- Locations: any data movement stage
- Frequency: per sparse block transfer
- Energy trade-off: encoding overhead vs. reduced data transfer

**Zero-Value Gating**
- Parameters: zero detection granularity, gating scope
- Energy components: zero detection logic, clock/power gate control
- Locations: throughout data path (DMAs, BlockMovers, Streamers, PEs)
- Frequency: per element or per vector
- Energy savings: clock tree, arithmetic units, routing when zeros propagate
- Dynamic: adapts to actual data sparsity without algorithm modification

### 3.6 Energy Event Summary by Pipeline Phase

For a complete operator execution, energy events occur in distinct pipeline phases:

**Input Phase**
- DRAM read (operand loading)
- DMA transfers (DRAM → L3)
- BlockMover transfers (L3 → L2)
- Streamer operations (L2 → L1)
- Token injection into fabric
- Optional: dequantization, sparse decoding

**Computation Phase**
- PE token signature matching (continuous)
- PE arithmetic operations (continuous)
- Inter-PE token routing (continuous)
- PE instruction decode (continuous, amortized in lock-step)
- Clock distribution (continuous, major contributor at high frequency)
- Optional: zero-value gating (dynamic savings)

**Output Phase**
- Token collection from fabric edge
- Optional: bias addition, activation (SFU)
- Streamer operations (L1 → L2)
- BlockMover transfers (L2 → L3)
- DMA transfers (L3 → DRAM)
- DRAM write (result storage)
- Optional: quantization, sparse encoding

**Overhead (Amortized)**
- Domain Flow Program loading (per kernel launch)
- System schedule programming (per operator)
- Configuration and synchronization
- Power management

---

## 4. Energy Modeling Considerations

When extracting energy events for modeling, consider:

1. **Frequency of Operations**: Some events occur once per operator (program loading), others occur millions of times per operator (PE arithmetic). Frequency multipliers are critical for accurate models.

2. **Data-Dependent Energy**: Quantization and sparsity optimizations produce data-dependent energy savings. Models should account for average-case, worst-case, and best-case scenarios.

3. **Amortization Effects**: Large tile sizes and long bursts amortize control and addressing overhead. Energy-per-byte improves with transfer granularity.

4. **Distance Effects**: Physical routing distance impacts energy for data movement operations. Distributed L3 and fabric size affect inter-component transfer energy.

5. **Concurrency**: L2 concurrent access, overlapped pipeline stages, and double buffering enable energy-efficient throughput by reducing stalls and maximizing utilization.

6. **Technology Scaling**: SRAM energy, DRAM energy, and arithmetic energy scale differently with process technology. Energy models must be parameterized by technology node.

7. **Clock Frequency**: Higher fabric frequencies increase performance but have quadratic impact on dynamic power (linear frequency × linear voltage increase for timing).

8. **Optimization Trade-offs**: SFU fusion, quantization, and sparsity add logic energy but save memory energy. Net energy depends on operator characteristics and data properties.

---

## 5. KPU Product Families and Configurations

The KPU architecture scales across multiple product families targeting different market segments and application requirements. The scalability is achieved through a modular tile-based design that maintains architectural consistency while varying compute capacity and memory resources.

### 5.1 Tile-Based Floorplan Architecture

**Checkerboard Pattern**
- 2D alternating arrangement of L3 memory tiles and compute tiles
- Pitch-matched tiles ensure efficient die coverage and routing
- Physical proximity minimizes L3-to-fabric communication energy and latency
- Scalable pattern: larger dies simply extend the checkerboard grid

**Tile Composition**
- Compute tile: Contains a systolic fabric array, L2 scratchpad, L1 stream buffers, Streamers, and BlockMovers
- L3 memory tile: Contains distributed scratchpad SRAM banks with local addressing and routing

**Design Constraint**
- Fabric dimensions (PE array size) are constrained by L3 tile and compute tile physical dimensions
- Tile size determines practical fabric sizes: 16×16 to 128×128 PE arrays
- Larger fabrics reduce hardware utilization due to increased idle cycles during boundary conditions and synchronization overhead

### 5.2 Product Family Specifications

#### 5.2.1 KPU-T64: Edge and Drone Applications

**Target Market**
- Autonomous drones and UAVs
- Edge AI smart sensors
- Low-power embedded vision systems

**Configuration**
- Compute tiles: 64
- Die organization: 8×8 checkerboard (4×4 compute tiles alternating with L3 tiles)
- Fabric size per compute tile: 16×16 to 32×32 PEs (total: 16K-64K PEs)
- Target process technology: 22nm (cost-optimized for edge deployment)

**Characteristics**
- Power budget: 5-15W typical
- Emphasis on energy efficiency over absolute performance
- Suitable for inference workloads with modest model sizes
- Cost-sensitive design for high-volume edge markets

#### 5.2.2 KPU-T256: Robotics and Embodied AI

**Target Market**
- Humanoid robots
- Quadruped robots and mobile manipulation platforms
- Autonomous mobile robots (AMR)
- Real-time perception and control systems

**Configuration**
- Compute tiles: 256
- Die organization: 16×16 checkerboard (8×8 compute tiles alternating with L3 tiles)
- Fabric size per compute tile: 32×32 to 64×64 PEs (total: 256K-1M PEs)
- Target process technology: 16nm, 7nm (automotive-certified processes)

**Characteristics**
- Power budget: 25-75W typical
- Balance of performance and efficiency for real-time robotics
- Sufficient compute for sensor fusion, SLAM, planning, and control
- Latency-optimized for closed-loop control applications

#### 5.2.3 KPU-T768: Automotive and High-Performance Applications

**Target Market**
- Autonomous vehicles (L3-L5 autonomy)
- Advanced driver assistance systems (ADAS)
- In-vehicle AI and infotainment processing
- Industrial automation and machine vision

**Configuration**
- Compute tiles: 768
- Die organization: Approximately 28×28 checkerboard (14×14 compute tiles alternating with L3 tiles)
- Fabric size per compute tile: 64×64 to 128×128 PEs (total: 3M-12M PEs)
- Target process technology: 16nm, 7nm, 4nm (automotive-certified processes)

**Characteristics**
- Power budget: 75-250W typical
- High throughput for multi-sensor fusion and multi-task AI workloads
- Automotive functional safety (ISO 26262) compliance requirements
- Redundancy and fault tolerance features for safety-critical applications
- Capable of running multiple concurrent AI workloads (e.g., perception, planning, occupant monitoring)

### 5.3 Technology Node Strategy

**Automotive-Certified Processes (Primary Focus)**
- 16nm: Mature, high-yield, cost-effective for automotive qualification
- 7nm: Balance of performance and power for mid-range automotive applications
- 4nm: Advanced node for premium automotive and high-performance robotics

**Edge-Optimized Processes**
- 22nm: Cost-optimized for high-volume edge and drone applications
- Emphasis on energy efficiency and low manufacturing cost
- Sufficient performance for inference-focused workloads

**Non-Target: Datacenter Nodes**
- No plans for 3nm or more advanced bleeding-edge processes
- Datacenter market (hyperscale training) not a strategic focus
- KPU positioning: edge-to-vehicle intelligence, not cloud training

### 5.4 Scalability Characteristics

**Memory Hierarchy Scaling**
- L3 capacity scales linearly with die size (more L3 tiles in larger configurations)
- L2 and L1 capacities scale with compute tile count
- Memory bandwidth scales with number of compute tiles (distributed access)

**Compute Capacity Scaling**
- Total PE count: O(N²) where N is checkerboard dimension
- Aggregate FLOPS/TOPS: Proportional to PE count and clock frequency
- Scalable from 16K PEs (T64) to 12M PEs (T768) with consistent architecture

**Energy Efficiency Scaling**
- Smaller configurations (T64): Maximum energy efficiency, lower absolute performance
- Larger configurations (T768): Higher absolute performance, efficiency maintained through spatial distribution
- Energy per operation remains relatively constant across families due to architectural consistency

**Die Size and Yield**
- T64: Smaller die, higher yield, lower cost per unit
- T768: Larger die, lower yield, higher cost, but proportionally higher value
- Checkerboard modularity aids in design reuse and defect tolerance strategies

### 5.5 Configuration Parameters for Energy Modeling

*Note: The following parameters are representative ranges. Specific implementations will vary based on process technology, market requirements, and design optimization.*

#### KPU-T64 Estimated Parameters
- L3 scratchpad total: 16-32 MB (distributed across L3 tiles)
- L2 scratchpad per compute tile: 128-256 KB
- L1 stream buffers per compute tile: 16-32 KB
- Compute fabric frequency: 1.0-1.5 GHz
- Memory controller frequency: 800-1200 MHz
- DRAM bandwidth: 50-100 GB/s (LPDDR4/LPDDR5)
- Peak performance: 2-8 TOPS INT8 (configuration dependent)

#### KPU-T256 Estimated Parameters
- L3 scratchpad total: 64-128 MB
- L2 scratchpad per compute tile: 256-512 KB
- L1 stream buffers per compute tile: 32-64 KB
- Compute fabric frequency: 1.5-2.0 GHz
- Memory controller frequency: 1.0-1.5 GHz
- DRAM bandwidth: 150-300 GB/s (LPDDR5/LPDDR5X)
- Peak performance: 32-128 TOPS INT8 (configuration dependent)

#### KPU-T768 Estimated Parameters
- L3 scratchpad total: 192-384 MB
- L2 scratchpad per compute tile: 512 KB - 1 MB
- L1 stream buffers per compute tile: 64-128 KB
- Compute fabric frequency: 1.5-2.5 GHz
- Memory controller frequency: 1.2-2.0 GHz
- DRAM bandwidth: 400-800 GB/s (LPDDR5X/GDDR6)
- Peak performance: 256-1000 TOPS INT8 (configuration dependent)

**Parameter Sensitivity**
- Process technology significantly impacts frequency, power, and memory density
- Fabric size (16×16 vs 128×128 per tile) dramatically affects compute tile characteristics
- PE type (scalar vs vector vs TensorCore) scales performance proportionally at similar energy
- Precision (INT4/8 vs FP16/32) impacts both performance (throughput) and energy per operation

---

## 6. Operator Coverage and Compilation Model

The KPU's programmable architecture enables complete operator coverage across modern deep learning frameworks, fundamentally simplifying the compilation flow compared to fixed-function accelerators or GPU-targeted approaches.

### 6.1 Complete Operator Support

**Framework Compatibility**
- TensorFlow: All operators in standard operator graph
- PyTorch: Complete operator coverage including custom operators expressible as linear algebra
- JAX: Full XLA operator set support
- ONNX: Complete ONNX operator specification

**Operator Categories**
- Linear layers and matrix operations
- Convolutional layers (1D, 2D, 3D, depthwise, grouped)
- Attention mechanisms (multi-head attention, flash attention variants)
- Recurrent operations (LSTM, GRU, vanilla RNN)
- Normalization (BatchNorm, LayerNorm, GroupNorm, RMSNorm)
- Activation functions (ReLU, GELU, Swish, Sigmoid, Tanh, etc.)
- Pooling operations (average, max, adaptive)
- Reduction operations (sum, mean, max, min along axes)
- Element-wise operations (add, multiply, divide, etc.)
- Reshape and transpose operations
- Embedding lookups and sparse operations

### 6.2 Direct Execution Model

**Architectural Advantage**
- KPU hardware adapts to operator requirements rather than forcing operators to conform to fixed execution patterns
- Each operator is directly executed as its natural SURE representation
- No need for complex graph rewriting or operator fusion passes in compiler
- Hardware performs automatic and implicit operator fusion through dataflow

**Contrast with GPU Compilation**
- GPU approach: Deep learning compilers (TVM, XLA, TensorRT) transform operator graphs into kernel sequences optimized for GPU execution model
- KPU approach: Minimal transformation—operators map directly to Domain Flow Programs
- Compiler complexity shifts from operator optimization to memory management

### 6.3 Simplified Compilation Flow

**KPU Compilation Stages**

1. **Graph Ingestion**
   - Parse operator graph from framework (TensorFlow, PyTorch, JAX, ONNX)
   - Preserve original operator semantics without fusion or rewriting

2. **Bufferization**
   - Analyze operator data dependencies
   - Allocate scratchpad memory (L3, L2) for tensors
   - Determine buffer lifetimes and reuse opportunities
   - Schedule memory allocation to minimize total on-chip memory requirements

3. **Domain Flow Program Generation**
   - Transform each operator into its SURE representation
   - Generate Domain Flow Program (PE instruction sequence)
   - Encode spatial embedding and recurrence variable mappings

4. **Schedule Generation**
   - Create system-level domain schedule for DMA engines
   - Determine tile sizes and blocking strategies for memory hierarchy
   - Program BlockMover and Streamer configurations
   - Schedule overlaps computation and data movement phases

5. **Code Generation**
   - Emit runtime code for operator dispatch
   - Generate configuration for DMA descriptors, BlockMover, and Streamers
   - Package Domain Flow Programs for fabric loading

**What the Compiler Does NOT Do**
- Operator fusion (hardware does this automatically via push-based dataflow)
- Kernel optimization for specific execution patterns
- Manual scheduling of computation across hardware resources
- Thread block sizing and occupancy optimization
- Shared memory management and banking conflict resolution

### 6.4 Automatic Operator Fusion

**Hardware-Driven Fusion**
- Results flow directly from one operator to the next through the dataflow pipeline
- No intermediate memory writes when operators are data-adjacent
- Bias addition, activation, normalization fuse naturally without compiler intervention
- Multi-operator sequences (e.g., Linear → LayerNorm → GELU) execute as single dataflow

**Examples of Automatic Fusion**
- GEMM + Bias + ReLU: Result tokens flow through fabric edge, bias added, ReLU applied by SFU, written once to L2
- Conv2D + BatchNorm: Convolution outputs stream directly into normalization logic
- Attention QKV computation: Three parallel GEMMs feed directly into attention mechanism without intermediate storage

**Energy Efficiency Benefit**
- Eliminates intermediate tensor writes to memory hierarchy
- Reduces DRAM bandwidth requirements
- Lowers energy per inference by avoiding redundant data movement
- Enables deeper fusion than typically achievable with explicit compiler passes

### 6.5 Scheduling and Concurrency Management

**Hardware-Managed Concurrency**
- Token signatures encode all necessary coordination information
- No global synchronization or centralized scheduling during computation
- Processing elements self-coordinate through token matching
- Distributed CAM eliminates centralized bottlenecks

**Compiler Responsibility: Memory Scheduling**
- Determine which tensors reside in which memory tier at what time
- Schedule DMA transfers to overlap with computation
- Manage double/triple buffering for continuous pipeline operation
- Balance memory capacity constraints with performance requirements

**Result: Simplified Software Stack**
- Compiler focuses on memory management (bufferization), not execution optimization
- Hardware handles execution efficiency through spatial dataflow
- Reduced compiler complexity compared to GPU-targeted deep learning compilers
- Faster compilation times and more predictable performance

### 6.6 Implications for AI Workloads

**Model Portability**
- Models trained in any framework run efficiently without framework-specific optimizations
- Same Domain Flow Program works across KPU product families (T64, T256, T768)
- Memory scheduling scales with available scratchpad capacity

**Emerging Model Support**
- New operators and model architectures supported immediately if expressible as linear algebra
- No waiting for vendor-specific kernel libraries or compiler updates
- Rapid adaptation to evolving AI research (e.g., new attention variants, novel architectures)

**Deployment Flexibility**
- Single compilation target across edge-to-automotive spectrum
- Same operator coverage on low-power T64 and high-performance T768
- Simplified deployment pipeline for heterogeneous robotic and automotive fleets

---

## 7. Example Operator Execution

*[Placeholder section for worked examples showing detailed execution traces]*

This section will contain detailed walkthroughs of representative operators executing on the KPU, showing:

### 7.1 Matrix-Matrix Multiply (GEMM) Example
*To be documented:*
- Tiled execution strategy
- System-level domain schedule (DMA programming)
- BlockMover staging sequence
- Streamer injection patterns (row/column streams)
- PE computation and token routing
- Result collection and writeback
- Energy event enumeration with frequencies
- Pipeline occupancy and utilization analysis

### 7.2 Convolutional Layer Example
*To be documented:*
- Input/filter/output tensor layout
- Spatial mapping to systolic array
- Streamer window generation for sliding windows
- PE computation pattern for conv kernel
- Handling of padding and stride
- Comparison with GEMM in terms of energy events

### 7.3 Multi-Head Attention Example
*To be documented:*
- QKV projection (parallel GEMMs)
- Attention score computation (batched matrix multiply)
- Softmax activation via SFU
- Value aggregation
- Output projection
- Automatic fusion opportunities in attention pipeline
- Energy comparison with unfused execution

### 7.4 Fused Operator Example
*To be documented:*
- Example: Linear → LayerNorm → GELU
- Data flow through combined operators without intermediate storage
- Energy savings quantification vs. separate operator execution
- Demonstration of hardware-driven fusion benefits

---

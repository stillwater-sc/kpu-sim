# NoC Integration Options Analysis

## Current State Summary

### NoC Implementations
| Implementation | Purpose | Used By | Complexity |
|----------------|---------|---------|------------|
| `WormholeNoC` | Flit-level wormhole switching | `BlockMoverArray` | ~1000 lines |
| `NoC` (generic) | Packet-switching 2D mesh | Tests only | ~1000 lines |
| `DataflowNoC` | Unified dataflow model | Tests only (new) | ~250 lines |

### Integration Points
```
┌─────────────────────────────────────────────────────────────────────┐
│  kpu-dfg-gen → kpu-dfg-sched → kpu-dfg-compile                      │
│       │              │                │                              │
│   TimingModel    DFGScheduler   BlockMoverCompiler                  │
│  (mesh topology)  (L3 concurrency)  (SEND_EAST, etc.)               │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BlockMoverArray                                                     │
│       │                                                              │
│       └── WormholeNoC (hardcoded)                                   │
│              └── inject_tile(), delivery_callback()                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Option 1: Runtime Configuration (Strategy Pattern)

### Description
Add an abstract `INoC` interface that all NoC implementations conform to. `BlockMoverArray` takes a configuration enum or factory that selects which NoC to instantiate.

### Implementation

```cpp
// Abstract NoC interface
class INoC {
public:
    virtual ~INoC() = default;

    virtual bool inject_tile(uint8_t src, uint8_t dst,
                             const TileDescriptor& tile, uint64_t cycle) = 0;
    virtual void step(uint64_t cycle) = 0;
    virtual void drain(uint64_t& cycle) = 0;
    virtual bool has_inflight_tiles() const = 0;

    using DeliveryCallback = std::function<void(uint8_t dst, const TileDescriptor&,
                                                 uint64_t inject, uint64_t complete)>;
    virtual void set_delivery_callback(DeliveryCallback cb) = 0;
};

// Configuration enum
enum class NoCType {
    WORMHOLE,     // Current default
    DATAFLOW,     // New unified model
    GENERIC       // Packet-switching
};

// Factory
std::unique_ptr<INoC> create_noc(NoCType type, const NoCConfig& config);

// BlockMoverArray with configurable NoC
class BlockMoverArray {
public:
    BlockMoverArray(const Config& config, NoCType noc_type = NoCType::WORMHOLE);
private:
    std::unique_ptr<INoC> noc_;  // Instead of WormholeNoC noc_;
};
```

### Pros
- **Runtime flexibility**: Switch NoC without recompilation
- **Clean separation**: NoC implementation hidden behind interface
- **Testing friendly**: Easy to mock or swap for tests
- **Gradual migration**: Keep WormholeNoC as default, experiment with DataflowNoC

### Cons
- **Virtual call overhead**: Each NoC call goes through vtable (minor)
- **Interface evolution**: Adding new NoC features requires interface changes
- **Lowest common denominator**: Interface must support all implementations
- **Adaptation required**: WormholeNoC and DataflowNoC have different callback signatures

### DFG Tools Impact
- **None**: DFG tools don't directly use NoC types
- Timing model remains the same (mesh-based hop counting)

---

## Option 2: Compile-Time Selection (Template/Policy Pattern)

### Description
Use C++ templates to select NoC implementation at compile time. Different build configurations produce different binaries.

### Implementation

```cpp
// NoC as template parameter
template<typename NoCPolicy>
class BlockMoverArray {
public:
    using NoCType = typename NoCPolicy::noc_type;

    BlockMoverArray(const Config& config)
        : noc_(NoCPolicy::create_config(config)) {}

private:
    NoCType noc_;
};

// Policy classes
struct WormholePolicy {
    using noc_type = noc::WormholeNoC;
    static auto create_config(const BlockMoverArray::Config& cfg) {
        noc::WormholeNoC::Config c;
        c.rows = cfg.mesh_rows;
        c.cols = cfg.mesh_cols;
        return c;
    }
};

struct DataflowPolicy {
    using noc_type = noc::DataflowNoC;
    static auto create_config(const BlockMoverArray::Config& cfg) {
        noc::DataflowNoC::Config c;
        c.rows = cfg.mesh_rows;
        c.cols = cfg.mesh_cols;
        return c;
    }
};

// Usage
using DefaultBlockMoverArray = BlockMoverArray<WormholePolicy>;
using DataflowBlockMoverArray = BlockMoverArray<DataflowPolicy>;
```

### CMake Configuration
```cmake
option(KPU_NOC_TYPE "NoC implementation" "WORMHOLE")

if(KPU_NOC_TYPE STREQUAL "DATAFLOW")
    add_compile_definitions(KPU_USE_DATAFLOW_NOC)
endif()
```

### Pros
- **Zero overhead**: No virtual calls, full inlining possible
- **Type safety**: Compile-time errors for incompatible usage
- **Optimization**: Compiler can optimize for specific NoC
- **Explicit**: Clear which NoC is being used at build time

### Cons
- **Build complexity**: Multiple binaries for different configurations
- **No runtime switching**: Must rebuild to change NoC
- **Template bloat**: Code duplication for each instantiation
- **Header dependencies**: NoC headers included everywhere

### DFG Tools Impact
- **None**: Same as Option 1

---

## Option 3: Parallel Implementations (Fork)

### Description
Keep both NoC implementations completely separate. Create parallel versions of `BlockMoverArray` and related components for each NoC.

### Implementation

```
include/sw/kpu/components/
├── stateful_block_mover.hpp       # Common BlockMover (no NoC)
├── wormhole_block_mover_array.hpp # WormholeNoC integration
└── dataflow_block_mover_array.hpp # DataflowNoC integration

src/components/
├── wormhole_block_mover_array.cpp
└── dataflow_block_mover_array.cpp
```

### Pros
- **No abstraction overhead**: Each implementation is self-contained
- **Independence**: Can evolve separately without coordination
- **Simplicity**: No interfaces or templates to maintain
- **Complete control**: Each can optimize for its NoC model

### Cons
- **Code duplication**: BlockMover logic duplicated
- **Maintenance burden**: Bug fixes needed in multiple places
- **Divergence risk**: Implementations may drift apart
- **Testing overhead**: Need to test both paths

### DFG Tools Impact
- **Minimal**: May need separate compile targets if timing differs significantly

---

## Option 4: Unified Dataflow Only (Replace)

### Description
Fully replace WormholeNoC with DataflowNoC. Remove the old implementation.

### Implementation

```cpp
// BlockMoverArray uses DataflowNoC directly
class BlockMoverArray {
    noc::DataflowNoC noc_;  // Replaced from WormholeNoC
};
```

### Migration Steps
1. Ensure DataflowNoC has feature parity with WormholeNoC
2. Update callback interfaces
3. Verify timing model compatibility
4. Run regression tests
5. Remove WormholeNoC code

### Pros
- **Conceptual simplicity**: One NoC model, one mental model
- **Reduced maintenance**: Only one implementation to maintain
- **Cleaner architecture**: Unified dataflow throughout
- **Smaller codebase**: Remove ~1000 lines of WormholeNoC

### Cons
- **Loss of fidelity**: WormholeNoC models real hardware more closely
- **Breaking change**: Existing tests/benchmarks may need updates
- **Risk**: If DataflowNoC has bugs, no fallback
- **Timing differences**: May affect performance projections

### DFG Tools Impact
- **Timing model**: May need adjustment if DataflowNoC has different latency characteristics
- **Validation**: Need to verify scheduling still produces correct results

---

## Option 5: Adapter Pattern (Wrapper)

### Description
Create an adapter that wraps DataflowNoC to present the WormholeNoC interface, enabling drop-in replacement.

### Implementation

```cpp
// Adapter that makes DataflowNoC look like WormholeNoC
class DataflowNoCAdapter {
public:
    // WormholeNoC-compatible interface
    noc::InjectResult inject_tile(uint8_t src, uint8_t dst,
                                   const TileDescriptor& tile, uint64_t cycle) {
        bool ok = dataflow_noc_.inject_tile(src, dst, tile, cycle);
        return ok ? noc::InjectResult::SUCCESS : noc::InjectResult::BLOCKED;
    }

    void set_tracer(noc::WormholeTracer* tracer) {
        // Adapt tracing events
    }

private:
    noc::DataflowNoC dataflow_noc_;
};

// In BlockMoverArray
#ifdef KPU_USE_DATAFLOW_NOC
    using NoCImpl = DataflowNoCAdapter;
#else
    using NoCImpl = noc::WormholeNoC;
#endif
```

### Pros
- **Minimal changes**: Existing code works unchanged
- **Gradual migration**: Swap implementation behind adapter
- **Testing**: Can compare behavior between implementations
- **Reusable**: Adapter pattern useful for other integrations

### Cons
- **Impedance mismatch**: Callback signatures differ
- **Event mapping complexity**: Wormhole events don't map 1:1 to dataflow
- **Partial fidelity**: Some WormholeNoC features may not translate
- **Extra layer**: Adapter adds conceptual overhead

### DFG Tools Impact
- **None**: Adapter handles interface translation

---

## Option 6: DFG Tool Variants (Parallel Toolchains)

### Description
Create separate DFG tool variants optimized for each NoC model.

### Implementation

```
tools/dfg/
├── common/                    # Shared JSON/spec parsing
├── kpu-dfg-gen/              # Common generator
├── kpu-dfg-sched-wormhole/   # Wormhole-optimized scheduler
├── kpu-dfg-sched-dataflow/   # Dataflow-optimized scheduler
└── kpu-dfg-compile/          # Common compiler
```

### Pros
- **Optimization**: Each scheduler can exploit NoC-specific properties
- **Flexibility**: Different scheduling algorithms per NoC
- **Independence**: Tools can evolve separately

### Cons
- **Duplication**: Similar logic in multiple tools
- **User confusion**: Which tool to use?
- **Maintenance**: Bug fixes needed in multiple places
- **Testing**: More test combinations

---

## Comparison Matrix

| Criterion | Option 1 (Runtime) | Option 2 (Compile-time) | Option 3 (Fork) | Option 4 (Replace) | Option 5 (Adapter) |
|-----------|-------------------|------------------------|-----------------|-------------------|-------------------|
| **Implementation effort** | Medium | Medium | High | Low-Medium | Medium |
| **Runtime flexibility** | High | None | None | N/A | Medium |
| **Performance overhead** | Low | None | None | N/A | Low |
| **Code duplication** | None | Low | High | None | None |
| **Maintenance burden** | Low | Low | High | Lowest | Low |
| **Risk** | Low | Low | Medium | Medium | Low |
| **Backwards compatibility** | Full | Partial | Full | Breaking | Full |
| **DFG tool changes** | None | None | Low | Low-Medium | None |

---

## Recommended Approach

### Short-term: Option 5 (Adapter) + Option 1 (Runtime Configuration)

**Rationale**:
1. Create a `DataflowNoCAdapter` to enable quick testing
2. Add `INoC` interface for clean abstraction
3. Both NoCs can be used, selected at runtime
4. Minimal disruption to existing code

### Medium-term: Option 4 (Replace) if DataflowNoC proves better

**Rationale**:
1. If DataflowNoC is more robust and conceptually cleaner, deprecate WormholeNoC
2. Simplifies the codebase long-term
3. Aligns with unified dataflow philosophy

### Implementation Phases

```
Phase 1: Adapter Layer (1-2 days)
├── Create INoC interface
├── Implement DataflowNoCAdapter
└── Add NoCType enum and factory

Phase 2: Integration (2-3 days)
├── Update BlockMoverArray to use INoC
├── Update callback signatures
└── Run existing tests with both NoCs

Phase 3: Validation (1-2 days)
├── Compare timing behavior
├── Run benchmarks with both
└── Document differences

Phase 4: Decision (ongoing)
├── Gather performance data
├── Evaluate conceptual benefits
└── Decide on long-term path
```

---

## DFG Tools: Required Changes

### If Timing Model Differs

The DFG tools embed timing assumptions in `TimingModel`:

```cpp
// Current assumptions (tile_dataflow_graph.hpp)
struct TimingModel {
    uint64_t l3_bandwidth_bytes_per_cycle = 64;
    uint64_t router_latency_cycles = 1;
    uint64_t link_latency_cycles = 1;
    bool store_and_forward = true;
};
```

**If DataflowNoC has different characteristics**, update:

1. **kpu-dfg-gen**: Allow CLI override of timing parameters
   ```bash
   kpu-dfg-gen --timing-model dataflow ...
   # Or: --router-latency 0 --store-and-forward false
   ```

2. **TimingModel**: Add NoC type awareness
   ```cpp
   enum class NoCModel { WORMHOLE, DATAFLOW };

   TimingModel create_timing_model(NoCModel model) {
       TimingModel tm;
       switch (model) {
           case NoCModel::DATAFLOW:
               tm.router_latency_cycles = 0;  // Stateless forwarding
               tm.store_and_forward = false;   // Tag-based reassembly
               break;
           // ...
       }
       return tm;
   }
   ```

3. **BlockMoverCompiler**: No changes needed (direction-based routing still applies)

### If Topology Changes

The tools assume 2D mesh. If DataflowNoC supports other topologies:

1. Add `--topology` option to kpu-dfg-gen
2. Generalize `hop_count()` calculation
3. Update `get_direction()` in BlockMoverCompiler

---

## Conclusion

The recommended path is:

1. **Immediate**: Create `INoC` interface and `DataflowNoCAdapter`
2. **Short-term**: Enable runtime selection between WormholeNoC and DataflowNoC
3. **Long-term**: Evaluate and potentially converge on DataflowNoC

This approach:
- Minimizes disruption
- Enables experimentation
- Preserves existing functionality
- Provides clear migration path

The DFG tools require minimal changes since they abstract the NoC through timing models and direction-based routing, not direct NoC type dependencies.

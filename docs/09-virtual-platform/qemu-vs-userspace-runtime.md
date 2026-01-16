# QEMU vs User-Space Runtime: Resource Management Analysis

## Executive Summary

This document analyzes where to place resource management in a KPU virtual platform and evaluates the trade-offs between QEMU-based emulation and a user-space runtime approach. **The conclusion: for functional simulation and algorithm development, a user-space runtime with pluggable backends is faster to build, easier to use, and sufficient for the task.**

QEMU becomes valuable later when testing the actual kernel driver code path, but adds complexity without proportional value during early bring-up.

---

## The Design Question

When building a virtual platform for a hardware accelerator, resource management can live in different places:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESOURCE MANAGEMENT SPECTRUM                         │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│   HOST-MANAGED      │      HYBRID         │       ON-CHIP MANAGED           │
│   (Driver Model)    │  (Command Processor)│       (RTOS Model)              │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ • Host driver owns  │ • Host sends cmd    │ • On-chip CPU runs RTOS         │
│   all state         │   buffers           │ • Host sends subprograms        │
│ • MMIO kicks ops    │ • On-chip FSM       │ • On-chip scheduler manages     │
│ • Accelerator is    │   executes cmds     │   all resources                 │
│   "dumb" executor   │ • Complex sched     │ • Host just monitors            │
│                     │   still on host     │                                 │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ QEMU complexity: ★☆☆│ QEMU complexity: ★★☆│ QEMU complexity: ★★★            │
│ Bring-up time: days │ Bring-up time: weeks│ Bring-up time: months           │
│ Realism: Low        │ Realism: Medium     │ Realism: High                   │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

---

## QEMU Approach Analysis

### Architecture: QEMU + Kernel Driver

```
┌──────────────────────┐      ┌──────────────────────────────────────┐
│     Host (Linux)     │      │           QEMU Instance              │
│                      │      │                                      │
│  ┌────────────────┐  │      │  ┌────────────────────────────────┐  │
│  │   KPU Driver   │  │ MMIO │  │      KPU Device Model          │  │
│  │                │  │      │  │                                │  │
│  │ • Buffer mgmt  │◄─┼──────┼─►│ • Register interface           │  │
│  │ • DMA setup    │  │      │  │ • IRQ generation               │  │
│  │ • Scheduling   │  │      │  │ • Calls kpu-sim backend        │  │
│  │ • Sync/fence   │  │      │  │                                │  │
│  └────────────────┘  │      │  └─────────────┬──────────────────┘  │
│                      │      │                │                     │
└──────────────────────┘      │                ▼                     │
                              │  ┌────────────────────────────────┐  │
                              │  │         kpu-sim                │  │
                              │  │   (BEHAVIORAL or CYCLE_ACC)    │  │
                              │  └────────────────────────────────┘  │
                              └──────────────────────────────────────┘
```

### What QEMU Provides

| Capability | Description | When Needed |
|------------|-------------|-------------|
| Real driver execution | Tests actual kernel driver code paths | When driver exists and needs testing |
| DMA/IOMMU semantics | Physical address translation, bounce buffers | When modeling system-level memory |
| Interrupt handling | Real IRQ delivery, coalescing, latency | When testing interrupt-driven designs |
| Multi-process sharing | Multiple processes using device | When testing resource arbitration |
| Boot/enumeration | Device tree, ACPI, probe sequence | When testing system integration |
| Security boundaries | User/kernel isolation, permissions | When testing privilege separation |

### QEMU Implementation Cost

**Device Model (1-2 weeks):**
```c
// qemu/hw/misc/kpu.c - Minimal MMIO register set
#define KPU_REG_CMD_QUEUE_BASE    0x000  // DMA descriptor ring base
#define KPU_REG_CMD_QUEUE_SIZE    0x008  // Ring size
#define KPU_REG_CMD_HEAD          0x010  // Producer index (host writes)
#define KPU_REG_CMD_TAIL          0x014  // Consumer index (device writes)
#define KPU_REG_STATUS            0x020  // Device status
#define KPU_REG_IRQ_STATUS        0x030  // Interrupt status
#define KPU_REG_IRQ_ENABLE        0x034  // Interrupt enable
```

**Kernel Driver (2-4 weeks):**
- Platform driver registration
- DMA buffer management
- IOCTL interface
- IRQ handling
- Sysfs attributes

**Total effort: 4-6 weeks minimum**

---

## User-Space Runtime Approach

### Architecture: Direct Runtime Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│            (PyTorch, TensorFlow, Custom Apps)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KPU Runtime Library                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Resource Management API                       │  │
│  │   • kpu_alloc_buffer()      • kpu_submit_graph()          │  │
│  │   • kpu_map_tensor()        • kpu_wait_completion()       │  │
│  │   • kpu_create_stream()     • kpu_sync_stream()           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Backend Interface (abstract)                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  kpu-sim    │     │  kpu-hw     │     │  kpu-remote │        │
│  │ (functional │     │ (real HW    │     │ (network to │        │
│  │  simulator) │     │  via ioctl) │     │  shared HW) │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: API Abstraction is What Matters

The same resource management API can target:
1. **Functional simulator** (kpu-sim) - for development
2. **Real hardware** (via kernel driver) - for production
3. **Remote hardware** (via network) - for shared access

This is exactly the pattern used by:
- **CUDA**: Runtime can target real GPU, software emulation, or remote GPU
- **ROCm**: HIP runtime abstracts AMD GPU access
- **oneAPI**: SYCL runtime targets CPU, GPU, FPGA backends

### Implementation

```cpp
// include/sw/kpu/runtime/backend.hpp
namespace sw::kpu::runtime {

class Backend {
public:
    virtual ~Backend() = default;

    // Memory management
    virtual BufferHandle alloc_buffer(size_t size, MemoryType type) = 0;
    virtual void free_buffer(BufferHandle h) = 0;
    virtual void* map_buffer(BufferHandle h) = 0;
    virtual void unmap_buffer(BufferHandle h) = 0;

    // Execution
    virtual StreamHandle create_stream() = 0;
    virtual void submit(StreamHandle s, const ComputeGraph& graph) = 0;
    virtual void sync(StreamHandle s) = 0;
    virtual bool query(StreamHandle s) = 0;
};

// Simulator backend - uses existing kpu-sim
class SimBackend : public Backend {
    KPUSimulator sim_;
public:
    SimBackend(SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL);
    // ... implementations call sim_ directly
};

// Hardware backend - for when silicon arrives
class HardwareBackend : public Backend {
    int fd_;  // /dev/kpu0
public:
    HardwareBackend() : fd_(open("/dev/kpu0", O_RDWR)) {}
    // ... implementations use ioctl()
};

// Runtime singleton with backend selection
class Runtime {
public:
    static Runtime& get();
    void init(BackendType type = BackendType::AUTO);
    Backend& backend();
};

}  // namespace sw::kpu::runtime
```

### User Code (Identical Regardless of Backend)

```cpp
#include <sw/kpu/runtime/kpu_runtime.hpp>

int main() {
    using namespace sw::kpu::runtime;

    // Auto-selects: sim if no hardware, hardware if /dev/kpu0 exists
    Runtime::get().init(BackendType::AUTO);
    auto& kpu = Runtime::get().backend();

    // Allocate, compute, read results - same API always
    auto input = kpu.alloc_buffer(1024, MemoryType::L3);
    auto output = kpu.alloc_buffer(1024, MemoryType::L3);

    ComputeGraph graph;
    graph.add_matmul(input, weights, output, dims);

    auto stream = kpu.create_stream();
    kpu.submit(stream, graph);
    kpu.sync(stream);

    return 0;
}
```

---

## Comparison Matrix

### Complexity vs Value

| Factor | User-Space Runtime | QEMU + Kernel Driver |
|--------|-------------------|----------------------|
| **Time to first boot** | 1-2 weeks | 4-6 weeks |
| **Debug ease** | Host GDB, IDE | QEMU monitor + kernel debug |
| **Portability** | Windows, Mac, Linux | Linux guests only |
| **CI/CD integration** | Trivial | Need VM infrastructure |
| **Iteration speed** | Fast | Slow |
| **Driver code testing** | No | Yes |
| **System integration** | No | Yes |

### What Each Approach Tests

| Capability | User-Space | QEMU |
|------------|------------|------|
| Functional correctness | ✓ | ✓ |
| Algorithm development | ✓ | ✓ |
| Performance modeling | ✓ | ✓ |
| Kernel driver code | ✗ | ✓ |
| DMA/IOMMU behavior | ✗ | ✓ |
| Interrupt handling | ✗ | ✓ |
| Multi-process sharing | ✗ | ✓ |
| Boot sequence | ✗ | ✓ |

---

## Recommendation

### Phased Approach

```
PHASE 1 (Now): User-space runtime + kpu-sim backend
├── Fast iteration, easy debugging, portable
├── Validates simulator architecture
├── Enables algorithm development
└── Timeline: 1-2 weeks

PHASE 2 (When driver needed): Add hardware backend
├── Same API, different Backend implementation
├── Runtime auto-selects based on device presence
└── Timeline: 2-3 weeks (driver development)

PHASE 3 (If needed): QEMU for driver testing
├── Only if kernel driver has bugs requiring isolation
├── Useful for system integration testing
└── Timeline: 4-6 weeks
```

### Decision Criteria

**Use User-Space Runtime when:**
- Primary goal is functional simulation
- Algorithm development and bring-up
- Fast iteration on API design
- CI/CD pipeline testing
- Cross-platform development

**Use QEMU when:**
- Testing actual kernel driver implementation
- Validating boot/enumeration sequences
- Testing security boundaries
- Multi-process resource sharing bugs
- System-level integration testing

### Bottom Line

**kpu-sim stays the same regardless of who's orchestrating it.**

The choice between user-space runtime and QEMU is about *where the orchestration happens*, not about the simulation itself. Start with the simpler approach (user-space runtime) and add QEMU complexity only when you have a kernel driver that needs testing.

---

## References

- `docs/09-virtual-platform/virtual_platform_analysis.md` - Overall virtual platform gap analysis
- `docs/07-runtime/resource-management.md` - Runtime resource management design
- `include/sw/kpu/runtime/` - Existing runtime implementation

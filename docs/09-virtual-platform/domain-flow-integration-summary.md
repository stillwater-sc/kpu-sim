# Domain Flow Integration Summary

## What We Built

This integration enables the KPU simulator to load computational graphs from the `domain_flow` repository (https://github.com/branes-ai/domain_flow) for compiler testing and schedule generation.

### Created Files

#### Infrastructure
- ✅ `cmake/DomainFlowIntegration.cmake` - CMake integration logic
  - Supports FetchContent (automatic download)
  - Supports local installation path
  - Optional integration (can disable)

#### Compiler Components
- ✅ `include/sw/compiler/graph_loader.hpp` - Graph loading API
  - `ComputationalGraph` class - IR representation
  - `GraphLoader` interface - Pluggable loaders
  - `DomainFlowGraphLoader` - MLIR/domain_flow support
  - `JSONGraphLoader` - Fallback format

- ✅ `src/compiler/graph_loader.cpp` - Implementation
  - Full JSON graph loading (working now)
  - Stub for MLIR loading (TODO: connect to domain_flow)

- ✅ `src/compiler/CMakeLists.txt` - Build configuration

#### Test Infrastructure
- ✅ `test_graphs/` directory structure
  - `simple/` - Basic operator tests
  - `networks/` - Complete network graphs
  - `benchmarks/` - Performance tests

- ✅ `test_graphs/simple/matmul.mlir` - Example MLIR graph
- ✅ `test_graphs/simple/matmul.json` - Example JSON graph
- ✅ `tests/compiler/test_graph_loader.cpp` - Unit tests
- ✅ `tests/compiler/CMakeLists.txt` - Test build config

#### Documentation
- ✅ `docs/domain-flow-integration.md` - Complete integration guide
- ✅ `test_graphs/README.md` - Test graph documentation
- ✅ `INTEGRATION_SUMMARY.md` - This file

## Integration Options

### Option 1: FetchContent (Recommended for CI/Production)
```bash
cmake -B build -GNinja
cmake --build build
```
- Downloads domain_flow automatically
- Requires CMake 3.28+ (domain_flow requirement)

### Option 2: Local Installation (Recommended for Development)
```bash
# Build domain_flow locally first
cd ~/dev
git clone https://github.com/branes-ai/domain_flow.git
cd domain_flow
cmake --preset user-ninja-release && cmake --build build/user-ninja-release

# Point KPU simulator to local build
cd ~/dev/KPU-simulator
cmake -B build -DKPU_DOMAIN_FLOW_LOCAL_PATH=~/dev/domain_flow -GNinja
cmake --build build
```

### Option 3: JSON Only (No domain_flow)
```bash
cmake -B build -DKPU_USE_DOMAIN_FLOW=OFF -GNinja
cmake --build build
```
- No MLIR dependencies
- Uses JSON test graphs only
- Good for quick iteration

## What Works Now

### ✅ Working Features
1. **JSON Graph Loading** - Full implementation
   ```cpp
   auto graph = load_graph("test_graphs/simple/matmul.json");
   graph->print();
   graph->validate();
   ```

2. **Graph Representation**
   - Operator graph with dependencies
   - Tensor metadata (shape, dtype, layout)
   - Operator attributes
   - Validation

3. **Test Infrastructure**
   - Unit tests for graph loading
   - Test graph examples
   - CTest integration

4. **CMake Integration**
   - Optional domain_flow support
   - Local and remote builds
   - Proper dependency management

### 🚧 Stub Implementation (TODO)
1. **MLIR Parsing** - Interface exists, needs domain_flow API integration
   ```cpp
   // This compiles but doesn't parse MLIR yet
   auto graph = load_graph("test_graphs/simple/matmul.mlir");
   ```

2. **Schedule Generation** - Not yet implemented
   - Need to connect graph → DMA/BlockMover/Streamer schedules

## Next Steps

### Immediate (You Can Do Now)

1. **Test JSON Graph Loading**
   ```bash
   cd /path/to/KPU-simulator
   cmake -B build -GNinja
   cmake --build build
   ctest --test-dir build -R graph_loader
   ```

2. **Add More Test Graphs**
   - Create JSON files in `test_graphs/simple/`
   - Add conv2d, attention, elementwise ops
   - Use as compiler test cases

3. **Decide on Integration Method**
   - Do you want FetchContent or local development?
   - Do you need to bump CMake to 3.28?

### Short Term (Next Development Phase)

1. **Connect MLIR Parsing**
   - Implement `DomainFlowGraphLoader::parse_mlir_module()`
   - Use domain_flow APIs to parse MLIR
   - Convert domain_flow ops to KPU operators

2. **Schedule Generation**
   - Create `schedule_generator.hpp/cpp`
   - Implement: `Schedule generate_schedule(Graph, KPUConfig)`
   - Generate DMA/BlockMover/Streamer commands

3. **Memory Planning**
   - Implement bufferization
   - Tensor lifetime analysis
   - L3/L2 allocation strategy

### Long Term (Compiler Feature Complete)

1. **Framework Importers**
   - ONNX → domain_flow → KPU
   - PyTorch → domain_flow → KPU
   - JAX → domain_flow → KPU

2. **Optimization Passes**
   - Operator fusion decisions
   - Tiling strategy selection
   - Memory reuse optimization

3. **Code Generation**
   - Emit executable schedules
   - Runtime integration

## Example Usage (What You Can Test Today)

```cpp
#include <sw/compiler/graph_loader.hpp>
#include <iostream>

int main() {
    using namespace sw::kpu::compiler;

    // Load a graph
    auto graph = load_graph("test_graphs/simple/matmul.json");

    // Inspect it
    std::cout << "Loaded graph: " << graph->name << "\n";
    graph->print();

    // Validate
    if (!graph->validate()) {
        std::cerr << "Graph validation failed!\n";
        return 1;
    }

    // Get execution order
    auto order = graph->get_execution_order();
    std::cout << "\nExecution order:\n";
    for (auto* op : order) {
        std::cout << "  " << op->name << " ("
                  << static_cast<int>(op->type) << ")\n";
    }

    return 0;
}
```

Compile and run:
```bash
g++ -std=c++20 -I include -o test_loader test_loader.cpp \
    -L build/src/compiler -lkpu_compiler
./test_loader
```

## Questions to Resolve

1. **CMake Version**: Can you upgrade to 3.28+ or should we stay on 3.20?
   - If 3.28+: Use FetchContent
   - If staying on 3.20: Use local installation method

2. **domain_flow Targets**: What are the actual exported targets?
   - Need to check domain_flow CMakeLists.txt
   - Update DomainFlowIntegration.cmake with correct target names

3. **MLIR Parsing**: What domain_flow APIs should we use?
   - Need to study domain_flow include files
   - Identify MLIR parsing functions

## Benefits of This Integration

### For Testing
- Well-defined test graphs in version control
- Same IR as production compiler pipeline
- Reproducible test cases

### For Development
- Clean separation: domain_flow (IR) vs KPU-simulator (execution)
- Each repo has clear responsibility
- Can develop compiler features against stable IR

### For Production
- Single source of truth for operator semantics (SUREs)
- Framework-agnostic compilation pipeline
- Leverage MLIR ecosystem

## Summary

You now have:
- ✅ Integration infrastructure in place
- ✅ Graph loading API designed and partially implemented
- ✅ Test framework ready
- ✅ Documentation complete
- ✅ JSON graphs working for immediate testing
- 🚧 MLIR integration stubbed (needs domain_flow API work)

**You can start using JSON graphs immediately** while we work on connecting the MLIR parsing in parallel.

The critical missing piece is implementing `DomainFlowGraphLoader::parse_mlir_module()` to actually use domain_flow's MLIR parsing. Once that's done, the full pipeline will work.

# Stillwater Knowledge Processing Unit functional simulator

## **Project Architecture Highlights**

### **Modular Component Structure**

- **`components`**: Independent hardware component libraries

  - `memory`: Memory hierarchy (DRAM, cache, scratchpad)

  - `compute`: Compute tiles (Domain Flow Architecture engines)

  - `fabric`: Interconnect and Network-on-Chip

  - `dma`: DMA controllers and streaming engines

  - `power`: Power modeling and thermal simulation

### **Modern CMake Build System**

- **Package-config integration**: Full `find_package()` support

- **Component-based building**: Build only what you need

- **Cross-platform support**: Windows, Linux, macOS

- **Feature detection**: OpenMP, CUDA, OpenCL auto-detection

- **Export targets**: Easy integration with other projects



### **Developer-Friendly Tools**

- **C++ Tools**: Profiler, debugger, assembler, benchmarks

- **Python Ecosystem**: Visualization, analysis, code generation

- **Build Scripts**: Automated, configurable build process

- **CI/CD Ready**: GitHub Actions integration



## **Key Design Principles**

### **1. Scalability**

```cmake

# Easy to add new components

add_subdirectory(components/new_component)

target_link_libraries(kpu_simulator PRIVATE stillwater_kpu::NewComponent)

```



### **2. Modern C++20 Standards**

```cpp

// Clean, modern interfaces

template<typename T> requires std::floating_point<T>

class MatrixUnit {

    auto multiply(std::span<const T> A, std::span<const T> B) -> std::vector<T>;

};

```



### **3. Package Manager Integration**

```cmake

find_package(StillwaterKPU REQUIRED COMPONENTS Memory Compute Fabric)

target_link_libraries(my_target COMPONENTS Memory Compute)

```



## **Usage Workflows**



### **Research \& Development**

```bash

# Quick development setup

git clone https://github.com/stillwater-sc/KPU-simulator.git

cd KPU-simulator

./scripts/build.sh --type Debug --sanitizers --docs

```



### **Production Integration**

```cmake

# In your CMakeLists.txt

find_package(StillwaterKPU 1.0 REQUIRED COMPONENTS Simulator)

target_link_libraries(my_app PRIVATE StillwaterKPU::Simulator)

```



### **Educational Use**

```python

# Simple Python interface

import stillwater_kpu as kpu

sim = kpu.create_simulator()

result = sim.matmul(matrix_a, matrix_b)

```



## **Advanced Features**



### **Component Isolation**

- Each component is a separate library

- Clean dependency management

- Independent testing and development

- Plug-and-play architecture



### **Multi-Language Support**

- **C++**: Core implementation

- **C API**: Language-agnostic interface

- **Python**: High-level scripting and analysis

- **MATLAB**: Engineering workflows



### **Development Tools**

- **Profiler**: Performance analysis

- **Debugger**: Interactive debugging

- **Visualizer**: System topology and data flow

- **Benchmark Suite**: Performance regression testing



## **Project Benefits**



1. **Modularity**: Develop components independently

2. **Maintainability**: Clear separation of concerns  

3. **Testability**: Component-level and integration testing

4. **Extensibility**: Easy to add new features and components

5. **Distribution**: Professional packaging and installation

6. **Documentation**: Automated docs with Doxygen/Sphinx



## **Next Steps**



This structure provides a solid foundation for:



1. **Implementing the core simulator components**

2. **Building a community around the project**

3. **Creating educational materials and tutorials**

4. **Developing advanced features like multi-GPU support**

5. **Establishing industry partnerships**



The architecture scales from simple educational examples to complex research simulations and production systems. Each component can be developed and tested independently while maintaining clean interfaces and dependencies.




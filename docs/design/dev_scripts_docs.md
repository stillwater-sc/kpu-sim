# Quick Start Guide

This guide will help you get the Stillwater KPU simulator up and running quickly.

## Prerequisites

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libomp-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    doxygen \
    graphviz
```

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 libomp-devel python3-devel python3-pip doxygen graphviz

# Fedora
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y cmake libomp-devel python3-devel python3-pip doxygen graphviz
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew dependencies
brew install cmake libomp python3 doxygen graphviz
```

#### Windows
```bash
# Using vcpkg
vcpkg install openmp python3 doxygen

# Or using Chocolatey
choco install cmake python3 doxygen.install
```

### Python Dependencies
```bash
pip3 install numpy matplotlib pytest black flake8 sphinx
```

## Building the Project

### Quick Build
```bash
# Clone the repository
git clone https://github.com/stillwater-sc/kpu-simulator.git
cd kpu-simulator

# Run the build script
chmod +x scripts/build.sh
./scripts/build.sh
```

### Custom Build Options
```bash
# Debug build with sanitizers
./scripts/build.sh --type Debug --sanitizers

# Release build with GPU support
./scripts/build.sh --cuda --opencl --install

# Build with documentation
./scripts/build.sh --docs

# Full development build
./scripts/build.sh --type Debug --sanitizers --docs --verbose
```

### Manual CMake Build
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKPU_BUILD_TESTS=ON \
    -DKPU_BUILD_PYTHON_BINDINGS=ON \
    -DKPU_ENABLE_OPENMP=ON

make -j$(nproc)
```

## Running Tests

### C++ Tests
```bash
cd build
ctest --output-on-failure
```

### Python Tests
```bash
cd tools/python
python3 -m pytest tests/
```

### Specific Test Categories
```bash
# Unit tests only
ctest -L unit

# Integration tests
ctest -L integration

# Performance benchmarks
ctest -L benchmark
```

## Basic Usage

### C++ API
```cpp
#include <stillwater/kpu/kpu.hpp>
using namespace stillwater::kpu;

int main() {
    // Create KPU simulator
    KPUSimulator kpu(1GB, 1MB);  // 1GB main memory, 1MB scratchpad
    
    // Access components
    auto& memory = kpu.main_memory();
    auto& compute = kpu.compute_engine();
    auto& dma = kpu.dma_engine();
    
    // Your simulation code here...
    return 0;
}
```

### Python API
```python
import stillwater_kpu as kpu
import numpy as np

# Create simulator
with kpu.Simulator() as sim:
    # Create test matrices
    A = np.random.randn(100, 200).astype(np.float32)
    B = np.random.randn(200, 150).astype(np.float32)
    
    # Perform matrix multiplication
    C = sim.matmul(A, B)
    
    # Verify results
    C_ref = A @ B
    print(f"Results match: {np.allclose(C, C_ref)}")
```

## Development Workflow

### Code Formatting
```bash
# Format C++ code
./scripts/format.sh

# Format Python code
cd tools/python
black stillwater_kpu/
flake8 stillwater_kpu/
```

### Running Static Analysis
```bash
# Build with static analysis
./scripts/build.sh --static-analysis

# Manual analysis
cd build
make clang-tidy
make cppcheck
```

### Creating Documentation
```bash
# Build documentation
./scripts/build.sh --docs

# View documentation
cd build/docs/html
python3 -m http.server 8000
# Open http://localhost:8000 in browser
```

## Project Structure Overview

```
stillwater-kpu/
├── cmake/              # CMake utilities and find modules
├── components/         # Hardware component libraries
│   ├── memory/        # Memory hierarchy (DRAM, cache, scratchpad)
│   ├── compute/       # Compute engines (matrix, vector, scalar)
│   ├── fabric/        # Interconnect and NoC
│   ├── dma/          # DMA controllers
│   └── power/        # Power modeling
├── src/               # Core simulator engine
│   ├── simulator/    # Main simulator implementation
│   └── bindings/     # C and Python APIs
├── tools/             # Development tools
│   ├── cpp/          # C++ tools (profiler, debugger, etc.)
│   └── python/       # Python tools and visualization
├── examples/          # Example applications and tutorials
├── tests/            # Test suites
├── benchmarks/       # Standard benchmarks
└── docs/             # Documentation
```

## Common Development Tasks

### Adding a New Component
1. Create directory under `components/`
2. Add `CMakeLists.txt` with library definition
3. Implement header files in `include/stillwater/kpu/component/`
4. Implement source files in `src/`
5. Add tests in `tests/`
6. Update main `CMakeLists.txt` to include component

### Adding a Python Tool
1. Create script in `tools/python/stillwater_kpu/`
2. Add to `setup.py` if it's a module
3. Create tests in `tools/python/tests/`
4. Update documentation

### Running Benchmarks
```bash
# Build benchmarks
./scripts/build.sh --benchmarks

# Run specific benchmark
./build/bin/benchmark_matrix_ops

# Run all benchmarks
cd build
make benchmark
```

## Troubleshooting

### Common Build Issues

#### CMake Version Too Old
```bash
# Install newer CMake
wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh
chmod +x cmake-3.25.1-linux-x86_64.sh
sudo ./cmake-3.25.1-linux-x86_64.sh --skip-license --prefix=/usr/local
```

#### Missing OpenMP
```bash
# Ubuntu/Debian
sudo apt install libomp-dev

# macOS
brew install libomp

# Set environment variables if needed
export OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include"
export OpenMP_CXX_LIB_NAMES="omp"
export OpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
```

#### Python Binding Issues
```bash
# Install pybind11
pip3 install pybind11

# Set Python paths if needed
export PYTHONPATH=$PYTHONPATH:/path/to/build/lib
```

### Performance Issues

#### Slow Matrix Operations
- Ensure OpenMP is enabled: `-DKPU_ENABLE_OPENMP=ON`
- Check CPU governor: `sudo cpupower frequency-set -g performance`
- Verify compiler optimizations: `-DCMAKE_BUILD_TYPE=Release`

#### Memory Issues
- Monitor with: `valgrind --tool=massif ./your_program`
- Use sanitizers: `--sanitizers` build flag
- Check for leaks: `valgrind --leak-check=full ./your_program`

## Advanced Features

### GPU Acceleration
```bash
# CUDA support
./scripts/build.sh --cuda

# OpenCL support  
./scripts/build.sh --opencl

# Both
./scripts/build.sh --cuda --opencl
```

### Distributed Simulation
```bash
# Build with MPI support (if available)
cmake .. -DKPU_ENABLE_MPI=ON
```

### Custom Number Formats
```cpp
#include <stillwater/kpu/compute/data_format.hpp>

// Use custom precision
using custom_float = stillwater::kpu::posit<16,2>;  // 16-bit posit
```

## Getting Help

### Documentation
- API Documentation: `build/docs/html/index.html`
- Architecture Guide: `docs/architecture/`
- Tutorial: `examples/tutorials/`

### Community
- GitHub Issues: https://github.com/stillwater-sc/kpu-simulator/issues
- Discussions: https://github.com/stillwater-sc/kpu-simulator/discussions
- Email: info@stillwater-sc.com

### Debugging
```bash
# Debug build
./scripts/build.sh --type Debug

# With debugger
gdb ./build/bin/example_program

# With profiler
perf record ./build/bin/example_program
perf report
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Build Matrix**: Tests on Ubuntu, macOS, Windows
- **Compiler Matrix**: GCC, Clang, MSVC
- **Python Matrix**: 3.7, 3.8, 3.9, 3.10, 3.11
- **Feature Matrix**: OpenMP, CUDA, OpenCL variants

### Local CI Testing
```bash
# Run the same checks as CI
./scripts/ci-check.sh

# Individual checks
./scripts/format.sh --check
./scripts/test.sh --all
./scripts/build.sh --sanitizers
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow coding standards: `./scripts/format.sh`
4. Add tests for new functionality
5. Ensure all tests pass: `./scripts/test.sh`
6. Create pull request

### Coding Standards

- **C++**: Follow Google C++ Style Guide
- **Python**: Follow PEP 8
- **CMake**: Modern CMake practices (3.18+)
- **Documentation**: Doxygen for C++, Sphinx for Python

---

**Happy Coding with Stillwater KPU!** 🚀
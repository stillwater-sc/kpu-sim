# Python Integration Architecture

## Directory Structure

```
stillwater-kpu/
├── src/bindings/python/               # Low-level C++ bindings
│   ├── CMakeLists.txt
│   ├── pybind_module.cpp              # pybind11 C++ module
│   └── __init__.py                    # Minimal init
├── tools/python/                      # High-level Python package
│   ├── setup.py                       # Package installation
│   ├── pyproject.toml                 # Modern Python packaging
│   ├── stillwater_kpu/               # Main Python package
│   │   ├── __init__.py               # Package exports
│   │   ├── simulator.py              # High-level simulator API (was kpu_python_api.py)
│   │   ├── core.py                   # Core functionality wrapper
│   │   ├── visualization/            # Visualization tools
│   │   │   ├── __init__.py
│   │   │   ├── performance_plot.py
│   │   │   └── memory_heatmap.py
│   │   ├── analysis/                 # Analysis tools
│   │   │   ├── __init__.py
│   │   │   └── performance_analyzer.py
│   │   └── utilities/                # Utility functions
│   │       ├── __init__.py
│   │       └── data_converter.py
│   ├── tests/                        # Python tests
│   │   ├── test_simulator.py
│   │   ├── test_core.py
│   │   └── conftest.py
│   └── examples/                     # Python examples
│       ├── basic_usage.py
│       └── neural_network.py
```

## Two-Layer Architecture

### Layer 1: C++ Bindings (`src/bindings/python/`)

**Purpose**: Minimal, direct wrapper around C++ API

```cpp
// src/bindings/python/pybind_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stillwater/kpu/kpu.hpp>

namespace py = pybind11;
using namespace stillwater::kpu;

PYBIND11_MODULE(stillwater_kpu_native, m) {
    m.doc() = "Stillwater KPU Simulator Native Bindings";
    
    py::class_<KPUSimulator>(m, "KPUSimulator")
        .def(py::init<size_t, size_t>(), "main_memory_size"_a=1ULL<<30, "scratchpad_size"_a=1ULL<<20)
        .def("main_memory_size", &KPUSimulator::main_memory_size)
        .def("scratchpad_size", &KPUSimulator::scratchpad_size);
    
    py::class_<MainMemory>(m, "MainMemory")
        .def("read", [](MainMemory& self, uint64_t addr, size_t size) {
            std::vector<uint8_t> data(size);
            self.read(addr, data.data(), size);
            return py::array_t<uint8_t>(size, data.data());
        })
        .def("write", [](MainMemory& self, uint64_t addr, py::array_t<uint8_t> data) {
            self.write(addr, data.data(), data.size());
        });
}
```

### Layer 2: High-Level Python API (`tools/python/stillwater_kpu/`)

**Purpose**: Pythonic interface with NumPy integration, error handling, convenience functions

```python
# tools/python/stillwater_kpu/simulator.py
# This is where the kpu_python_api.py content should go

import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path

# Import the native C++ module
try:
    from . import stillwater_kpu_native as _native
except ImportError:
    import stillwater_kpu_native as _native

class KPUSimulator:
    """High-level Python interface to KPU Simulator."""
    
    def __init__(self, main_memory_size: int = 1<<30, scratchpad_size: int = 1<<20):
        self._native = _native.KPUSimulator(main_memory_size, scratchpad_size)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup handled by C++ destructor
        pass
    
    @property
    def main_memory_size(self) -> int:
        return self._native.main_memory_size()
    
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """High-level matrix multiplication with automatic memory management."""
        # Implementation using native bindings
        pass
```

## Package Installation Methods

### Method 1: Development Installation

```bash
cd tools/python
pip install -e .
```

This creates an editable installation that imports both:
1. The compiled C++ module (`stillwater_kpu_native`)
2. The Python package (`stillwater_kpu`)

### Method 2: Build Integration

The CMake build system handles both:

```cmake
# src/bindings/python/CMakeLists.txt
pybind11_add_module(stillwater_kpu_native pybind_module.cpp)
target_link_libraries(stillwater_kpu_native PRIVATE StillwaterKPU::Simulator)

# tools/python/CMakeLists.txt
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(python_package
    COMMAND ${Python3_EXECUTABLE} -m pip install -e .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS stillwater_kpu_native
)
```

## Usage Examples

### User Perspective

```python
# User imports the high-level package
import stillwater_kpu as kpu
import numpy as np

# High-level API
with kpu.Simulator() as sim:
    A = np.random.randn(100, 200).astype(np.float32)
    B = np.random.randn(200, 150).astype(np.float32)
    C = sim.matmul(A, B)

# Visualization tools
kpu.visualization.plot_performance(benchmark_results)

# Analysis tools  
analyzer = kpu.analysis.PerformanceAnalyzer()
report = analyzer.analyze(simulation_data)
```

### Package Structure

```python
# tools/python/stillwater_kpu/__init__.py
"""Stillwater KPU Simulator Python API."""

from .simulator import KPUSimulator as Simulator
from .core import *
from . import visualization
from . import analysis
from . import utilities

__version__ = "1.0.0"
__all__ = ["Simulator", "visualization", "analysis", "utilities"]
```

## Build Process

### 1. C++ Module Build

```bash
cd build
cmake --build . --target stillwater_kpu_native
```

This creates: `build/lib/stillwater_kpu_native.so` (Linux) or `.pyd` (Windows)

### 2. Python Package Build

```bash
cd tools/python
pip install -e .
```

This installs the Python package and links to the native module.

### 3. Combined Build

```bash
cmake --preset release
cmake --build --preset release --target python_package
```

## Testing

```bash
cd tools/python
python -m pytest tests/ -v
```

## Key Benefits

1. **Separation of Concerns**:
   - C++ bindings = minimal, fast
   - Python layer = convenient, Pythonic

2. **Maintainability**:
   - C++ changes don't affect Python API
   - Python tools can be developed independently

3. **Performance**:
   - Critical paths in C++
   - Convenience functions in Python

4. **Distribution**:
   - Can distribute just the Python package
   - Users don't need to build C++

## File Mapping

| Original File | New Location | Purpose |
|---------------|--------------|---------|
| `kpu_python_api.py` | `tools/python/stillwater_kpu/simulator.py` | High-level simulator API |
| N/A | `src/bindings/python/pybind_module.cpp` | Low-level C++ bindings |
| N/A | `tools/python/setup.py` | Package installation |
| N/A | `tools/python/stillwater_kpu/__init__.py` | Package exports |
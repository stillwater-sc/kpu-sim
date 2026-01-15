# How to Build and Use KPU Python Bindings

## Overview

The KPU Simulator provides Python bindings built with `pybind11` that expose the C++ API to Python. This guide shows you how to build and use these bindings to test the DMA engine and other simulator features.

## Build System Architecture

```
KPU-Simulator/
├── src/bindings/python/
│   ├── kpu_bindings.cpp           # Main KPU simulator bindings
│   ├── toplevel_bindings.cpp      # System-level bindings
│   └── CMakeLists.txt             # Build configuration
└── build/                          # Linux/WSL build output
    └── src/bindings/python/
        └── Release/
            ├── stillwater_kpu.so             # KPU module (Linux)
            └── stillwater_toplevel.so        # System module (Linux)

OR

└── build_msvc/                     # Windows/MSVC build output
    └── src/bindings/python/
        ├── Debug/
        │   ├── stillwater_kpu.pyd           # KPU module (Debug)
        │   └── stillwater_toplevel.pyd      # System module (Debug)
        └── Release/
            ├── stillwater_kpu.pyd           # KPU module (Release)
            └── stillwater_toplevel.pyd      # System module (Release)
```

## Building Python Bindings

### Prerequisites

#### Linux/WSL
```bash
# Install Python development headers
sudo apt update
sudo apt install python3-dev python3-pip

# Verify Python is found
python3 --version
python3-config --includes
```

#### Windows
Python development headers are included with standard Python installation.

### Build Steps

#### Option 1: Build on Linux/WSL

```bash
cd /path/to/KPU-simulator

# Create build directory
mkdir -p build && cd build

# Configure with Python bindings enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DKPU_BUILD_PYTHON_BINDINGS=ON

# Build the Python modules
cmake --build . --target stillwater_kpu -j8
cmake --build . --target stillwater_toplevel -j8

# Or build everything
cmake --build . -j8
```

**Output Location**: `build/src/bindings/python/Release/`
- `stillwater_kpu.so` (or `.pyd` on Windows)
- `stillwater_toplevel.so` (or `.pyd` on Windows)

#### Option 2: Build on Windows with MSVC

```cmd
cd C:\path\to\KPU-simulator

# Create build directory
mkdir build_msvc
cd build_msvc

# Configure (Python bindings ON by default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release --target stillwater_kpu
cmake --build . --config Release --target stillwater_toplevel
```

**Output Location**: `build_msvc\src\bindings\python\Release\`
- `stillwater_kpu.pyd`
- `stillwater_toplevel.pyd`

## Using the Python Modules

### Method 1: Direct PYTHONPATH (For Testing)

This is the simplest method for testing and development:

#### Linux/WSL
```bash
# Set PYTHONPATH to the build output directory
export PYTHONPATH=/path/to/KPU-simulator/build/src/bindings/python/Release:$PYTHONPATH

# Or for a single command
PYTHONPATH=/path/to/KPU-simulator/build/src/bindings/python/Release python3 your_script.py
```

#### Windows (PowerShell)
```powershell
# Set PYTHONPATH
$env:PYTHONPATH = "C:\path\to\KPU-simulator\build_msvc\src\bindings\python\Release;$env:PYTHONPATH"

# Run your script
python your_script.py
```

#### Windows (Command Prompt)
```cmd
set PYTHONPATH=C:\path\to\KPU-simulator\build_msvc\src\bindings\python\Release;%PYTHONPATH%
python your_script.py
```

### Method 2: Add to sys.path (In Script)

Add this at the top of your Python script:

```python
import sys
import os

# Adjust path to your build directory
build_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../build/src/bindings/python/Release'  # Linux
    # OR for Windows:
    # '../../build_msvc/src/bindings/python/Release'
))

if build_path not in sys.path:
    sys.path.insert(0, build_path)

# Now you can import
import stillwater_kpu as kpu
```

### Method 3: Install in Development Mode (Advanced)

Create a simple setup.py to install the module:

```python
# setup.py (create in project root)
from setuptools import setup
import os

# Find the built module
build_dir = 'build/src/bindings/python/Release'  # or build_msvc/...
module_path = os.path.join(build_dir, 'stillwater_kpu.pyd')  # or .so

setup(
    name='stillwater_kpu',
    version='0.1.0',
    py_modules=['stillwater_kpu'],
    package_data={'': [module_path]},
)
```

Then:
```bash
pip install -e .
```

## Testing the DMA Engine

### Quick Test Script

Create `test_dma.py`:

```python
#!/usr/bin/env python3
"""Test the address-based DMA API"""

import sys
import os

# Add build path (adjust for your system)
build_path = os.path.abspath('./build_msvc/src/bindings/python/Release')
sys.path.insert(0, build_path)

import stillwater_kpu as kpu

def test_basic_dma():
    """Test basic DMA transfer External → Scratchpad"""
    print("=" * 60)
    print("Testing Address-Based DMA API")
    print("=" * 60)

    # Create simulator with simple config
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 64
    config.scratchpad_count = 2
    config.scratchpad_capacity_kb = 256
    config.dma_engine_count = 4

    sim = kpu.KPUSimulator(config)

    print(f"\nSimulator Configuration:")
    print(f"  External banks: {sim.get_memory_bank_count()}")
    print(f"  Scratchpads: {sim.get_scratchpad_count()}")
    print(f"  DMA engines: {sim.get_dma_engine_count()}")

    # Prepare test data
    test_data = [float(i) for i in range(256)]

    # Write to external memory
    print(f"\n1. Writing {len(test_data)} floats to external memory...")
    sim.write_memory_bank(0, 0, test_data)

    # Compute global addresses
    src_addr = sim.get_external_bank_base(0) + 0
    dst_addr = sim.get_scratchpad_base(0) + 0
    transfer_size = len(test_data) * 4  # 4 bytes per float

    print(f"\n2. Starting DMA transfer...")
    print(f"   Source: 0x{src_addr:08x} (External[0])")
    print(f"   Dest:   0x{dst_addr:08x} (Scratchpad[0])")
    print(f"   Size:   {transfer_size} bytes")

    # Start DMA transfer
    complete = [False]  # Use list for closure
    def on_complete():
        complete[0] = True
        print("   ✓ DMA transfer completed!")

    sim.dma_external_to_scratchpad(0, src_addr, dst_addr, transfer_size, on_complete)

    # Run simulation
    print(f"\n3. Running simulation...")
    sim.run_until_idle()

    # Verify data
    print(f"\n4. Verifying data in scratchpad...")
    result = sim.read_scratchpad(0, 0, len(test_data))

    if result == test_data:
        print("   ✓ Data verification PASSED")
    else:
        print("   ✗ Data verification FAILED")
        return False

    # Print stats
    print(f"\n5. Statistics:")
    print(f"   Cycles: {sim.get_current_cycle()}")
    print(f"   Time: {sim.get_elapsed_time_ms():.2f} ms")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_basic_dma()
    sys.exit(0 if success else 1)
```

### Running the Test

```bash
# Linux/WSL
PYTHONPATH=build/src/bindings/python/Release python3 test_dma.py

# Windows
set PYTHONPATH=build_msvc\src\bindings\python\Release
python test_dma.py
```

### Expected Output

```
============================================================
Testing Address-Based DMA API
============================================================

Simulator Configuration:
  External banks: 2
  Scratchpads: 2
  DMA engines: 4

1. Writing 256 floats to external memory...

2. Starting DMA transfer...
   Source: 0x100000000 (External[0])
   Dest:   0x180100000 (Scratchpad[0])
   Size:   1024 bytes

3. Running simulation...
   ✓ DMA transfer completed!

4. Verifying data in scratchpad...
   ✓ Data verification PASSED

5. Statistics:
   Cycles: 145
   Time: 0.15 ms

============================================================
✓ All tests passed!
============================================================
```

## Running Existing Examples

Once PYTHONPATH is set, run the address-based DMA example:

```bash
cd examples/python
python address_based_dma_example.py
```

## Troubleshooting

### Module Not Found

**Error**: `ModuleNotFoundError: No module named 'stillwater_kpu'`

**Solutions**:
1. Check PYTHONPATH is set correctly
2. Verify the `.pyd` or `.so` file exists in the path
3. Make sure you built with `DKPU_BUILD_PYTHON_BINDINGS=ON`

### Python Version Mismatch

**Error**: Module loads but crashes or gives import errors

**Solution**: Ensure CMake found the same Python version you're using:
```bash
# Check CMake's Python
cmake .. -DKPU_BUILD_PYTHON_BINDINGS=ON 2>&1 | grep Python

# Check your Python
which python3
python3 --version
```

### Missing python3-dev (Linux only)

**Error**: `Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Development)`

**Solution**:
```bash
sudo apt install python3-dev
```

### DLL/SO Loading Errors

**Error**: Module found but fails to load

**Solution**: Check dependencies are satisfied:
```bash
# Linux
ldd build/src/bindings/python/Release/stillwater_kpu.so

# Windows (use Dependency Walker or similar)
```

## Advanced Usage

### Using in Jupyter Notebooks

```python
# First cell - setup path
import sys
sys.path.insert(0, '/path/to/build/src/bindings/python/Release')

# Second cell - import and use
import stillwater_kpu as kpu
import numpy as np

sim = kpu.KPUSimulator()
# ... your code ...
```

### Batch Testing

Create a `run_tests.sh` script:

```bash
#!/bin/bash
export PYTHONPATH=build/src/bindings/python/Release:$PYTHONPATH

echo "Running DMA tests..."
python3 test_dma.py

echo "Running matrix tests..."
python3 examples/python/address_based_dma_example.py

echo "All tests complete!"
```

## Module API Quick Reference

### Available Modules

1. **stillwater_kpu** - Main KPU simulator
   - Full memory hierarchy (host, external, L3, L2, scratchpad)
   - Address-based DMA API
   - Matrix operations
   - Configuration and statistics

2. **stillwater_toplevel** - System-level simulator
   - System initialization
   - Self-tests
   - Context manager support

### Key Classes

```python
# Configuration
config = kpu.SimulatorConfig()
config.memory_bank_count = 4
config.scratchpad_count = 2

# Simulator
sim = kpu.KPUSimulator(config)

# Operations
sim.write_memory_bank(bank_id, addr, data)
sim.dma_external_to_scratchpad(dma_id, src, dst, size, callback)
sim.run_until_idle()
sim.print_component_status()
```

## See Also

- [Address-Based DMA Quick Start](address-based-dma-quickstart.md)
- [Python Example](../examples/python/address_based_dma_example.py)
- [API Documentation](unified-address-space.md)

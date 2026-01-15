# Cross-Platform Build Guide

This project uses modern CMake (3.20+) with presets for easy cross-platform building. No platform-specific scripts required!

## Quick Start

### Prerequisites

#### All Platforms
- CMake 3.20 or newer
- C++20 compatible compiler
- Git

#### Windows
```powershell
# Using winget
winget install Kitware.CMake
winget install Git.Git
winget install Microsoft.VisualStudio.2022.Community

# Using Chocolatey
choco install cmake git visualstudio2022community
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y cmake build-essential git ninja-build
```

#### Linux (CentOS/RHEL/Fedora)  
```bash
# CentOS/RHEL 8+
sudo dnf install -y cmake gcc-c++ git ninja-build

# Older versions
sudo yum install -y cmake3 gcc-c++ git ninja-build
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Using Homebrew
brew install cmake ninja
```

## Building with CMake Presets

### 1. Configure

Choose a preset based on your needs:

```bash
# List available presets
cmake --list-presets=configure

# Quick development build
cmake --preset debug

# Optimized release build  
cmake --preset release

# Full-featured build
cmake --preset full

# Minimal build (core only)
cmake --preset minimal

# Platform-specific
cmake --preset windows-msvc    # Windows with MSVC
cmake --preset linux-gcc       # Linux with GCC
cmake --preset macos           # macOS with Xcode
```

### 2. Build

```bash
# Build using preset
cmake --build --preset debug

# Or specify configuration manually
cmake --build build --config Release

# Parallel build (auto-detects cores)
cmake --build build --parallel
```

### 3. Test

```bash  
# Run all tests
ctest --preset default

# Run specific test categories
ctest --preset unit           # Unit tests only
ctest --preset integration    # Integration tests
ctest --preset performance    # Performance benchmarks
```

### 4. Package

```bash
# Create packages
cpack --preset default

# Platform-specific packages
cpack --preset windows        # NSIS installer + ZIP
cpack --preset linux          # DEB + RPM + tarball
cpack --preset macos          # DMG + tarball
```

## Platform-Specific Instructions

### Windows

#### Using Visual Studio
```powershell
# Configure for Visual Studio
cmake --preset windows-msvc

# Open in Visual Studio
start build/StillwaterKPU.sln

# Or build from command line
cmake --build build --config Release
```

#### Using Visual Studio Code
```powershell
# Install CMake Tools extension
code --install-extension ms-vscode.cmake-tools

# Open project
code .

# Select preset in VS Code command palette (Ctrl+Shift+P):
# "CMake: Select Configure Preset" -> choose desired preset
```

#### Using CLion
1. Open project directory in CLion
2. CLion will automatically detect `CMakePresets.json`
3. Select desired preset from the dropdown
4. Build using Ctrl+F9

### Linux

#### Command Line
```bash
# Standard build
cmake --preset linux-gcc
cmake --build --preset release

# With specific compiler
CC=gcc-11 CXX=g++-11 cmake --preset linux-gcc
cmake --build build --parallel

# Install system-wide
sudo cmake --build build --target install
```

#### IDE Integration
Most Linux IDEs (Qt Creator, KDevelop, etc.) support CMake presets automatically.

### macOS

#### Command Line
```bash
# Configure for Xcode
cmake --preset macos

# Build with Xcode
cmake --build build --config Release

# Or use Ninja for faster builds
cmake --preset default -G Ninja
cmake --build build
```

#### Xcode
```bash
cmake --preset macos
open build/StillwaterKPU.xcodeproj
```

## Custom Configuration

### Environment Variables

```bash
# Customize install prefix
export CMAKE_INSTALL_PREFIX=/opt/stillwater-kpu

# Use specific compiler
export CC=clang
export CXX=clang++

# Then configure
cmake --preset release
```

### CMake Cache Variables

```bash
# Override preset options
cmake --preset release \
  -DKPU_ENABLE_CUDA=ON \
  -DKPU_BUILD_DOCS=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local
```

### Custom Preset

Create `CMakeUserPresets.json` (not tracked by git):

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "my-config",
      "displayName": "My Custom Configuration",
      "inherits": "release",
      "cacheVariables": {
        "KPU_ENABLE_CUDA": "ON",
        "CMAKE_INSTALL_PREFIX": "/home/user/kpu"
      }
    }
  ]
}
```

## Feature Options

All feature flags can be controlled via CMake cache variables:

| Option | Default | Description |
|--------|---------|-------------|
| `KPU_BUILD_TESTS` | ON | Build test suite |
| `KPU_BUILD_EXAMPLES` | ON | Build example applications |
| `KPU_BUILD_TOOLS` | ON | Build development tools |
| `KPU_BUILD_PYTHON_BINDINGS` | ON | Build Python API |
| `KPU_BUILD_BENCHMARKS` | ON | Build benchmark suite |
| `KPU_BUILD_DOCS` | OFF | Build documentation |
| `KPU_ENABLE_OPENMP` | ON | Enable OpenMP parallelization |
| `KPU_ENABLE_CUDA` | OFF | Enable CUDA support |
| `KPU_ENABLE_OPENCL` | OFF | Enable OpenCL support |
| `KPU_ENABLE_SANITIZERS` | OFF | Enable runtime sanitizers |
| `KPU_ENABLE_PROFILING` | OFF | Enable profiling support |

## Dependency Management

### Automatic Dependencies

CMake will automatically download and build required dependencies:

- **spdlog**: Logging framework
- **nlohmann/json**: JSON parsing
- **fmt**: String formatting
- **Catch2**: Testing framework (if tests enabled)
- **pybind11**: Python bindings (if Python enabled)

### System Dependencies

#### Optional: OpenMP
```bash
# Linux
sudo apt install libomp-dev        # Ubuntu/Debian
sudo dnf install libomp-devel       # Fedora

# macOS
brew install libomp

# Windows
# Included with MSVC or install Intel OpenMP
```

#### Optional: CUDA
```bash
# Download from NVIDIA website or use package manager
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-2
```

## Troubleshooting

### CMake Version Issues
```bash
# Check version
cmake --version

# Upgrade on Ubuntu
sudo snap install cmake --classic

# Upgrade on macOS  
brew upgrade cmake

# Windows: Download from kitware.com
```

### Compiler Issues

#### Missing C++20 Support
```bash
# Ubuntu: Install newer GCC
sudo apt install gcc-11 g++-11
export CC=gcc-11 CXX=g++-11

# Or use Clang
sudo apt install clang-14
export CC=clang-14 CXX=clang++-14
```

#### Windows MSVC Not Found
```powershell
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Or use Clang
winget install LLVM.LLVM
```

### Python Binding Issues
```bash
# Ensure Python development headers
# Ubuntu
sudo apt install python3-dev

# CentOS/RHEL
sudo dnf install python3-devel

# Windows: Use official Python installer (includes dev headers)
```

### Performance Issues

#### Slow Builds
```bash
# Use Ninja generator (faster than Make)
cmake --preset default -G Ninja

# Increase parallel jobs
cmake --build build --parallel 16
```

#### Runtime Performance
```bash
# Ensure release build
cmake --preset release

# Enable all optimizations
cmake --preset release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# For specific CPU
cmake --preset release -DCMAKE_CXX_FLAGS="-march=native"
```

## IDE Integration

### Visual Studio Code

1. Install extensions:
   ```
   ms-vscode.cmake-tools
   ms-vscode.cpptools
   ```

2. Open project folder
3. Select preset: `Ctrl+Shift+P` → "CMake: Select Configure Preset"
4. Build: `F7` or `Ctrl+Shift+P` → "CMake: Build"

### CLion
- Automatically detects `CMakePresets.json`
- Select preset from dropdown in settings

### Qt Creator
- Open `CMakeLists.txt` as project
- Configure with desired preset

## Continuous Integration

The project includes GitHub Actions workflows that test all presets:

- **Windows**: MSVC 2022, Clang
- **Linux**: GCC 11/12, Clang 14/15  
- **macOS**: Xcode 14/15

Local CI simulation:
```bash
# Test all configurations
cmake --preset debug && cmake --build --preset debug
cmake --preset release && cmake --build --preset release
ctest --preset default
```

---

This approach is much better than platform-specific scripts because:

1. ✅ **Native CMake**: Uses CMake's built-in cross-platform capabilities
2. ✅ **IDE Integration**: All major IDEs understand CMake presets
3. ✅ **Maintainable**: Single configuration file for all platforms
4. ✅ **Flexible**: Easy to customize without editing scripts
5. ✅ **Standard**: Follows modern CMake best practices
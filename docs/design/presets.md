# CMake Build Configuration

CMake is specifically designed to handle cross-platform builds. This project is a proper CMake-first approach:

## ✅ **Why CMake Presets > Platform Scripts**



### **1. Native Cross-Platform Support**

  - **CMake handles platform differences automatically**

  - **No need for separate `.sh`, `.ps1`, `.bat` files**

  - **Single `CMakePresets.json` works everywhere**



### **2. IDE Integration**

  - **Visual Studio**: Automatically detects presets

  - **VS Code**: CMake Tools extension supports presets

  - **CLion**: Native preset support

  - **Qt Creator**: Understands CMake presets



### **3. Standardized Approach**

```bash

# Same commands work on Windows, Linux, macOS

cmake --preset debug

cmake --build --preset debug

ctest --preset unit

```



### **4. Developer Experience**

```bash

# List available configurations

cmake --list-presets=configure



# No need to remember script arguments

cmake --preset full  # vs ./build.sh --cuda --opencl --docs --install

```

## **Key Benefits of This Approach**



### **Configuration Management**

```json
{
  "configurePresets": [
    {
      "name": "debug",
      "cacheVariables": {
        "KPU_ENABLE_SANITIZERS": "ON",
        "CMAKE_BUILD_TYPE": "Debug"
      }
    }
  ]
}
```



### **Platform-Specific Handling**

```json
{
  "name": "windows-msvc",
  "condition": {
    "type": "equals",
    "lhs": "${hostSystemName}",
    "rhs": "Windows"
  },
  "generator": "Visual Studio 17 2022"
}
```

### **User Customization**

Users can create `CMakeUserPresets.json` for personal configurations without modifying the main project.



## **Modern Developer Workflow**



```bash
# Clone and build - same on all platforms

git clone https://github.com/stillwater-sc/kpu-simulator.git
cd kpu-simulator
cmake --preset release
cmake --build --preset release
ctest --preset default

```

The only platform-specific scripts we keep are for:

  - **Code formatting** (clang-format wrappers)

  - **Static analysis** (optional development tools)

  - **CI helpers** (GitHub Actions specific)

This approach follows modern CMake best practices and makes the project much more accessible to developers on all platforms.

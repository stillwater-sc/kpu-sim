# Config Formatter and Memory Map API Implementation

## Overview

This document describes the implementation of the config formatter and memory map reporting system, which provides automatic, comprehensive configuration output and eliminates error-prone manual printing in client code.

## Problem Statement

**Before**: Clients had to manually format configuration output:
```cpp
std::cout << "  System: " << config.system.name << "\n";
std::cout << "  KPU Components:\n";
std::cout << "    Memory banks: " << config.accelerators[0].kpu_config->memory.banks.size() << "\n";
std::cout << "    L3 tiles: " << config.accelerators[0].kpu_config->memory.l3_tiles.size() << "\n";
// ... 10+ more error-prone lines
```

**Issues**:
- Error-prone indexing (`accelerators[0]`)
- Requires deep knowledge of config structure
- No access to memory map information
- Repetitive code in every client
- Maintenance burden when adding new fields

## Solution Architecture

### Layer 1: Stream Operators (`operator<<`)

Added comprehensive stream operators for all config structures:

**New Files**:
- `include/sw/system/config_formatter.hpp` - Declarations
- `src/system/config_formatter.cpp` - Implementations

**Features**:
- Standard C++ idiom
- Composable (nested configs format recursively)
- Works with any output stream (cout, files, stringstreams)
- Automatic hierarchical formatting

**Usage**:
```cpp
SystemConfig config = SystemConfig::create_minimal_kpu();
std::cout << config;  // Clean, automatic formatting!
```

### Layer 2: Memory Map Reporting

Exposed the existing `AddressDecoder::to_string()` through `SystemSimulator`:

**New Methods** in `SystemSimulator`:
- `std::string get_memory_map(size_t kpu_index = 0) const`
  - Returns formatted memory map for a specific KPU
  - Shows unified address space with all regions
  - Includes region types, IDs, and names

- `std::string get_system_report() const`
  - Comprehensive report: config + memory maps + runtime state
  - One-call access to complete system information

- `void print_full_report(std::ostream& os) const`
  - Stream-based output for files or custom destinations

**New Accessor** in `KPUSimulator`:
- `const AddressDecoder* get_address_decoder() const`
  - Provides read-only access to memory map

## Implementation Details

### Files Created
1. `include/sw/system/config_formatter.hpp` - Interface definitions
2. `src/system/config_formatter.cpp` - Formatting implementations
3. `tests/system/test_config_formatting.cpp` - Comprehensive unit tests
4. `examples/basic/config_formatter_demo.cpp` - Usage demonstration

### Files Modified
1. `include/sw/kpu/kpu_simulator.hpp` - Added get_address_decoder()
2. `include/sw/system/toplevel.hpp` - Added reporting methods
3. `src/system/toplevel.cpp` - Implemented reporting, simplified print_config()
4. `src/system/CMakeLists.txt` - Added config_formatter.cpp
5. `tests/system/CMakeLists.txt` - Added test_config_formatting
6. `examples/basic/CMakeLists.txt` - Added config_formatter_demo
7. `models/kpu/host_t100.cpp` - Simplified to use new API
8. `tests/integration/test_python_cpp_integration.cpp` - Added API validation

## Benefits

### 1. Eliminates Error-Prone Code
**Before**:
```cpp
// Unsafe - crashes if accelerators is empty or kpu_config is nullopt
std::cout << config.accelerators[0].kpu_config->memory.banks.size();
```

**After**:
```cpp
std::cout << config;  // Handles all cases automatically
```

### 2. Memory Map Visibility
**Before**: No way to see the unified address space

**After**:
```cpp
SystemSimulator sim(config);
sim.initialize();
std::cout << sim.get_memory_map();  // Complete memory map displayed
```

**Output**:
```
Memory Map (21 regions):
  Address Range          | Size      | Type        | ID | Name
  ---------------------- | --------- | ----------- | -- | ----
  0x00000000 - 0xffffffff | 4 GB     | HOST        | 00 | Host Memory Region 0
  0x100000000 - 0x13fffffff | 1 GB     | EXTERNAL    | 00 | External Bank 0
  0x140000000 - 0x17fffffff | 1 GB     | EXTERNAL    | 01 | External Bank 1
  0x180000000 - 0x18001ffff | 128 KB     | L3_TILE     | 00 | L3 Tile 0
  ...
```

### 3. Comprehensive System Reports
```cpp
std::string report = sim.get_system_report();
// Includes: configuration + memory maps + runtime state
// Perfect for debugging, documentation, and logging
```

### 4. Standard C++ Patterns
- Uses `operator<<` idiom familiar to all C++ developers
- Works with iostreams infrastructure
- Supports output redirection to files

### 5. Maintenance
- Single source of truth for formatting
- Adding new config fields only requires updating formatter
- No need to update multiple client sites

## Test Coverage

### Unit Tests (test_config_formatting.cpp)
- ✅ SystemInfo formatting
- ✅ CPUConfig formatting
- ✅ MemoryModuleConfig formatting
- ✅ KPUMemoryConfig with all hierarchy levels
- ✅ ComputeTileConfig with systolic arrays
- ✅ DMAEngineConfig with bandwidth
- ✅ AcceleratorType enum formatting
- ✅ InterconnectConfig with PCIe
- ✅ Complete SystemConfig formatting
- ✅ Factory configurations (minimal, edge, datacenter)
- ✅ Custom configurations
- ✅ Empty/minimal configurations
- ✅ to_string() and print_config() functions

**Result**: 57 assertions, all passing

### Integration Tests
- ✅ Updated `test_python_cpp_integration.cpp` to validate new APIs
- ✅ Verified memory map reporting works after initialization
- ✅ Confirmed system reports contain expected content

### Example Programs
- ✅ `example_config_formatter_demo` - Shows all API features
- ✅ `host_t100` - Simplified to use new operators

## Usage Examples

### Basic Configuration Output
```cpp
auto config = SystemConfig::create_minimal_kpu();
std::cout << config;
```

### Memory Map Access
```cpp
SystemSimulator sim(config);
sim.initialize();

// Get memory map for KPU 0
std::string memory_map = sim.get_memory_map(0);
std::cout << memory_map;
```

### Complete System Report
```cpp
// Get everything in one call
std::string full_report = sim.get_system_report();
std::cout << full_report;

// Or write to file
std::ofstream file("system_report.txt");
sim.print_full_report(file);
```

### Integration with Existing Code
```cpp
// Old way (verbose, error-prone)
std::cout << "  System: " << config.system.name << "\n";
std::cout << "  KPU Components:\n";
std::cout << "    Memory banks: "
          << config.accelerators[0].kpu_config->memory.banks.size() << "\n";
// ... many more lines

// New way (clean, automatic)
std::cout << config;
```

## Future Enhancements

### Verbosity Levels (Planned)
```cpp
enum class FormatDetail {
    Summary,   // Just counts and totals
    Standard,  // IDs and key specs (current default)
    Full       // Everything including computed values
};

// Usage
std::cout << to_string(config, FormatDetail::Summary);
```

### Custom Formatting (Extensible)
The architecture allows for:
- JSON output via `operator<<` to JSON stream
- XML output for tooling integration
- Machine-readable formats for automation

## Build System Integration

### CMake Changes
- Added `config_formatter.cpp` to `src/system/CMakeLists.txt`
- Added test target in `tests/system/CMakeLists.txt`
- Added example in `examples/basic/CMakeLists.txt`

### Compilation
No special flags required. The implementation uses standard C++20 features.

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code continues to work unchanged
- New functionality is additive only
- No breaking changes to existing APIs

## Performance

- **Formatting**: O(n) where n is the number of config elements
- **Memory overhead**: Minimal - mostly stack-based string streams
- **Memory map access**: O(1) - direct pointer access
- **System report**: O(m) where m is the number of KPU instances

## Documentation

- API documented in header files with Doxygen comments
- Example program demonstrates all features
- This implementation guide provides architecture overview
- Test cases serve as usage examples

## Conclusion

The config formatter and memory map API implementation successfully:
1. ✅ Eliminates error-prone manual formatting
2. ✅ Exposes memory map information automatically
3. ✅ Provides comprehensive system reporting in one call
4. ✅ Uses standard C++ idioms (operator<<)
5. ✅ Maintains backward compatibility
6. ✅ Includes comprehensive tests (57 assertions passing)
7. ✅ Integrates cleanly with existing codebase

**Client code is now cleaner, safer, and more maintainable.**

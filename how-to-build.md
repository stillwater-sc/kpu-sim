# How to build the KPU simulator

```bash
# Linux (default: GCC with Ninja)
cmake --preset=release
cmake --build --preset=release

# Linux with Clang
cmake --preset=linux-clang
cmake --build build

# Windows (Visual Studio)
cmake --preset=windows-msvc
cmake --build build --config Release

# macOS (Xcode)
cmake --preset=macos
cmake --build build --config Release

# Debug build (with sanitizers)
cmake --preset=debug
cmake --build --preset=debug
```

## Alternative without Ninja

```bash
# Use Unix Makefiles or Visual Studio instead
cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
cmake --build build
```

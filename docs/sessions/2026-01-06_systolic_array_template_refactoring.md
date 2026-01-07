# Session Log: SystolicArray Template Refactoring

**Date:** 2026-01-06
**Duration:** ~1 hour
**Focus:** Refactor SystolicArray to be a template class parameterized by Scalar type

## Summary

Refactored the `SystolicArray` class from a non-templated class with a hardcoded `Scalar` typedef to a fully parameterized template class. This enables instantiation with different numeric types (float, double, int8_t, int32_t, posit, etc.) while maintaining identical systolic array structure and behavior.

## Context

The original `SystolicArray` class had:
```cpp
class SystolicArray {
    using Scalar = double;  // Hardcoded type
    // ...
};
```

This design limited flexibility since a systolic array's structure is fundamentally orthogonal to its scalar type. The same PE array topology, data flow patterns, and timing characteristics apply whether the elements are floats, doubles, integers, or custom types like posits.

## Changes Made

### Phase 1: systolic_array.hpp Template Conversion

**Key changes:**
- Converted `class SystolicArray` to `template<typename Scalar> class SystolicArray`
- Removed the internal `using Scalar = double;` typedef
- Changed bus types from `std::queue<float>` to `std::queue<Scalar>`
- Updated all literal values from `0.0f` to `Scalar{0}` for type-safety
- Moved all method implementations inline (required for C++ templates)
- Added necessary includes (`<algorithm>`, `<stdexcept>`, `<cmath>`)

**ProcessingElement changes:**
- Already was a template class `ProcessingElement<Scalar>`
- Updated literal comparisons from `0.0f` to `Scalar{0}`
- Added value initialization for member variables

### Phase 2: systolic_array.cpp Simplification

Reduced to explicit template instantiations:
```cpp
namespace sw::kpu {
    template class SystolicArray<int8_t>;
    template class SystolicArray<int32_t>;
    template class SystolicArray<float>;
    template class SystolicArray<double>;
}
```

### Phase 3: compute_fabric.hpp/cpp Updates

Updated to use explicit template parameter:
- `std::unique_ptr<SystolicArray>` → `std::unique_ptr<SystolicArray<float>>`
- `SystolicArray::DEFAULT_ROWS` → `SystolicArray<float>::DEFAULT_ROWS`
- `SystolicArray::MatMulConfig` → `SystolicArray<float>::MatMulConfig`
- Return type of `get_systolic_array()` → `SystolicArray<float>*`

## Files Modified

| File | Changes |
|------|---------|
| `include/sw/kpu/components/systolic_array.hpp` | Major rewrite - template class with inline implementation |
| `src/components/compute/systolic_array.cpp` | Reduced to explicit instantiations only |
| `include/sw/kpu/components/compute_fabric.hpp` | Updated to use `SystolicArray<float>` |
| `src/components/compute/compute_fabric.cpp` | Updated instantiations and type references |

## Testing

All existing tests pass:
- `systolic_array_test`: 27 assertions in 3 test cases
- `compute_basic_test`: 2 assertions in 1 test case

Test coverage includes:
- Small matrix multiplication (2x2)
- Large matrix multiplication (8x8)
- Configuration queries
- Error handling (bounds checks, busy state)

## Benefits

1. **Type flexibility**: Can now create systolic arrays for different scalar types:
   ```cpp
   SystolicArray<float> float_array(16, 16);
   SystolicArray<double> double_array(16, 16);
   SystolicArray<int8_t> int8_array(16, 16);  // For quantized inference
   ```

2. **Future extensibility**: Support for custom numeric types like:
   - Posit numbers (universal number format)
   - Block floating point
   - Custom fixed-point types

3. **Compile-time type safety**: Type mismatches caught at compile time

4. **No runtime overhead**: Template instantiation produces optimal code for each type

## Design Notes

### Why Templates Over Virtual Dispatch?

Templates were chosen over a virtual base class approach because:
1. **Zero overhead**: No vtable indirection for inner-loop operations
2. **Full optimization**: Compiler can inline and optimize for specific types
3. **Type safety**: Catch errors at compile time, not runtime
4. **Simplicity**: Single class definition covers all types

### Why Explicit Instantiations?

Explicit instantiations in the .cpp file provide:
1. **Faster compile times**: Template code compiled once, not per translation unit
2. **Smaller binaries**: Single instantiation shared across the library
3. **Clear API**: Documents supported types for library users

## Lessons Learned

1. C++ template classes require implementations in headers (or explicit instantiation)
2. Use `Scalar{0}` instead of `0.0f` for type-generic zero values
3. Nested types in templates need full qualification: `SystolicArray<float>::MatMulConfig`
4. Explicit instantiations are useful for library code to control compile times

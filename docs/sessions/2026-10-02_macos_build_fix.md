# Session: macOS build fix (Apple Clang 21)

## What was done
1. **fmt 10.1.1 → 11.2.0** (`cmake/Dependencies.cmake`). fmt 10.x fails to compile under
   Apple Clang 21: `FMT_STRING(...)` "is not a constant expression" in its consteval format
   checks. 11.2.0 is the version spdlog v1.15.3 already bundles. No project source calls
   `fmt::` directly; it is only linked by `src/system`.
2. **`l0::parse_float` reads via `double`** (`include/sw/kpu/program/serialize/l0_format.hpp`).
   With the build fixed, `test_l0_serialization` failed: libc++'s `operator>>` into a
   `float` sets failbit on a denormal (strtof reports ERANGE), so `denorm_min` did not
   round-trip. CI has no macOS runner, so libstdc++/MSVC never showed it.

## Decisions and alternatives
- `std::from_chars(float)` was tried first. Rejected: Apple's libc++ marks it unavailable
  below macOS 26, and raising the deployment target for a parser is the wrong trade.
- Parsing to `double` and narrowing is exact for everything `exact_float` writes (9
  significant digits sit far from any float rounding midpoint).

## Wrong decisions
- The first overflow guard (`|d| > FLT_MAX`) rejected FLT_MAX itself: its 9-digit text
  is slightly above FLT_MAX as a double but rounds back to it. Fixed by checking
  `isinf` after narrowing. Lesson: judge range on the narrowed value, not the wide one.

## Validation
- `cmake --build build`: clean.
- `ctest`: 187/187 passed.

## Follow-up
- Consider a macOS job in `.github/workflows/cmake-multi-platform.yml`; both failures
  were macOS-only and invisible to CI.

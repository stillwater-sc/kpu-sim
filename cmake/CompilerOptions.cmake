# cmake/CompilerOptions.cmake
# Modern C++ compiler options and optimizations

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Common GCC/Clang options
    set(KPU_CXX_FLAGS_DEBUG "-O0 -g3 -ggdb")
    set(KPU_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(KPU_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(KPU_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
    
    # Warning flags
    set(KPU_WARNING_FLAGS
        -Wall -Wextra -Wpedantic
        -Wcast-align -Wcast-qual -Wctor-dtor-privacy
        -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs
        -Wnoexcept -Wold-style-cast -Woverloaded-virtual
        -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo
        -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default
        -Wundef -Wno-unused
    )

    # On newer x86 CI runners, `-march=native` enables a partial AVX10.1 target
    # feature (`+avx10.1-256`), which Clang flags via -Winvalid-feature-combination
    # ("will be promoted to avx10.1-512") and -Werror then turns fatal. It is a
    # benign codegen-tuning note, not a source issue, so silence it for Clang.
    # (GCC has no such warning; it builds cleanly with the same -march=native.)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND KPU_WARNING_FLAGS -Wno-invalid-feature-combination)
    endif()


    # Architecture-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND KPU_CXX_FLAGS_RELEASE "-march=native -mtune=native")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        list(APPEND KPU_CXX_FLAGS_RELEASE "-mcpu=native")
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC options
    set(KPU_CXX_FLAGS_DEBUG "/Od /Zi /RTC1")
    set(KPU_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(KPU_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi /DNDEBUG")
    set(KPU_CXX_FLAGS_MINSIZEREL "/O1 /DNDEBUG")
    
    set(KPU_WARNING_FLAGS /W4 /permissive-)
endif()

# Apply compiler flags
function(kpu_set_target_options target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE ${KPU_WARNING_FLAGS})
    
    # Build-type specific flags
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:${KPU_CXX_FLAGS_DEBUG}>
        $<$<CONFIG:Release>:${KPU_CXX_FLAGS_RELEASE}>
        $<$<CONFIG:RelWithDebInfo>:${KPU_CXX_FLAGS_RELWITHDEBINFO}>
        $<$<CONFIG:MinSizeRel>:${KPU_CXX_FLAGS_MINSIZEREL}>
    )
    
    # Enable position-independent code
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Link-time optimization for release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION ON
        )
    endif()
    
    # Sanitizers
    if(KPU_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE 
                -fsanitize=address,undefined,leak
                -fno-omit-frame-pointer
            )
            target_link_options(${target} PRIVATE 
                -fsanitize=address,undefined,leak
            )
        endif()
    endif()
    
    # Profiling support
    if(KPU_ENABLE_PROFILING)
        target_compile_definitions(${target} PRIVATE KPU_ENABLE_PROFILING)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -pg)
            target_link_options(${target} PRIVATE -pg)
        endif()
    endif()
endfunction()


# Makefile for GPU MLP Implementations
# Supports NVIDIA CUDA and OpenCL (for Mali)

# Detect available GPU frameworks
HAS_CUDA := $(shell which nvcc 2>/dev/null)
HAS_OPENCL := $(shell echo '\#include <CL/cl.h>' | gcc -E - 2>/dev/null && echo yes)

.PHONY: all cuda opencl clean info help

# Default target
all: info
	@echo ""
	@echo "Building available targets..."
	@$(MAKE) -s build_available

build_available:
	@if [ -n "$(HAS_CUDA)" ]; then $(MAKE) cuda; fi
	@if [ -n "$(HAS_OPENCL)" ]; then $(MAKE) opencl; fi
	@if [ -z "$(HAS_CUDA)" ] && [ -z "$(HAS_OPENCL)" ]; then \
		echo "ERROR: No GPU framework found. Install CUDA Toolkit or OpenCL."; \
		exit 1; \
	fi

# ============================================================================
# NVIDIA CUDA
# ============================================================================

NVCC := nvcc
CUDA_ARCH := sm_80
CUDA_FLAGS := -arch=$(CUDA_ARCH) -O3 -lineinfo
CUDA_SRC := cuda/mlp_cuda_example.cu
CUDA_BIN := bin/mlp_cuda

cuda: $(CUDA_BIN)
	@echo "✓ CUDA example built successfully"

$(CUDA_BIN): $(CUDA_SRC) | bin
	@echo "Building CUDA example (Ampere $(CUDA_ARCH))..."
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Compile PTX assembly
PTX_SRC := nvidia_ampere/mlp_forward.ptx
PTX_OBJ := bin/mlp_forward.ptx.o

ptx: $(PTX_OBJ)
	@echo "✓ PTX compiled successfully"

$(PTX_OBJ): $(PTX_SRC) | bin
	@echo "Compiling PTX assembly..."
	$(NVCC) -arch=$(CUDA_ARCH) -c $< -o $@

# Generate SASS from CUDA source
sass: cuda
	@echo "Generating SASS assembly..."
	cuobjdump -sass $(CUDA_BIN) > bin/mlp_cuda.sass
	@echo "✓ SASS generated: bin/mlp_cuda.sass"

# Run CUDA example
run-cuda: $(CUDA_BIN)
	@echo "Running CUDA example..."
	@$(CUDA_BIN)

# Benchmark CUDA with different configurations
benchmark-cuda: $(CUDA_BIN)
	@echo "Benchmarking CUDA with different problem sizes..."
	@for size in 256 512 1024 2048; do \
		echo ""; \
		echo "=== Input size: $$size ==="; \
		$(CUDA_BIN); \
	done

# Profile CUDA kernel
profile-cuda: $(CUDA_BIN)
	@echo "Profiling with NVIDIA Nsight Compute..."
	ncu --set full -o profile_cuda $(CUDA_BIN)
	@echo "Profile saved to: profile_cuda.ncu-rep"
	@echo "View with: ncu-ui profile_cuda.ncu-rep"

# ============================================================================
# OpenCL (for ARM Mali and other GPUs)
# ============================================================================

CC := gcc
OPENCL_FLAGS := -O3 -Wall
OPENCL_LIBS := -lOpenCL
OPENCL_SRC := opencl/mlp_opencl_example.c
OPENCL_BIN := bin/mlp_opencl

opencl: $(OPENCL_BIN)
	@echo "✓ OpenCL example built successfully"

$(OPENCL_BIN): $(OPENCL_SRC) | bin
	@echo "Building OpenCL example..."
	$(CC) $(OPENCL_FLAGS) $< -o $@ $(OPENCL_LIBS)

# Run OpenCL example
run-opencl: $(OPENCL_BIN)
	@echo "Running OpenCL example..."
	@$(OPENCL_BIN)

# Android cross-compilation for Mali
ANDROID_NDK := $(ANDROID_NDK_ROOT)
ANDROID_API := 29
ANDROID_ARCH := arm64-v8a

android: check-ndk
	@echo "Cross-compiling for Android (Mali)..."
	@mkdir -p bin/android
	$(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(ANDROID_API)-clang \
		-O3 $(OPENCL_SRC) -o bin/android/mlp_opencl -lOpenCL
	@echo "✓ Android binary built: bin/android/mlp_opencl"
	@echo "Deploy with: adb push bin/android/mlp_opencl /data/local/tmp/"

check-ndk:
	@if [ -z "$(ANDROID_NDK)" ]; then \
		echo "ERROR: ANDROID_NDK_ROOT not set"; \
		echo "Download from: https://developer.android.com/ndk"; \
		exit 1; \
	fi

# ============================================================================
# Utilities
# ============================================================================

bin:
	@mkdir -p bin bin/android

# Show PTX assembly
show-ptx: cuda
	@echo "=== PTX Assembly ==="
	@cuobjdump -ptx $(CUDA_BIN) | head -200
	@echo ""
	@echo "(Showing first 200 lines. Full output: cuobjdump -ptx $(CUDA_BIN))"

# Show SASS assembly
show-sass: cuda
	@echo "=== SASS Assembly (Ampere) ==="
	@cuobjdump -sass $(CUDA_BIN) | head -200
	@echo ""
	@echo "(Showing first 200 lines. Full output: cuobjdump -sass $(CUDA_BIN))"

# Device information
device-info:
	@echo "=== GPU Device Information ==="
	@if [ -n "$(HAS_CUDA)" ]; then \
		echo ""; \
		echo "NVIDIA CUDA Devices:"; \
		nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || \
		echo "  nvidia-smi not found"; \
	fi
	@if [ -n "$(HAS_OPENCL)" ]; then \
		echo ""; \
		echo "OpenCL Devices:"; \
		clinfo -l 2>/dev/null || \
		echo "  clinfo not found (install clinfo package)"; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf bin/
	@echo "✓ Clean complete"

# Show build information
info:
	@echo "GPU MLP Build System"
	@echo "===================="
	@echo ""
	@echo "Available frameworks:"
	@if [ -n "$(HAS_CUDA)" ]; then \
		echo "  ✓ NVIDIA CUDA ($(shell nvcc --version | grep release | awk '{print $$5}'))"; \
		echo "    Target: $(CUDA_ARCH)"; \
	else \
		echo "  ✗ NVIDIA CUDA (not found)"; \
	fi
	@if [ -n "$(HAS_OPENCL)" ]; then \
		echo "  ✓ OpenCL"; \
	else \
		echo "  ✗ OpenCL (not found)"; \
	fi
	@echo ""

# Help message
help:
	@echo "GPU MLP Assembly - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all available targets (default)"
	@echo "  cuda             - Build CUDA example"
	@echo "  opencl           - Build OpenCL example"
	@echo "  ptx              - Compile PTX assembly"
	@echo "  sass             - Generate SASS assembly from CUDA"
	@echo ""
	@echo "  run-cuda         - Build and run CUDA example"
	@echo "  run-opencl       - Build and run OpenCL example"
	@echo "  benchmark-cuda   - Run CUDA benchmarks"
	@echo "  profile-cuda     - Profile CUDA kernel (requires Nsight Compute)"
	@echo ""
	@echo "  android          - Cross-compile for Android (Mali)"
	@echo "  device-info      - Show GPU device information"
	@echo ""
	@echo "  show-ptx         - Display PTX assembly"
	@echo "  show-sass        - Display SASS assembly"
	@echo ""
	@echo "  clean            - Remove build artifacts"
	@echo "  info             - Show build configuration"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Architecture Selection:"
	@echo "  CUDA_ARCH=sm_XX  - Set CUDA architecture (default: sm_80 for Ampere)"
	@echo "    sm_75 - Turing"
	@echo "    sm_80 - Ampere"
	@echo "    sm_86 - Ampere (RTX 3090)"
	@echo "    sm_89 - Ada Lovelace (RTX 4090)"
	@echo "    sm_90 - Hopper (H100)"
	@echo ""
	@echo "Examples:"
	@echo "  make cuda CUDA_ARCH=sm_89    # Build for RTX 4090"
	@echo "  make run-cuda                # Build and run CUDA"
	@echo "  make opencl && make run-opencl  # Build and run OpenCL"
	@echo "  make show-sass > ampere.sass # Save SASS to file"
	@echo ""

.DEFAULT_GOAL := help

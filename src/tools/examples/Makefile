# Makefile for MLP Assembly Implementations
# Supports x86-64 (AVX/SSE) and ARM (NEON) architectures

# Detect architecture
ARCH := $(shell uname -m)

# Compiler settings
CC = gcc
AS = as
CFLAGS = -Wall -O2 -I./include
LDFLAGS = -lm

# Architecture-specific settings
ifeq ($(ARCH),x86_64)
    ASM_SRC = x86_64/mlp_avx.s
    CFLAGS += -mavx -mfma
    TARGET = mlp_example_x86_64
    $(info Building for x86-64 with AVX support)
else ifeq ($(ARCH),aarch64)
    ASM_SRC = arm/mlp_neon_aarch64.s
    CFLAGS += -march=armv8-a
    TARGET = mlp_example_aarch64
    $(info Building for ARM AArch64 with NEON support)
else ifeq ($(ARCH),armv7l)
    ASM_SRC = arm/mlp_neon_armv7.s
    CFLAGS += -march=armv7-a -mfpu=neon
    TARGET = mlp_example_armv7
    $(info Building for ARM ARMv7 with NEON support)
else
    $(warning Unknown architecture: $(ARCH))
    $(warning Building without assembly optimization)
    ASM_SRC =
    TARGET = mlp_example_generic
endif

# Source files
C_SRC = src/example.c
OBJ_DIR = obj
BIN_DIR = bin

# Object files
ASM_OBJ = $(if $(ASM_SRC),$(OBJ_DIR)/$(notdir $(ASM_SRC:.s=.o)),)
C_OBJ = $(OBJ_DIR)/example.o

# Default target
.PHONY: all
all: $(BIN_DIR)/$(TARGET)

# Create directories
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile C source
$(C_OBJ): $(C_SRC) include/mlp_asm.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Assemble x86-64
$(OBJ_DIR)/mlp_avx.o: x86_64/mlp_avx.s | $(OBJ_DIR)
	$(AS) $< -o $@

# Assemble ARM AArch64
$(OBJ_DIR)/mlp_neon_aarch64.o: arm/mlp_neon_aarch64.s | $(OBJ_DIR)
	$(AS) $< -o $@

# Assemble ARM ARMv7
$(OBJ_DIR)/mlp_neon_armv7.o: arm/mlp_neon_armv7.s | $(OBJ_DIR)
	$(AS) $< -o $@

# Link executable
$(BIN_DIR)/$(TARGET): $(C_OBJ) $(ASM_OBJ) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Cross-compilation targets
.PHONY: x86_64
x86_64:
	$(MAKE) ARCH=x86_64 CC=gcc AS=as

.PHONY: aarch64
aarch64:
	$(MAKE) ARCH=aarch64 CC=aarch64-linux-gnu-gcc AS=aarch64-linux-gnu-as

.PHONY: armv7
armv7:
	$(MAKE) ARCH=armv7l CC=arm-linux-gnueabihf-gcc AS=arm-linux-gnueabihf-as

# Run the example
.PHONY: run
run: $(BIN_DIR)/$(TARGET)
	./$(BIN_DIR)/$(TARGET)

# Benchmark mode
.PHONY: benchmark
benchmark: CFLAGS += -O3 -march=native
benchmark: clean all
	@echo "Running optimized benchmark..."
	./$(BIN_DIR)/$(TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned build artifacts"

# Show assembly listing
.PHONY: listing
listing: $(ASM_SRC)
	@echo "=== Assembly Listing ==="
	@cat $(ASM_SRC) | grep -v "^[[:space:]]*$$" | head -100
	@echo "..."
	@echo "(Showing first 100 non-empty lines)"

# Show architecture info
.PHONY: info
info:
	@echo "Architecture: $(ARCH)"
	@echo "Assembly source: $(ASM_SRC)"
	@echo "Target binary: $(BIN_DIR)/$(TARGET)"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

# Help
.PHONY: help
help:
	@echo "MLP Assembly Implementation - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build for current architecture (default)"
	@echo "  run         - Build and run the example program"
	@echo "  benchmark   - Build with full optimizations and run"
	@echo "  x86_64      - Cross-compile for x86-64"
	@echo "  aarch64     - Cross-compile for ARM AArch64"
	@echo "  armv7       - Cross-compile for ARM ARMv7"
	@echo "  clean       - Remove build artifacts"
	@echo "  listing     - Show assembly source code"
	@echo "  info        - Show build configuration"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Current configuration:"
	@echo "  Architecture: $(ARCH)"
	@echo "  Target: $(TARGET)"

.DEFAULT_GOAL := all

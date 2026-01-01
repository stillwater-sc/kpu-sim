#!/bin/bash
# install-tools.sh - Build and install KPU tools to local ./bin directory
#
# Usage:
#   ./scripts/install-tools.sh          # Build and install all tools
#   ./scripts/install-tools.sh dfg      # Build and install only DFG tools
#   ./scripts/install-tools.sh clean    # Remove installed tools

set -e

# Find project root (where this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$PROJECT_ROOT/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Handle clean command
if [[ "$1" == "clean" ]]; then
    info "Removing installed tools from $BIN_DIR"
    rm -rf "$BIN_DIR"
    info "Done"
    exit 0
fi

# Ensure build directory exists and is configured
if [[ ! -f "$BUILD_DIR/Makefile" ]] && [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
    info "Configuring build..."
    cmake -B "$BUILD_DIR" -S "$PROJECT_ROOT" -DCMAKE_BUILD_TYPE=Release
fi

# Determine which tools to build
if [[ "$1" == "dfg" ]]; then
    TARGETS="dfg-gen dfg-sched dfg-compile dfg-viz dfg-analyze"
    info "Building DFG tools..."
elif [[ "$1" == "all" ]] || [[ -z "$1" ]]; then
    # Build all tools by building the entire project
    TARGETS=""
    info "Building all tools..."
else
    TARGETS="$1"
    info "Building $TARGETS..."
fi

# Build
if [[ -n "$TARGETS" ]]; then
    cmake --build "$BUILD_DIR" --target $TARGETS --parallel
else
    cmake --build "$BUILD_DIR" --parallel
fi

# Create bin directory
mkdir -p "$BIN_DIR"

# Function to install a tool
install_tool() {
    local src="$1"
    local name="$2"
    if [[ -f "$src" ]]; then
        cp "$src" "$BIN_DIR/$name"
        chmod +x "$BIN_DIR/$name"
        info "Installed: $name"
    fi
}

# Install DFG tools
install_tool "$BUILD_DIR/tools/dfg/kpu-dfg-gen" "kpu-dfg-gen"
install_tool "$BUILD_DIR/tools/dfg/kpu-dfg-sched" "kpu-dfg-sched"
install_tool "$BUILD_DIR/tools/dfg/kpu-dfg-compile" "kpu-dfg-compile"
install_tool "$BUILD_DIR/tools/dfg/kpu-dfg-viz" "kpu-dfg-viz"
install_tool "$BUILD_DIR/tools/dfg/kpu-dfg-analyze" "kpu-dfg-analyze"

# Install other tools if they exist
install_tool "$BUILD_DIR/tools/runner/kpu-runner" "kpu-runner"
install_tool "$BUILD_DIR/tools/configuration/kpu-config" "kpu-config"
install_tool "$BUILD_DIR/tools/analysis/kpu-kpubin-disasm" "kpu-kpubin-disasm"

# Count installed tools
TOOL_COUNT=$(ls -1 "$BIN_DIR" 2>/dev/null | wc -l)

info ""
info "Installed $TOOL_COUNT tools to: $BIN_DIR"
info ""
info "To use these tools, either:"
info "  1. Add to PATH:  export PATH=\"$BIN_DIR:\$PATH\""
info "  2. Source env:   source $PROJECT_ROOT/scripts/env.sh"
info "  3. Use full path: $BIN_DIR/kpu-dfg-gen --help"

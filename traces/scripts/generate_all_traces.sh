#!/bin/bash
#
# generate_all_traces.sh
#
# Generates trace files for all memory patterns (LPDDR5 and GDDR6)
#
# Usage:
#   ./generate_all_traces.sh           # Generate all traces
#   ./generate_all_traces.sh lpddr5    # Generate only LPDDR5 traces
#   ./generate_all_traces.sh gddr6     # Generate only GDDR6 traces
#   ./generate_all_traces.sh --clean   # Remove all traces before generating
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TRACES_DIR="$PROJECT_ROOT/traces"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0

# Parse arguments
GENERATE_LPDDR5=true
GENERATE_GDDR6=true
CLEAN_FIRST=false

for arg in "$@"; do
    case $arg in
        lpddr5)
            GENERATE_GDDR6=false
            ;;
        gddr6)
            GENERATE_LPDDR5=false
            ;;
        --clean)
            CLEAN_FIRST=true
            ;;
        --help|-h)
            echo "Usage: $0 [lpddr5|gddr6] [--clean]"
            echo ""
            echo "Options:"
            echo "  lpddr5    Generate only LPDDR5 traces"
            echo "  gddr6     Generate only GDDR6 traces"
            echo "  --clean   Remove existing traces before generating"
            echo "  --help    Show this help message"
            exit 0
            ;;
    esac
done

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}ERROR: Build directory not found at $BUILD_DIR${NC}"
    echo "Please run: cmake --preset release && cmake --build --preset release"
    exit 1
fi

# Clean traces if requested
if [ "$CLEAN_FIRST" = true ]; then
    echo -e "${YELLOW}Cleaning existing traces...${NC}"
    find "$TRACES_DIR/memory" -name "*.json" -type f -delete 2>/dev/null || true
fi

# Function to run a pattern and check result
run_pattern() {
    local name=$1
    local binary=$2

    if [ ! -x "$binary" ]; then
        echo -e "  ${YELLOW}SKIP${NC} $name (binary not found)"
        return
    fi

    # Run pattern and capture output
    if output=$("$binary" 2>&1); then
        if echo "$output" | grep -q "=== PASS ==="; then
            echo -e "  ${GREEN}PASS${NC} $name"
            ((PASS_COUNT++))
        else
            echo -e "  ${YELLOW}WARN${NC} $name (completed but no PASS marker)"
            ((PASS_COUNT++))
        fi
    else
        echo -e "  ${RED}FAIL${NC} $name"
        ((FAIL_COUNT++))
    fi
}

echo "=============================================="
echo "Memory Pattern Trace Generator"
echo "=============================================="
echo ""

# LPDDR5 Patterns
if [ "$GENERATE_LPDDR5" = true ]; then
    echo -e "${BLUE}=== LPDDR5 Patterns ===${NC}"

    echo "Level 1 - Single Bank:"
    run_pattern "page_hits" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_page_hits"
    run_pattern "page_conflicts" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_page_conflicts"
    run_pattern "mixed_rw" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_mixed_rw"

    echo "Level 2 - Two Bank:"
    run_pattern "same_group" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_same_group"
    run_pattern "diff_groups" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_diff_groups"

    echo "Level 3 - Three Bank:"
    run_pattern "three_same_group" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_three_same_group"
    run_pattern "three_mixed_groups" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_three_mixed_groups"

    echo "Level 4 - Four Bank:"
    run_pattern "full_group" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_full_group"
    run_pattern "across_groups" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_across_groups"
    run_pattern "page_hit_burst" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_page_hit_burst"

    echo "Level 5 - Dual Channel:"
    run_pattern "independent" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_independent"
    run_pattern "interleaved" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_interleaved"

    echo "Level 6 - Complex:"
    run_pattern "strided" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_strided"
    run_pattern "random" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_random"
    run_pattern "tile_load" "$BUILD_DIR/patterns/memory/lpddr5/lpddr5_tile_load"

    echo ""
fi

# GDDR6 Patterns
if [ "$GENERATE_GDDR6" = true ]; then
    echo -e "${BLUE}=== GDDR6 Patterns ===${NC}"

    echo "Level 1 - Single Bank:"
    run_pattern "page_hits" "$BUILD_DIR/patterns/memory/gddr6/gddr6_page_hits"
    run_pattern "page_conflicts" "$BUILD_DIR/patterns/memory/gddr6/gddr6_page_conflicts"
    run_pattern "mixed_rw" "$BUILD_DIR/patterns/memory/gddr6/gddr6_mixed_rw"

    echo "Level 2 - Two Bank:"
    run_pattern "two_same_group" "$BUILD_DIR/patterns/memory/gddr6/gddr6_two_same_group"
    run_pattern "two_diff_groups" "$BUILD_DIR/patterns/memory/gddr6/gddr6_two_diff_groups"

    echo "Level 3 - Three Bank:"
    run_pattern "three_same_group" "$BUILD_DIR/patterns/memory/gddr6/gddr6_three_same_group"
    run_pattern "three_mixed_groups" "$BUILD_DIR/patterns/memory/gddr6/gddr6_three_mixed_groups"

    echo "Level 4 - Four Bank:"
    run_pattern "full_group" "$BUILD_DIR/patterns/memory/gddr6/gddr6_full_group"
    run_pattern "across_groups" "$BUILD_DIR/patterns/memory/gddr6/gddr6_across_groups"
    run_pattern "page_hit_burst" "$BUILD_DIR/patterns/memory/gddr6/gddr6_page_hit_burst"

    echo "Level 5 - Dual Channel:"
    run_pattern "dual_independent" "$BUILD_DIR/patterns/memory/gddr6/gddr6_dual_independent"
    run_pattern "dual_interleaved" "$BUILD_DIR/patterns/memory/gddr6/gddr6_dual_interleaved"

    echo "Level 6 - Complex:"
    run_pattern "strided" "$BUILD_DIR/patterns/memory/gddr6/gddr6_strided"
    run_pattern "random" "$BUILD_DIR/patterns/memory/gddr6/gddr6_random"
    run_pattern "tile_load" "$BUILD_DIR/patterns/memory/gddr6/gddr6_tile_load"

    echo ""
fi

# Summary
echo "=============================================="
echo "Summary"
echo "=============================================="
echo -e "  ${GREEN}PASS${NC}: $PASS_COUNT"
echo -e "  ${RED}FAIL${NC}: $FAIL_COUNT"
echo ""

# List generated traces
echo "Generated traces:"
find "$TRACES_DIR/memory" -name "*.json" -type f 2>/dev/null | sort | while read -r trace; do
    size=$(du -h "$trace" | cut -f1)
    echo "  $size  ${trace#$TRACES_DIR/}"
done

echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Some patterns failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All patterns passed!${NC}"
    exit 0
fi

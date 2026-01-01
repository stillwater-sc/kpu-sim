#!/bin/bash
# matmul_pipeline.sh - Example DFG pipeline for matrix multiplication
#
# This script demonstrates how to use the DFG toolchain to:
# 1. Generate a matmul DFG
# 2. Schedule it
# 3. Compile to BlockMover programs
# 4. Analyze the results
# 5. Generate visualizations
#
# Usage:
#   ./examples/pipelines/matmul_pipeline.sh [M] [N] [K] [tiles]
#   ./examples/pipelines/matmul_pipeline.sh 2048 2048 2048 8x8x8

set -e

# === Environment Setup ===
# Find and source the KPU environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/kpu-env.sh" || {
    echo "Error: Cannot find KPU environment. Are you in the kpu-sim project?"
    exit 1
}

# Verify required tools are installed
kpu_require kpu-dfg-gen kpu-dfg-sched kpu-dfg-compile kpu-dfg-analyze kpu-dfg-viz || {
    echo ""
    echo "Tools not installed. Run:"
    echo "  $KPU_ROOT/scripts/install-tools.sh"
    exit 1
}

# === Configuration ===
M="${1:-1024}"
N="${2:-1024}"
K="${3:-1024}"
TILES="${4:-4x4x4}"
OUTPUT_DIR="${5:-./dfg_output}"
ALGORITHM="${6:-ASAP}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== KPU Matmul Pipeline ==="
echo "Matrix: ${M}x${K} * ${K}x${N}"
echo "Tiles:  $TILES"
echo "Output: $OUTPUT_DIR"
echo ""

# === Step 1: Generate DFG ===
echo "[1/5] Generating DFG..."
kpu-dfg-gen \
    --template matmul \
    -M "$M" -N "$N" -K "$K" \
    --tiles "$TILES" \
    -o "$OUTPUT_DIR/matmul.dfg.json" \
    -v

# === Step 2: Schedule ===
echo ""
echo "[2/5] Scheduling ($ALGORITHM)..."
kpu-dfg-sched \
    -i "$OUTPUT_DIR/matmul.dfg.json" \
    -o "$OUTPUT_DIR/matmul.sched.json" \
    --algorithm "$ALGORITHM" \
    --validate \
    -v

# === Step 3: Compile ===
echo ""
echo "[3/5] Compiling to BlockMover programs..."
kpu-dfg-compile \
    -i "$OUTPUT_DIR/matmul.sched.json" \
    -o "$OUTPUT_DIR/matmul.prog.json" \
    -v

# === Step 4: Analyze ===
echo ""
echo "[4/5] Analyzing..."
kpu-dfg-analyze \
    -i "$OUTPUT_DIR/matmul.dfg.json" \
    --stats --critical-path

# === Step 5: Visualize ===
echo ""
echo "[5/5] Generating visualizations..."

# DOT graph
kpu-dfg-viz \
    -i "$OUTPUT_DIR/matmul.dfg.json" \
    -o "$OUTPUT_DIR/matmul.dot" \
    --format dot

# Chrome Trace (for Perfetto)
kpu-dfg-viz \
    -i "$OUTPUT_DIR/matmul.sched.json" \
    -o "$OUTPUT_DIR/matmul_timeline.json" \
    --format chrome-trace

echo ""
echo "=== Pipeline Complete ==="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "To view timeline: Open $OUTPUT_DIR/matmul_timeline.json in https://ui.perfetto.dev"
echo "To view graph:    dot -Tpng $OUTPUT_DIR/matmul.dot -o $OUTPUT_DIR/matmul.png"

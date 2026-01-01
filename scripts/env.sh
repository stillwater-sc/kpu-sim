#!/bin/bash
# env.sh - Set up environment for KPU tools
#
# Usage:
#   source scripts/env.sh      # From project root
#   source ./env.sh            # From scripts directory
#
# IMPORTANT: This script must be SOURCED, not executed!
#   CORRECT:   source scripts/env.sh
#   WRONG:     ./scripts/env.sh
#
# This script can be sourced from anywhere - it will find the project root.

# Detect if script is being executed instead of sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed!"
    echo ""
    echo "Usage:"
    echo "  source $0"
    echo "  # or"
    echo "  . $0"
    echo ""
    echo "This allows the PATH changes to persist in your current shell."
    exit 1
fi

# Find the directory where this script lives
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    _ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Fallback for zsh or other shells
    _ENV_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# Project root is parent of scripts/
export KPU_ROOT="$(cd "$_ENV_SCRIPT_DIR/.." && pwd)"
export KPU_BIN="$KPU_ROOT/bin"
export KPU_BUILD="$KPU_ROOT/build"

# Add bin to PATH if not already there
if [[ ":$PATH:" != *":$KPU_BIN:"* ]]; then
    export PATH="$KPU_BIN:$PATH"
    echo "Added $KPU_BIN to PATH"
fi

# Also add Python tools if they exist
if [[ -d "$KPU_ROOT/tools/python" ]]; then
    if [[ ":$PYTHONPATH:" != *":$KPU_ROOT/tools/python:"* ]]; then
        export PYTHONPATH="$KPU_ROOT/tools/python:$PYTHONPATH"
    fi
fi

# Helper function to rebuild and reinstall tools
kpu-rebuild() {
    echo "Rebuilding KPU tools..."
    "$KPU_ROOT/scripts/install-tools.sh" "$@"
}

# Helper function to get project root from any subdirectory
kpu-root() {
    echo "$KPU_ROOT"
}

# Show what's available
echo "KPU environment configured:"
echo "  KPU_ROOT:  $KPU_ROOT"
echo "  KPU_BIN:   $KPU_BIN"
echo "  KPU_BUILD: $KPU_BUILD"
echo ""
echo "Available commands:"
if [[ -d "$KPU_BIN" ]]; then
    ls -1 "$KPU_BIN" 2>/dev/null | sed 's/^/  /'
else
    echo "  (no tools installed - run: ./scripts/install-tools.sh)"
fi

# Cleanup temp variable
unset _ENV_SCRIPT_DIR

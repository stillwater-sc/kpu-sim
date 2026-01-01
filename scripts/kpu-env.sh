#!/bin/bash
# kpu-env.sh - Lightweight environment setup for pipeline scripts
#
# Include this at the top of your scripts to automatically find KPU tools:
#
#   #!/bin/bash
#   source "$(dirname "$0")/../scripts/kpu-env.sh" || exit 1
#
# Or for scripts that might be in subdirectories:
#
#   #!/bin/bash
#   # Find project root by looking for .git or CMakeLists.txt
#   _find_kpu_root() {
#       local dir="$PWD"
#       while [[ "$dir" != "/" ]]; do
#           if [[ -f "$dir/scripts/kpu-env.sh" ]]; then
#               echo "$dir"
#               return 0
#           fi
#           dir="$(dirname "$dir")"
#       done
#       return 1
#   }
#   KPU_ROOT="$(_find_kpu_root)" || { echo "Error: Cannot find KPU project root"; exit 1; }
#   source "$KPU_ROOT/scripts/kpu-env.sh"

# Determine project root
_find_project_root() {
    local dir="${1:-$PWD}"
    while [[ "$dir" != "/" ]]; do
        # Look for characteristic files
        if [[ -f "$dir/CMakeLists.txt" ]] && [[ -d "$dir/tools/dfg" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

# Try to find root from script location first, then from PWD
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    KPU_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
else
    KPU_ROOT="$(_find_project_root)" || {
        echo "Error: Cannot find KPU project root" >&2
        return 1 2>/dev/null || exit 1
    }
fi

export KPU_ROOT
export KPU_BIN="$KPU_ROOT/bin"
export KPU_BUILD="$KPU_ROOT/build"

# Add bin to PATH (silently)
if [[ ":$PATH:" != *":$KPU_BIN:"* ]]; then
    export PATH="$KPU_BIN:$PATH"
fi

# Verify tools exist
_check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "Warning: $1 not found. Run: $KPU_ROOT/scripts/install-tools.sh" >&2
        return 1
    fi
    return 0
}

# Export check function for scripts to use
kpu_require() {
    for tool in "$@"; do
        _check_tool "$tool" || return 1
    done
}

unset _SCRIPT_DIR

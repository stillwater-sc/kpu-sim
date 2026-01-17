# python/kpu/_native/__init__.py
"""
Native bindings package for KPU Python API.

This module provides optional C++ acceleration for DFX program execution.
The pure Python implementation in kpu.runtime works without this module,
but the native bindings can provide better performance.

Usage:
    try:
        from kpu._native import _native
        runtime = _native.create_runtime()
        # Use native execution
    except ImportError:
        # Fall back to pure Python
        pass
"""

try:
    from ._native import *
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False

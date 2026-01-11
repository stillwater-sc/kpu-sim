"""
Stillwater KPU Simulator Python Bindings
"""

try:
    # Try importing the SystemSimulator (clean orchestration API)
    from .stillwater_toplevel import *
    __all__ = ["SystemSimulator"]
except ImportError:
    # Fallback to empty if bindings not built
    __all__ = []
    pass

# TODO: Re-enable when KPU components are implemented
# from .stillwater_kpu_native import *
# __all__.extend(["KpuSimulator", "MemoryManager", "MemoryPool"])

__version__ = "0.1.0"
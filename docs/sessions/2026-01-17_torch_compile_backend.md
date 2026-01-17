# Session Log: torch.compile Backend for KPU

**Date:** 2026-01-17
**Duration:** ~2 hours
**Focus:** Implementation and debugging of torch.compile backend for KPU simulator

## Summary

Implemented a fully functional `torch.compile` backend for the KPU simulator, enabling PyTorch models to be compiled and executed on the KPU behavioral simulator. This provides the Exaloop compiler team with the infrastructure to walk PyTorch FX IR and convert to KPU kernels for hardware/software co-design.

## Context

The user explicitly rejected creating a custom JSON model serialization format, stating:
> "It is not advisable to create yet another DNN serialization format in JSON... Torch.compile will give the compiler team the infrastructure to walk the IR and convert to KPU kernels."

The goal is to run PyTorch models live through `torch.compile` as the first step, with offline export (ONNX/flatbuffers) as a future goal for deployments.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    torch.compile(model, backend="kpu")          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PyTorch Dynamo                          │
│   Captures model execution as FX GraphModule with placeholders  │
│   for both parameters AND inputs                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         KPU Backend                             │
│   kpu/torch_backend.py - Registered as "kpu" backend            │
│   Receives: gm (FX GraphModule), example_inputs (all tensors)   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FXToKPUConverter                           │
│   kpu/fx_converter.py - Walks FX graph, emits KPU operations    │
│   Handles: torch._C._nn.linear, torch.conv2d, F.relu, etc.      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NumPy-based Execution                        │
│   Converts torch tensors ↔ numpy arrays                         │
│   Executes operations using NumPy implementations               │
│   Returns results as torch tensors in tuple format              │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created/Modified

| File | Purpose |
|------|---------|
| `kpu/torch_backend.py` | **NEW**: Backend registration, KPUBackend class |
| `kpu/fx_converter.py` | **NEW**: FX graph to KPU converter (~1055 lines) |
| `kpu/__init__.py` | Added torch_backend imports with torch detection |
| `examples/torch_compile_demo.py` | **NEW**: Demo with MLP, CNN, function compilation |

## Key Challenges and Fixes

### Challenge 1: Dynamo Graph Structure

**Discovery:** Dynamo-captured graphs differ significantly from `fx.symbolic_trace` graphs:

```python
# Dynamo puts ALL parameters as placeholders
def forward(self,
    L_self_modules_fc1_parameters_weight_: "f32[128, 784]",  # param placeholder
    L_self_modules_fc1_parameters_bias_: "f32[128]",         # param placeholder
    L_x_: "f32[2, 784]",                                      # input placeholder
    ...):
```

**Initial Wrong Assumption:** I assumed only actual inputs would be placeholders.

**Fix:** Record all placeholders in order. At runtime, Dynamo passes ALL values (params + inputs) as args.

### Challenge 2: Output Format

**Problem:** MLP output shape was `[10]` instead of `[32, 10]`.

**Root Cause:** Our `_build_executable` returned `outputs[0]` for single outputs, but Dynamo expects a tuple.

**Debug Session:**
```python
# Inside executable: shape is correct
outputs[0].shape  # torch.Size([2, 10])

# After return: shape is wrong
result.shape      # torch.Size([10]) - WHY?
```

**Discovery:** `gm.forward` returns `tuple` not tensor:
```python
gm.forward result type: <class 'tuple'>
Tuple length: 1
  [0]: shape=torch.Size([2, 10])
```

**Fix:**
```python
# Before (wrong)
if len(outputs) == 1:
    return outputs[0]
return tuple(outputs)

# After (correct)
return tuple(outputs)  # Always return tuple
```

### Challenge 3: conv2d Parameters as Args

**Problem:** CNN output shape was `(2, 16, 26, 26)` instead of `(2, 16, 28, 28)`.

**Root Cause:** Our `_emit_conv2d` only read stride/padding from kwargs, but Dynamo passes them as positional args:

```python
# FX graph shows:
torch.conv2d(l_x_, weight, bias, (1, 1), (1, 1), (1, 1), 1)
#            input  weight bias  stride padding dilation groups
```

**Fix:**
```python
# Get stride and padding from args or kwargs
if len(node.args) > 3:
    stride = node.args[3]
else:
    stride = kwargs.get('stride', (1, 1))

if len(node.args) > 4:
    padding = node.args[4]
else:
    padding = kwargs.get('padding', (0, 0))
```

## Test Results

**torch.compile Demo:**
```
============================================================
Demo 1: Simple MLP with torch.compile
============================================================
Input shape: torch.Size([32, 784])
KPU output shape: torch.Size([32, 10])
Max difference from PyTorch: 1.34e-07
VALIDATION PASSED

============================================================
Demo 2: CNN with torch.compile
============================================================
Input shape: torch.Size([4, 1, 28, 28])
KPU output shape: torch.Size([4, 10])
Max difference from PyTorch: 1.80e-07
VALIDATION PASSED

============================================================
Demo 4: Compile a Function
============================================================
KPU output shape: torch.Size([16, 10])
Max difference from PyTorch: 7.63e-06
VALIDATION PASSED

============================================================
SUMMARY
============================================================
  Simple MLP: PASS
  CNN: PASS
  Function: PASS
============================================================
All demos passed!
```

**Existing Unit Tests:** All 38 tests pass (no regressions)

## Supported Operations

| Category | Operations |
|----------|------------|
| **Activations** | relu, gelu, silu, sigmoid, tanh, softmax |
| **Convolutions** | conv2d (nn.Conv2d or F.conv2d) |
| **Pooling** | max_pool2d, avg_pool2d, adaptive_avg_pool2d |
| **Normalization** | batch_norm, layer_norm |
| **Linear** | linear (F.linear or torch._C._nn.linear), matmul, mm, bmm |
| **Elementwise** | add, sub, mul, div |
| **Shape** | reshape, view, flatten, transpose, permute |
| **Reductions** | mean, sum |
| **Other** | concat, getitem |

## Usage Example

```python
import torch
import torch.nn as nn

# Define any PyTorch model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleMLP().eval()

# Compile with KPU backend
compiled_model = torch.compile(model, backend="kpu")

# Execute - Dynamo captures graph, KPU backend runs on simulator
x = torch.randn(32, 784)
with torch.no_grad():
    output = compiled_model(x)
print(output.shape)  # torch.Size([32, 10])
```

## Next Steps

1. **Transactional Mode**: Connect to C++ kpu-sim for timing statistics
2. **More Operations**: Attention, grouped conv, depthwise separable conv
3. **Graph Inspection API**: Expose the captured FX graph for analysis
4. **Offline Export**: ONNX or flatbuffers for deployment (future)

## Related Documents

- `docs/09-virtual-platform/unified-dnn-roadmap.md` - Updated with torch.compile section
- `python/kpu/torch_backend.py` - Backend implementation
- `python/kpu/fx_converter.py` - FX to KPU conversion logic
- `python/examples/torch_compile_demo.py` - Working demo

---

*Session completed successfully. All demos pass, all existing tests pass.*

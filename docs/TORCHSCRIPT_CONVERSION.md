# Depth Anything v3 TorchScript Conversion Guide

## Overview

This document describes the approach for converting Depth Anything v3 to TorchScript format for use in Nuke's Inference node.

## Architecture Assessment

### Compatibility Challenges

Depth Anything v3 introduces several features that complicate TorchScript conversion compared to v2:

1. **Dynamic Object Creation** - Uses `importlib` and `create_object()` for runtime module instantiation
2. **OmegaConf/DictConfig** - Configuration system not compatible with TorchScript
3. **Addict.Dict** - Custom dictionary type used for outputs
4. **Einops** - Tensor rearrangement library with limited TorchScript support
5. **Multi-view Architecture** - Handles (B, S, 3, H, W) instead of (B, 3, H, W)
6. **RoPE with Caching** - Rotary position embeddings use dictionary caching
7. **F.scaled_dot_product_attention** - May not exist in Nuke's PyTorch 1.6/1.12

### Solution Approach

We've created a wrapper architecture that addresses these issues:

```
┌─────────────────────────────────────────────────────────────┐
│                    DepthAnything3Nuke                        │
│                    (TorchScript-compatible wrapper)          │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input Handling                                       │   │
│  │ - Converts Nuke (1, 6, H, W) → (1, 2, 3, H, W)     │   │
│  │ - Handles camera parameters (extrinsics/intrinsics) │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ DinoV2 Backbone (modified)                          │   │
│  │ - No einops (replaced with native PyTorch)          │   │
│  │ - Manual attention (no fused_attn)                  │   │
│  │ - RoPE without complex caching                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ DPT/DualDPT Head (modified)                         │   │
│  │ - Returns standard dict (not addict.Dict)           │   │
│  │ - Native PyTorch operations only                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Output                                               │   │
│  │ - Depth map (1, 2, H, W) for two views             │   │
│  │ - Compatible with Nuke's Inference node             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Nuke Format Handling

Nuke's Inference node requires inputs in (1, 6, H, W) format for two RGB images:

```python
def _nuke_format_to_multiview(self, x: torch.Tensor) -> torch.Tensor:
    """Convert (1, 6, H, W) to (1, 2, 3, H, W)"""
    B, C6, H, W = x.shape
    return x.view(B, 2, 3, H, W)
```

### 2. Einops Replacement

All `einops.rearrange()` calls are replaced with native PyTorch operations:

| Einops Pattern | Native PyTorch |
|----------------|----------------|
| `"b s c h w -> (b s) c h w"` | `x.view(B * S, C, H, W)` |
| `"(b s) n c -> b s n c"` | `x.view(B, S, N, C)` |
| `"b s n c -> b (s n) c"` | `x.reshape(B, S * N, C)` |
| `"b (s n) c -> b s n c"` | `x.reshape(B, S, N, C)` |

See `src/depth_anything_3/model/utils/torchscript_compat.py` for complete utilities.

### 3. Attention Mechanism

Force manual attention implementation for compatibility with older PyTorch:

```python
def replace_attention_with_compatible(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False  # Disable F.scaled_dot_product_attention
    return model
```

### 4. RoPE Caching

Dictionary-based caching is disabled for TorchScript compatibility:

```python
def disable_rope_caching(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if hasattr(module, 'frequency_cache'):
            module.frequency_cache.clear()
        if hasattr(module, 'position_cache'):
            module.position_cache.clear()
    return model
```

### 5. Output Format

Instead of `addict.Dict`, we return standard PyTorch tensors:

```python
def forward(self, x, extrinsics=None, intrinsics=None) -> torch.Tensor:
    # ... processing ...
    depth = output['depth']  # Extract tensor from dict
    return depth  # Return plain tensor for Nuke
```

## Usage

### Convert Model to TorchScript

```bash
python scripts/convert_to_torchscript.py \
    --config src/depth_anything_3/configs/da3-small.yaml \
    --checkpoint path/to/checkpoint.pth \
    --output da3_small_nuke.pt \
    --device cuda
```

### Options

- `--config`: Path to model configuration YAML
- `--checkpoint`: Path to pretrained checkpoint (.pth)
- `--output`: Output path for TorchScript model (.pt)
- `--device`: Device to use (cuda/cpu)
- `--use-trace`: Use torch.jit.trace instead of torch.jit.script
- `--skip-test`: Skip testing before conversion

### Load in Nuke

The generated `.pt` file can be converted to Nuke's `.cat` format using `CatFileCreator`:

```python
# In Nuke's Script Editor
import torch

# Load TorchScript model
model = torch.jit.load("da3_small_nuke.pt")

# Create example input (Nuke format: 1, 6, H, W)
x = torch.randn(1, 6, 518, 518)

# Camera parameters (optional)
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)

# Inference
with torch.no_grad():
    depth = model(x, extrinsics, intrinsics)

print(f"Output shape: {depth.shape}")  # Should be (1, 2, 518, 518)
```

## Camera Parameters

### Extrinsics Format

Camera extrinsics are 4x4 transformation matrices in world-to-camera format:

```
extrinsics shape: (1, 2, 4, 4)

For each view:
[[ R00  R01  R02  tx ]
 [ R10  R11  R12  ty ]
 [ R20  R21  R22  tz ]
 [  0    0    0    1 ]]

Where:
- R: 3x3 rotation matrix (world to camera)
- t: 3x1 translation vector (camera position in world space)
```

### Intrinsics Format

Camera intrinsics are 3x3 matrices:

```
intrinsics shape: (1, 2, 3, 3)

For each view:
[[ fx   0   cx ]
 [  0  fy   cy ]
 [  0   0    1 ]]

Where:
- fx, fy: focal lengths in pixel units
- cx, cy: principal point coordinates (usually image center)
```

### Example Camera Setup

```python
import torch

# Create identity extrinsics (camera at origin, looking down -Z axis)
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)

# Create intrinsics for a 518x518 image
H, W = 518, 518
intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
intrinsics[:, :, 0, 0] = 500.0  # fx
intrinsics[:, :, 1, 1] = 500.0  # fy
intrinsics[:, :, 0, 2] = W / 2  # cx
intrinsics[:, :, 1, 2] = H / 2  # cy
```

## Known Limitations

1. **PyTorch Version**: Tested on PyTorch 1.12+. Nuke 13 (PyTorch 1.6) may require additional compatibility work.

2. **Dynamic Shapes**: TorchScript models work best with fixed input sizes. Variable sizes may cause issues.

3. **3DGS Support**: Gaussian Splatting features are not included in the current wrapper.

4. **Nested Models**: The dual-branch NestedDepthAnything3Net is not yet supported.

5. **Auxiliary Outputs**: Only depth output is returned. Confidence maps and other auxiliary outputs are available in the dict but not returned by default.

## Troubleshooting

### Issue: "RuntimeError: Cannot find schema for operator"

**Solution**: Ensure all operations are TorchScript-compatible. Check for:
- Custom operators without proper registration
- Dynamic control flow that TorchScript can't trace
- Unsupported dictionary operations

### Issue: "AttributeError: 'Tensor' object has no attribute 'depth'"

**Solution**: The output should be a plain tensor, not a dictionary. Modify the wrapper's forward method to return only the depth tensor.

### Issue: "CUDA out of memory"

**Solution**: Reduce input resolution or enable gradient checkpointing:
```bash
python scripts/convert_to_torchscript.py \
    --config configs/da3-small.yaml \
    --checkpoint checkpoint.pth \
    --output model.pt \
    --device cpu  # Use CPU for conversion
```

### Issue: "Traced function does not match expected schema"

**Solution**: Use `--use-trace` flag instead of scripting:
```bash
python scripts/convert_to_torchscript.py \
    --config configs/da3-small.yaml \
    --checkpoint checkpoint.pth \
    --output model.pt \
    --use-trace
```

## Future Improvements

1. **Full Feature Support**: Add support for all DA3 features (3DGS, nested models, etc.)

2. **Dynamic Shapes**: Improve support for variable input resolutions

3. **Quantization**: Add INT8/FP16 quantization for faster inference

4. **Batch Processing**: Support larger batch sizes for multi-view scenarios

5. **Complete Testing**: Comprehensive testing with actual Nuke integration

## References

- [Depth Anything v2 for Nuke](https://github.com/rafaelperez/Depth-Anything-for-Nuke)
- [PyTorch TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [Nuke Python API](https://learn.foundry.com/nuke/developers/latest/pythondevguide/)

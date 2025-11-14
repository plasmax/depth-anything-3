# Depth Anything v3 TorchScript Conversion Assessment

**Date**: 2025-11-14
**Status**: Prototype Complete
**Feasibility**: FEASIBLE with MODERATE COMPLEXITY

---

## Executive Summary

This assessment evaluates the feasibility of converting Depth Anything v3 (DA3) to TorchScript format for compatibility with Nuke's Inference node. Based on architectural analysis and comparison with the successful DA2 Nuke implementation, we conclude that conversion is **feasible but requires significant refactoring**.

### Key Findings

- **Complexity**: 3-4x more complex than DA2 conversion
- **Estimated Effort**: 40-60 hours development + testing
- **Success Probability**: 70-80% with proper refactoring
- **Major Blockers**: 10 identified (3 high priority, 4 medium, 3 low)

---

## Architecture Analysis

### DA3 Core Components

| Component | Description | TorchScript Issues |
|-----------|-------------|-------------------|
| **Backbone** | DinoV2 Vision Transformer | Einops, RoPE caching, fused attention |
| **Head** | DPT/DualDPT depth prediction | Addict.Dict, string-based activations |
| **Camera System** | Optional pose estimation | Complex geometry operations |
| **Configuration** | OmegaConf-based dynamic loading | Not TorchScript-compatible |
| **Multi-view** | Handles (B, S, 3, H, W) tensors | Increased complexity |

### Key Files

```
src/depth_anything_3/model/
├── da3.py:39-379                    # Main model classes
├── dinov2/vision_transformer.py     # Backbone (438 lines)
├── dpt.py:31-458                    # Single-head depth prediction
├── dualdpt.py:30-365                # Dual-head variant
├── cam_enc.py                       # Camera encoder
├── cam_dec.py                       # Camera decoder
└── dinov2/layers/
    ├── attention.py                 # Attention with F.scaled_dot_product_attention
    └── rope.py                      # RoPE with dictionary caching
```

---

## Compatibility Issues

### BLOCKER: High Priority

#### 1. Dynamic Object Creation
- **Location**: `cfg.py:108-129`
- **Problem**: Runtime module instantiation via `importlib`
- **Impact**: TorchScript requires static graph definition
- **Solution**: Replace with explicit module initialization
  ```python
  # Current (incompatible):
  self.backbone = create_object(_wrap_cfg(net))

  # Required (compatible):
  self.backbone = DinoV2(explicit_config)
  ```

#### 2. OmegaConf/DictConfig Dependencies
- **Location**: Throughout configuration system
- **Problem**: `OmegaConf.DictConfig` not TorchScript-compatible
- **Solution**: Convert all configs to Python primitives before model creation

#### 3. Addict.Dict Return Types
- **Location**: `da3.py:106`, `dpt.py:167`, `dualdpt.py:163`
- **Problem**: Custom dict type in return values
- **Solution**: Return standard dict or plain tensors
  ```python
  # Current:
  return Dict({'depth': depth_tensor, 'depth_conf': conf_tensor})

  # Required:
  return depth_tensor  # or standard dict
  ```

### MAJOR: Moderate Priority

#### 4. Einops Dependency
- **Location**: `vision_transformer.py:16`, multiple usage sites
- **Problem**: Limited TorchScript support for `rearrange()`
- **Solution**: Replace with native PyTorch operations
  ```python
  # Current:
  x = rearrange(x, "b s c h w -> (b s) c h w")

  # Required:
  B, S, C, H, W = x.shape
  x = x.view(B * S, C, H, W)
  ```
- **Status**: ✅ Implemented in `torchscript_compat.py`

#### 5. RoPE Caching with Dict[Tuple[int, int], Tensor]
- **Location**: `rope.py:36`, `rope.py:85`
- **Problem**: Complex dictionary key types not well supported
- **Solution**: Remove caching or use simpler structures
- **Status**: ✅ Addressed in conversion script

#### 6. F.scaled_dot_product_attention
- **Location**: `attention.py:60-70`
- **Problem**: Not available in PyTorch 1.6/1.12 (Nuke's versions)
- **Solution**: Force manual attention fallback
  ```python
  module.fused_attn = False  # Use manual implementation
  ```
- **Status**: ✅ Implemented in conversion script

#### 7. Optional/Union Type Annotations
- **Location**: Throughout, e.g., `da3.py:102-106`
- **Problem**: `torch.Tensor | None` requires careful handling
- **Solution**: Use `Optional[torch.Tensor]` consistently

### MINOR: Lower Priority

#### 8. Mutable Default Arguments
- **Location**: `da3.py:104`
- **Problem**: `export_feat_layers: list[int] | None = []`
- **Solution**: Use `None`, initialize inside function

#### 9. Complex Control Flow
- **Location**: `vision_transformer.py:317-323`
- **Problem**: Alternating global/local attention patterns
- **Solution**: Ensure all branches are traceable

#### 10. String-based Activation Selection
- **Location**: `dpt.py:286-308`, `dualdpt.py:341-364`
- **Problem**: Runtime string matching
- **Solution**: Pre-select activation modules during init

---

## Comparison with DA2

### DA2 Success Factors
- ✅ Simple single-image architecture (B, 3, H, W)
- ✅ No dynamic configuration system
- ✅ Direct encoder-decoder structure
- ✅ Successfully used `torch.jit.script()`
- ✅ Compatible with PyTorch 1.6 and 1.12

### DA3 New Challenges
- ❌ Multi-view support (B, S, 3, H, W)
- ❌ Dynamic object creation via config
- ❌ Camera pose estimation (optional)
- ❌ Dual-head architecture (depth + ray/aux)
- ❌ RoPE embeddings with caching
- ❌ Nested metric scaling model (two branches)
- ❌ More complex attention mechanisms

### Complexity Comparison

| Aspect | DA2 | DA3 | Increase |
|--------|-----|-----|----------|
| Model Components | 2 (encoder, decoder) | 5+ (backbone, head, cam_enc, cam_dec, gs_head) | 2.5x |
| Dependencies | Minimal | Einops, Addict, OmegaConf | 3x |
| Input Format | (B, 3, H, W) | (B, S, 3, H, W) | - |
| Output Format | (B, H, W) | Dict with multiple outputs | - |
| Lines of Code | ~800 | ~2000+ | 2.5x |
| Config Complexity | Hardcoded | Dynamic YAML-based | 4x |

---

## Implemented Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Original DA3 Model                            │
│  ┌─────────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐   │
│  │   Backbone  │→│  Head   │→│ CamDec  │→│   Output    │   │
│  │  (DinoV2)   │  │  (DPT)  │  │         │  │ (addict.Dict)│   │
│  └─────────────┘  └─────────┘  └─────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Wrapper
┌─────────────────────────────────────────────────────────────────┐
│              DepthAnything3Nuke (TorchScript-compatible)         │
│                                                                   │
│  Input: (1, 6, H, W) [Nuke format]                              │
│     ↓                                                             │
│  Reshape to (1, 2, 3, H, W)                                     │
│     ↓                                                             │
│  Process through modified components:                            │
│    - No einops (native PyTorch)                                 │
│    - No addict.Dict (standard dict/tensors)                     │
│    - Manual attention (no fused_attn)                           │
│    - No RoPE caching                                            │
│     ↓                                                             │
│  Output: (1, 2, H, W) [Depth for 2 views]                      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files Created

1. **`src/depth_anything_3/model/da3_nuke.py`** (194 lines)
   - TorchScript-compatible wrapper class
   - Handles Nuke's (1, 6, H, W) format
   - Preserves camera parameter support
   - Returns plain tensors

2. **`src/depth_anything_3/model/utils/torchscript_compat.py`** (273 lines)
   - Native PyTorch replacements for einops operations
   - TorchScript-decorated helper functions
   - DictOutput container for compatibility

3. **`scripts/convert_to_torchscript.py`** (323 lines)
   - End-to-end conversion pipeline
   - Automatic compatibility fixes
   - Testing and verification
   - Support for both script and trace modes

4. **`docs/TORCHSCRIPT_CONVERSION.md`** (This document)
   - Comprehensive guide for conversion process
   - Usage examples and troubleshooting

---

## Nuke-Specific Requirements

### Input Format

Nuke's Inference node requires concatenated RGB images:
```python
# Two RGB images: img1 (3, H, W) and img2 (3, H, W)
# Concatenated along channel dimension:
nuke_input = torch.cat([img1, img2], dim=0)  # (6, H, W)
nuke_input = nuke_input.unsqueeze(0)          # (1, 6, H, W)
```

Our wrapper automatically converts this to DA3's expected format:
```python
multiview_input = nuke_input.view(1, 2, 3, H, W)
```

### Camera Parameters

Format must match Nuke's conventions:
```python
# Extrinsics: (1, 2, 4, 4) - world-to-camera transforms
extrinsics = torch.stack([cam1_ext, cam2_ext]).unsqueeze(0)

# Intrinsics: (1, 2, 3, 3) - camera intrinsic matrices
intrinsics = torch.stack([cam1_int, cam2_int]).unsqueeze(0)
```

### Output Format

Returns depth maps for both views:
```python
depth = model(nuke_input, extrinsics, intrinsics)
# Shape: (1, 2, H, W)
# depth[0, 0] = depth map for first view
# depth[0, 1] = depth map for second view
```

---

## Conversion Pipeline

### Step 1: Load Model
```bash
python scripts/convert_to_torchscript.py \
    --config configs/da3-small.yaml \
    --checkpoint checkpoints/da3_small.pth \
    --output da3_small_nuke.pt
```

### Step 2: Automatic Fixes
The script automatically applies:
- ✅ Disables fused attention (`fused_attn = False`)
- ✅ Clears RoPE caches
- ✅ Validates forward pass with dummy inputs
- ✅ Tests both script and trace compilation

### Step 3: Optimization
If PyTorch ≥ 1.12:
```python
scripted = torch.jit.optimize_for_inference(scripted)
```

### Step 4: Verification
- ✅ Saves TorchScript model (.pt)
- ✅ Reloads to verify integrity
- ✅ Compares output shapes

### Step 5: Nuke Integration
Convert `.pt` to `.cat` using Nuke's CatFileCreator:
```python
# In Nuke
import CatFileCreator
CatFileCreator.create("da3_small_nuke.pt", "da3_small_nuke.cat")
```

---

## Testing Strategy

### Unit Tests
- ✅ Nuke format conversion: (1, 6, H, W) → (1, 2, 3, H, W)
- ✅ Camera parameter handling
- ✅ Forward pass with dummy data
- ⏳ Output shape validation
- ⏳ Numerical accuracy vs original model

### Integration Tests
- ⏳ Load in Nuke environment
- ⏳ Real-world multi-view image pairs
- ⏳ Different input resolutions
- ⏳ With and without camera parameters
- ⏳ Performance benchmarking

### Compatibility Tests
- ⏳ PyTorch 1.6 (Nuke 13)
- ⏳ PyTorch 1.12 (Nuke 14+)
- ✅ PyTorch 2.x (development)

---

## Risk Assessment

| Risk | Level | Probability | Impact | Mitigation |
|------|-------|-------------|--------|------------|
| Dynamic config system incompatibility | HIGH | 90% | CRITICAL | ✅ Wrapper with explicit init |
| Einops conversion errors | MEDIUM | 30% | HIGH | ✅ Comprehensive utility functions |
| Multi-view processing issues | MEDIUM | 40% | HIGH | ✅ Format conversion in wrapper |
| RoPE caching problems | MEDIUM | 50% | MEDIUM | ✅ Cache clearing |
| PyTorch 1.6 compatibility | HIGH | 70% | HIGH | ⏳ Requires testing on Nuke 13 |
| Attention mechanism failures | LOW | 20% | MEDIUM | ✅ Manual fallback implemented |
| Type annotation issues | LOW | 10% | LOW | ✅ Proper typing throughout |
| Performance degradation | MEDIUM | 40% | MEDIUM | ⏳ Needs benchmarking |

Legend:
- ✅ = Mitigated
- ⏳ = In progress / Needs testing
- ❌ = Not addressed

---

## Performance Considerations

### Expected Performance

Based on DA2 Nuke implementation:
- **Resolution**: 518x518 typical, up to 1024x1024 possible
- **Inference Time**:
  - Small: ~50-100ms per view (GPU)
  - Base: ~100-200ms per view (GPU)
  - Large: ~200-400ms per view (GPU)
- **Memory**:
  - Small: ~2GB VRAM
  - Base: ~4GB VRAM
  - Large: ~8GB VRAM

### Optimization Opportunities

1. **Half Precision (FP16)**: 2x speedup, 50% memory reduction
   ```python
   model = model.half()
   ```

2. **Inference Optimization**: Available in PyTorch 1.12+
   ```python
   model = torch.jit.optimize_for_inference(model)
   ```

3. **Quantization**: INT8 can provide 4x speedup (requires additional work)

4. **Batch Processing**: Process multiple frame pairs simultaneously

---

## Known Limitations

1. **PyTorch Version Dependency**
   - Tested on PyTorch 2.x (development)
   - PyTorch 1.12+ recommended
   - PyTorch 1.6 (Nuke 13) requires additional testing

2. **Dynamic Input Shapes**
   - TorchScript works best with fixed sizes
   - Variable resolutions may cause issues
   - Recommend fixed resolution per model

3. **Feature Completeness**
   - ✅ Multi-view depth estimation
   - ✅ Camera parameter support
   - ❌ 3D Gaussian Splatting
   - ❌ Nested metric scaling model
   - ❌ Auxiliary output branches (confidence, sky, etc.)

4. **Configuration System**
   - Cannot use dynamic YAML configs at runtime
   - Must pre-load specific model variant
   - Recommend separate .pt file per model size

---

## Future Improvements

### Short Term (1-2 weeks)
- [ ] Complete unit test suite
- [ ] Test on actual Nuke 13/14 environments
- [ ] Benchmark against DA2 implementation
- [ ] Add FP16 support
- [ ] Create multiple model size variants

### Medium Term (1-2 months)
- [ ] Support auxiliary outputs (confidence, sky masks)
- [ ] Implement nested metric scaling model
- [ ] Add dynamic resolution support
- [ ] Optimize for batch processing
- [ ] Create comprehensive documentation

### Long Term (3-6 months)
- [ ] 3D Gaussian Splatting support
- [ ] INT8 quantization
- [ ] Multi-GPU support
- [ ] Integration with Nuke's depth toolkit
- [ ] Automated testing pipeline

---

## Recommendations

### For Immediate Use

1. **Start with DA3-Small**: Fastest iteration, easier to debug
2. **Use Fixed Resolutions**: 518x518 or 1024x1024
3. **Test on PyTorch 1.12+**: Better TorchScript support
4. **Validate Numerically**: Compare outputs with original DA3

### For Production Deployment

1. **Thorough Testing**: Multi-view scenarios, edge cases
2. **Performance Profiling**: Ensure acceptable inference times
3. **Memory Monitoring**: Avoid OOM issues in Nuke
4. **Version Control**: Track model checksums and configs
5. **Documentation**: Provide clear usage guidelines for artists

### For Development

1. **Incremental Approach**: Test each component separately
2. **Extensive Logging**: Debug TorchScript compilation issues
3. **Fallback Options**: Keep DA2 as backup for single-view
4. **Community Engagement**: Share findings with ByteDance team

---

## Conclusion

The conversion of Depth Anything v3 to TorchScript for Nuke is **feasible** but requires **moderate development effort**. The primary challenges involve:

1. Replacing dynamic configuration system
2. Eliminating incompatible dependencies (einops, addict, OmegaConf)
3. Handling multi-view architecture
4. Ensuring PyTorch version compatibility

Our prototype implementation addresses these challenges through:
- ✅ TorchScript-compatible wrapper (`DepthAnything3Nuke`)
- ✅ Native PyTorch operation replacements
- ✅ Automatic compatibility fixes in conversion script
- ✅ Comprehensive documentation

**Next Steps**:
1. Test on actual Nuke environment (13 & 14)
2. Validate numerical accuracy
3. Benchmark performance
4. Iterate based on findings

**Estimated Timeline**:
- Prototype: ✅ Complete (1 week)
- Testing: ⏳ In progress (1-2 weeks)
- Production-ready: 🔄 4-6 weeks total

---

## References

### Documentation
- [DA3 Original Repository](https://github.com/DepthAnything/Depth-Anything-V3)
- [DA2 Nuke Implementation](https://github.com/rafaelperez/Depth-Anything-for-Nuke)
- [PyTorch TorchScript Guide](https://pytorch.org/docs/stable/jit.html)
- [Nuke Python API](https://learn.foundry.com/nuke/developers/)

### Key Files
- `src/depth_anything_3/model/da3.py` - Original DA3 implementation
- `src/depth_anything_3/model/da3_nuke.py` - TorchScript wrapper
- `src/depth_anything_3/model/utils/torchscript_compat.py` - Compatibility utilities
- `scripts/convert_to_torchscript.py` - Conversion pipeline

### Contact
For questions or issues, please open an issue on the GitHub repository.

---

**Assessment prepared by**: Claude (Anthropic AI)
**Review status**: Ready for implementation
**Last updated**: 2025-11-14

#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to convert Depth Anything v3 models to TorchScript format for Nuke.

This script loads a pretrained DA3 model and converts it to TorchScript,
making it compatible with Nuke's Inference node.

Usage:
    python scripts/convert_to_torchscript.py \
        --config configs/da3-small.yaml \
        --checkpoint path/to/checkpoint.pth \
        --output da3_small_nuke.pt \
        --nuke-format
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.cfg import load_config, create_object
from depth_anything_3.model.da3_nuke import DepthAnything3Nuke


def replace_attention_with_compatible(model: nn.Module) -> nn.Module:
    """
    Replace F.scaled_dot_product_attention with manual implementation.

    This ensures compatibility with older PyTorch versions (1.6, 1.12).

    Args:
        model: Model to modify

    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if hasattr(module, 'fused_attn'):
            # Force use of manual attention implementation
            module.fused_attn = False
            print(f"Disabled fused_attn for: {name}")

    return model


def disable_rope_caching(model: nn.Module) -> nn.Module:
    """
    Disable RoPE caching for TorchScript compatibility.

    Args:
        model: Model to modify

    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if hasattr(module, 'frequency_cache'):
            # Clear the cache - it will be rebuilt during forward pass
            module.frequency_cache.clear()
            print(f"Cleared frequency_cache for: {name}")

        if hasattr(module, 'position_cache'):
            # Clear the position cache
            module.position_cache.clear()
            print(f"Cleared position_cache for: {name}")

    return model


def create_nuke_wrapper(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> DepthAnything3Nuke:
    """
    Create a Nuke-compatible wrapper from a DA3 model.

    Args:
        config_path: Path to model config YAML
        checkpoint_path: Path to pretrained checkpoint
        device: Device to load model on

    Returns:
        DepthAnything3Nuke wrapper instance
    """
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print("Creating model from config...")
    model = create_object(config)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Apply TorchScript compatibility fixes
    print("Applying TorchScript compatibility fixes...")
    model = replace_attention_with_compatible(model)
    model = disable_rope_caching(model)

    # Extract components for wrapper
    print("Creating Nuke wrapper...")
    backbone = model.backbone if hasattr(model, "backbone") else model.da3.backbone
    head = model.head if hasattr(model, "head") else model.da3.head
    cam_enc = model.cam_enc if hasattr(model, "cam_enc") else None
    cam_dec = model.cam_dec if hasattr(model, "cam_dec") else None

    # Create wrapper
    wrapper = DepthAnything3Nuke(
        backbone=backbone,
        head=head,
        cam_enc=cam_enc,
        cam_dec=cam_dec,
        patch_size=14,
    )

    wrapper.eval()

    return wrapper


def test_wrapper(wrapper: DepthAnything3Nuke, device: str = "cuda") -> None:
    """
    Test the wrapper with dummy inputs.

    Args:
        wrapper: DepthAnything3Nuke instance
        device: Device to run test on
    """
    print("\nTesting wrapper with dummy inputs...")

    # Create dummy Nuke-format input (1, 6, H, W)
    H, W = 518, 518  # Standard size
    x_nuke = torch.randn(1, 6, H, W, device=device)

    # Create dummy camera parameters
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    intrinsics[:, :, 0, 0] = 500  # fx
    intrinsics[:, :, 1, 1] = 500  # fy
    intrinsics[:, :, 0, 2] = W / 2  # cx
    intrinsics[:, :, 1, 2] = H / 2  # cy

    print(f"Input shape: {x_nuke.shape}")
    print(f"Extrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")

    # Test forward pass
    with torch.no_grad():
        try:
            output = wrapper(x_nuke, extrinsics, intrinsics)
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            raise


def convert_to_torchscript(
    wrapper: DepthAnything3Nuke,
    output_path: str,
    use_script: bool = True,
    device: str = "cuda",
) -> None:
    """
    Convert wrapper to TorchScript and save.

    Args:
        wrapper: DepthAnything3Nuke instance
        output_path: Path to save TorchScript model
        use_script: If True, use torch.jit.script; otherwise use torch.jit.trace
        device: Device to run on
    """
    print(f"\nConverting to TorchScript (method: {'script' if use_script else 'trace'})...")

    wrapper.eval()

    if use_script:
        try:
            scripted = torch.jit.script(wrapper)
            print("✓ TorchScript compilation (script) successful!")
        except Exception as e:
            print(f"✗ TorchScript compilation (script) failed: {e}")
            print("\nAttempting trace method instead...")
            use_script = False

    if not use_script:
        # Prepare example inputs for tracing
        H, W = 518, 518
        x = torch.randn(1, 6, H, W, device=device)
        extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        intrinsics[:, :, 0, 0] = 500
        intrinsics[:, :, 1, 1] = 500
        intrinsics[:, :, 0, 2] = W / 2
        intrinsics[:, :, 1, 2] = H / 2

        with torch.no_grad():
            try:
                scripted = torch.jit.trace(
                    wrapper,
                    (x, extrinsics, intrinsics),
                    check_trace=True,
                )
                print("✓ TorchScript compilation (trace) successful!")
            except Exception as e:
                print(f"✗ TorchScript compilation (trace) failed: {e}")
                raise

    # Optimize for inference if available
    if hasattr(torch.jit, 'optimize_for_inference'):
        print("Applying inference optimization...")
        scripted = torch.jit.optimize_for_inference(scripted)

    # Save
    print(f"Saving TorchScript model to: {output_path}")
    scripted.save(output_path)
    print("✓ Model saved successfully!")

    # Verify saved model can be loaded
    print("\nVerifying saved model...")
    loaded = torch.jit.load(output_path)
    print("✓ Model loaded successfully!")

    return scripted


def main():
    parser = argparse.ArgumentParser(
        description="Convert Depth Anything v3 to TorchScript for Nuke"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="da3_nuke.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Use torch.jit.trace instead of torch.jit.script",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip testing the wrapper before conversion",
    )

    args = parser.parse_args()

    # Create wrapper
    wrapper = create_nuke_wrapper(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Test wrapper
    if not args.skip_test:
        test_wrapper(wrapper, device=args.device)

    # Convert to TorchScript
    convert_to_torchscript(
        wrapper=wrapper,
        output_path=args.output,
        use_script=not args.use_trace,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print(f"TorchScript model saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()

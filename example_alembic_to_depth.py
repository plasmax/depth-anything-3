#!/usr/bin/env python3
"""
Complete example: Load Alembic camera data and run DepthAnything3 inference

This script demonstrates the full pipeline:
1. Load camera data from an Alembic file
2. Load corresponding images
3. Run DepthAnything3 inference with camera data
4. Export results
"""

import numpy as np
from pathlib import Path
from alembic_camera_converter import read_alembic_cameras
from src.depth_anything_3.api import DepthAnything3


def main():
    """Example pipeline using Alembic camera data with DepthAnything3."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run DepthAnything3 inference using Alembic camera data'
    )
    parser.add_argument('alembic_file', help='Path to Alembic (.abc) file with camera data')
    parser.add_argument('image_dir', help='Directory containing input images')
    parser.add_argument('--model', default='da3-large',
                        help='Model name (default: da3-large)')
    parser.add_argument('--sample', type=int, default=0,
                        help='Alembic frame/sample index (default: 0)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Image width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Image height in pixels (default: 1080)')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory for results (default: output)')
    parser.add_argument('--export-format', default='glb',
                        help='Export format: glb, ply, npz, etc. (default: glb)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    print("="*70)
    print("DEPTHANYTHING3 with ALEMBIC CAMERA DATA")
    print("="*70)

    # Step 1: Load camera data from Alembic
    print(f"\n[1/4] Loading camera data from: {args.alembic_file}")
    extrinsics, intrinsics = read_alembic_cameras(
        args.alembic_file,
        sample_index=args.sample,
        image_width=args.width,
        image_height=args.height
    )
    print(f"  ✓ Loaded {len(extrinsics)} cameras")
    print(f"    - Extrinsics shape: {extrinsics.shape}")
    print(f"    - Intrinsics shape: {intrinsics.shape}")

    # Step 2: Load images
    print(f"\n[2/4] Loading images from: {args.image_dir}")
    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob('*.png')) + \
                  sorted(image_dir.glob('*.jpg')) + \
                  sorted(image_dir.glob('*.jpeg'))

    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    # Convert to strings for DepthAnything3
    image_list = [str(p) for p in image_paths[:len(extrinsics)]]
    print(f"  ✓ Found {len(image_list)} images")

    if len(image_list) != len(extrinsics):
        print(f"  ⚠ Warning: Number of images ({len(image_list)}) != "
              f"number of cameras ({len(extrinsics)})")

    # Step 3: Load DepthAnything3 model
    print(f"\n[3/4] Loading DepthAnything3 model: {args.model}")
    model = DepthAnything3(model_name=args.model)
    model = model.to(args.device)
    print(f"  ✓ Model loaded on {args.device}")

    # Step 4: Run inference
    print(f"\n[4/4] Running inference...")
    print(f"  Export format: {args.export_format}")
    print(f"  Output directory: {args.output_dir}")

    prediction = model.inference(
        image=image_list,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        export_dir=args.output_dir,
        export_format=args.export_format,
        show_cameras=True,  # Show camera wireframes in GLB export
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Depth maps shape: {prediction.depth.shape}")
    print(f"  Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"  Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"\n  ✓ Results exported to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Round-trip example: Export DepthAnything3 prediction cameras back to Alembic

This script demonstrates:
1. Running DepthAnything3 inference (which produces extrinsics/intrinsics)
2. Exporting the predicted camera data to an Alembic file
3. Optionally: Re-importing and verifying the data

Use case: After running depth estimation, export the predicted camera poses
to visualize in 3D software like Blender, Maya, or Houdini.
"""

import numpy as np
from pathlib import Path
from export_to_alembic_cameras import export_cameras_to_alembic


def export_prediction_cameras(
    prediction,
    output_file,
    image_width=1920,
    image_height=1080,
    camera_names=None
):
    """
    Export camera data from a DepthAnything3 Prediction to Alembic.

    Args:
        prediction: DepthAnything3 Prediction object
        output_file: Path to output Alembic (.abc) file
        image_width: Image width in pixels
        image_height: Image height in pixels
        camera_names: Optional list of camera names
    """
    # Extract extrinsics and intrinsics from prediction
    extrinsics = prediction.extrinsics
    intrinsics = prediction.intrinsics

    # Convert to camera-to-world if needed (add bottom row)
    if extrinsics.shape[1:] == (3, 4):
        # Add [0, 0, 0, 1] row to make it 4x4
        num_cams = extrinsics.shape[0]
        extrinsics_4x4 = np.zeros((num_cams, 4, 4))
        extrinsics_4x4[:, :3, :] = extrinsics
        extrinsics_4x4[:, 3, 3] = 1.0
        extrinsics = extrinsics_4x4

    # Export to Alembic
    export_cameras_to_alembic(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_file=output_file,
        image_width=image_width,
        image_height=image_height,
        camera_names=camera_names
    )


def main():
    """Example: Export camera data from DepthAnything3 prediction."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Export DepthAnything3 prediction cameras to Alembic'
    )
    parser.add_argument('prediction_file', help='Path to .npz file with DepthAnything3 prediction')
    parser.add_argument('output_file', help='Path to output Alembic (.abc) file')
    parser.add_argument('--width', type=int, default=1920,
                        help='Image width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Image height in pixels (default: 1080)')

    args = parser.parse_args()

    print("="*70)
    print("EXPORT DEPTHANYTHING3 CAMERAS TO ALEMBIC")
    print("="*70)

    # Load prediction data
    print(f"\n[1/2] Loading prediction from: {args.prediction_file}")
    data = np.load(args.prediction_file)

    # Check what keys are available
    print(f"  Available keys: {list(data.keys())}")

    # Try to extract camera data
    if 'extrinsics' in data and 'intrinsics' in data:
        extrinsics = data['extrinsics']
        intrinsics = data['intrinsics']
        print(f"  ✓ Loaded extrinsics: {extrinsics.shape}")
        print(f"  ✓ Loaded intrinsics: {intrinsics.shape}")
    else:
        raise ValueError(
            f"Prediction file must contain 'extrinsics' and 'intrinsics'. "
            f"Found: {list(data.keys())}"
        )

    # Export to Alembic
    print(f"\n[2/2] Exporting to: {args.output_file}")
    export_cameras_to_alembic(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_file=args.output_file,
        image_width=args.width,
        image_height=args.height
    )

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n1. Import {args.output_file} into your 3D software:")
    print("   - Blender: File → Import → Alembic Cache (.abc)")
    print("   - Maya: File → Import → Alembic")
    print("   - Houdini: File → Import → Alembic Scene")
    print("\n2. The cameras will appear in your scene with correct:")
    print("   - Position and orientation (from extrinsics)")
    print("   - Focal length and sensor size (from intrinsics)")
    print("\n3. You can use these cameras to:")
    print("   - Visualize the camera setup")
    print("   - Render from predicted viewpoints")
    print("   - Align with 3D geometry")
    print("="*70)


if __name__ == '__main__':
    main()

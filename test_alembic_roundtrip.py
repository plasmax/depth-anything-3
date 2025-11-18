#!/usr/bin/env python3
"""
Test round-trip conversion: numpy → Alembic → numpy

This script verifies that camera data can be correctly exported to Alembic
and re-imported without losing information.
"""

import numpy as np
import tempfile
import os
from pathlib import Path

from export_to_alembic_cameras import export_cameras_to_alembic
from alembic_camera_converter import read_alembic_cameras


def create_synthetic_cameras(num_cameras=5, image_width=1920, image_height=1080):
    """
    Create synthetic camera extrinsics and intrinsics for testing.

    Args:
        num_cameras: Number of cameras to create
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        tuple: (extrinsics, intrinsics)
    """
    print(f"Creating {num_cameras} synthetic cameras...")

    extrinsics_list = []
    intrinsics_list = []

    for i in range(num_cameras):
        # Create a camera looking at origin from different positions
        # Position cameras in a circle around the origin
        angle = (2 * np.pi * i) / num_cameras
        radius = 5.0
        height = 2.0

        # Camera position
        cam_pos = np.array([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ])

        # Create rotation matrix (looking at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)  # Look at origin
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # Build rotation matrix (camera to world)
        rotation = np.column_stack([right, up, forward])

        # Build world-to-camera transform (extrinsics)
        # This is the inverse of the camera-to-world transform
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = rotation
        cam_to_world[:3, 3] = cam_pos

        extrinsics = np.linalg.inv(cam_to_world)
        extrinsics_list.append(extrinsics)

        # Create intrinsics (simple pinhole camera)
        focal_length_px = 1000.0  # pixels
        cx = image_width / 2.0
        cy = image_height / 2.0

        intrinsics = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ])
        intrinsics_list.append(intrinsics)

    extrinsics_array = np.stack(extrinsics_list, axis=0)
    intrinsics_array = np.stack(intrinsics_list, axis=0)

    print(f"  ✓ Created extrinsics: {extrinsics_array.shape}")
    print(f"  ✓ Created intrinsics: {intrinsics_array.shape}")

    return extrinsics_array, intrinsics_array


def compare_matrices(original, recovered, name, tolerance=1e-3):
    """
    Compare two matrices and report differences.

    Args:
        original: Original matrix
        recovered: Recovered matrix
        name: Name for reporting
        tolerance: Acceptable difference threshold

    Returns:
        bool: True if matrices match within tolerance
    """
    diff = np.abs(original - recovered)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n{name}:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < tolerance:
        print(f"  ✓ PASS (within tolerance {tolerance})")
        return True
    else:
        print(f"  ✗ FAIL (exceeds tolerance {tolerance})")
        print(f"\n  Original:\n{original}")
        print(f"\n  Recovered:\n{recovered}")
        print(f"\n  Difference:\n{diff}")
        return False


def test_roundtrip():
    """
    Test complete round-trip: numpy → Alembic → numpy.
    """
    print("="*70)
    print("ALEMBIC CAMERA ROUND-TRIP TEST")
    print("="*70)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Step 1: Create synthetic camera data
        print("\n[1/4] Creating synthetic camera data...")
        num_cameras = 5
        image_width = 1920
        image_height = 1080

        original_extrinsics, original_intrinsics = create_synthetic_cameras(
            num_cameras, image_width, image_height
        )

        # Step 2: Export to Alembic
        print("\n[2/4] Exporting to Alembic...")
        alembic_file = tmp_path / "test_cameras.abc"

        export_cameras_to_alembic(
            extrinsics=original_extrinsics,
            intrinsics=original_intrinsics,
            output_file=str(alembic_file),
            image_width=image_width,
            image_height=image_height,
            camera_names=[f"test_cam_{i}" for i in range(num_cameras)]
        )

        # Step 3: Re-import from Alembic
        print("\n[3/4] Re-importing from Alembic...")
        recovered_extrinsics, recovered_intrinsics = read_alembic_cameras(
            str(alembic_file),
            sample_index=0,
            image_width=image_width,
            image_height=image_height
        )

        # Step 4: Compare results
        print("\n[4/4] Comparing original vs recovered...")
        print("="*70)

        all_passed = True

        # Compare each camera
        for i in range(num_cameras):
            print(f"\n--- Camera {i} ---")

            # Compare extrinsics
            ext_pass = compare_matrices(
                original_extrinsics[i],
                recovered_extrinsics[i],
                f"Extrinsics (Camera {i})",
                tolerance=1e-3
            )

            # Compare intrinsics
            int_pass = compare_matrices(
                original_intrinsics[i],
                recovered_intrinsics[i],
                f"Intrinsics (Camera {i})",
                tolerance=1e-1  # More tolerance for intrinsics due to unit conversions
            )

            all_passed = all_passed and ext_pass and int_pass

        # Final result
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)

        if all_passed:
            print("\n✓ ALL TESTS PASSED")
            print("\nRound-trip conversion is working correctly!")
            print("Camera data can be reliably exported and re-imported.")
        else:
            print("\n✗ SOME TESTS FAILED")
            print("\nThere were differences between original and recovered data.")
            print("This may be due to:")
            print("  - Precision loss in Alembic format")
            print("  - Unit conversion issues")
            print("  - Coordinate system differences")

        print("="*70)

        return all_passed


def main():
    """Run the round-trip test."""
    success = test_roundtrip()
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

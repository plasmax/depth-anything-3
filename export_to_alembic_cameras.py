#!/usr/bin/env python3
"""
Export Camera Data to Alembic

This script takes extrinsics and intrinsics numpy arrays and exports them to an Alembic file
as camera objects. This is the reverse operation of alembic_camera_converter.py.

Input formats:
- extrinsics: (N, 4, 4) numpy array - world-to-camera transformation matrices
- intrinsics: (N, 3, 3) numpy array - camera intrinsic matrices

Output:
- Alembic (.abc) file with N camera objects
"""

import numpy as np
import alembic
from alembic import Abc
from alembic import AbcGeom
import imath


def camera_to_world_matrix(extrinsics):
    """
    Convert camera-space (extrinsics) matrix to world-space transformation.

    Args:
        extrinsics: 4x4 camera/view matrix

    Returns:
        4x4 world transformation matrix (inverse of extrinsics)
    """
    return np.linalg.inv(extrinsics)


def decompose_intrinsics(intrinsics_matrix, image_width, image_height):
    """
    Decompose intrinsics matrix to camera properties needed for Alembic.

    The intrinsics matrix K has the form:
    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

    Args:
        intrinsics_matrix: 3x3 intrinsics matrix
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        dict with keys: focal_length, h_aperture, v_aperture
    """
    # Extract focal lengths and principal point
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    # Choose a standard focal length (35mm is common)
    # We'll use 35mm as the reference
    focal_length_mm = 35.0

    # Calculate sensor size based on focal length and field of view
    # fx = (focal_length / sensor_width) * image_width
    # => sensor_width = (focal_length / fx) * image_width
    h_aperture_mm = (focal_length_mm / fx) * image_width
    v_aperture_mm = (focal_length_mm / fy) * image_height

    # Convert from mm to cm for Alembic
    h_aperture_cm = h_aperture_mm / 10.0
    v_aperture_cm = v_aperture_mm / 10.0

    return {
        'focal_length': focal_length_mm,
        'h_aperture': h_aperture_cm,
        'v_aperture': v_aperture_cm,
    }


def numpy_to_imath_matrix(np_matrix):
    """
    Convert numpy 4x4 matrix to Imath M44d matrix.

    Args:
        np_matrix: 4x4 numpy array

    Returns:
        Imath.M44d matrix
    """
    m = imath.M44d()
    for i in range(4):
        for j in range(4):
            m[i][j] = float(np_matrix[i, j])
    return m


def export_cameras_to_alembic(
    extrinsics,
    intrinsics,
    output_file,
    image_width=1920,
    image_height=1080,
    camera_names=None,
    fps=24.0
):
    """
    Export camera extrinsics and intrinsics to an Alembic file.

    Args:
        extrinsics: numpy array of shape (N, 4, 4) - camera-to-world matrices
        intrinsics: numpy array of shape (N, 3, 3) - camera calibration matrices
        output_file: Path to output Alembic (.abc) file
        image_width: Image width in pixels (default: 1920)
        image_height: Image height in pixels (default: 1080)
        camera_names: List of camera names (default: ['camera_0', 'camera_1', ...])
        fps: Frame rate for time sampling (default: 24.0)
    """
    # Validate inputs
    assert extrinsics.shape[0] == intrinsics.shape[0], \
        f"Number of extrinsics ({extrinsics.shape[0]}) must match intrinsics ({intrinsics.shape[0]})"
    assert extrinsics.shape == (extrinsics.shape[0], 4, 4), \
        f"Extrinsics must have shape (N, 4, 4), got {extrinsics.shape}"
    assert intrinsics.shape == (intrinsics.shape[0], 3, 3), \
        f"Intrinsics must have shape (N, 3, 3), got {intrinsics.shape}"

    num_cameras = extrinsics.shape[0]

    # Generate camera names if not provided
    if camera_names is None:
        camera_names = [f'camera_{i}' for i in range(num_cameras)]
    else:
        assert len(camera_names) == num_cameras, \
            f"Number of camera names ({len(camera_names)}) must match number of cameras ({num_cameras})"

    print(f"Exporting {num_cameras} camera(s) to: {output_file}")

    # Create Alembic archive
    archive = Abc.OArchive(output_file)
    time_sampling = Abc.TimeSampling(1.0 / fps, 0.0)
    ts_index = archive.addTimeSampling(time_sampling)

    # Get root object
    root = archive.getTop()

    # Create cameras
    for i, (ext, int_mat, cam_name) in enumerate(zip(extrinsics, intrinsics, camera_names)):
        print(f"  Creating camera {i+1}/{num_cameras}: {cam_name}")

        # Create transform for camera
        xform = AbcGeom.OXform(root, cam_name)
        xform_schema = xform.getSchema()

        # Convert extrinsics to world transform
        world_transform = camera_to_world_matrix(ext)
        imath_matrix = numpy_to_imath_matrix(world_transform)

        # Create XformSample and set the matrix
        xform_sample = AbcGeom.XformSample()
        xform_sample.setMatrix(imath_matrix)

        # Set the transform
        xform_schema.set(xform_sample)

        # Create camera under the transform
        camera = AbcGeom.OCamera(xform, f"{cam_name}Shape")
        cam_schema = camera.getSchema()

        # Decompose intrinsics to camera properties
        cam_props = decompose_intrinsics(int_mat, image_width, image_height)

        # Create camera sample
        cam_sample = AbcGeom.CameraSample()
        cam_sample.setFocalLength(cam_props['focal_length'])
        cam_sample.setHorizontalAperture(cam_props['h_aperture'])
        cam_sample.setVerticalAperture(cam_props['v_aperture'])

        # Set camera sample
        cam_schema.set(cam_sample)

    print(f"\n✓ Successfully exported {num_cameras} camera(s) to {output_file}")


def main():
    """Example usage of the Alembic camera exporter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Export numpy extrinsics/intrinsics to Alembic camera file'
    )
    parser.add_argument('input_file', help='Path to .npz file containing extrinsics and intrinsics')
    parser.add_argument('output_file', help='Path to output Alembic (.abc) file')
    parser.add_argument('--width', type=int, default=1920,
                        help='Image width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Image height in pixels (default: 1080)')
    parser.add_argument('--fps', type=float, default=24.0,
                        help='Frame rate for time sampling (default: 24.0)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Camera names (e.g., --names cam1 cam2 cam3)')

    args = parser.parse_args()

    # Load camera data from npz file
    print(f"Loading camera data from: {args.input_file}")
    data = np.load(args.input_file)

    if 'extrinsics' not in data or 'intrinsics' not in data:
        raise ValueError(
            f"Input file must contain 'extrinsics' and 'intrinsics' arrays. "
            f"Found keys: {list(data.keys())}"
        )

    extrinsics = data['extrinsics']
    intrinsics = data['intrinsics']

    print(f"  Loaded extrinsics: {extrinsics.shape}")
    print(f"  Loaded intrinsics: {intrinsics.shape}")

    # Export to Alembic
    export_cameras_to_alembic(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_file=args.output_file,
        image_width=args.width,
        image_height=args.height,
        camera_names=args.names,
        fps=args.fps
    )

    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nYou can now import {args.output_file} into:")
    print("  - Blender (File → Import → Alembic)")
    print("  - Maya (File → Import → Alembic)")
    print("  - Houdini (File → Import → Alembic)")
    print("  - Any other DCC tool that supports Alembic")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

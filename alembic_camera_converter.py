#!/usr/bin/env python3
"""
Alembic Camera Data Converter

This script reads camera data from an Alembic file and converts it to numpy arrays
in the extrinsics/intrinsics format expected by DepthAnything3.inference().

Expected formats:
- extrinsics: (N, 4, 4) numpy array - world-to-camera transformation matrices
- intrinsics: (N, 3, 3) numpy array - camera intrinsic matrices

Alembic camera data includes:
- Camera transforms (position, rotation) -> extrinsics
- Camera properties (focal length, sensor size, etc.) -> intrinsics
"""

import numpy as np
import alembic
from alembic.Abc import IArchive, IObject
from alembic.AbcGeom import ICamera, IXform
import imath


def extract_camera_transform(xform, sample_index=0):
    """
    Extract the transformation matrix from an Alembic Xform object.

    Args:
        xform: Alembic IXform object
        sample_index: Frame/sample index to extract

    Returns:
        4x4 numpy array representing the transformation matrix
    """
    schema = xform.getSchema()
    sample = schema.getValue(sample_index)
    matrix = sample.getMatrix()

    # Convert Imath matrix to numpy array
    transform = np.array([
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]]
    ])

    return transform


def compute_intrinsics_matrix(camera_sample, image_width, image_height):
    """
    Compute the camera intrinsics matrix from Alembic camera properties.

    The intrinsics matrix K has the form:
    [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

    Args:
        camera_sample: Alembic camera sample containing properties
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        3x3 numpy array representing the intrinsics matrix
    """
    # Get camera properties
    focal_length = camera_sample.getFocalLength()  # in mm
    h_aperture = camera_sample.getHorizontalAperture()  # in cm, convert to mm
    v_aperture = camera_sample.getVerticalAperture()  # in cm, convert to mm

    # Convert aperture from cm to mm
    h_aperture_mm = h_aperture * 10.0
    v_aperture_mm = v_aperture * 10.0

    # Calculate focal length in pixels
    fx = (focal_length / h_aperture_mm) * image_width
    fy = (focal_length / v_aperture_mm) * image_height

    # Principal point (usually image center)
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Build intrinsics matrix
    intrinsics = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])

    return intrinsics


def world_to_camera_matrix(world_matrix):
    """
    Convert world-space transformation matrix to camera-space (view matrix).

    Args:
        world_matrix: 4x4 world transformation matrix

    Returns:
        4x4 camera/view matrix (inverse of world matrix)
    """
    # The extrinsics matrix is typically the inverse of the world transform
    # (it transforms from world space to camera space)
    return np.linalg.inv(world_matrix)


def find_cameras_recursive(obj, cameras_info, parent_xform=None):
    """
    Recursively traverse the Alembic hierarchy to find all cameras.

    Args:
        obj: Current Alembic object to check
        cameras_info: List to accumulate camera information
        parent_xform: Parent transformation (for hierarchy)
    """
    # Check if this object is a camera
    if ICamera.matches(obj.getMetaData()):
        camera = ICamera(obj, alembic.Abc.WrapExistingFlag.kWrapExisting)
        cameras_info.append({
            'name': obj.getName(),
            'camera': camera,
            'parent_xform': parent_xform
        })

    # Check if this object has a transform
    current_xform = None
    if IXform.matches(obj.getMetaData()):
        current_xform = IXform(obj, alembic.Abc.WrapExistingFlag.kWrapExisting)

    # Recursively process children
    for i in range(obj.getNumChildren()):
        child = obj.getChild(i)
        find_cameras_recursive(child, cameras_info, current_xform or parent_xform)


def read_alembic_cameras(alembic_file, sample_index=0, image_width=1920, image_height=1080):
    """
    Read camera data from an Alembic file and convert to extrinsics/intrinsics format.

    Args:
        alembic_file: Path to the Alembic (.abc) file
        sample_index: Frame/sample index to extract (default: 0 for first frame)
        image_width: Target image width in pixels (default: 1920)
        image_height: Target image height in pixels (default: 1080)

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics: numpy array of shape (N, 4, 4)
            - intrinsics: numpy array of shape (N, 3, 3)
    """
    # Open the Alembic archive
    archive = IArchive(alembic_file)
    root = archive.getTop()

    # Find all cameras in the hierarchy
    cameras_info = []
    find_cameras_recursive(root, cameras_info)

    if not cameras_info:
        raise ValueError(f"No cameras found in Alembic file: {alembic_file}")

    print(f"Found {len(cameras_info)} camera(s) in {alembic_file}")

    # Extract camera data
    extrinsics_list = []
    intrinsics_list = []

    for cam_info in cameras_info:
        camera = cam_info['camera']
        parent_xform = cam_info['parent_xform']
        cam_name = cam_info['name']

        print(f"Processing camera: {cam_name}")

        # Get camera sample
        cam_schema = camera.getSchema()
        cam_sample = cam_schema.getValue(sample_index)

        # Compute intrinsics
        intrinsics = compute_intrinsics_matrix(cam_sample, image_width, image_height)
        intrinsics_list.append(intrinsics)

        # Get camera transform
        if parent_xform is not None:
            world_transform = extract_camera_transform(parent_xform, sample_index)
        else:
            # If no parent transform, use identity
            world_transform = np.eye(4)

        # Convert to camera space (view matrix)
        extrinsics = world_to_camera_matrix(world_transform)
        extrinsics_list.append(extrinsics)

    # Convert lists to numpy arrays
    extrinsics_array = np.stack(extrinsics_list, axis=0)  # Shape: (N, 4, 4)
    intrinsics_array = np.stack(intrinsics_list, axis=0)  # Shape: (N, 3, 3)

    return extrinsics_array, intrinsics_array


def main():
    """Example usage of the Alembic camera converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Alembic camera data to numpy arrays for DepthAnything3'
    )
    parser.add_argument('alembic_file', help='Path to the Alembic (.abc) file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Frame/sample index to extract (default: 0)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Image width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Image height in pixels (default: 1080)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz file to save camera data (optional)')

    args = parser.parse_args()

    # Read camera data
    extrinsics, intrinsics = read_alembic_cameras(
        args.alembic_file,
        sample_index=args.sample,
        image_width=args.width,
        image_height=args.height
    )

    # Print results
    print("\n" + "="*60)
    print("CAMERA DATA EXTRACTED")
    print("="*60)
    print(f"\nExtrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")

    print("\n--- Extrinsics (World-to-Camera) ---")
    for i, ext in enumerate(extrinsics):
        print(f"\nCamera {i}:")
        print(ext)

    print("\n--- Intrinsics (Calibration) ---")
    for i, int_mat in enumerate(intrinsics):
        print(f"\nCamera {i}:")
        print(int_mat)

    # Save to file if requested
    if args.output:
        np.savez(args.output, extrinsics=extrinsics, intrinsics=intrinsics)
        print(f"\n✓ Camera data saved to: {args.output}")
        print("  Load with: data = np.load('file.npz'); extrinsics = data['extrinsics']; intrinsics = data['intrinsics']")

    print("\n" + "="*60)
    print("USAGE WITH DEPTHANYTHING3")
    print("="*60)
    print("\nYou can now use this data with DepthAnything3:")
    print("""
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/Depth-Anything-V2-Large")
model = model.to('cuda')

# Use the extracted camera data
prediction = model.inference(
    image=your_images,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir="output",
    export_format="glb"
)
""")


if __name__ == '__main__':
    main()

# Alembic Camera Data Converter for DepthAnything3

This guide explains how to use Alembic camera data with DepthAnything3 for depth estimation with known camera parameters.

## Overview

The converter reads camera data from Alembic files and converts it to the format expected by DepthAnything3:
- **Extrinsics**: `(N, 4, 4)` numpy array - World-to-camera transformation matrices
- **Intrinsics**: `(N, 3, 3)` numpy array - Camera calibration matrices

## Installation

Install the required dependencies:

```bash
pip install alembic imath numpy
```

## Scripts

### 1. `alembic_camera_converter.py`

Core converter that reads Alembic camera data and converts it to numpy arrays.

**Usage:**

```bash
# Basic usage
python alembic_camera_converter.py cameras.abc

# Specify frame and resolution
python alembic_camera_converter.py cameras.abc --sample 10 --width 1920 --height 1080

# Save to file
python alembic_camera_converter.py cameras.abc --output camera_data.npz
```

**Arguments:**
- `alembic_file`: Path to the Alembic (.abc) file
- `--sample`: Frame/sample index to extract (default: 0)
- `--width`: Image width in pixels (default: 1920)
- `--height`: Image height in pixels (default: 1080)
- `--output`: Output .npz file to save camera data (optional)

### 2. `example_alembic_to_depth.py`

Complete example that combines Alembic camera data with DepthAnything3 inference.

**Usage:**

```bash
# Run inference with Alembic cameras
python example_alembic_to_depth.py cameras.abc ./images/ --output-dir results

# Specify model and export format
python example_alembic_to_depth.py cameras.abc ./images/ \
    --model da3-giant \
    --export-format glb \
    --output-dir results
```

**Arguments:**
- `alembic_file`: Path to Alembic file with camera data
- `image_dir`: Directory containing input images
- `--model`: Model name (default: da3-large)
- `--sample`: Alembic frame index (default: 0)
- `--width`: Image width (default: 1920)
- `--height`: Image height (default: 1080)
- `--output-dir`: Output directory (default: output)
- `--export-format`: Export format: glb, ply, npz, etc. (default: glb)
- `--device`: Device: cuda or cpu (default: cuda)

## Python API Usage

### Using the converter programmatically:

```python
from alembic_camera_converter import read_alembic_cameras
import numpy as np

# Read camera data
extrinsics, intrinsics = read_alembic_cameras(
    'cameras.abc',
    sample_index=0,
    image_width=1920,
    image_height=1080
)

# extrinsics: (N, 4, 4) array of world-to-camera matrices
# intrinsics: (N, 3, 3) array of camera calibration matrices
```

### Using with DepthAnything3:

```python
from depth_anything_3.api import DepthAnything3
from alembic_camera_converter import read_alembic_cameras

# Load camera data
extrinsics, intrinsics = read_alembic_cameras('cameras.abc')

# Load model
model = DepthAnything3(model_name='da3-large')
model = model.to('cuda')

# Run inference
prediction = model.inference(
    image=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir='output',
    export_format='glb'
)
```

### Loading saved camera data:

```python
import numpy as np

# Save camera data
np.savez('camera_data.npz', extrinsics=extrinsics, intrinsics=intrinsics)

# Load camera data
data = np.load('camera_data.npz')
extrinsics = data['extrinsics']
intrinsics = data['intrinsics']
```

## Camera Data Format

### Extrinsics (N, 4, 4)

World-to-camera transformation matrix for each camera:

```
[[R11, R12, R13, tx],
 [R21, R22, R23, ty],
 [R31, R32, R33, tz],
 [  0,   0,   0,  1]]
```

Where:
- `R`: 3x3 rotation matrix
- `t`: 3x1 translation vector

### Intrinsics (N, 3, 3)

Camera calibration matrix for each camera:

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

Where:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (image center)

## Alembic File Requirements

The Alembic file should contain:
1. **Camera objects** with properties:
   - Focal length (mm)
   - Horizontal aperture (cm)
   - Vertical aperture (cm)

2. **Transform hierarchy** defining camera positions and orientations

## Example Workflow

```bash
# 1. Export camera data from your 3D software to Alembic format
# (e.g., from Blender, Maya, Houdini, etc.)

# 2. Convert Alembic to numpy format
python alembic_camera_converter.py cameras.abc --output camera_data.npz

# 3. Run DepthAnything3 with camera data
python example_alembic_to_depth.py cameras.abc ./images/ \
    --export-format glb \
    --output-dir results

# 4. View the results
# Open results/*.glb in a 3D viewer
```

## Troubleshooting

### No cameras found
- Ensure your Alembic file contains camera objects
- Check that cameras are properly exported from your 3D software

### Image/camera count mismatch
- Ensure the number of images matches the number of cameras
- Images should correspond to each camera's viewpoint

### Resolution mismatch
- Specify the correct image resolution with `--width` and `--height`
- Resolution should match your actual images

## Notes

- The converter assumes standard Alembic camera conventions
- Camera transforms are converted from world space to camera space
- Intrinsics are computed from focal length and aperture settings
- Multiple cameras in a single Alembic file are supported
- Frame/sample selection allows extracting camera data from animated sequences

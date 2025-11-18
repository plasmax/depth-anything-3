# Alembic Camera Data Converter for DepthAnything3

This guide explains how to use Alembic camera data with DepthAnything3 for depth estimation with known camera parameters, and how to export results back to Alembic format.

## Overview

The converters provide bidirectional conversion between Alembic camera files and numpy arrays:

### Import (Alembic → numpy)
Read camera data from Alembic files and convert to DepthAnything3 format:
- **Extrinsics**: `(N, 4, 4)` numpy array - World-to-camera transformation matrices
- **Intrinsics**: `(N, 3, 3)` numpy array - Camera calibration matrices

### Export (numpy → Alembic)
Export camera extrinsics/intrinsics (e.g., from DepthAnything3 predictions) back to Alembic format for visualization in 3D software like Blender, Maya, or Houdini.

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

### 3. `export_to_alembic_cameras.py`

Export numpy extrinsics/intrinsics arrays to an Alembic camera file. This is the reverse operation - useful for visualizing camera data in 3D software.

**Usage:**

```bash
# Export from .npz file
python export_to_alembic_cameras.py camera_data.npz output_cameras.abc

# Specify resolution and camera names
python export_to_alembic_cameras.py camera_data.npz output_cameras.abc \
    --width 1920 --height 1080 \
    --names cam1 cam2 cam3
```

**Arguments:**
- `input_file`: Path to .npz file containing extrinsics and intrinsics
- `output_file`: Path to output Alembic (.abc) file
- `--width`: Image width in pixels (default: 1920)
- `--height`: Image height in pixels (default: 1080)
- `--fps`: Frame rate for time sampling (default: 24.0)
- `--names`: Camera names (optional)

### 4. `example_camera_roundtrip.py`

Export camera data from a DepthAnything3 prediction to Alembic format.

**Usage:**

```bash
# Export prediction cameras to Alembic
python example_camera_roundtrip.py prediction.npz cameras.abc

# Then import into Blender, Maya, Houdini, etc.
```

**Arguments:**
- `prediction_file`: Path to .npz file with DepthAnything3 prediction
- `output_file`: Path to output Alembic (.abc) file
- `--width`: Image width (default: 1920)
- `--height`: Image height (default: 1080)

### 5. `test_alembic_roundtrip.py`

Test script that verifies round-trip conversion (numpy → Alembic → numpy) works correctly.

**Usage:**

```bash
# Run the test
python test_alembic_roundtrip.py
```

This creates synthetic camera data, exports it to Alembic, re-imports it, and verifies the data matches within acceptable tolerances.

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

### Exporting camera data to Alembic:

```python
from export_to_alembic_cameras import export_cameras_to_alembic
import numpy as np

# After running DepthAnything3 inference
# prediction = model.inference(...)

# Export predicted cameras to Alembic
export_cameras_to_alembic(
    extrinsics=prediction.extrinsics,
    intrinsics=prediction.intrinsics,
    output_file='predicted_cameras.abc',
    image_width=1920,
    image_height=1080,
    camera_names=['cam0', 'cam1', 'cam2']
)

# Now you can import predicted_cameras.abc into Blender, Maya, etc.
```

### Complete round-trip workflow:

```python
from alembic_camera_converter import read_alembic_cameras
from export_to_alembic_cameras import export_cameras_to_alembic
from depth_anything_3.api import DepthAnything3

# 1. Load camera data from Alembic
extrinsics, intrinsics = read_alembic_cameras('input_cameras.abc')

# 2. Run depth estimation
model = DepthAnything3(model_name='da3-large').to('cuda')
prediction = model.inference(
    image=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir='output',
    export_format='glb'
)

# 3. Export predicted cameras back to Alembic
export_cameras_to_alembic(
    extrinsics=prediction.extrinsics,
    intrinsics=prediction.intrinsics,
    output_file='output_cameras.abc',
    image_width=1920,
    image_height=1080
)
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

## Example Workflows

### Workflow 1: Import Alembic → DepthAnything3

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

### Workflow 2: Export DepthAnything3 → Alembic

```bash
# 1. Run DepthAnything3 inference (produces extrinsics/intrinsics)
# This saves results as .npz files in the output directory

# 2. Export predicted cameras to Alembic
python export_to_alembic_cameras.py results/prediction.npz cameras_out.abc

# 3. Import cameras_out.abc into Blender, Maya, or Houdini
# - Blender: File → Import → Alembic Cache (.abc)
# - Maya: File → Import → Alembic
# - Houdini: File → Import → Alembic Scene

# 4. Visualize camera positions and orientations in 3D
```

### Workflow 3: Round-trip (Alembic → DepthAnything3 → Alembic)

```bash
# 1. Start with Alembic cameras
python alembic_camera_converter.py input_cameras.abc --output cam_data.npz

# 2. Run depth estimation
python example_alembic_to_depth.py input_cameras.abc ./images/ \
    --export-format npz \
    --output-dir results

# 3. Export predicted cameras back to Alembic
python example_camera_roundtrip.py results/prediction.npz output_cameras.abc

# 4. Compare input_cameras.abc vs output_cameras.abc in your 3D software
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

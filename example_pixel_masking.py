"""
Example script demonstrating pixel masking in Depth Anything V3.

This script shows how to use the mask parameter to tell the model which pixels to ignore
during depth estimation. Masked regions will be processed using learnable mask tokens,
allowing the model to estimate depth while being aware of which regions should be ignored.
"""

import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3


def create_circular_mask(image_shape, center, radius):
    """
    Create a circular mask for an image.

    Args:
        image_shape: Tuple of (height, width)
        center: Tuple of (y, x) for circle center
        radius: Radius of the circle in pixels

    Returns:
        Boolean numpy array where True indicates pixels to mask
    """
    h, w = image_shape
    y, x = np.ogrid[:h, :w]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    return mask


def create_rectangular_mask(image_shape, top_left, bottom_right):
    """
    Create a rectangular mask for an image.

    Args:
        image_shape: Tuple of (height, width)
        top_left: Tuple of (y, x) for top-left corner
        bottom_right: Tuple of (y, x) for bottom-right corner

    Returns:
        Boolean numpy array where True indicates pixels to mask
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    y1, x1 = top_left
    y2, x2 = bottom_right
    mask[y1:y2, x1:x2] = True
    return mask


def main():
    # Initialize the model
    print("Loading Depth Anything V3 model...")
    model = DepthAnything3.from_pretrained("depth-anything/Depth-Anything-V3-Small")

    # Example 1: Single image with circular mask
    print("\n=== Example 1: Single image with circular mask ===")

    # Load an image (replace with your own image path)
    image_path = "path/to/your/image.jpg"
    image = Image.open(image_path)

    # Create a circular mask in the center
    h, w = image.size[1], image.size[0]
    center = (h // 2, w // 2)
    radius = min(h, w) // 4
    mask = create_circular_mask((h, w), center, radius)

    print(f"Image shape: {image.size}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked pixels: {mask.sum()} / {mask.size} ({100 * mask.sum() / mask.size:.1f}%)")

    # Run inference with mask
    prediction = model.inference(
        image=[image],
        mask=[mask],
        process_res=518,
    )

    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Depth range: [{prediction.depth.min():.3f}, {prediction.depth.max():.3f}]")

    # Example 2: Multiple images with different masks
    print("\n=== Example 2: Multiple images with rectangular masks ===")

    # Load multiple images (replace with your own image paths)
    images = [
        Image.open("path/to/image1.jpg"),
        Image.open("path/to/image2.jpg"),
    ]

    # Create different masks for each image
    masks = []
    for img in images:
        h, w = img.size[1], img.size[0]
        # Mask the top-left quarter
        mask = create_rectangular_mask((h, w), (0, 0), (h // 2, w // 2))
        masks.append(mask)

    # Run inference with masks
    prediction = model.inference(
        image=images,
        mask=masks,
        process_res=518,
    )

    print(f"Number of depth maps: {len(prediction.depth)}")

    # Example 3: Load mask from image file
    print("\n=== Example 3: Load mask from binary image ===")

    # You can also load masks from binary images (white = masked, black = unmasked)
    mask_image = Image.open("path/to/mask.png").convert('L')
    mask_array = np.array(mask_image) > 128  # Threshold at 128

    prediction = model.inference(
        image=[image],
        mask=[mask_array],
        process_res=518,
    )

    print("Inference with mask from image file completed successfully!")

    # Example 4: No mask (for comparison)
    print("\n=== Example 4: No mask (baseline) ===")

    prediction_no_mask = model.inference(
        image=[image],
        process_res=518,
    )

    print("Inference without mask completed for comparison.")

    print("\n=== Usage Tips ===")
    print("1. Masks should be boolean numpy arrays (or convertible to boolean)")
    print("2. True values indicate pixels to IGNORE (mask out)")
    print("3. Masks are automatically resized to match the processed image resolution")
    print("4. You can provide one mask per image, or a single mask for all images")
    print("5. Masked regions use learnable mask tokens instead of image patches")


if __name__ == "__main__":
    main()

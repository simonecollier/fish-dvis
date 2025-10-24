#!/usr/bin/env python3
"""
Helper script to calculate query patch indices for specific pixel locations
"""

def pixel_to_patch(pixel_x, pixel_y, image_width=384, image_height=288, patch_size=16):
    """
    Convert pixel coordinates to patch index
    
    Args:
        pixel_x, pixel_y: Pixel coordinates in the original image
        image_width, image_height: Original image dimensions
        patch_size: Size of each patch (typically 16x16)
    
    Returns:
        patch_index: Index of the patch containing the pixel
    """
    # Calculate patch grid dimensions
    patches_w = image_width // patch_size
    patches_h = image_height // patch_size
    
    # Convert pixel to patch coordinates
    patch_x = min(pixel_x // patch_size, patches_w - 1)
    patch_y = min(pixel_y // patch_size, patches_h - 1)
    
    # Convert to linear patch index
    patch_index = patch_y * patches_w + patch_x
    
    print(f"Pixel ({pixel_x}, {pixel_y}) -> Patch ({patch_x}, {patch_y}) -> Index {patch_index}")
    print(f"Image: {image_width}×{image_height}, Patches: {patches_w}×{patches_h} = {patches_w * patches_h} total")
    
    return patch_index

def patch_to_pixel(patch_index, image_width=384, image_height=288, patch_size=16):
    """
    Convert patch index back to pixel coordinates (center of patch)
    """
    patches_w = image_width // patch_size
    
    patch_y = patch_index // patches_w
    patch_x = patch_index % patches_w
    
    # Get center pixel of the patch
    center_x = patch_x * patch_size + patch_size // 2
    center_y = patch_y * patch_size + patch_size // 2
    
    print(f"Patch {patch_index} -> Patch ({patch_x}, {patch_y}) -> Center pixel ({center_x}, {center_y})")
    
    return center_x, center_y

if __name__ == "__main__":
    print("=== Query Patch Calculator ===")
    print()
    
    # Common query patches
    print("Common query patch locations:")
    print("Top-left corner:", end=" ")
    pixel_to_patch(0, 0)
    
    print("Top-right corner:", end=" ")
    pixel_to_patch(383, 0)
    
    print("Center:", end=" ")
    pixel_to_patch(192, 144)
    
    print("Bottom-left corner:", end=" ")
    pixel_to_patch(0, 287)
    
    print("Bottom-right corner:", end=" ")
    pixel_to_patch(383, 287)
    
    print()
    print("=== Reverse lookup ===")
    print("What pixel locations do common patch indices represent?")
    
    for patch_idx in [0, 23, 216, 408, 431]:
        patch_to_pixel(patch_idx)
    
    print()
    print("=== Interactive Mode ===")
    print("Enter pixel coordinates to find the corresponding patch:")
    
    try:
        while True:
            x = int(input("Pixel X (0-383): "))
            y = int(input("Pixel Y (0-287): "))
            patch_idx = pixel_to_patch(x, y)
            print(f"Use --query-patch {patch_idx} to visualize this region")
            print()
    except (KeyboardInterrupt, EOFError, ValueError):
        print("\nDone!")

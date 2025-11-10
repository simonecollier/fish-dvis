import os
import argparse
import re
import json
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def load_rolled_out_spatial_attention(directory, video_id):
    """
    Load the rolled out spatial attention map for a video.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder
                  (e.g., /path/to/attn_extract_3231)
        video_id: Video ID number
    
    Returns:
        Attention map with shape [num_frames, num_patches, num_patches]
    """
    attention_maps_dir = os.path.join(directory, "attention_maps")
    rolled_out_dir = os.path.join(attention_maps_dir, "rolled_out")
    npz_path = os.path.join(rolled_out_dir, f"video_{video_id}_backbone_vit_rolled_out.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find rolled out attention map at {npz_path}")
    
    data = np.load(npz_path)
    
    if 'attention_weights' not in data.keys():
        raise ValueError(f"Expected 'attention_weights' key in {npz_path}")
    
    attention_map = data['attention_weights']
    
    # Verify shape is [num_frames, num_patches, num_patches]
    if len(attention_map.shape) != 3:
        raise ValueError(f"Expected 3D array [num_frames, num_patches, num_patches], got shape {attention_map.shape}")
    
    return attention_map


def load_image_dimensions_from_eval(directory, video_id):
    """
    Load image and patch dimensions from image_dimensions.json file in inference directory.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder (e.g., /path/to/attn_extract_3231)
        video_id: Video ID number
    
    Returns:
        dict with keys:
        - 'padded_image_height', 'padded_image_width' (padded dimensions used by model)
        - 'original_image_height', 'original_image_width' (original image dimensions)
        - 'H_patches', 'W_patches' (patch grid dimensions)
        - 'pad_height', 'pad_width', 'pad_top', 'pad_left', 'pad_bottom', 'pad_right' (padding info)
        Raises FileNotFoundError if not found
    """
    # image_dimensions.json is in inference/ directory relative to the specified directory
    inference_dir = os.path.join(directory, "inference")
    dims_path = os.path.join(inference_dir, "image_dimensions.json")
    
    if not os.path.exists(dims_path):
        raise FileNotFoundError(f"Could not find image_dimensions.json at {dims_path}")
    
    with open(dims_path, 'r') as f:
        dims_dict = json.load(f)
        video_key = str(video_id)
        if video_key not in dims_dict:
            raise KeyError(f"Video {video_id} not found in image_dimensions.json")
        return dims_dict[video_key]


def find_val_json_in_model_dir(directory):
    """
    Find the val.json file in the model directory.
    
    The model directory is determined by going up from the directory until we find
    a directory containing a val.json file.
    
    Args:
        directory: Path to directory (e.g., /path/to/attn_extract_3231)
    
    Returns:
        Path to val.json file
        Raises FileNotFoundError if not found or multiple found
    """
    # Go up from directory to find model directory
    current_dir = Path(directory).resolve()
    
    # Look for val.json files
    val_json_files = []
    for parent in [current_dir] + list(current_dir.parents):
        val_json_path = parent / "val.json"
        if val_json_path.exists():
            val_json_files.append(str(val_json_path))
    
    if len(val_json_files) == 0:
        raise FileNotFoundError(f"Could not find val.json in model directory (searched from {directory})")
    elif len(val_json_files) > 1:
        raise ValueError(f"Multiple val.json files found: {val_json_files}. Please specify model directory.")
    
    return val_json_files[0]


def get_video_info_from_val_json(val_json_path, video_id):
    """
    Get video information from val.json file.
    
    Args:
        val_json_path: Path to val.json file
        video_id: Video ID number
    
    Returns:
        dict with video info including 'file_names' list
        Raises KeyError if video not found
    """
    with open(val_json_path, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    if video_id not in videos:
        raise KeyError(f"Video {video_id} not found in val.json")
    
    return videos[video_id]


def get_image_root_directory(model_dir):
    """
    Determine the image root directory based on model directory name.
    
    Args:
        model_dir: Path to model directory (e.g., /path/to/model_camera_ft123)
    
    Returns:
        Path to image root directory:
        - /home/simone/shared-data/fishway_ytvis/all_videos if model_dir contains "camera"
        - /home/simone/shared-data/fishway_ytvis/all_videos_mask if model_dir contains "silhouette"
        Raises ValueError if neither found
    """
    model_dir_str = str(model_dir)
    if "camera" in model_dir_str:
        return "/home/simone/shared-data/fishway_ytvis/all_videos"
    elif "silhouette" in model_dir_str:
        return "/home/simone/shared-data/fishway_ytvis/all_videos_mask"
    else:
        raise ValueError(f"Could not determine image root directory from model directory: {model_dir}. "
                         f"Model directory must contain 'camera' or 'silhouette'.")


def load_resize_and_pad_image(image_path, original_height, original_width, 
                              pad_top=0, pad_left=0, pad_bottom=0, pad_right=0):
    """
    Load an image, resize it to original dimensions, and apply padding.
    
    Args:
        image_path: Path to image file
        original_height: Target height after resize (before padding)
        original_width: Target width after resize (before padding)
        pad_top, pad_left, pad_bottom, pad_right: Padding values in pixels
    
    Returns:
        PIL Image object (resized and padded)
    """
    # Load image
    img = Image.open(image_path)
    
    # Resize to original dimensions
    img_resized = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
    
    # Apply padding
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        # Calculate final dimensions
        final_width = original_width + pad_left + pad_right
        final_height = original_height + pad_top + pad_bottom
        
        # Create new image with padding (black background)
        img_padded = Image.new('RGB', (final_width, final_height), (0, 0, 0))
        
        # Paste resized image at the correct position
        img_padded.paste(img_resized, (pad_left, pad_top))
        
        return img_padded
    else:
        return img_resized


def try_load_spatial_shape_from_metadata(directory, video_id):
    """
    Try to load spatial shape (H_patches, W_patches) from metadata files.
    
    Note: For adapter format, spatial_shape is typically None, so this may not work.
    But we check anyway in case it was captured.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder
        video_id: Video ID number
    
    Returns:
        (H_patches, W_patches) if found in metadata, None otherwise
    """
    import json
    attention_maps_dir = os.path.join(directory, "attention_maps")
    
    # Look for any metadata file for this video
    # Support both old and new naming formats
    pattern_old = re.compile(rf'video_{video_id}_frames\d+-\d+_layer_backbone_vit_module_blocks_\d+_attn\.meta\.json')
    pattern_new = re.compile(rf'video_{video_id}_frames\d+-\d+_backbone_vit_layer_\d+_attn\.meta\.json')
    
    for filename in os.listdir(attention_maps_dir):
        if pattern_old.match(filename) or pattern_new.match(filename):
            meta_path = os.path.join(attention_maps_dir, filename)
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    spatial_shape = meta.get('spatial_shape')
                    if spatial_shape is not None and isinstance(spatial_shape, list) and len(spatial_shape) == 2:
                        H_patches, W_patches = spatial_shape
                        return H_patches, W_patches
            except Exception:
                continue
    
    return None


def average_columns_per_frame(attention_map):
    """
    Average down the columns of the attention matrix for each frame.
    
    For each frame, we have a [num_patches, num_patches] attention matrix.
    Averaging down the columns means: for each column (patch), we average
    across all rows. This gives us a [num_patches] vector per frame.
    
    This is equivalent to .mean(axis=1) for each frame's matrix.
    
    Args:
        attention_map: Attention map with shape [num_frames, num_patches, num_patches]
    
    Returns:
        Averaged attention map with shape [num_frames, num_patches]
    """
    if len(attention_map.shape) != 3:
        raise ValueError(f"Expected 3D array [num_frames, num_patches, num_patches], got shape {attention_map.shape}")
    
    # For each frame, average across rows (axis=1) to get column averages
    # This gives us [num_frames, num_patches]
    column_averages = attention_map.mean(axis=1)
    
    return column_averages


def normalize_per_frame(column_averages):
    """
    Normalize each frame's column averages between 0 and 1 using min-max normalization.
    
    For each frame (each row), applies min-max normalization:
    normalized = (values - min) / (max - min)
    
    Args:
        column_averages: Array with shape [num_frames, num_patches]
    
    Returns:
        Normalized array with shape [num_frames, num_patches], values in [0, 1]
    """
    if len(column_averages.shape) != 2:
        raise ValueError(f"Expected 2D array [num_frames, num_patches], got shape {column_averages.shape}")
    
    num_frames, num_patches = column_averages.shape
    normalized = np.zeros_like(column_averages, dtype=np.float32)
    
    for frame_idx in range(num_frames):
        frame_values = column_averages[frame_idx]
        frame_min = frame_values.min()
        frame_max = frame_values.max()
        frame_range = frame_max - frame_min
        
        if frame_range == 0:
            # All values are the same, set to 0 (or could be 0.5, but 0 is safer)
            normalized[frame_idx] = np.zeros_like(frame_values)
        else:
            normalized[frame_idx] = (frame_values - frame_min) / frame_range
    
    return normalized


def compute_patch_grid_dimensions(num_patches, patch_size=16, has_cls_token=True):
    """
    Compute patch grid dimensions (H_patches, W_patches) from total number of patches.
    
    Based on DVIS-DAQ codebase analysis:
    - CLS token is PREPENDED to patch tokens (index 0 is CLS, indices 1+ are spatial patches)
    - PatchEmbed converts image (B, C, H_img, W_img) to patches (B, H_patches*W_patches, C)
    - Where H_patches = H_img // patch_size, W_patches = W_img // patch_size
    - Total tokens = 1 (CLS) + H_patches * W_patches (spatial patches)
    
    Args:
        num_patches: Total number of tokens (1 CLS token + spatial patches)
        patch_size: Size of each patch in pixels (default: 16 for ViT-L)
        has_cls_token: Whether the first token is a CLS token (default: True)
    
    Returns:
        (H_patches, W_patches, image_height, image_width): 
            - H_patches: Number of patches in height dimension
            - W_patches: Number of patches in width dimension
            - image_height: Image height in pixels (H_patches * patch_size)
            - image_width: Image width in pixels (W_patches * patch_size)
    """
    if has_cls_token:
        spatial_patches = num_patches - 1
    else:
        spatial_patches = num_patches
    
    # Find factors of spatial_patches that make sense for image dimensions
    # Try to find a reasonable aspect ratio (common: 3:4, 4:3, 1:1, etc.)
    best_h, best_w = None, None
    min_diff = float('inf')
    
    # Try common aspect ratios
    for h in range(1, int(np.sqrt(spatial_patches)) + 1):
        if spatial_patches % h == 0:
            w = spatial_patches // h
            # Prefer aspect ratios close to common image ratios
            aspect_ratio = w / h if h > 0 else 0
            # Common ratios: 1.0 (square), 1.33 (4:3), 0.75 (3:4), 1.5, 0.67, etc.
            diff = min(abs(aspect_ratio - 1.0), abs(aspect_ratio - 1.33), 
                      abs(aspect_ratio - 0.75), abs(aspect_ratio - 1.5), 
                      abs(aspect_ratio - 0.67))
            if diff < min_diff:
                min_diff = diff
                best_h, best_w = h, w
    
    if best_h is None:
        # Fallback: use square grid
        grid_size = int(np.sqrt(spatial_patches))
        if grid_size * grid_size == spatial_patches:
            best_h = best_w = grid_size
        else:
            raise ValueError(f"Cannot find valid patch grid dimensions for {spatial_patches} spatial patches")
    
    H_patches = best_h
    W_patches = best_w
    image_height = H_patches * patch_size
    image_width = W_patches * patch_size
    
    return H_patches, W_patches, image_height, image_width


def patch_index_to_image_coords(patch_idx, H_patches, W_patches, patch_size=16, has_cls_token=True,
                                pad_top=0, pad_left=0):
    """
    Convert patch index to image coordinates (bounding box).
    
    Based on DVIS-DAQ codebase:
    - Token sequence: [CLS_token (index 0), patch_0 (index 1), patch_1 (index 2), ..., patch_N-1]
    - Spatial patches are arranged in row-major order (flattened from 2D grid)
    - Patch at grid position (row, col) has linear index = row * W_patches + col
    - So spatial patch index i corresponds to: row = i // W_patches, col = i % W_patches
    
    IMPORTANT: By default, coordinates are for the PADDED image (after size_divisibility padding).
    The model pads images to be multiples of 32 (size_divisibility), and patches are computed
    from this padded image. 
    
    If pad_top and pad_left are provided, coordinates are adjusted to map to the ORIGINAL image.
    Typically, padding is added at the bottom/right, so pad_top=0 and pad_left=0, but this
    function supports arbitrary padding offsets.
    
    Args:
        patch_idx: Token index (0-based). If has_cls_token=True, 0 is CLS token, 1+ are spatial patches.
        H_patches: Number of patches in height dimension (from padded image)
        W_patches: Number of patches in width dimension (from padded image)
        patch_size: Size of each patch in pixels (default: 16)
        has_cls_token: Whether the first token is a CLS token (default: True)
        pad_top: Padding offset at top (default: 0). If non-zero, coordinates are adjusted for original image.
        pad_left: Padding offset at left (default: 0). If non-zero, coordinates are adjusted for original image.
    
    Returns:
        (x_min, y_min, x_max, y_max): Bounding box coordinates in image pixels
        - If pad_top=0 and pad_left=0: coordinates are in PADDED image space
        - If pad_top or pad_left are non-zero: coordinates are adjusted for ORIGINAL image space
        Returns None if patch_idx is 0 and has_cls_token=True (CLS token has no spatial location)
    """
    if has_cls_token and patch_idx == 0:
        # CLS token has no spatial location (it's prepended before all spatial patches)
        return None
    
    if has_cls_token:
        spatial_idx = patch_idx - 1  # Convert from token index to spatial patch index
    else:
        spatial_idx = patch_idx
    
    # Convert linear spatial index to 2D grid coordinates (row-major order)
    patch_row = spatial_idx // W_patches
    patch_col = spatial_idx % W_patches
    
    # Convert to image coordinates (pixels) in padded image space
    x_min = patch_col * patch_size
    y_min = patch_row * patch_size
    x_max = x_min + patch_size
    y_max = y_min + patch_size
    
    # Adjust for padding offset if provided (to map to original image)
    x_min -= pad_left
    y_min -= pad_top
    x_max -= pad_left
    y_max -= pad_top
    
    return (x_min, y_min, x_max, y_max)


def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial attention maps from rolled out attention data"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing 'attention_maps' folder (e.g., /path/to/attn_extract_3231)"
    )
    parser.add_argument(
        "--video-id",
        type=int,
        help="Video ID number (if not provided, will try to find from files in attention_maps folder)"
    )
    parser.add_argument(
        "--image-height",
        type=int,
        help="Actual image height in pixels (if known). If not provided, will be computed from num_patches."
    )
    parser.add_argument(
        "--image-width",
        type=int,
        help="Actual image width in pixels (if known). If not provided, will be computed from num_patches."
    )
    
    args = parser.parse_args()
    
    directory = args.directory
    attention_maps_dir = os.path.join(directory, "attention_maps")
    
    if not os.path.exists(attention_maps_dir):
        raise FileNotFoundError(f"Could not find 'attention_maps' folder in {directory}")
    
    # If video_id not provided, try to find it from files
    video_id = args.video_id
    if video_id is None:
        # Look for files matching pattern: video_<id>_backbone_vit_rolled_out.npz in rolled_out subdirectory
        rolled_out_dir = os.path.join(attention_maps_dir, "rolled_out")
        pattern = re.compile(r'video_(\d+)_backbone_vit_rolled_out\.npz')
        found_ids = []
        if os.path.exists(rolled_out_dir):
            for filename in os.listdir(rolled_out_dir):
                match = pattern.match(filename)
                if match:
                    found_ids.append(int(match.group(1)))
        
        if len(found_ids) == 0:
            raise ValueError(f"No rolled out attention maps found in {rolled_out_dir}")
        elif len(found_ids) > 1:
            print(f"Found multiple video IDs: {found_ids}")
            print(f"Using first one: {found_ids[0]}")
            video_id = found_ids[0]
        else:
            video_id = found_ids[0]
            print(f"Found video ID: {video_id}")
    
    # Load the rolled out attention map
    print(f"Loading rolled out spatial attention for video {video_id}...")
    attention_map = load_rolled_out_spatial_attention(directory, video_id)
    print(f"Loaded attention map with shape: {attention_map.shape}")
    
    # Average columns for each frame
    print("Averaging columns for each frame...")
    column_averages = average_columns_per_frame(attention_map)
    print(f"Column averages shape: {column_averages.shape}")
    
    # Remove CLS token (first entry, index 0) before normalization
    print("Removing CLS token (index 0) from column averages...")
    column_averages_spatial = column_averages[:, 1:]  # Shape: [num_frames, num_patches - 1]
    print(f"Column averages (spatial only) shape: {column_averages_spatial.shape}")
    
    # Normalize each frame's column averages between 0 and 1 (per frame, independently)
    print("Normalizing each frame's column averages between 0 and 1 (per frame)...")
    normalized_averages = normalize_per_frame(column_averages_spatial)
    print(f"Normalized averages shape: {normalized_averages.shape}")
    print(f"Normalized value range: [{normalized_averages.min():.4f}, {normalized_averages.max():.4f}]")
    
    # Compute patch grid dimensions
    num_patches = attention_map.shape[1]  # Get num_patches from attention map
    print(f"\nComputing patch grid dimensions for {num_patches} patches...")
    
    # Load image dimensions from inference/image_dimensions.json (required)
    print(f"\nLoading image dimensions from inference/image_dimensions.json...")
    try:
        eval_dims = load_image_dimensions_from_eval(directory, video_id)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find image_dimensions.json: {e}")
    except KeyError as e:
        raise KeyError(f"Video {video_id} not found in image_dimensions.json: {e}")
    
    # Extract dimensions
    original_image_height = eval_dims['original_image_height']
    original_image_width = eval_dims['original_image_width']
    padded_image_height = eval_dims.get('padded_image_height', original_image_height)
    padded_image_width = eval_dims.get('padded_image_width', original_image_width)
    H_patches = eval_dims['H_patches']
    W_patches = eval_dims['W_patches']
    pad_info = {
        'pad_top': eval_dims.get('pad_top', 0),
        'pad_left': eval_dims.get('pad_left', 0),
        'pad_bottom': eval_dims.get('pad_bottom', 0),
        'pad_right': eval_dims.get('pad_right', 0),
    }
    
    print(f"Found dimensions from image_dimensions.json:")
    print(f"  Original image: {original_image_height} x {original_image_width} pixels")
    print(f"  Padded image: {padded_image_height} x {padded_image_width} pixels")
    print(f"  Patch grid: {H_patches} x {W_patches} patches")
    print(f"  Padding: top={pad_info['pad_top']}, left={pad_info['pad_left']}, "
          f"bottom={pad_info['pad_bottom']}, right={pad_info['pad_right']}")
    
    # Verify dimensions match num_patches
    expected_patches = 1 + H_patches * W_patches  # 1 CLS token + spatial patches
    if expected_patches != num_patches:
        print(f"WARNING: Dimensions give {expected_patches} patches, but attention map has {num_patches} patches")
    
    # Load, resize, and pad all frames and create attention overlays
    print(f"\nLoading frame images and creating attention overlays...")
    
    # Find val.json in model directory
    val_json_path = find_val_json_in_model_dir(directory)
    print(f"Found val.json at: {val_json_path}")
    
    # Get video info from val.json
    video_info = get_video_info_from_val_json(val_json_path, video_id)
    file_names = video_info['file_names']
    num_frames = len(file_names)
    print(f"Found {num_frames} frames in video")
    
    # Determine image root directory based on model directory
    model_dir = os.path.dirname(val_json_path)
    image_root = get_image_root_directory(model_dir)
    print(f"Image root directory: {image_root}")
    
    # Output directory for plots
    output_dir = os.path.join(directory, "inference", "attention_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame
    print(f"\nProcessing {num_frames} frames...")
    for frame_idx in range(num_frames):
        frame_file = file_names[frame_idx]
        frame_path = os.path.join(image_root, frame_file)
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_idx} image not found: {frame_path}, skipping")
            continue
        
        # Load, resize, and pad the frame image
        img_processed = load_resize_and_pad_image(
            frame_path,
            original_image_height,
            original_image_width,
            pad_top=pad_info['pad_top'],
            pad_left=pad_info['pad_left'],
            pad_bottom=pad_info['pad_bottom'],
            pad_right=pad_info['pad_right']
        )
        
        # Get attention values for this frame (CLS token already removed)
        frame_attention = normalized_averages[frame_idx]  # Shape: [num_patches - 1]
        
        # Reshape attention to patch grid (H_patches x W_patches)
        attention_grid = frame_attention.reshape(H_patches, W_patches)
        
        # Create the plot with overlay
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Display the processed image
        ax.imshow(img_processed, aspect='auto')
        
        # Resize attention grid to match image dimensions for overlay
        img_height, img_width = img_processed.size[1], img_processed.size[0]  # PIL uses (width, height)
        attention_overlay = zoom(attention_grid, (img_height / H_patches, img_width / W_patches), order=1)
        
        # Overlay attention heatmap with transparency (transparent to red)
        # Use 'Reds' colormap and make alpha proportional to attention values
        # Low values (0) will be transparent, high values (1) will be opaque red
        im = ax.imshow(attention_overlay, cmap='Reds', alpha=attention_overlay, aspect='auto', 
                       interpolation='bilinear', vmin=0, vmax=1)
        
        ax.set_title(f'Video {video_id}, Frame {frame_idx}\nSpatial Attention Overlay', fontsize=12)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Attention')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f"video_{video_id}_frame_{frame_idx:05d}_attention.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
    
    print(f"\nSaved {num_frames} attention overlay plots to: {output_dir}")
    
    # Continue with patch coordinate mapping
    # Use original dimensions if available, otherwise use padded
    image_height = original_image_height if original_image_height is not None else padded_image_height
    image_width = original_image_width if original_image_width is not None else padded_image_width
    
    print(f"\nPatch grid: {H_patches} x {W_patches} patches")
    if original_image_height is not None:
        print(f"Image dimensions (for plotting): {image_height} x {image_width} pixels (original)")
        if padded_image_height != original_image_height or padded_image_width != original_image_width:
            print(f"  Note: Model used padded dimensions {padded_image_height} x {padded_image_width} pixels")
    else:
        print(f"Image dimensions: {image_height} x {image_width} pixels (padded)")
    print(f"Patch size: 16 x 16 pixels")
    
    # Test patch to image coordinate mapping for a few patches
    print(f"\nTesting patch to image coordinate mapping:")
    # Use padding offsets if available to map to original image
    pad_top = pad_info.get('pad_top', 0) if pad_info else 0
    pad_left = pad_info.get('pad_left', 0) if pad_info else 0
    coord_space = "original image" if (pad_top != 0 or pad_left != 0) or original_image_height is not None else "padded image"
    print(f"  Coordinates are in {coord_space} space")
    test_patches = [0, 1, 2, H_patches, H_patches * W_patches // 2, num_patches - 1]
    for patch_idx in test_patches:
        if patch_idx < num_patches:
            coords = patch_index_to_image_coords(
                patch_idx, H_patches, W_patches, patch_size=16, has_cls_token=True,
                pad_top=pad_top, pad_left=pad_left
            )
            if coords is None:
                print(f"  Patch {patch_idx}: CLS token (no spatial location)")
            else:
                x_min, y_min, x_max, y_max = coords
                print(f"  Patch {patch_idx}: image coords ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    print("\nFunction completed successfully. Processed first frame image saved.")


if __name__ == "__main__":
    main()


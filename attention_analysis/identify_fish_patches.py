import os
import json
import argparse
import numpy as np
from pycocotools import mask as mask_utils


def load_image_dimensions(image_dimensions_path):
    """
    Load image dimensions JSON file.
    
    Args:
        image_dimensions_path: Path to image_dimensions.json
    
    Returns:
        Dictionary mapping video_id to dimension info
    """
    with open(image_dimensions_path, 'r') as f:
        data = json.load(f)
    return data


def load_ytvis_annotations(ytvis_json_path):
    """
    Load YTVIS format JSON file with annotations.
    
    Args:
        ytvis_json_path: Path to YTVIS JSON file
    
    Returns:
        Dictionary with 'videos' and 'annotations' keys
    """
    with open(ytvis_json_path, 'r') as f:
        data = json.load(f)
    return data


def decode_rle_to_mask(rle, target_height, target_width):
    """
    Decode RLE segmentation to binary mask and resize.
    
    Args:
        rle: RLE segmentation dict with 'size' and 'counts' keys, or None
        target_height: Target height after resizing
        target_width: Target width after resizing
    
    Returns:
        Binary mask of shape (target_height, target_width), or None if rle is None
    """
    if rle is None:
        return None
    
    # Decode RLE to binary mask
    # RLE format: {'size': [height, width], 'counts': '...'}
    original_mask = mask_utils.decode(rle)  # Shape: (height, width)
    
    # Resize mask to target dimensions
    # Use nearest neighbor interpolation to preserve binary values
    from PIL import Image
    mask_pil = Image.fromarray(original_mask.astype(np.uint8) * 255)
    mask_resized = mask_pil.resize((target_width, target_height), Image.NEAREST)
    mask_resized = np.array(mask_resized) > 127  # Convert back to binary
    
    return mask_resized.astype(np.uint8)


def compute_patch_mask(mask, patch_size=16):
    """
    Compute which patches contain segmentation.
    
    Args:
        mask: Binary mask of shape (height, width)
        patch_size: Size of each patch (default 16)
    
    Returns:
        Binary array of shape (num_patches,) where 1 indicates patch contains segmentation
    """
    height, width = mask.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    # Initialize patch mask
    patch_mask = np.zeros(h_patches * w_patches, dtype=np.uint8)
    
    # For each patch, check if any pixel in the patch is part of segmentation
    for h_idx in range(h_patches):
        for w_idx in range(w_patches):
            patch_h_start = h_idx * patch_size
            patch_h_end = (h_idx + 1) * patch_size
            patch_w_start = w_idx * patch_size
            patch_w_end = (w_idx + 1) * patch_size
            
            # Extract patch region
            patch_region = mask[patch_h_start:patch_h_end, patch_w_start:patch_w_end]
            
            # Check if any pixel in patch is 1 (part of segmentation)
            if np.any(patch_region > 0):
                # Convert 2D patch index to 1D patch index
                # Patches are flattened row-major: patch_idx = h_idx * w_patches + w_idx
                patch_idx = h_idx * w_patches + w_idx
                patch_mask[patch_idx] = 1
    
    return patch_mask


def process_video_annotations(
    annotations,
    video_id,
    image_dimensions,
    patch_size=16
):
    """
    Process annotations for a single video to create patch mask matrix.
    
    Args:
        annotations: List of annotation dicts from YTVIS JSON
        video_id: Video ID to process
        image_dimensions: Dictionary from image_dimensions.json
        patch_size: Size of each patch (default 16)
    
    Returns:
        Matrix of shape (num_frames, num_patches) with 1 where patch contains segmentation
    """
    # Get dimension info for this video
    if str(video_id) not in image_dimensions:
        raise ValueError(f"Video ID {video_id} not found in image_dimensions.json")
    
    dim_info = image_dimensions[str(video_id)]
    target_height = dim_info['original_image_height']
    target_width = dim_info['original_image_width']
    h_patches = dim_info['H_patches']
    w_patches = dim_info['W_patches']
    num_patches = h_patches * w_patches
    
    # Find annotations for this video
    video_annotations = [ann for ann in annotations if ann['video_id'] == video_id]
    
    if len(video_annotations) == 0:
        raise ValueError(f"No annotations found for video ID {video_id}")
    
    # Get the first annotation (assuming all annotations for same video have same length)
    # We'll combine all annotations for the same video
    first_ann = video_annotations[0]
    num_frames = first_ann['length']
    
    # Initialize output matrix: (num_frames, num_patches)
    patch_matrix = np.zeros((num_frames, num_patches), dtype=np.uint8)
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Combine all annotations for this frame (union of all segmentations)
        frame_mask = None
        
        for ann in video_annotations:
            if frame_idx < len(ann['segmentations']):
                seg = ann['segmentations'][frame_idx]
                if seg is not None:
                    # Decode and resize segmentation
                    mask = decode_rle_to_mask(seg, target_height, target_width)
                    if mask is not None:
                        if frame_mask is None:
                            frame_mask = mask.copy()
                        else:
                            # Union: combine masks (OR operation)
                            frame_mask = np.logical_or(frame_mask, mask).astype(np.uint8)
        
        if frame_mask is not None:
            # Compute which patches contain segmentation
            patch_mask = compute_patch_mask(frame_mask, patch_size=patch_size)
            patch_matrix[frame_idx] = patch_mask
    
    return patch_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Identify patches containing fish segmentation"
    )
    parser.add_argument(
        "image_dimensions_json",
        type=str,
        help="Path to image_dimensions.json file"
    )
    parser.add_argument(
        "ytvis_json",
        type=str,
        help="Path to YTVIS format JSON file with annotations"
    )
    parser.add_argument(
        "--video-id",
        type=int,
        help="Video ID to process (if not provided, processes all videos in image_dimensions.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for patch matrix (.npz file). If not provided, saves as video_<id>_fish_patches.npz in the same directory as image_dimensions.json"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Size of each patch (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Load files
    print(f"Loading image dimensions from: {args.image_dimensions_json}")
    image_dimensions = load_image_dimensions(args.image_dimensions_json)
    
    print(f"Loading YTVIS annotations from: {args.ytvis_json}")
    ytvis_data = load_ytvis_annotations(args.ytvis_json)
    annotations = ytvis_data.get('annotations', [])
    
    # Determine which video IDs to process
    if args.video_id is not None:
        video_ids = [args.video_id]
    else:
        video_ids = [int(vid) for vid in image_dimensions.keys()]
    
    print(f"Processing {len(video_ids)} video(s)")
    print()
    
    # Process each video
    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")
        
        try:
            # Process annotations
            patch_matrix = process_video_annotations(
                annotations,
                video_id,
                image_dimensions,
                patch_size=args.patch_size
            )
            
            print(f"  Created patch matrix: shape {patch_matrix.shape}")
            print(f"  Patches with fish: {np.sum(patch_matrix)} / {patch_matrix.size} ({100 * np.sum(patch_matrix) / patch_matrix.size:.2f}%)")
            
            # Determine output path
            if args.output is not None:
                output_path = args.output
            else:
                # Save in the same directory as image_dimensions.json
                image_dimensions_dir = os.path.dirname(os.path.abspath(args.image_dimensions_json))
                output_filename = f"video_{video_id}_fish_patches.npz"
                output_path = os.path.join(image_dimensions_dir, output_filename)
            
            # Save matrix
            np.savez_compressed(
                output_path,
                patch_matrix=patch_matrix,
                video_id=video_id,
                num_frames=patch_matrix.shape[0],
                num_patches=patch_matrix.shape[1],
                patch_size=args.patch_size
            )
            
            print(f"  Saved to: {output_path}")
            print()
            
        except Exception as e:
            print(f"  ERROR processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("Done!")


if __name__ == "__main__":
    main()


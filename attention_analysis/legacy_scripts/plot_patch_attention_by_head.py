#!/usr/bin/env python3
"""
Plot patch-by-patch attention maps for each head across all layers for a specific frame.

For each of the 16 attention heads, creates a single PNG file containing 24 subplots
(one per layer) showing the patch-by-patch attention heatmap for the specified frame.
"""

import os
import argparse
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_video_for_frame(directory, frame_num):
    """
    Find which video and frame range contains the specified frame number.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder
        frame_num: Frame number to find
    
    Returns:
        (video_id, frame_range_str, frame_idx_in_array, frame_start) or None if not found
    """
    attention_maps_dir = os.path.join(directory, "attention_maps")
    
    # Look for all metadata files
    meta_files = glob.glob(os.path.join(attention_maps_dir, "*_backbone_vit_layer_*_attn.meta.json"))
    
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            frame_indices = meta.get('frame_idx', [])
            if frame_num in frame_indices:
                video_id = meta.get('video_id')
                if video_id is None:
                    # Try to extract from filename
                    match = re.search(r'video_(\d+)_frames', os.path.basename(meta_path))
                    if match:
                        video_id = int(match.group(1))
                    else:
                        continue
                
                # Extract frame range from filename
                match = re.search(r'frames(\d+)-(\d+)', os.path.basename(meta_path))
                if match:
                    frame_start = int(match.group(1))
                    frame_end = int(match.group(2))
                    frame_range_str = f"{frame_start}-{frame_end}"
                else:
                    frame_range_str = "unknown"
                    frame_start = frame_indices[0] if frame_indices else 0
                
                # Calculate array index: frame_num - frame_start
                frame_idx_in_array = frame_num - frame_start
                
                # Verify the index is valid
                if frame_idx_in_array < 0 or frame_idx_in_array >= len(frame_indices):
                    continue
                
                return video_id, frame_range_str, frame_idx_in_array, frame_start
        except Exception as e:
            continue
    
    return None


def load_image_dimensions(directory, video_id):
    """
    Load image and patch dimensions from image_dimensions.json.
    
    Args:
        directory: Path to directory
        video_id: Video ID number
    
    Returns:
        dict with H_patches, W_patches, etc.
    """
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


def load_attention_for_frame_and_head(directory, video_id, frame_range_str, frame_idx_in_array, layer_idx, head_idx):
    """
    Load attention map for a specific frame, layer, and head.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder
        video_id: Video ID number
        frame_range_str: Frame range string (e.g., "100-199")
        frame_idx_in_array: Index of frame in the attention array
        layer_idx: Layer index (0-23)
        head_idx: Head index (0-15)
    
    Returns:
        Attention matrix with shape [num_patches, num_patches] for the specified head
    """
    attention_maps_dir = os.path.join(directory, "attention_maps")
    npz_path = os.path.join(attention_maps_dir, 
                           f"video_{video_id}_frames{frame_range_str}_backbone_vit_layer_{layer_idx}_attn.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find attention file: {npz_path}")
    
    data = np.load(npz_path)
    
    # Get the attention weights array
    if 'attention_weights' in data:
        attn_array = data['attention_weights']
    else:
        # Try first key
        attn_array = data[list(data.keys())[0]]
    
    # Shape should be [num_frames, num_heads, num_patches, num_patches]
    if len(attn_array.shape) != 4:
        raise ValueError(f"Expected 4D array [num_frames, num_heads, num_patches, num_patches], got shape {attn_array.shape}")
    
    # Extract frame and head
    frame_attn = attn_array[frame_idx_in_array]  # Shape: [num_heads, num_patches, num_patches]
    head_attn = frame_attn[head_idx]  # Shape: [num_patches, num_patches]
    
    return head_attn


def plot_patch_attention_by_head(directory, frame_num, output_dir=None):
    """
    Plot patch-by-patch attention maps for each head across all layers.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder
        frame_num: Frame number to visualize
        output_dir: Output directory (default: directory/inference/attention_plots)
    """
    # Find video and frame info
    print(f"Finding video containing frame {frame_num}...")
    result = find_video_for_frame(directory, frame_num)
    if result is None:
        raise ValueError(f"Could not find frame {frame_num} in any video")
    
    video_id, frame_range_str, frame_idx_in_array, frame_start = result
    print(f"Found frame {frame_num} in video {video_id}, frame range {frame_range_str}, array index {frame_idx_in_array}")
    
    # Load image dimensions
    print(f"Loading image dimensions for video {video_id}...")
    dims = load_image_dimensions(directory, video_id)
    H_patches = dims['H_patches']
    W_patches = dims['W_patches']
    num_patches = H_patches * W_patches
    print(f"Patch grid: {H_patches} x {W_patches} = {num_patches} patches")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(directory, "inference", "attention_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load attention for all layers and heads
    print(f"Loading attention maps for all 24 layers...")
    
    # For each head (0-15), create a plot with 24 subplots (one per layer)
    num_heads = 16
    num_layers = 24
    
    for head_idx in range(num_heads):
        print(f"\nProcessing head {head_idx}...")
        
        # Create figure with 24 subplots (6 rows x 4 columns)
        fig, axes = plt.subplots(6, 4, figsize=(20, 30))
        fig.suptitle(f'Patch-by-Patch Attention Maps - Frame {frame_num}, Head {head_idx}\n'
                    f'Video {video_id}, All 24 Layers', fontsize=16, y=0.995)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for layer_idx in range(num_layers):
            try:
                # Load attention for this layer and head
                head_attn = load_attention_for_frame_and_head(
                    directory, video_id, frame_range_str, frame_idx_in_array, layer_idx, head_idx
                )
                
                # Remove CLS token if present (first patch)
                # Check if shape suggests CLS token (num_patches + 1)
                if head_attn.shape[0] == num_patches + 1:
                    # Remove CLS token (first row and column)
                    head_attn = head_attn[1:, 1:]
                elif head_attn.shape[0] != num_patches:
                    print(f"Warning: Layer {layer_idx} has {head_attn.shape[0]} patches, expected {num_patches} or {num_patches + 1}")
                
                # For patch-by-patch visualization, show the full attention matrix
                # head_attn has shape [num_patches, num_patches] where [i, j] is attention from patch i to patch j
                # We'll visualize this as a 2D heatmap
                
                # Plot the full attention matrix as a heatmap
                ax = axes_flat[layer_idx]
                im = ax.imshow(head_attn, cmap='viridis', aspect='auto', interpolation='nearest')
                ax.set_title(f'Layer {layer_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add colorbar for each subplot
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
            except Exception as e:
                print(f"Error loading layer {layer_idx} for head {head_idx}: {e}")
                # Create empty subplot
                ax = axes_flat[layer_idx]
                ax.text(0.5, 0.5, f'Layer {layer_idx}\nError', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        output_path = os.path.join(output_dir, f"video_{video_id}_frame_{frame_num:05d}_head_{head_idx:02d}_patch_attention.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved head {head_idx} plot to {output_path}")
    
    print(f"\nCompleted! Generated {num_heads} PNG files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot patch-by-patch attention maps for each head across all layers for a specific frame"
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs='?',
        default="/home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_240_115/attn_extract_3029_vid12",
        help="Path to directory containing 'attention_maps' folder (default: /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_240_115/attn_extract_3029_vid12)"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=116,
        help="Frame number to visualize (default: 116)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <directory>/inference/attention_plots)"
    )
    
    args = parser.parse_args()
    
    plot_patch_attention_by_head(args.directory, args.frame, args.output_dir)


if __name__ == "__main__":
    main()


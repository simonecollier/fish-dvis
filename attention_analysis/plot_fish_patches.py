import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_patch_matrix(npz_path):
    """
    Load the patch matrix from .npz file.
    
    Args:
        npz_path: Path to .npz file
    
    Returns:
        Dictionary with patch_matrix and metadata
    """
    data = np.load(npz_path)
    return {
        'patch_matrix': data['patch_matrix'],
        'video_id': int(data['video_id']),
        'num_frames': int(data['num_frames']),
        'num_patches': int(data['num_patches']),
        'patch_size': int(data['patch_size'])
    }


def load_image_dimensions(image_dimensions_path, video_id):
    """
    Load image dimensions for a specific video.
    
    Args:
        image_dimensions_path: Path to image_dimensions.json
        video_id: Video ID
    
    Returns:
        Dictionary with dimension info
    """
    with open(image_dimensions_path, 'r') as f:
        data = json.load(f)
    
    if str(video_id) not in data:
        raise ValueError(f"Video ID {video_id} not found in image_dimensions.json")
    
    return data[str(video_id)]


def plot_fish_patches(
    patch_matrix,
    frame_idx,
    image_dimensions,
    output_path=None
):
    """
    Plot patches containing fish for a specific frame.
    
    Args:
        patch_matrix: Array of shape (num_frames, num_patches)
        frame_idx: Frame index to plot
        image_dimensions: Dictionary with H_patches, W_patches, original_image_height, original_image_width, patch_size
        output_path: Optional path to save the plot
    """
    num_frames, num_patches = patch_matrix.shape
    
    if frame_idx >= num_frames:
        raise ValueError(f"Frame index {frame_idx} is out of range (max: {num_frames - 1})")
    
    # Extract patches for this frame
    frame_patches = patch_matrix[frame_idx]  # Shape: (num_patches,)
    
    # Get dimensions
    h_patches = image_dimensions['H_patches']
    w_patches = image_dimensions['W_patches']
    img_height = image_dimensions['original_image_height']
    img_width = image_dimensions['original_image_width']
    patch_size = image_dimensions['patch_size']
    
    # Verify dimensions match
    if h_patches * w_patches != num_patches:
        raise ValueError(f"Patch count mismatch: {h_patches} * {w_patches} = {h_patches * w_patches}, but matrix has {num_patches} patches")
    
    # Reshape to 2D patch grid
    patch_grid = frame_patches.reshape(h_patches, w_patches)  # Shape: (H_patches, W_patches)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Create a full-size image showing patch boundaries
    # Initialize with zeros (black background)
    vis_image = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Fill in patches that contain fish
    for h_idx in range(h_patches):
        for w_idx in range(w_patches):
            patch_h_start = h_idx * patch_size
            patch_h_end = (h_idx + 1) * patch_size
            patch_w_start = w_idx * patch_size
            patch_w_end = (w_idx + 1) * patch_size
            
            # Check if this patch contains fish
            if patch_grid[h_idx, w_idx] > 0:
                # Fill patch with white (or a color) to indicate fish presence
                vis_image[patch_h_start:patch_h_end, patch_w_start:patch_w_end] = 1.0
    
    # Display the image with explicit extent to match image dimensions exactly
    ax.imshow(vis_image, cmap='gray', origin='upper', vmin=0, vmax=1,
              extent=[0, img_width, img_height, 0], aspect='equal')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Reversed because origin='upper'
    ax.set_title(f'Frame {frame_idx} - Patches Containing Fish\n'
                 f'Image size: {img_width}x{img_height}, Patches: {h_patches}x{w_patches} ({patch_size}x{patch_size} each)',
                 fontsize=12)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    # Add grid lines to show patch boundaries
    for h_idx in range(h_patches + 1):
        y = h_idx * patch_size
        ax.axhline(y, color='red', linewidth=0.5, alpha=0.5)
    
    for w_idx in range(w_patches + 1):
        x = w_idx * patch_size
        ax.axvline(x, color='red', linewidth=0.5, alpha=0.5)
    
    # Add text annotations showing patch indices for patches with fish
    for h_idx in range(h_patches):
        for w_idx in range(w_patches):
            if patch_grid[h_idx, w_idx] > 0:
                # Center of patch
                center_x = w_idx * patch_size + patch_size / 2
                center_y = h_idx * patch_size + patch_size / 2
                # Add patch index
                patch_idx = h_idx * w_patches + w_idx
                ax.text(center_x, center_y, str(patch_idx), 
                       ha='center', va='center', 
                       color='yellow', fontsize=6, fontweight='bold')
    
    # Add statistics
    num_fish_patches = np.sum(frame_patches)
    total_patches = len(frame_patches)
    stats_text = f'Patches with fish: {num_fish_patches}/{total_patches} ({100 * num_fish_patches / total_patches:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio and remove extra padding
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot patches containing fish for a specific frame"
    )
    parser.add_argument(
        "patch_matrix_npz",
        type=str,
        help="Path to .npz file containing patch matrix"
    )
    parser.add_argument(
        "image_dimensions_json",
        type=str,
        help="Path to image_dimensions.json file"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=80,
        help="Frame index to plot (default: 80)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the plot. If not provided, displays interactively"
    )
    
    args = parser.parse_args()
    
    # Load patch matrix
    print(f"Loading patch matrix from: {args.patch_matrix_npz}")
    patch_data = load_patch_matrix(args.patch_matrix_npz)
    video_id = patch_data['video_id']
    patch_matrix = patch_data['patch_matrix']
    
    print(f"  Video ID: {video_id}")
    print(f"  Matrix shape: {patch_matrix.shape}")
    print(f"  Patch size: {patch_data['patch_size']}")
    
    # Load image dimensions
    print(f"\nLoading image dimensions from: {args.image_dimensions_json}")
    image_dimensions = load_image_dimensions(args.image_dimensions_json, video_id)
    
    print(f"  Image size: {image_dimensions['original_image_width']}x{image_dimensions['original_image_height']}")
    print(f"  Patches: {image_dimensions['W_patches']}x{image_dimensions['H_patches']}")
    
    # Determine output path
    if args.output is None:
        # Save in same directory as patch matrix
        patch_matrix_dir = os.path.dirname(os.path.abspath(args.patch_matrix_npz))
        output_filename = f"video_{video_id}_frame_{args.frame}_fish_patches.png"
        output_path = os.path.join(patch_matrix_dir, output_filename)
    else:
        output_path = args.output
    
    # Create plot
    print(f"\nPlotting frame {args.frame}...")
    plot_fish_patches(
        patch_matrix,
        args.frame,
        image_dimensions,
        output_path=output_path
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()


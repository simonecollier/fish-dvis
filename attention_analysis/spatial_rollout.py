import os
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def simulate_residual_connection(attn_matrix):
    """
    Simulate residual connection by adding identity matrix and row normalize.
    
    This simulates the residual connection in attention layers by adding the identity
    matrix to the attention matrix, then row normalizing so each row sums to 1.
    
    Args:
        attn_matrix: Single attention matrix with shape [num_patches, num_patches]
    
    Returns:
        Attention matrix with residual connection simulated, same shape [num_patches, num_patches]
    """
    num_patches = attn_matrix.shape[0]
    
    # Create identity matrix
    identity = np.eye(num_patches, dtype=attn_matrix.dtype)
    
    # Add identity matrix to simulate residual connection
    attn_with_residual = attn_matrix + identity
    
    # Row normalize: divide each row by its sum
    row_sums = attn_with_residual.sum(axis=1, keepdims=True)
    # Avoid division by zero (shouldn't happen after adding identity, but be safe)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    attn_with_residual /= row_sums
    
    return attn_with_residual


def parse_filename(filename):
    """
    Parse filename to extract video_id, frame range, and layer index.
    
    Example: "video_66_frames0-30_layer_backbone_vit_module_blocks_0_attn.npz"
    or: "video_66_frames0-30_layer_backbone_vit_module_blocks_0_attn_collapsed.npz"
    Returns: (video_id, frame_start, frame_end, layer_idx)
    """
    # Pattern: video_<id>_frames<start>-<end>_layer_backbone_vit_module_blocks_<layer>_attn[.npz or _collapsed.npz]
    pattern = r'video_(\d+)_frames(\d+)-(\d+)_layer_backbone_vit_module_blocks_(\d+)_attn(?:|_collapsed)\.npz'
    match = re.match(pattern, filename)
    
    if match:
        video_id = int(match.group(1))
        frame_start = int(match.group(2))
        frame_end = int(match.group(3))
        layer_idx = int(match.group(4))
        return video_id, frame_start, frame_end, layer_idx
    else:
        return None


def group_files_by_video_and_frames(spatial_files):
    """
    Group spatial attention map files by video_id and frame range.
    
    For each unique (video_id, frame_start, frame_end) combination, this function
    collects all 24 layers. This is necessary because videos are processed in windows
    (e.g., frames 0-30, 31-61, etc.), and each window has its own set of 24 layer files.
    
    Args:
        spatial_files: List of file paths to .npz files
    
    Returns:
        Dictionary: {(video_id, frame_start, frame_end): {layer_idx: file_path}}
        For each frame range, the dictionary contains all 24 layers (0-23)
    """
    groups = defaultdict(dict)
    
    for file_path in spatial_files:
        filename = os.path.basename(file_path)
        parsed = parse_filename(filename)
        
        if parsed is None:
            print(f"Warning: Could not parse filename {filename}, skipping")
            continue
        
        video_id, frame_start, frame_end, layer_idx = parsed
        # Group by (video_id, frame_start, frame_end) - this ensures we collect
        # all 24 layers for each specific frame range
        key = (video_id, frame_start, frame_end)
        groups[key][layer_idx] = file_path
    
    return groups


def collapse_heads_in_memory(attn_weights):
    """
    Collapse the num_heads dimension by averaging across heads in memory.
    
    Args:
        attn_weights: Attention weights with shape [num_frames, num_heads, num_patches, num_patches]
    
    Returns:
        Averaged attention map with shape [num_frames, num_patches, num_patches]
    """
    if len(attn_weights.shape) != 4:
        raise ValueError(f"Expected 4D array [num_frames, num_heads, num_patches, num_patches], got shape {attn_weights.shape}")
    
    # Average across the num_heads dimension (axis=1)
    return np.mean(attn_weights, axis=1)


def rollout_across_layers(layer_attentions, num_layers=24):
    """
    Rollout attention maps across layers by multiplying them in order.
    
    For each frame, takes the 433x433 matrix from each layer, adds identity and normalizes,
    then multiplies them in order: layer_23 * layer_22 * ... * layer_1 * layer_0
    
    Args:
        layer_attentions: Dictionary mapping layer_idx to attention array with shape [num_frames, num_patches, num_patches]
        num_layers: Expected number of layers (default 24)
    
    Returns:
        Final rolled out attention map with shape [num_frames, num_patches, num_patches]
    """
    # Check that we have all layers
    if len(layer_attentions) != num_layers:
        raise ValueError(f"Expected {num_layers} layers, found {len(layer_attentions)}")
    
    # Get dimensions from first layer
    num_frames, num_patches, _ = list(layer_attentions.values())[0].shape
    
    # For each frame, rollout across layers
    final_attn = np.zeros((num_frames, num_patches, num_patches), dtype=np.float32)
    
    for frame_idx in range(num_frames):
        # Start with identity matrix
        result = np.eye(num_patches, dtype=np.float32)
        
        # Multiply in order: A_23 @ A_22 @ ... @ A_1 @ A_0
        # Where A_0 = blocks_0, A_1 = blocks_1, ..., A_23 = blocks_23
        # So: blocks_23 @ blocks_22 @ ... @ blocks_1 @ blocks_0
        # We iterate from layer 0 to 23, multiplying left-to-right:
        #   Start: result = I (identity matrix)
        #   layer 0: result = A_0 @ I = A_0  (multiplying by I gives A_0)
        #   layer 1: result = A_1 @ A_0
        #   layer 2: result = A_2 @ A_1 @ A_0
        #   ...
        #   layer 23: result = A_23 @ A_22 @ ... @ A_1 @ A_0
        for layer_idx in range(num_layers):  # From 0 to 23
            # Get the attention matrix for this frame and layer
            attn_matrix = layer_attentions[layer_idx][frame_idx]
            
            # Simulate residual connection (add identity and normalize)
            attn_rolled = simulate_residual_connection(attn_matrix)
            
            # Multiply: result = attn_rolled @ result
            # This builds up: A_0, then A_1 @ A_0, ..., finally A_23 @ ... @ A_1 @ A_0
            result = attn_rolled @ result
        
        # Row normalize the final result
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        result /= row_sums
        
        final_attn[frame_idx] = result
    
    return final_attn


def process_window(group_files, attention_maps_dir, num_layers=24):
    """
    Process a single window (frame range): load raw files, collapse heads in memory, rollout, and save.
    
    Args:
        group_files: Dictionary mapping layer_idx to file_path for a video/frame range
        attention_maps_dir: Directory where attention maps are stored
        num_layers: Expected number of layers (default 24)
    
    Returns:
        True if successful, False otherwise
    """
    video_id, frame_start, frame_end = None, None, None
    
    # Load all raw layers and collapse heads in memory
    layer_attentions = {}
    num_frames = None
    num_patches = None
    
    for layer_idx in range(num_layers):
        if layer_idx not in group_files:
            raise ValueError(f"Missing layer {layer_idx}")
        
        file_path = group_files[layer_idx]
        
        # Extract video_id, frame_start, frame_end from first file
        if video_id is None:
            filename = os.path.basename(file_path)
            parsed = parse_filename(filename)
            if parsed is None:
                raise ValueError(f"Could not parse filename: {filename}")
            video_id, frame_start, frame_end, _ = parsed
        
        # Load raw attention weights
        data = np.load(file_path)
        
        if 'attention_weights' not in data.keys():
            raise ValueError(f"Expected 'attention_weights' key in {file_path}")
        
        attn_weights = data['attention_weights']
        
        # Check if already collapsed (3D) or raw (4D)
        if len(attn_weights.shape) == 4:
            # Raw file: collapse heads in memory
            attn_collapsed = collapse_heads_in_memory(attn_weights)
        elif len(attn_weights.shape) == 3:
            # Already collapsed: use as is
            attn_collapsed = attn_weights
        else:
            raise ValueError(f"Unexpected shape: {attn_weights.shape}")
        
        # Check dimensions
        if num_frames is None:
            num_frames, num_patches, _ = attn_collapsed.shape
        else:
            if attn_collapsed.shape != (num_frames, num_patches, num_patches):
                raise ValueError(f"Shape mismatch: layer {layer_idx} has shape {attn_collapsed.shape}, expected ({num_frames}, {num_patches}, {num_patches})")
        
        layer_attentions[layer_idx] = attn_collapsed
        
        # Free memory from raw data
        del attn_weights, data
    
    # Rollout across layers
    final_attn = rollout_across_layers(layer_attentions, num_layers=num_layers)
    
    # Free memory from layer attentions
    del layer_attentions
    
    # Verify rows sum to 1
    all_rows_sum_to_one = True
    for frame_idx in range(final_attn.shape[0]):
        row_sums = final_attn[frame_idx].sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            all_rows_sum_to_one = False
            break
    
    # Save the rolled out attention maps
    output_filename = f"video_{video_id}_frames{frame_start}-{frame_end}_backbone_vit_rolled_out.npz"
    output_path = os.path.join(attention_maps_dir, output_filename)
    np.savez_compressed(output_path, attention_weights=final_attn)
    
    # Free memory
    del final_attn
    
    return video_id, frame_start, frame_end, all_rows_sum_to_one


def process_directory(directory):
    """
    Process all spatial attention maps in a directory.
    
    Processes one window (frame range) at a time:
    1. Load all 24 raw layer files for that window
    2. Collapse heads in memory (no intermediate files)
    3. Rollout across layers
    4. Save individual window result
    5. Free memory and move to next window
    
    At the end, combines all windows into a single file and deletes individual window files.
    
    Args:
        directory: Path to the directory containing attention_maps folder
    """
    # Find the attention_maps folder
    attention_maps_dir = os.path.join(directory, "attention_maps")
    
    if not os.path.exists(attention_maps_dir):
        raise FileNotFoundError(f"Could not find 'attention_maps' folder in {directory}")
    
    print(f"Processing spatial attention maps in: {attention_maps_dir}")
    print()
    
    # Find all original .npz files (raw or collapsed)
    original_files = []
    for filename in os.listdir(attention_maps_dir):
        if (filename.endswith('.npz') and 
            'layer_backbone_vit_module_blocks' in filename and 
            '_collapsed' not in filename and
            '_rolled_out' not in filename):
            original_files.append(os.path.join(attention_maps_dir, filename))
    
    if len(original_files) == 0:
        print(f"No original spatial attention maps found (files with 'layer_backbone_vit_module_blocks' in name)")
        return
    
    print(f"Found {len(original_files)} spatial attention map file(s)")
    print()
    
    # Group files by video_id and frame range
    groups = group_files_by_video_and_frames(original_files)
    
    print(f"Found {len(groups)} window(s) (video_id, frame_range)")
    print()
    
    # Process each window one at a time
    successful = 0
    failed = 0
    window_files = []  # Track individual window files to combine later
    video_id = None
    
    for group_key, group_files in sorted(groups.items()):
        try:
            print(f"Processing window: {group_key}")
            print(f"  Layers found: {sorted(group_files.keys())}")
            
            # Check if we have all 24 layers
            if len(group_files) != 24:
                print(f"  WARNING: Expected 24 layers, found {len(group_files)}. Skipping this window.")
                failed += 1
                continue
            
            # Process window: load, collapse, rollout, save
            window_video_id, frame_start, frame_end, all_rows_sum_to_one = process_window(
                group_files, attention_maps_dir, num_layers=24
            )
            
            # Track video_id (should be same for all windows)
            if video_id is None:
                video_id = window_video_id
            elif video_id != window_video_id:
                print(f"  WARNING: Multiple video IDs found ({video_id} and {window_video_id})")
            
            print(f"  Rolled out across layers -> shape (num_frames, num_patches, num_patches)")
            
            if all_rows_sum_to_one:
                print(f"  ✓ All rows sum to 1 for all frames")
            else:
                print(f"  ✗ WARNING: Some rows do not sum to 1")
            
            output_filename = f"video_{window_video_id}_frames{frame_start}-{frame_end}_backbone_vit_rolled_out.npz"
            window_file_path = os.path.join(attention_maps_dir, output_filename)
            window_files.append((window_file_path, frame_start, frame_end))
            print(f"  Saved rolled out attention maps to {output_filename}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ERROR processing window: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
        print()
    
    print(f"Completed: {successful} window(s) processed successfully, {failed} window(s) failed")
    print()
    
    # Combine all windows into a single file
    if successful > 0 and video_id is not None:
        print(f"Combining {len(window_files)} window(s) into a single file...")
        print()
        
        # Sort windows by frame_start to ensure correct order
        window_files.sort(key=lambda x: x[1])
        
        # Load and concatenate all windows
        all_attentions = []
        for window_file_path, frame_start, frame_end in window_files:
            if not os.path.exists(window_file_path):
                print(f"  WARNING: Window file not found: {os.path.basename(window_file_path)}, skipping")
                continue
            
            data = np.load(window_file_path)
            attn = data['attention_weights']
            all_attentions.append(attn)
            print(f"  Loaded window frames {frame_start}-{frame_end}: shape {attn.shape}")
        
        if len(all_attentions) > 0:
            # Concatenate along frame dimension
            combined_attn = np.concatenate(all_attentions, axis=0)
            print(f"  Combined shape: {combined_attn.shape}")
            
            # Save combined file
            combined_filename = f"video_{video_id}_backbone_vit_rolled_out.npz"
            combined_path = os.path.join(attention_maps_dir, combined_filename)
            np.savez_compressed(combined_path, attention_weights=combined_attn)
            print(f"  Saved combined file: {combined_filename}")
            print()
            
            # Delete individual window files
            print(f"Deleting {len(window_files)} individual window file(s)...")
            for window_file_path, frame_start, frame_end in window_files:
                try:
                    if os.path.exists(window_file_path):
                        os.remove(window_file_path)
                        print(f"  Deleted: {os.path.basename(window_file_path)}")
                except Exception as e:
                    print(f"  WARNING: Could not delete {os.path.basename(window_file_path)}: {e}")
            
            print()
            print(f"Final combined file: {combined_filename} with shape {combined_attn.shape}")
        else:
            print("  WARNING: No windows to combine")


def main():
    parser = argparse.ArgumentParser(
        description="Process spatial attention maps: collapse heads and rollout across layers"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing 'attention_maps' folder (e.g., /path/to/attn_extract_3231)"
    )
    
    args = parser.parse_args()
    
    process_directory(args.directory)


if __name__ == "__main__":
    main()


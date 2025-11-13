import os
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def remove_cls_token(attn_weights):
    """
    Remove CLS token from attention matrices by removing first row and first column,
    then row-normalize so each row sums to 1.0.
    
    In ViT backbones, the CLS token is prepended at index 0, so it occupies:
    - First row (index 0) of the attention matrix
    - First column (index 0) of the attention matrix
    
    After removing the CLS token column, the remaining rows no longer sum to 1.0
    (since we removed the attention weight allocated to the CLS token). This function
    re-normalizes each row so they sum to 1.0 again.
    
    Args:
        attn_weights: Attention weights with shape:
            - [num_frames, num_heads, num_patches, num_patches] (4D, with heads)
            - [num_frames, num_patches, num_patches] (3D, heads already collapsed)
    
    Returns:
        Attention weights with CLS token removed and row-normalized, shape:
            - [num_frames, num_heads, num_patches-1, num_patches-1] (4D)
            - [num_frames, num_patches-1, num_patches-1] (3D)
    """
    if len(attn_weights.shape) == 4:
        # Shape: [num_frames, num_heads, num_patches, num_patches]
        # Remove first row and first column (CLS token at index 0)
        attn_no_cls = attn_weights[:, :, 1:, 1:]
        
        # Row normalize: for each frame, head, and row, divide by row sum
        # Sum over last dimension (columns) for each row
        row_sums = attn_no_cls.sum(axis=-1, keepdims=True)  # Shape: [num_frames, num_heads, num_patches-1, 1]
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        attn_normalized = attn_no_cls / row_sums
        
        return attn_normalized
    elif len(attn_weights.shape) == 3:
        # Shape: [num_frames, num_patches, num_patches]
        # Remove first row and first column (CLS token at index 0)
        attn_no_cls = attn_weights[:, 1:, 1:]
        
        # Row normalize: for each frame and row, divide by row sum
        # Sum over last dimension (columns) for each row
        row_sums = attn_no_cls.sum(axis=-1, keepdims=True)  # Shape: [num_frames, num_patches-1, 1]
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        attn_normalized = attn_no_cls / row_sums
        
        return attn_normalized
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {attn_weights.shape}")


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
    
    Supports both old and new naming formats:
    - Old: "video_66_frames0-30_layer_backbone_vit_module_blocks_0_attn.npz"
    - New: "video_66_frames0-30_backbone_vit_layer_0_attn.npz"
    - Also supports: "video_66_frames0-30_layer_backbone_vit_module_blocks_0_attn_collapsed.npz"
    
    Returns: (video_id, frame_start, frame_end, layer_idx)
    """
    # Try new format first: video_<id>_frames<start>-<end>_backbone_vit_layer_<layer>_attn[.npz or _collapsed.npz]
    new_pattern = r'video_(\d+)_frames(\d+)-(\d+)_backbone_vit_layer_(\d+)_attn(?:|_collapsed)\.npz'
    match = re.match(new_pattern, filename)
    
    if match:
        video_id = int(match.group(1))
        frame_start = int(match.group(2))
        frame_end = int(match.group(3))
        layer_idx = int(match.group(4))
        return video_id, frame_start, frame_end, layer_idx
    
    # Try old format: video_<id>_frames<start>-<end>_layer_backbone_vit_module_blocks_<layer>_attn[.npz or _collapsed.npz]
    old_pattern = r'video_(\d+)_frames(\d+)-(\d+)_layer_backbone_vit_module_blocks_(\d+)_attn(?:|_collapsed)\.npz'
    match = re.match(old_pattern, filename)
    
    if match:
        video_id = int(match.group(1))
        frame_start = int(match.group(2))
        frame_end = int(match.group(3))
        layer_idx = int(match.group(4))
        return video_id, frame_start, frame_end, layer_idx
    
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


def collapse_heads_with_weights(attn_weights, weights):
    """
    Collapse the num_heads dimension by weighted sum using entropy-based weights.
    
    For each frame, computes: weighted_attn[frame] = sum_head(weights[frame, head] * attn_weights[frame, head, :, :])
    
    Args:
        attn_weights: Attention weights with shape [num_frames, num_heads, num_patches, num_patches]
        weights: Weights with shape [num_frames, num_heads] (e.g., from compute_attention_head_entropy)
                 Should sum to 1.0 across heads for each frame
    
    Returns:
        Weighted attention map with shape [num_frames, num_patches, num_patches]
    """
    if len(attn_weights.shape) != 4:
        raise ValueError(f"Expected 4D array [num_frames, num_heads, num_patches, num_patches], got shape {attn_weights.shape}")
    
    if len(weights.shape) != 2:
        raise ValueError(f"Expected 2D array [num_frames, num_heads], got shape {weights.shape}")
    
    if attn_weights.shape[:2] != weights.shape:
        raise ValueError(f"Shape mismatch: attn_weights has shape {attn_weights.shape[:2]}, weights has shape {weights.shape}")
    
    # Weighted sum: for each frame, sum over heads with weights
    # weights[:, :, None, None] expands to [num_frames, num_heads, 1, 1] for broadcasting
    weighted_attn = np.sum(attn_weights * weights[:, :, None, None], axis=1)
    
    return weighted_attn


def compute_attention_head_entropy(attn_weights, remove_cls=True):
    """
    Compute entropy for each attention head and convert to weights using softmax(-entropy).
    
    For each (frame, head) combination:
    1. Compute row-wise entropy: for each row i, entropy_i = -sum_j(a_ij * log(a_ij))
       where a_ij is the attention weight at row i, column j (treating 0 * log(0) = 0)
    2. Average across all rows to get the entropy for that head
    3. Apply softmax(-entropy) across heads for each frame to get weights
    
    Args:
        attn_weights: Attention weights with shape [num_frames, num_heads, num_patches, num_patches]
                     Rows should already be normalized (sum to 1.0)
        remove_cls: If True, remove CLS token (first row and column) before computation
    
    Returns:
        entropy: Array with shape [num_frames, num_heads] containing entropy for each head
        weights: Array with shape [num_frames, num_heads] containing softmax(-entropy) weights
                 (normalized so each frame sums to 1.0 across heads)
    """
    if len(attn_weights.shape) != 4:
        raise ValueError(f"Expected 4D array [num_frames, num_heads, num_patches, num_patches], got shape {attn_weights.shape}")
    
    num_frames, num_heads, num_patches, _ = attn_weights.shape
    
    # Optionally remove CLS token
    if remove_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]  # Remove first row and column
        num_patches = num_patches - 1
    
    # Ensure rows are normalized (should already be, but be safe)
    row_sums = attn_weights.sum(axis=-1, keepdims=True)  # Shape: [num_frames, num_heads, num_patches, 1]
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    attn_normalized = attn_weights / row_sums
    
    # Compute entropy for each (frame, head)
    # For each row i: entropy_i = -sum_j(a_ij * log(a_ij))
    # We'll compute this for all rows at once using vectorized operations
    
    # Compute log of attention weights, handling zeros (0 * log(0) = 0)
    # Use np.log with small epsilon to avoid log(0), then multiply by attn_normalized
    # This automatically handles the 0 * log(0) = 0 case
    epsilon = 1e-10
    log_attn = np.log(attn_normalized + epsilon)  # Shape: [num_frames, num_heads, num_patches, num_patches]
    
    # Compute -a_ij * log(a_ij) for all i, j
    entropy_per_row = -attn_normalized * log_attn  # Shape: [num_frames, num_heads, num_patches, num_patches]
    
    # Sum over columns (axis=-1) to get entropy for each row
    # Shape: [num_frames, num_heads, num_patches]
    row_entropies = entropy_per_row.sum(axis=-1)
    
    # Average across rows (axis=-1) to get entropy for each head
    # Shape: [num_frames, num_heads]
    entropy = row_entropies.mean(axis=-1)
    
    # Compute weights using softmax(-entropy) across heads for each frame
    # softmax(-entropy) = exp(-entropy) / sum(exp(-entropy))
    # Use log-sum-exp trick for numerical stability
    neg_entropy = -entropy  # Shape: [num_frames, num_heads]
    # Subtract max for numerical stability
    neg_entropy_shifted = neg_entropy - neg_entropy.max(axis=-1, keepdims=True)
    exp_neg_entropy = np.exp(neg_entropy_shifted)
    weights = exp_neg_entropy / exp_neg_entropy.sum(axis=-1, keepdims=True)
    
    return entropy, weights


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
        
        final_attn[frame_idx] = result
    
    return final_attn


def process_window(group_files, attention_maps_dir, num_layers=24):
    """
    Process a single window (frame range): load raw files, collapse heads using entropy-based weights, rollout, and save.
    
    For each layer, computes entropy-based weights for each head and uses weighted sum instead of simple averaging
    when collapsing heads. Heads with lower entropy (more focused attention) receive higher weights.
    
    Args:
        group_files: Dictionary mapping layer_idx to file_path for a video/frame range
        attention_maps_dir: Directory where attention maps are stored
        num_layers: Expected number of layers (default 24)
    
    Returns:
        video_id, frame_start, frame_end
    """
    video_id, frame_start, frame_end = None, None, None
    
    # Load all raw layers and collapse heads using entropy-based weights
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
        
        # Remove CLS token as the first step (remove first row and first column)
        attn_weights = remove_cls_token(attn_weights)
        
        # Check if already collapsed (3D) or raw (4D)
        if len(attn_weights.shape) == 4:
            # Raw file: collapse heads using entropy-based weights
            # Compute entropy and weights for this layer
            _, head_weights = compute_attention_head_entropy(attn_weights, remove_cls=False)
            # Note: remove_cls=False because we already removed CLS token above
            
            # Collapse heads using weighted sum
            attn_collapsed = collapse_heads_with_weights(attn_weights, head_weights)
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
    
    # Save the rolled out attention maps in rolled_out subdirectory
    rolled_out_dir = os.path.join(attention_maps_dir, "rolled_out")
    os.makedirs(rolled_out_dir, exist_ok=True)
    output_filename = f"video_{video_id}_frames{frame_start}-{frame_end}_backbone_vit_rolled_out.npz"
    output_path = os.path.join(rolled_out_dir, output_filename)
    np.savez_compressed(output_path, attention_weights=final_attn)
    
    # Free memory
    del final_attn
    
    return video_id, frame_start, frame_end


def process_directory(directory):
    """
    Process all spatial attention maps in a directory.
    
    Processes one window (frame range) at a time:
    1. Load all 24 raw layer files for that window
    2. Collapse heads using entropy-based weights (no intermediate files)
       - Computes entropy for each head and uses softmax(-entropy) as weights
       - Heads with lower entropy (more focused) receive higher weights
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
    # Support both old and new naming formats
    original_files = []
    for filename in os.listdir(attention_maps_dir):
        if (filename.endswith('.npz') and 
            ('backbone_vit_layer' in filename or 'layer_backbone_vit_module_blocks' in filename) and 
            '_collapsed' not in filename and
            '_rolled_out' not in filename):
            original_files.append(os.path.join(attention_maps_dir, filename))
    
    if len(original_files) == 0:
        print(f"No original spatial attention maps found (files with 'backbone_vit_layer' or 'layer_backbone_vit_module_blocks' in name)")
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
            window_video_id, frame_start, frame_end = process_window(
                group_files, attention_maps_dir, num_layers=24
            )
            
            # Track video_id (should be same for all windows)
            if video_id is None:
                video_id = window_video_id
            elif video_id != window_video_id:
                print(f"  WARNING: Multiple video IDs found ({video_id} and {window_video_id})")
            
            print(f"  Rolled out across layers -> shape (num_frames, num_patches, num_patches)")
            
            rolled_out_dir = os.path.join(attention_maps_dir, "rolled_out")
            output_filename = f"video_{window_video_id}_frames{frame_start}-{frame_end}_backbone_vit_rolled_out.npz"
            window_file_path = os.path.join(rolled_out_dir, output_filename)
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
            
            # Save combined file in rolled_out subdirectory
            rolled_out_dir = os.path.join(attention_maps_dir, "rolled_out")
            os.makedirs(rolled_out_dir, exist_ok=True)
            combined_filename = f"video_{video_id}_backbone_vit_rolled_out.npz"
            combined_path = os.path.join(rolled_out_dir, combined_filename)
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
        description="Process spatial attention maps: collapse heads using entropy-based weights and rollout across layers"
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


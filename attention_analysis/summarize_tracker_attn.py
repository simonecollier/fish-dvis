#!/usr/bin/env python3
"""
Plot tracker cross-attention maps for the top-scoring prediction per video.

This script:
1. Loads results_temporal.json and finds the top-scoring prediction for each video_id
2. Maps refiner_id to sequence ID using refiner_id_mappings.json
3. Finds the tracker_id corresponding to that sequence ID
4. Extracts attention rows for all layers (0-5) for a specified frame
5. Creates line plots showing attention patterns across query indices
"""

import os
import json
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def load_results_temporal(run_dir):
    """Load results_temporal.json from the run directory."""
    inf_dir = os.path.join(run_dir, "inference")
    results_path = os.path.join(inf_dir, "results_temporal.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results_temporal.json at {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)


def pick_top_predictions_by_video(predictions):
    """Find the top-scoring prediction for each video_id."""
    top_by_video = {}
    for pred in predictions:
        vid = pred.get("video_id")
        score = float(pred.get("score", -1.0))
        if vid is None:
            continue
        if vid not in top_by_video or score > top_by_video[vid]["score"]:
            top_by_video[vid] = pred
    return top_by_video


def load_refiner_id_mappings(run_dir):
    """Load refiner_id_mappings.json from the run directory."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    mappings_path = os.path.join(attn_dir, "refiner_id_mappings.json")
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Could not find refiner_id_mappings.json at {mappings_path}")
    with open(mappings_path, "r") as f:
        return json.load(f)


def get_sequence_id_from_refiner_id(mappings, video_id, refiner_id):
    """
    Extract sequence ID for a given refiner_id.
    
    Note: The frame_range key is not accurate, but sequence IDs are consistent
    across the entire video, so we can use any available frame range.
    """
    refiner_maps = mappings.get("_refiner_id_to_seq_id_maps", {})
    video_maps = refiner_maps.get(str(video_id), {})
    
    if not video_maps:
        raise ValueError(f"No mappings found for video_id {video_id}")
    
    # Use the first available frame range (they're all the same)
    frame_range_key = list(video_maps.keys())[0]
    seq_id_list = video_maps[frame_range_key]
    
    # Ensure refiner_id is an integer
    refiner_id = int(refiner_id)
    
    if refiner_id >= len(seq_id_list):
        raise ValueError(f"refiner_id {refiner_id} is out of range (max: {len(seq_id_list) - 1})")
    
    return int(seq_id_list[refiner_id])


def load_tracker_meta(run_dir, video_id, frame):
    """Load tracker cross-attention metadata JSON for a specific frame and layer 0."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    meta_path = os.path.join(attn_dir, f"video_{video_id}_frame_{frame}_tracker_cross_layer_0_attn.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find tracker meta file at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def find_tracker_id_from_sequence_id(meta, sequence_id):
    """
    Find the tracker_id that corresponds to the given sequence ID.
    
    Returns the tracker_id (as integer) if found, None otherwise.
    """
    tracker_id_to_seq_id = meta.get("tracker_id_to_seq_id")
    
    if tracker_id_to_seq_id is None:
        raise ValueError("tracker_id_to_seq_id is null in metadata (this may occur for frame 0)")
    
    # Ensure sequence_id is an integer for comparison
    sequence_id = int(sequence_id)
    
    # Search for the key (tracker_id) whose value matches the sequence ID
    for tracker_id_str, seq_id_value in tracker_id_to_seq_id.items():
        if int(seq_id_value) == sequence_id:
            return int(tracker_id_str)
    
    return None


def load_tracker_attention_npz(run_dir, video_id, frame, layer):
    """Load tracker cross-attention NPZ file for a specific frame and layer."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    npz_path = os.path.join(attn_dir, f"video_{video_id}_frame_{frame}_tracker_cross_layer_{layer}_attn.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find attention NPZ file at {npz_path}")
    data = np.load(npz_path)
    return data['attention_weights']


def extract_attention_row(attention_weights, tracker_id):
    """
    Extract a single row from the attention matrix.
    
    Args:
        attention_weights: Array of shape [1, 200, 200]
        tracker_id: Index into the second dimension (0-199)
    
    Returns:
        1D array of shape (200,) containing the attention row
    """
    if attention_weights.shape != (1, 200, 200):
        raise ValueError(f"Expected shape (1, 200, 200), got {attention_weights.shape}")
    
    # Ensure tracker_id is a Python int (not numpy int) for indexing
    tracker_id = int(tracker_id)
    
    if tracker_id < 0 or tracker_id >= 200:
        raise ValueError(f"tracker_id {tracker_id} is out of range (0-199)")
    
    return attention_weights[0, tracker_id, :]


def find_available_frames(run_dir, video_id):
    """Find all available frames for a video by checking for layer 0 meta files."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    pattern = os.path.join(attn_dir, f"video_{video_id}_frame_*_tracker_cross_layer_0_attn.meta.json")
    files = glob.glob(pattern)
    frames = []
    for f in files:
        # Extract frame number from filename
        match = re.search(rf'video_{video_id}_frame_(\d+)_tracker_cross_layer_0_attn\.meta\.json', f)
        if match:
            frames.append(int(match.group(1)))
    return sorted(frames)


def process_frame(run_dir, video_id, refiner_id, sequence_id, tracker_id, frame, plot_dir, create_plots=False):
    """
    Process a single frame: extract attention, find top 5 indices, and optionally plot.
    
    Args:
        run_dir: Path to attention analysis directory
        video_id: Video ID
        refiner_id: Refiner ID
        sequence_id: Sequence ID
        tracker_id: Tracker ID
        frame: Frame number
        plot_dir: Directory to save plots (only used if create_plots=True)
        create_plots: Whether to create plots (default: False)
    
    Returns:
        dict with top5 data and full attention vectors for each layer, or None if frame couldn't be processed
    """
    try:
        # Extract attention rows for all layers
        attention_rows = {}
        for layer in range(6):  # Layers 0-5
            try:
                attention_weights = load_tracker_attention_npz(run_dir, video_id, frame, layer)
                attention_rows[layer] = extract_attention_row(attention_weights, tracker_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}, skipping layer {layer}")
                continue
        
        if not attention_rows:
            return None
        
        # Find top 5 indices for each layer
        frame_data = {}
        top_indices_per_layer = []
        
        # Calculate element-wise sum across all layers
        attention_sum_across_layers = np.zeros(200, dtype=np.float32)
        
        for layer_idx, attention_row in sorted(attention_rows.items()):
            top5_indices = np.argsort(attention_row)[-5:][::-1]  # Get top 5, descending order
            top5_values = attention_row[top5_indices].tolist()
            
            frame_data[layer_idx] = {
                'top5_indices': top5_indices.tolist(),
                'top5_values': top5_values,
                'attention_weights': attention_row.tolist()  # Full attention vector for this layer
            }
            
            # Accumulate attention row to sum
            attention_sum_across_layers += attention_row
            
            # Store top index for comparison
            top_indices_per_layer.append(int(top5_indices[0]))
        
        # Store the element-wise sum across all layers
        frame_data['sum_across_layers'] = {
            'attention_sum': attention_sum_across_layers.tolist()
        }
        
        # Only create plot if requested and layers differ in top index
        if create_plots:
            # Check if all layers have the same top index
            all_same_top = len(set(top_indices_per_layer)) == 1
            
            if not all_same_top:
                # Create visualization
                num_layers = len(attention_rows)
                
                # Arrange subplots: 2 rows, 3 columns
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for layer_idx, attention_row in sorted(attention_rows.items()):
                    ax = axes[layer_idx]
                    
                    # Create x-axis (query indices 0-199)
                    x = np.arange(200)
                    
                    # Plot line
                    ax.plot(x, attention_row, linewidth=1.5)
                    
                    # Get top 5 indices and values
                    top5_indices = frame_data[layer_idx]['top5_indices']
                    top5_values = frame_data[layer_idx]['top5_values']
                    
                    # Annotate top 5 points
                    for idx, val in zip(top5_indices, top5_values):
                        ax.plot(idx, val, 'ro', markersize=8)  # Mark with red dot
                        ax.annotate(
                            f'{idx}\n{val:.4f}',
                            xy=(idx, val),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
                    
                    ax.set_title(f"Layer {layer_idx}", fontsize=12)
                    ax.set_xlabel("Query Index", fontsize=10)
                    ax.set_ylabel("Attention", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 199)
                
                # Hide unused subplots
                for idx in range(num_layers, 6):
                    axes[idx].set_visible(False)
                
                # Add overall title
                fig.suptitle(
                    f"Tracker Cross-Attention: Video {video_id}, Frame {frame}\n"
                    f"Refiner ID: {refiner_id}, Sequence ID: {sequence_id}, Tracker ID: {tracker_id}",
                    fontsize=14
                )
                
                plt.tight_layout()
                
                # Save plot
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"tracker_attn_video_{video_id}_frame_{frame}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {plot_path}")
                plt.close()
            else:
                print(f"Frame {frame}: All layers have same top index ({top_indices_per_layer[0]}), skipping plot")
        
        return frame_data
        
    except Exception as e:
        print(f"Error processing frame {frame}: {e}")
        return None


def plot_tracker_attention(run_dir, output_json_path=None, create_plots=False):
    """
    Main function to process all frames and plot tracker attention maps.
    
    Args:
        run_dir: Path to attention analysis directory
        output_json_path: Optional path for JSON output (default: tracker_attention_top5.json in run_dir/inference)
        create_plots: Whether to create plots (default: False)
    """
    # Load results and find top predictions
    predictions = load_results_temporal(run_dir)
    top_by_video = pick_top_predictions_by_video(predictions)
    
    if not top_by_video:
        raise ValueError("No predictions found in results_temporal.json")
    
    # Load refiner ID mappings
    mappings = load_refiner_id_mappings(run_dir)
    
    # Set default JSON output path
    if output_json_path is None:
        inf_dir = os.path.join(run_dir, "inference")
        output_json_path = os.path.join(inf_dir, "tracker_attention_top5.json")
    
    # Set plot directory
    plot_dir = os.path.join(run_dir, "inference", "tracker_attention_plots")
    
    # Process each video
    all_results = {}
    
    for video_id, top_pred in top_by_video.items():
        refiner_id = top_pred.get("refiner_id")
        if refiner_id is None:
            print(f"Warning: No refiner_id found for video {video_id}, skipping")
            continue
        
        # Ensure refiner_id is an integer
        try:
            refiner_id = int(refiner_id)
        except (ValueError, TypeError):
            print(f"Warning: Invalid refiner_id {refiner_id} for video {video_id}, skipping")
            continue
        
        try:
            # Get sequence ID from refiner_id
            sequence_id = get_sequence_id_from_refiner_id(mappings, video_id, refiner_id)
            
            # Find all available frames
            available_frames = find_available_frames(run_dir, video_id)
            print(f"Found {len(available_frames)} frames for video {video_id}")
            
            video_results = {}
            
            # Process each frame
            for frame in available_frames:
                try:
                    # Load tracker meta to find tracker_id
                    meta = load_tracker_meta(run_dir, video_id, frame)
                    tracker_id = find_tracker_id_from_sequence_id(meta, sequence_id)
                    
                    if tracker_id is None:
                        print(f"Warning: Could not find tracker_id for sequence_id {sequence_id} in video {video_id}, frame {frame}, skipping")
                        continue
                    
                    # Process frame
                    frame_data = process_frame(run_dir, video_id, refiner_id, sequence_id, tracker_id, frame, plot_dir, create_plots)
                    
                    if frame_data is not None:
                        video_results[frame] = frame_data
                        
                except ValueError as e:
                    # Handle frame 0 case where tracker_id_to_seq_id is null
                    if "tracker_id_to_seq_id is null" in str(e):
                        print(f"Frame {frame}: tracker_id_to_seq_id is null, skipping")
                    else:
                        print(f"Error processing frame {frame}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing frame {frame}: {e}")
                    continue
            
            all_results[video_id] = {
                'refiner_id': refiner_id,
                'sequence_id': sequence_id,
                'frames': video_results
            }
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
    
    # Save results to JSON
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved top 5 indices data to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process all frames and plot tracker cross-attention maps for top-scoring predictions"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to attention analysis directory (e.g., /path/to/eval_attn_6059_vid75)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path for top 5 indices (default: {run_dir}/inference/tracker_attention_top5.json)"
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Enable plot creation (disabled by default)"
    )
    
    args = parser.parse_args()
    
    plot_tracker_attention(args.run_dir, args.output_json, args.create_plots)


if __name__ == "__main__":
    main()


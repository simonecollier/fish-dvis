#!/usr/bin/env python3
"""
Create a CSV file with top frames based on activation_proj values.

This script:
1. Loads activation_proj_top_predictions.json
2. For each video, finds frames that reach cumulative sum of 0.75 when denormalized
3. Creates a CSV with video_id, pred_category, true_category, frame_num, frame_activation, frame_activation_norm
4. Outputs to top_frames.csv in the same directory as input JSON
"""

import json
import os
import argparse
import csv
import numpy as np
from pathlib import Path


def denormalize(value, original_min, original_max):
    """Reverse min-max normalization."""
    return value * (original_max - original_min) + original_min


def get_top_frames(activation_vector, original_min, original_max, threshold=0.75, max_frames=None):
    """
    Get frames that reach cumulative sum threshold when denormalized.
    
    Args:
        activation_vector: List of normalized values (0-1)
        original_min: Minimum value before normalization
        original_max: Maximum value before normalization
        threshold: Cumulative sum threshold (default 0.75)
        max_frames: Maximum number of frames to select (default None, no cap)
    
    Returns:
        List of tuples: (frame_index_0_based, denormalized_value, normalized_value)
        Sorted by denormalized value (descending)
    """
    # Convert to numpy array
    normalized = np.array(activation_vector)
    
    # Denormalize
    denormalized = denormalize(normalized, original_min, original_max)
    
    # Sort by importance (descending) - keep track of original indices
    sorted_indices = np.argsort(denormalized)[::-1]
    sorted_values = denormalized[sorted_indices]
    
    # Calculate cumulative sum
    cumulative = np.cumsum(sorted_values)
    
    # Find where cumulative exceeds threshold
    total_sum = cumulative[-1]
    target_sum = threshold * total_sum
    
    num_frames_needed = np.searchsorted(cumulative, target_sum) + 1
    num_frames_needed = min(num_frames_needed, len(denormalized))
    
    # Apply max_frames cap if specified
    if max_frames is not None:
        num_frames_needed = min(num_frames_needed, max_frames)
    
    # Get selected frames with their values
    selected_frames = []
    for i in range(num_frames_needed):
        orig_idx = sorted_indices[i]
        denorm_val = sorted_values[i]
        norm_val = normalized[orig_idx]
        selected_frames.append((orig_idx, denorm_val, norm_val))
    
    # Sort by denormalized value descending (already sorted, but ensure)
    selected_frames.sort(key=lambda x: x[1], reverse=True)
    
    return selected_frames


def load_val_categories(val_json_path):
    """
    Load true categories from val JSON file.
    
    Args:
        val_json_path: Path to val JSON file
    
    Returns:
        Dictionary mapping video_id -> category_id
    """
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Create mapping from annotations
    video_to_category = {}
    if 'annotations' in val_data:
        for ann in val_data['annotations']:
            video_id = ann.get('video_id')
            category_id = ann.get('category_id')
            if video_id is not None and category_id is not None:
                # If multiple annotations for same video, use first one
                if video_id not in video_to_category:
                    video_to_category[video_id] = category_id
    
    return video_to_category


def find_val_json(input_json_path):
    """
    Find val JSON file in the same directory as input JSON.
    
    Args:
        input_json_path: Path to input JSON file
    
    Returns:
        Path to val JSON file or None if not found
    """
    input_dir = os.path.dirname(input_json_path)
    
    # Look for val_*.json files in the directory
    val_files = list(Path(input_dir).glob('val_*.json'))
    
    if val_files:
        # Prefer val_fold6_all_frames.json if it exists, otherwise use first match
        preferred = [f for f in val_files if 'all_frames' in f.name]
        if preferred:
            return str(preferred[0])
        return str(val_files[0])
    
    # Also check parent directory (one level up from inference/)
    parent_dir = os.path.dirname(input_dir)
    val_files = list(Path(parent_dir).glob('val_*.json'))
    if val_files:
        # Prefer val_fold6_all_frames.json
        preferred = [f for f in val_files if 'all_frames' in f.name]
        if preferred:
            return str(preferred[0])
        return str(val_files[0])
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV with top frames based on activation_proj values'
    )
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to activation_proj_top_predictions.json file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Cumulative sum threshold (default: 0.75)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to select per video (default: None, no cap)'
    )
    parser.add_argument(
        '--val-json',
        type=str,
        default=None,
        help='Path to val JSON file (auto-detected if not provided)'
    )
    
    args = parser.parse_args()
    
    # Load activation_proj data
    print(f"Loading activation_proj data from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        activation_data = json.load(f)
    
    # Find val JSON file
    if args.val_json:
        val_json_path = args.val_json
    else:
        val_json_path = find_val_json(args.input_json)
    
    if val_json_path is None:
        print("Warning: Could not find val JSON file. true_category will be empty.")
        video_to_category = {}
    else:
        print(f"Loading val categories from {val_json_path}...")
        video_to_category = load_val_categories(val_json_path)
    
    # Prepare CSV data
    csv_rows = []
    
    # Process each video
    for video_id_str, video_data in activation_data.items():
        video_id = int(video_id_str)
        activation_vector = video_data['activation_vector']
        original_min = video_data['original_min']
        original_max = video_data['original_max']
        pred_category = video_data.get('category_id', None)
        
        # Get true category
        true_category = video_to_category.get(video_id, None)
        
        # Get top frames
        top_frames = get_top_frames(
            activation_vector,
            original_min,
            original_max,
            threshold=args.threshold,
            max_frames=args.max_frames
        )
        
        # Calculate sum of selected frame activations for normalization
        selected_activations = [frame_activation_denorm for _, frame_activation_denorm, _ in top_frames]
        total_selected = sum(selected_activations)
        
        # Add rows for each frame (already sorted by frame_activation descending)
        for frame_idx_0_based, frame_activation_denorm, frame_activation_norm in top_frames:
            frame_num = frame_idx_0_based + 1  # Convert to 1-indexed
            
            # Normalize weight to sum to 1.0 for this video's selected frames
            frame_activation_norm_weight = frame_activation_denorm / total_selected if total_selected > 0 else 0.0
            
            csv_rows.append({
                'video_id': video_id,
                'pred_category': pred_category if pred_category is not None else '',
                'true_category': true_category if true_category is not None else '',
                'frame_num': frame_num,
                'frame_activation': frame_activation_denorm,
                'frame_activation_norm': frame_activation_norm_weight
            })
    
    # Sort rows: first by video_id, then by frame_activation (descending)
    csv_rows.sort(key=lambda x: (x['video_id'], -x['frame_activation']))
    
    # Write CSV
    output_dir = os.path.dirname(args.input_json)
    threshold_str = str(int(args.threshold * 100))
    
    # Build filename with threshold and optional cap
    filename_parts = [f'top_frames_{threshold_str}']
    if args.max_frames is not None:
        filename_parts.append(f'cap{args.max_frames}')
    filename = '_'.join(filename_parts) + '.csv'
    
    output_path = os.path.join(output_dir, filename)
    
    print(f"Writing CSV to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['video_id', 'pred_category', 'true_category', 'frame_num', 'frame_activation', 'frame_activation_norm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Successfully created {output_path} with {len(csv_rows)} rows")
    print(f"Processed {len(activation_data)} videos")


if __name__ == '__main__':
    main()


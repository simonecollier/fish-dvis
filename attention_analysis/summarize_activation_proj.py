#!/usr/bin/env python3
"""
Extract activation_proj weights for top predictions from each video.

This script:
1. Reads results_temporal.json to find top prediction (highest score) for each video
2. Extracts the refiner_id for each top prediction
3. Loads the corresponding activation_proj_weights.npz file
4. Extracts the activation vector [0, 0, :, refiner_id, 0] for the top prediction
5. Saves all vectors to activation_proj_top_predictions.json
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def find_val_json(base_dir: str) -> Optional[str]:
    """
    Find the val JSON file in the base directory by looking for files with 'val' in the name.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        Path to the val JSON file, or None if not found
    """
    base_path = Path(base_dir)
    
    # Look for JSON files with 'val' in the name
    json_files = list(base_path.glob("*.json"))
    
    for json_file in json_files:
        if 'val' in json_file.name.lower():
            return str(json_file)
    
    return None


def load_results_temporal(results_path: str) -> List[Dict]:
    """Load results_temporal.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_frames_with_annotations(val_json_path: str) -> Dict[int, Dict]:
    """
    Analyze which frames have non-null annotations for each video.
    
    Args:
        val_json_path: Path to the val JSON file
        
    Returns:
        Dictionary mapping video_id -> {
            'num_frames_with_fish': int,
            'frame_indices': List[int],
            'file_names': List[str]
        }
    """
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    videos = {video['id']: video for video in val_data['videos']}
    annotations = {ann['video_id']: ann for ann in val_data['annotations']}
    
    result = {}
    
    for video_id, video in videos.items():
        if video_id not in annotations:
            continue
        
        annotation = annotations[video_id]
        file_names = video.get('file_names', [])
        
        # Check segmentations for non-null values (most reliable indicator)
        segmentations = annotation.get('segmentations', [])
        
        # Find frames with non-null annotations
        frame_indices = []
        frame_file_names = []
        
        for frame_idx, seg in enumerate(segmentations):
            if seg is not None:
                frame_indices.append(frame_idx)
                if frame_idx < len(file_names):
                    frame_file_names.append(file_names[frame_idx])
        
        result[video_id] = {
            'num_frames_with_fish': len(frame_indices),
            'frame_indices': frame_indices,
            'file_names': frame_file_names
        }
    
    return result


def find_top_predictions_per_video(results: List[Dict]) -> Dict[int, Dict]:
    """
    Find the prediction with highest score for each video_id.
    
    Returns:
        Dictionary mapping video_id -> {score, refiner_id, category_id}
    """
    top_predictions = {}
    
    for entry in results:
        video_id = entry['video_id']
        score = entry['score']
        
        # If this video_id hasn't been seen, or this score is higher, update
        if video_id not in top_predictions or score > top_predictions[video_id]['score']:
            top_predictions[video_id] = {
                'score': score,
                'refiner_id': entry['refiner_id'],
                'category_id': entry.get('category_id', None)
            }
    
    return top_predictions


def load_activation_proj_weights(npz_path: str) -> Optional[np.ndarray]:
    """Load activation_proj weights from npz file."""
    if not os.path.exists(npz_path):
        print(f"Warning: Activation proj weights file not found: {npz_path}")
        return None
    
    try:
        data = np.load(npz_path)
        if 'activation_weights' in data:
            return data['activation_weights']
        else:
            print(f"Warning: 'activation_weights' key not found in {npz_path}")
            print(f"Available keys: {list(data.keys())}")
            return None
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None


def extract_activation_vector(activation_weights: np.ndarray, refiner_id: int) -> Optional[np.ndarray]:
    """
    Extract activation vector for a specific refiner_id.
    
    Args:
        activation_weights: Array of shape (l, b, t, q, 1)
        refiner_id: Index into the queries dimension (q)
    
    Returns:
        Vector of shape (t,) - frame importance weights
    """
    if activation_weights is None:
        return None
    
    # Check shape
    if len(activation_weights.shape) != 5:
        print(f"Warning: Unexpected activation_weights shape: {activation_weights.shape}")
        return None
    
    l, b, t, q, _ = activation_weights.shape
    
    # Check if refiner_id is valid
    if refiner_id < 0 or refiner_id >= q:
        print(f"Warning: refiner_id {refiner_id} out of range [0, {q-1}]")
        return None
    
    # Extract [0, 0, :, refiner_id, 0] - last layer, first batch, all frames, specific query, remove last dim
    activation_vector = activation_weights[0, 0, :, refiner_id, 0]
    
    return activation_vector


def plot_activation_vector(activation_vector: np.ndarray, video_id: int, refiner_id: int, 
                          score: float, category_id: Optional[int], output_path: str,
                          frames_with_fish: Optional[List[int]] = None):
    """
    Plot activation vector as a line plot with frame number on x-axis.
    
    Args:
        activation_vector: Array of shape (t,) - frame importance weights (normalized to [0, 1])
        video_id: Video ID
        refiner_id: Refiner ID used
        score: Prediction score
        category_id: Category ID (optional)
        output_path: Path to save the plot
        frames_with_fish: List of frame indices that contain fish annotations (optional)
    """
    num_frames = len(activation_vector)
    frame_numbers = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Add pale red background for frames with fish
    if frames_with_fish is not None:
        for frame_idx in frames_with_fish:
            if 0 <= frame_idx < num_frames:
                ax.axvspan(frame_idx - 0.5, frame_idx + 0.5, color='red', alpha=0.15, zorder=0)
    
    ax.plot(frame_numbers, activation_vector, linewidth=2, color='steelblue', zorder=2)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Activation Weight (Frame Importance, Normalized [0,1])', fontsize=12)
    ax.set_ylim(0, 1)  # Ensure y-axis shows [0, 1] range
    
    # Create title with video info
    title = f'Video {video_id} - Activation Proj Weights (Normalized)'
    title += f'\nRefiner ID: {refiner_id}, Score: {score:.6f}'
    if category_id is not None:
        title += f', Category: {category_id}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to {output_path}")


def plot_activation_vector_sorted_bar(activation_vector: np.ndarray, video_id: int, refiner_id: int, 
                                     score: float, category_id: Optional[int], output_path: str,
                                     frames_with_fish: Optional[List[int]] = None):
    """
    Plot activation vector as a bar plot with values sorted from largest to smallest.
    
    Args:
        activation_vector: Array of shape (t,) - unnormalized frame importance weights
        video_id: Video ID
        refiner_id: Refiner ID used
        score: Prediction score
        category_id: Category ID (optional)
        output_path: Path to save the plot
        frames_with_fish: List of original frame indices that contain fish annotations (optional)
    """
    # Sort values from largest to smallest
    sorted_indices = np.argsort(activation_vector)[::-1]  # Descending order
    sorted_values = activation_vector[sorted_indices]
    
    num_frames = len(activation_vector)
    bar_positions = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Map original frame indices to their new positions after sorting
    if frames_with_fish is not None:
        # Create a mapping from original index to sorted position
        original_to_sorted = {orig_idx: sorted_pos for sorted_pos, orig_idx in enumerate(sorted_indices)}
        
        # Find sorted positions for frames with fish
        sorted_positions_with_fish = []
        for orig_idx in frames_with_fish:
            if orig_idx in original_to_sorted:
                sorted_positions_with_fish.append(original_to_sorted[orig_idx])
        
        # Add pale red background for bars corresponding to frames with fish
        for sorted_pos in sorted_positions_with_fish:
            if 0 <= sorted_pos < num_frames:
                ax.axvspan(sorted_pos - 0.5, sorted_pos + 0.5, color='red', alpha=0.15, zorder=0)
    
    ax.bar(bar_positions, sorted_values, color='steelblue', alpha=0.7, zorder=2)
    ax.set_xlabel('Frame Index (Sorted by Activation Value)', fontsize=12)
    ax.set_ylabel('Activation Weight (Unnormalized)', fontsize=12)
    
    # Create title with video info
    title = f'Video {video_id} - Activation Proj Weights (Sorted, Unnormalized)'
    title += f'\nRefiner ID: {refiner_id}, Score: {score:.6f}'
    if category_id is not None:
        title += f', Category: {category_id}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved sorted bar plot to {output_path}")


def process_directory(base_dir: str, plot: bool = False) -> Dict:
    """
    Process all videos in the directory and extract activation_proj vectors for top predictions.
    
    Args:
        base_dir: Base directory containing inference/ and attention_maps/ subdirectories
        plot: If True, create plots for each video
    
    Returns:
        Dictionary with video_id -> {refiner_id, activation_vector, score, category_id}
    """
    base_path = Path(base_dir)
    
    # Paths
    results_path = base_path / "inference" / "results_temporal.json"
    attention_maps_dir = base_path / "attention_maps"
    
    # Check if files exist
    if not results_path.exists():
        raise FileNotFoundError(f"results_temporal.json not found at {results_path}")
    
    if not attention_maps_dir.exists():
        raise FileNotFoundError(f"attention_maps directory not found at {attention_maps_dir}")
    
    # Load results
    print(f"Loading results from {results_path}...")
    results = load_results_temporal(str(results_path))
    print(f"Found {len(results)} total predictions")
    
    # Find top predictions per video
    print("Finding top predictions per video...")
    top_predictions = find_top_predictions_per_video(results)
    print(f"Found top predictions for {len(top_predictions)} videos")
    
    # Find and analyze val JSON file
    print("\nLooking for val JSON file...")
    val_json_path = find_val_json(str(base_path))
    frames_with_fish_data = {}
    
    if val_json_path:
        print(f"Found val JSON file: {val_json_path}")
        print("Analyzing frames with non-null annotations...")
        frames_with_fish_data = analyze_frames_with_annotations(val_json_path)
        print(f"Analyzed {len(frames_with_fish_data)} videos from val JSON")
        
        # Save frames with fish data
        inference_dir = base_path / "inference"
        frames_output_path = inference_dir / "val_frames_with_fish.json"
        with open(frames_output_path, 'w') as f:
            json.dump(frames_with_fish_data, f, indent=2)
        print(f"Saved frames with fish data to {frames_output_path}")
    else:
        print("Warning: No val JSON file found in base directory")
    
    # Create plots directory if plotting is enabled
    plots_dir = None
    if plot:
        inference_dir = base_path / "inference"
        plots_dir = inference_dir / "activation_proj_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to {plots_dir}")
    
    # Extract activation vectors
    output_data = {}
    
    for video_id, pred_info in top_predictions.items():
        refiner_id = pred_info['refiner_id']
        score = pred_info['score']
        category_id = pred_info['category_id']
        
        # Load activation_proj weights
        npz_path = attention_maps_dir / f"video_{video_id}_activation_proj_weights.npz"
        
        if not npz_path.exists():
            print(f"Warning: Activation proj weights not found for video {video_id}: {npz_path}")
            continue
        
        print(f"Processing video {video_id}: refiner_id={refiner_id}, score={score:.6f}")
        
        # Load weights
        activation_weights = load_activation_proj_weights(str(npz_path))
        
        if activation_weights is None:
            continue
        
        # Extract vector
        activation_vector = extract_activation_vector(activation_weights, refiner_id)
        
        if activation_vector is None:
            print(f"Warning: Failed to extract activation vector for video {video_id}, refiner_id {refiner_id}")
            continue
        
        # Normalize activation vector to [0, 1] using min-max normalization
        activation_min = activation_vector.min()
        activation_max = activation_vector.max()
        if activation_max > activation_min:
            activation_vector_normalized = (activation_vector - activation_min) / (activation_max - activation_min)
        else:
            # If all values are the same, set to 0.5 (or could use zeros/ones)
            activation_vector_normalized = np.full_like(activation_vector, 0.5)
        
        print(f"  Extracted vector of length {len(activation_vector)}")
        print(f"  Original range: [{activation_min:.6f}, {activation_max:.6f}]")
        print(f"  Normalized range: [{activation_vector_normalized.min():.6f}, {activation_vector_normalized.max():.6f}]")
        
        # Create plots if requested
        if plot and plots_dir is not None:
            # Get frame indices with fish for this video (if available)
            frames_with_fish = None
            if video_id in frames_with_fish_data:
                frames_with_fish = frames_with_fish_data[video_id].get('frame_indices', None)
            
            # Line plot with normalized vector (original frame order)
            plot_path = plots_dir / f"video_{video_id}_activation_proj_plot.png"
            plot_activation_vector(activation_vector_normalized, video_id, refiner_id, score, category_id, str(plot_path), frames_with_fish)
            
            # Bar plot with unnormalized vector (sorted largest to smallest)
            bar_plot_path = plots_dir / f"video_{video_id}_activation_proj_sorted_bar.png"
            plot_activation_vector_sorted_bar(activation_vector, video_id, refiner_id, score, category_id, str(bar_plot_path), frames_with_fish)
        
        # Store normalized results
        output_data[video_id] = {
            'refiner_id': int(refiner_id),
            'score': float(score),
            'category_id': int(category_id) if category_id is not None else None,
            'activation_vector': activation_vector_normalized.tolist(),  # Convert normalized numpy array to list
            'num_frames': int(len(activation_vector_normalized)),
            'original_min': float(activation_min),
            'original_max': float(activation_max)
        }
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract activation_proj weights for top predictions from each video"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Base directory containing inference/ and attention_maps/ subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <directory>/inference/activation_proj_top_predictions.json)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create line plots of activation vectors (saved to <directory>/activation_proj_plots/)"
    )
    
    args = parser.parse_args()
    
    # Process directory
    print(f"Processing directory: {args.directory}")
    output_data = process_directory(args.directory, plot=args.plot)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.directory, "inference", "activation_proj_top_predictions.json")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully processed {len(output_data)} videos")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()


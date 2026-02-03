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


def extract_frame_number(filename: str) -> Optional[int]:
    """
    Extract frame number from filename.
    
    Args:
        filename: Filename like "path/to/00200.jpg" or "Credit__2024__.../00200.jpg"
    
    Returns:
        Frame number (e.g., 200) or None if extraction fails
    """
    try:
        # Get the last part after the last slash
        basename = filename.split('/')[-1]
        # Remove .jpg extension and convert to int
        frame_num_str = basename.replace('.jpg', '').strip()
        return int(frame_num_str)
    except (ValueError, AttributeError):
        return None


def create_unscramble_mapping(file_names: List[str]) -> Optional[np.ndarray]:
    """
    Create mapping from unscrambled (correct) order to scrambled order.
    
    The file_names are in scrambled order. We extract frame numbers from each filename,
    sort them from smallest to largest, and track which scrambled position each frame
    came from. This handles videos that don't start at 00000.jpg.
    
    Args:
        file_names: List of filenames in scrambled order (as they appear in val.json)
    
    Returns:
        Array of indices: unscrambled_index -> scrambled_index, or None if mapping fails
        Example: mapping[0] = 5 means the first frame in correct order is at position 5 in scrambled order
    """
    if not file_names:
        return None
    
    # Extract frame numbers for each position in scrambled order
    # Each element is (scrambled_index, frame_number)
    scrambled_frame_numbers = []
    for scrambled_idx, filename in enumerate(file_names):
        frame_num = extract_frame_number(filename)
        if frame_num is None:
            print(f"Warning: Could not extract frame number from {filename}")
            return None
        scrambled_frame_numbers.append((scrambled_idx, frame_num))
    
    # Sort by frame number (smallest to largest) to get correct order
    # After sorting, the first element has the smallest frame number, etc.
    scrambled_frame_numbers.sort(key=lambda x: x[1])
    
    # Create mapping: unscrambled_index -> scrambled_index
    # unscrambled_index is the position after sorting (0 = smallest frame number)
    # scrambled_index is the original position in the file_names array
    unscramble_mapping = np.array([scrambled_idx for scrambled_idx, _ in scrambled_frame_numbers])
    
    return unscramble_mapping


def unscramble_activation_vector(activation_vector: np.ndarray, unscramble_mapping: np.ndarray) -> np.ndarray:
    """
    Unscramble activation vector using the mapping.
    
    Args:
        activation_vector: Activation vector in scrambled order
        unscramble_mapping: Mapping from unscrambled index to scrambled index
    
    Returns:
        Activation vector in correct (unscrambled) order
    """
    return activation_vector[unscramble_mapping]


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
        activation_vector: Array of shape (t,) - frame importance weights (unnormalized)
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
                ax.axvspan(frame_idx - 0.5, frame_idx + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
    
    ax.plot(frame_numbers, activation_vector, linewidth=2, color='steelblue', zorder=2)
    ax.set_xlabel('Frame Index', fontsize=24)
    ax.set_ylabel('Frame Importance Score', fontsize=24)
    ax.set_title(f'Video {video_id}', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    
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
                ax.axvspan(sorted_pos - 0.5, sorted_pos + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
        
        # Find position where we've reached 75% of frames with fish
        num_frames_with_fish = len(sorted_positions_with_fish)
        if num_frames_with_fish > 0:
            # Sort the positions to count them in order
            sorted_positions_with_fish_sorted = sorted(sorted_positions_with_fish)
            target_count = int(np.ceil(0.75 * num_frames_with_fish))
            
            if target_count > 0 and target_count <= len(sorted_positions_with_fish_sorted):
                # Find the sorted position where we've reached 75% of frames with fish
                position_75_percent = sorted_positions_with_fish_sorted[target_count - 1]
                
                # Add vertical line at this position
                ax.axvline(x=position_75_percent, color='green', linestyle='--', linewidth=2, 
                          label=f'75% of frames with fish (position {position_75_percent})', zorder=3)
                ax.legend(loc='upper right', fontsize=20)
    
    ax.bar(bar_positions, sorted_values, color='steelblue', alpha=0.7, edgecolor='steelblue', linewidth=0, zorder=2)
    ax.set_xlabel('Frame Rank', fontsize=24)
    ax.set_ylabel('Frame Importance Score', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved sorted bar plot to {output_path}")


def format_model_label_for_legend(model_name1: str, model_name2: str) -> Tuple[str, str]:
    """
    Format model names for legend labels.
    
    Rules:
    - For Silhouette vs Camera (both original): remove "_original" from both
    - For original vs scrambled: format "Silhouette_original" to "Silhouette Original Video"
    
    Args:
        model_name1: First model name
        model_name2: Second model name
    
    Returns:
        Tuple of (formatted_label1, formatted_label2)
    """
    name1_lower = model_name1.lower()
    name2_lower = model_name2.lower()
    
    # Check if both are original (neither is scrambled)
    both_original = 'scrambled' not in name1_lower and 'scramble' not in name1_lower and \
                    'scrambled' not in name2_lower and 'scramble' not in name2_lower
    
    if both_original:
        # Remove "_original" from both labels
        label1 = model_name1.replace('_original', '').replace('_Original', '')
        label2 = model_name2.replace('_original', '').replace('_Original', '')
        # Capitalize first letter
        label1 = label1.capitalize()
        label2 = label2.capitalize()
    else:
        # Original vs scrambled: format nicely
        def format_label(name: str) -> str:
            name_lower = name.lower()
            if 'silhouette' in name_lower:
                if 'scrambled' in name_lower or 'scramble' in name_lower:
                    return 'Silhouette Scrambled Video'
                else:
                    return 'Silhouette Original Video'
            elif 'camera' in name_lower:
                if 'scrambled' in name_lower or 'scramble' in name_lower:
                    return 'Camera Scrambled Video'
                else:
                    return 'Camera Original Video'
            else:
                # Fallback: capitalize and replace underscores
                return name.replace('_', ' ').title()
        
        label1 = format_label(model_name1)
        label2 = format_label(model_name2)
    
    return label1, label2


def plot_activation_vector_comparison(activation_vector1: np.ndarray, activation_vector2: np.ndarray,
                                     video_id: int, refiner_id1: int, refiner_id2: int,
                                     score1: float, score2: float, category_id1: Optional[int],
                                     category_id2: Optional[int], output_path: str,
                                     model_name1: str = "model1", model_name2: str = "model2",
                                     frames_with_fish: Optional[List[int]] = None):
    """
    Plot two activation vectors on the same line plot for comparison.
    
    Args:
        activation_vector1: Array of shape (t,) - unnormalized frame importance weights for model 1
        activation_vector2: Array of shape (t,) - unnormalized frame importance weights for model 2
        video_id: Video ID
        refiner_id1: Refiner ID for model 1
        refiner_id2: Refiner ID for model 2
        score1: Prediction score for model 1
        score2: Prediction score for model 2
        category_id1: Category ID for model 1 (optional)
        category_id2: Category ID for model 2 (optional)
        output_path: Path to save the plot
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        frames_with_fish: List of frame indices that contain fish annotations (optional)
    """
    num_frames = max(len(activation_vector1), len(activation_vector2))
    frame_numbers = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Add pale red background for frames with fish
    if frames_with_fish is not None:
        for frame_idx in frames_with_fish:
            if 0 <= frame_idx < num_frames:
                ax.axvspan(frame_idx - 0.5, frame_idx + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
    
    # Always use blue and red for line plots
    color1 = 'steelblue'  # Blue for model 1
    color2 = 'red'  # Red for model 2
    
    # Format model names for legend
    label1, label2 = format_model_label_for_legend(model_name1, model_name2)
    
    # Plot both models
    ax.plot(frame_numbers[:len(activation_vector1)], activation_vector1, 
            linewidth=2, color=color1, label=label1, zorder=2)
    ax.plot(frame_numbers[:len(activation_vector2)], activation_vector2, 
            linewidth=2, color=color2, label=label2, zorder=2)
    
    ax.set_xlabel('Frame Index', fontsize=24)
    ax.set_ylabel('Frame Importance Score', fontsize=24)
    ax.set_title(f'Video {video_id}', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot to {output_path}")


def plot_activation_vector_sorted_bar_comparison(activation_vector1: np.ndarray, activation_vector2: np.ndarray,
                                                video_id: int, refiner_id1: int, refiner_id2: int,
                                                score1: float, score2: float, category_id1: Optional[int],
                                                category_id2: Optional[int], output_path: str,
                                                model_name1: str = "model1", model_name2: str = "model2",
                                                frames_with_fish: Optional[List[int]] = None):
    """
    Plot two activation vectors as bar plots on the same figure for comparison (sorted).
    
    Args:
        activation_vector1: Array of shape (t,) - unnormalized frame importance weights for model 1
        activation_vector2: Array of shape (t,) - unnormalized frame importance weights for model 2
        video_id: Video ID
        refiner_id1: Refiner ID for model 1
        refiner_id2: Refiner ID for model 2
        score1: Prediction score for model 1
        score2: Prediction score for model 2
        category_id1: Category ID for model 1 (optional)
        category_id2: Category ID for model 2 (optional)
        output_path: Path to save the plot
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        frames_with_fish: List of original frame indices that contain fish annotations (optional)
    """
    # Sort values from largest to smallest for both models
    sorted_indices1 = np.argsort(activation_vector1)[::-1]
    sorted_values1 = activation_vector1[sorted_indices1]
    
    sorted_indices2 = np.argsort(activation_vector2)[::-1]
    sorted_values2 = activation_vector2[sorted_indices2]
    
    num_frames = max(len(activation_vector1), len(activation_vector2))
    bar_positions = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Map original frame indices to their new positions after sorting for model 1
    if frames_with_fish is not None:
        original_to_sorted1 = {orig_idx: sorted_pos for sorted_pos, orig_idx in enumerate(sorted_indices1)}
        sorted_positions_with_fish1 = []
        for orig_idx in frames_with_fish:
            if orig_idx in original_to_sorted1:
                sorted_positions_with_fish1.append(original_to_sorted1[orig_idx])
        
        # Add pale red background for bars corresponding to frames with fish (model 1)
        for sorted_pos in sorted_positions_with_fish1:
            if 0 <= sorted_pos < num_frames:
                ax.axvspan(sorted_pos - 0.5, sorted_pos + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
    
    # Determine colors based on model names (scrambled vs original, and silhouette vs camera)
    def get_color(model_name: str) -> str:
        """Get color for model based on whether it's scrambled/original and silhouette/camera."""
        name_lower = model_name.lower()
        is_scrambled = 'scrambled' in name_lower or 'scramble' in name_lower
        is_silhouette = 'silhouette' in name_lower
        
        if is_scrambled:
            return 'orange'  # Orange for scrambled versions
        elif is_silhouette:
            return 'red'  # Red for original silhouette
        else:
            return 'steelblue'  # Blue for original camera
    
    color1 = get_color(model_name1)
    color2 = get_color(model_name2)
    
    # Format model names for legend
    label1, label2 = format_model_label_for_legend(model_name1, model_name2)
    
    # Plot both models as bars side by side
    width = 0.35
    x1 = bar_positions[:len(sorted_values1)] - width/2
    x2 = bar_positions[:len(sorted_values2)] + width/2
    
    ax.bar(x1, sorted_values1, width, color=color1, alpha=0.7, edgecolor=color1, linewidth=0, label=label1, zorder=2)
    ax.bar(x2, sorted_values2, width, color=color2, alpha=0.7, edgecolor=color2, linewidth=0, label=label2, zorder=2)
    
    ax.set_xlabel('Frame Rank', fontsize=24)
    ax.set_ylabel('Frame Importance Score', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison sorted bar plot to {output_path}")








def process_directory(base_dir: str, plot: bool = False, model_name: str = "model1", plots_base_dir: Optional[str] = None) -> Dict:
    """
    Process all videos in the directory and extract activation_proj vectors for top predictions.
    
    Args:
        base_dir: Base directory containing inference/ and attention_maps/ subdirectories
        plot: If True, create plots for each video
        model_name: Name identifier for this model (used for organizing output)
        plots_base_dir: Optional base directory for plots (if None, uses <base_dir>/inference/activation_proj_plots/)
    
    Returns:
        Tuple of (output_data dictionary, frames_with_fish_data dictionary)
        output_data: Dictionary with video_id -> {refiner_id, activation_vector, score, category_id}
        frames_with_fish_data: Dictionary with video_id -> frame annotation info
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
    
    # Check if this is a scrambled directory
    is_scrambled = 'scramble' in base_dir.lower()
    
    # Find and analyze val JSON file
    print("\nLooking for val JSON file...")
    val_json_path = find_val_json(str(base_path))
    frames_with_fish_data = {}
    video_file_names = {}  # For unscrambling: video_id -> file_names list
    
    if val_json_path:
        print(f"Found val JSON file: {val_json_path}")
        print("Analyzing frames with non-null annotations...")
        frames_with_fish_data = analyze_frames_with_annotations(val_json_path)
        print(f"Analyzed {len(frames_with_fish_data)} videos from val JSON")
        
        # If scrambled, also load file_names for unscrambling
        if is_scrambled:
            print("Scrambled directory detected - loading file_names for unscrambling...")
            with open(val_json_path, 'r') as f:
                val_data = json.load(f)
            
            videos = {video['id']: video for video in val_data['videos']}
            for video_id, video in videos.items():
                if 'file_names' in video:
                    video_file_names[video_id] = video['file_names']
            print(f"Loaded file_names for {len(video_file_names)} videos")
        
        # Save frames with fish data
        inference_dir = base_path / "inference"
        frames_output_path = inference_dir / "val_frames_with_fish.json"
        with open(frames_output_path, 'w') as f:
            json.dump(frames_with_fish_data, f, indent=2)
        print(f"Saved frames with fish data to {frames_output_path}")
    else:
        print("Warning: No val JSON file found in base directory")
        if is_scrambled:
            print("Warning: Scrambled directory detected but no val.json found - cannot unscramble!")
    
    # Create plots directory if plotting is enabled
    plots_dir = None
    if plot:
        if plots_base_dir:
            plots_dir = Path(plots_base_dir) / model_name
        else:
            inference_dir = base_path / "inference"
            plots_dir = inference_dir / "activation_proj_plots" / model_name
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
        
        # Extract vector (this is in scrambled order if directory is scrambled)
        activation_vector_scrambled = extract_activation_vector(activation_weights, refiner_id)
        
        if activation_vector_scrambled is None:
            print(f"Warning: Failed to extract activation vector for video {video_id}, refiner_id {refiner_id}")
            continue
        
        # Unscramble if needed
        activation_vector = activation_vector_scrambled  # Default: use as-is
        is_unscrambled = False
        
        if is_scrambled and video_id in video_file_names:
            file_names = video_file_names[video_id]
            if len(file_names) == len(activation_vector_scrambled):
                unscramble_mapping = create_unscramble_mapping(file_names)
                if unscramble_mapping is not None:
                    activation_vector = unscramble_activation_vector(activation_vector_scrambled, unscramble_mapping)
                    is_unscrambled = True
                    print(f"  Unscrambled activation vector for video {video_id}")
                else:
                    print(f"  Warning: Failed to create unscramble mapping for video {video_id}, using scrambled version")
            else:
                print(f"  Warning: Mismatch in frame counts for video {video_id} (file_names: {len(file_names)}, vector: {len(activation_vector_scrambled)}), using scrambled version")
        
        # Normalize activation vector to [0, 1] using min-max normalization
        # Use min/max from scrambled version (same values, just rearranged)
        activation_min = activation_vector_scrambled.min()
        activation_max = activation_vector_scrambled.max()
        if activation_max > activation_min:
            activation_vector_normalized = (activation_vector - activation_min) / (activation_max - activation_min)
            activation_vector_scrambled_normalized = (activation_vector_scrambled - activation_min) / (activation_max - activation_min)
        else:
            # If all values are the same, set to 0.5 (or could use zeros/ones)
            activation_vector_normalized = np.full_like(activation_vector, 0.5)
            activation_vector_scrambled_normalized = np.full_like(activation_vector_scrambled, 0.5)
        
        print(f"  Extracted vector of length {len(activation_vector)}")
        print(f"  Original range: [{activation_min:.6f}, {activation_max:.6f}]")
        print(f"  Normalized range: [{activation_vector_normalized.min():.6f}, {activation_vector_normalized.max():.6f}]")
        
        # Create plots if requested (use unscrambled version)
        if plot and plots_dir is not None:
            # Get frame indices with fish for this video (if available)
            frames_with_fish = None
            if video_id in frames_with_fish_data:
                frames_with_fish = frames_with_fish_data[video_id].get('frame_indices', None)
            
            # Line plot with unnormalized vector (unscrambled frame order)
            plot_path = plots_dir / f"video_{video_id}_activation_proj_plot.png"
            plot_activation_vector(activation_vector, video_id, refiner_id, score, category_id, str(plot_path), frames_with_fish)
        
        # Store results with model name identifier
        # For backward compatibility, activation_vector and activation_vector_unnormalized are unscrambled
        output_dict = {
            'model_name': model_name,
            'refiner_id': int(refiner_id),
            'score': float(score),
            'category_id': int(category_id) if category_id is not None else None,
            'activation_vector': activation_vector_normalized.tolist(),  # Unscrambled, normalized
            'activation_vector_unnormalized': activation_vector.tolist(),  # Unscrambled, unnormalized
            'num_frames': int(len(activation_vector)),
            'original_min': float(activation_min),
            'original_max': float(activation_max)
        }
        
        # If scrambled, also store scrambled versions
        if is_scrambled:
            output_dict['activation_vector_scrambled'] = activation_vector_scrambled.tolist()
            output_dict['activation_vector_scrambled_normalized'] = activation_vector_scrambled_normalized.tolist()
            output_dict['is_scrambled'] = True
            output_dict['is_unscrambled'] = is_unscrambled
        else:
            output_dict['is_scrambled'] = False
        
        output_data[video_id] = output_dict
    
    return output_data, frames_with_fish_data


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
    parser.add_argument(
        "--dir2",
        type=str,
        default=None,
        help="Additional directory for another model (same structure as base directory). Requires --plot flag."
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <directory>/inference/activation_proj_plots/ for individual plots, or comparison/activation_proj_plots/ for comparison plots)"
    )
    
    args = parser.parse_args()
    
    # Validate flag combinations
    if args.dir2 and not args.plot:
        parser.error("--dir2 requires --plot flag to be enabled")
    
    # Extract model names from directory paths (look for 'camera' or 'silhouette' in path, and 'scrambled'/'scramble' for scrambled versions)
    def extract_model_name(directory_path: str) -> str:
        """Extract model name from directory path (camera/silhouette, original/scrambled)."""
        path_lower = directory_path.lower()
        is_scrambled = 'scrambled' in path_lower or 'scramble' in path_lower
        
        if 'camera' in path_lower:
            return 'camera_scrambled' if is_scrambled else 'camera_original'
        elif 'silhouette' in path_lower:
            return 'silhouette_scrambled' if is_scrambled else 'silhouette_original'
        else:
            # Fallback: use last directory name
            return Path(directory_path).name
    
    model_name1 = extract_model_name(args.directory)
    
    # Determine plots directory
    if args.plots_dir:
        plots_base_dir = args.plots_dir
    else:
        plots_base_dir = None  # Use default behavior
    
    # Process base directory
    print(f"Processing base directory: {args.directory}")
    output_data, frames_with_fish_data1 = process_directory(args.directory, plot=args.plot, model_name=model_name1, plots_base_dir=plots_base_dir)
    
    # Process additional directory if provided
    additional_output_data = None
    frames_with_fish_data2 = None
    model_name2 = None
    if args.dir2:
        model_name2 = extract_model_name(args.dir2)
        print(f"\nProcessing additional directory: {args.dir2}")
        # Create individual plots for the second directory as well
        additional_output_data, frames_with_fish_data2 = process_directory(args.dir2, plot=args.plot, model_name=model_name2, plots_base_dir=plots_base_dir)
        
        # Create comparison plots (only when both directories are provided and plot is enabled)
        if args.plot:
            # Create comparison directory with model names: <model1>_vs_<model2>
            comparison_dir_name = f"{model_name1}_vs_{model_name2}"
            comparison_plots_dir = Path("/home/simone/store/simone/dvis-model-outputs/top_fold_results/comparison/activation_proj_plots") / comparison_dir_name
            comparison_plots_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nCreating comparison plots in {comparison_plots_dir}...")
            
            # Use frames_with_fish_data from first directory (or second if first doesn't have it)
            frames_with_fish_data = frames_with_fish_data1 if frames_with_fish_data1 else frames_with_fish_data2
            
            # Find videos that exist in both models
            common_videos = set(output_data.keys()) & set(additional_output_data.keys())
            print(f"Found {len(common_videos)} videos in both models for comparison")
            
            for video_id in common_videos:
                data1 = output_data[video_id]
                data2 = additional_output_data[video_id]
                
                # Get frame indices with fish for this video (if available)
                frames_with_fish = None
                if frames_with_fish_data and video_id in frames_with_fish_data:
                    frames_with_fish = frames_with_fish_data[video_id].get('frame_indices', None)
                
                # Convert lists back to numpy arrays (use unnormalized for both plots)
                vec1_unnorm = np.array(data1['activation_vector_unnormalized'])
                vec2_unnorm = np.array(data2['activation_vector_unnormalized'])
                
                # Create comparison line plot only (using unnormalized vectors)
                plot_path = comparison_plots_dir / f"video_{video_id}_activation_proj_comparison_plot.png"
                plot_activation_vector_comparison(
                    vec1_unnorm, vec2_unnorm, video_id,
                    data1['refiner_id'], data2['refiner_id'],
                    data1['score'], data2['score'],
                    data1['category_id'], data2['category_id'],
                    str(plot_path), model_name1, model_name2, frames_with_fish
                )
                
        
        # Merge results - if same video_id exists in both, keep both with model identifiers
        # The output_data already has model_name="model1", so we just need to handle conflicts
        for video_id, data in additional_output_data.items():
            if video_id in output_data:
                # If video exists in both, create a combined entry
                # Store as a list of models for this video
                if isinstance(output_data[video_id], list):
                    output_data[video_id].append(data)
                else:
                    # Convert single entry to list and add both
                    output_data[video_id] = [output_data[video_id], data]
            else:
                output_data[video_id] = data
    
    # Save results only if --plot is not used
    if not args.plot:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(args.directory, "inference", "activation_proj_top_predictions.json")
        
        # Save results
        print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        total_videos = len(output_data)
        print(f"Successfully processed {total_videos} videos")
        print(f"Results saved to {output_path}")
    else:
        total_videos = len(output_data)
        print(f"\nSuccessfully processed {total_videos} videos")
        print("Skipping JSON file save (--plot mode enabled)")


if __name__ == "__main__":
    main()


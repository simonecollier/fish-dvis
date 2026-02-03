#!/usr/bin/env python3
"""
CLR (Centered Log-Ratio) permutation test for activation projection vectors.

This script:
1. Loads top_scoring_activation_proj.json files for silhouette and camera models
2. Runs permutation test on CLR-transformed activation vectors
3. Optionally creates CLR comparison plots
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_activation_proj_json(json_path: str) -> Dict:
    """
    Load activation projection data from JSON file.
    
    Args:
        json_path: Path to the JSON file (e.g., activation_proj_top_predictions.json or top_scoring_activation_proj.json)
    
    Returns:
        Dictionary with video_id (as int) -> activation data
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle case where data might be a list (convert to dict keyed by video_id)
    if isinstance(data, list):
        data_dict = {}
        for entry in data:
            # Extract video_id from entry (assuming it has a 'video_id' key)
            if 'video_id' in entry:
                video_id = entry['video_id']
                # Convert to int if it's a string
                if isinstance(video_id, str):
                    video_id = int(video_id)
                data_dict[video_id] = entry
            else:
                raise ValueError("List format not supported - entries must have 'video_id' key")
        return data_dict
    
    # If it's already a dict, convert string keys to integers
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            # Convert string keys to integers
            if isinstance(key, str) and key.isdigit():
                converted_data[int(key)] = value
            else:
                converted_data[key] = value
        return converted_data
    
    return data


def get_unnormalized_activation_vector(data: Dict) -> np.ndarray:
    """
    Get unnormalized activation vector from data dictionary.
    
    The original activation vector summed to 1 (proportions). It was then min-max normalized
    to [0, 1] range. To get back to the original proportions that sum to 1:
    1. Reverse min-max normalization: x = x_norm * (max - min) + min
    2. Normalize to sum to 1 to restore proportions
    
    If 'activation_vector_unnormalized' exists, use it directly (it already sums to 1).
    Otherwise, unnormalize 'activation_vector' using 'original_min' and 'original_max',
    then normalize to sum to 1.
    
    Args:
        data: Dictionary containing activation vector data
    
    Returns:
        Unnormalized activation vector as numpy array (sums to 1, proportions)
    """
    if 'activation_vector_unnormalized' in data:
        return np.array(data['activation_vector_unnormalized'])
    elif 'activation_vector' in data and 'original_min' in data and 'original_max' in data:
        # Reverse min-max normalization: x = x_norm * (max - min) + min
        vec_norm = np.array(data['activation_vector'])
        min_val = data['original_min']
        max_val = data['original_max']
        
        if max_val > min_val:
            # Reverse min-max normalization
            vec_unnorm = vec_norm * (max_val - min_val) + min_val
        else:
            # If min == max, all values were the same in original
            # Set to equal proportions (sum to 1)
            vec_unnorm = np.full_like(vec_norm, 1.0 / len(vec_norm))
        
        # Normalize to sum to 1 to restore original proportions
        sum_val = vec_unnorm.sum()
        if sum_val > 0:
            vec_unnorm = vec_unnorm / sum_val
        else:
            # If sum is zero, set to equal proportions
            vec_unnorm = np.full_like(vec_unnorm, 1.0 / len(vec_unnorm))
        
        return vec_unnorm
    else:
        raise ValueError("Data dictionary must contain either 'activation_vector_unnormalized' or "
                        "('activation_vector', 'original_min', 'original_max')")


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
                        

def compute_clr_values(activation_vector: np.ndarray) -> np.ndarray:
    """
    Compute CLR (centered log-ratio) values from an activation vector.
    
    CLR transformation: clr_i = log(p_i / g(p)), where:
    - p_i is the proportion (normalized activation so it sums to 1)
    - g(p) is the geometric mean: [p_1 * p_2 * ... * p_k]^(1/k)
    
    Args:
        activation_vector: Array of shape (t,) - unnormalized frame importance weights
    
    Returns:
        Array of shape (t,) - CLR transformed values
    """
    epsilon = 1e-10
    sum_val = activation_vector.sum()
    
    if sum_val > 0:
        proportions = activation_vector / sum_val
        # Replace zeros with epsilon to avoid log(0)
        proportions = np.where(proportions == 0, epsilon, proportions)
    else:
        proportions = np.full_like(activation_vector, 1.0 / len(activation_vector))
    
    # Compute geometric mean: g(p) = exp(mean(log(p_i)))
    log_proportions = np.log(proportions)
    geometric_mean = np.exp(np.mean(log_proportions))
    
    # Apply CLR transformation: clr_i = log(p_i / g(p))
    clr_values = np.log(proportions / geometric_mean)
    
    return clr_values


def compute_mean_clr_distance(activation_vectors1: Dict[int, np.ndarray], 
                               activation_vectors2: Dict[int, np.ndarray],
                               common_videos: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean Euclidean distance between CLR-transformed activation vectors across all videos.
    
    For each video:
    1. Compute CLR values for both models
    2. Calculate Euclidean distance D between CLR vectors
    3. Compute mean distance across all videos
    
    Args:
        activation_vectors1: Dictionary mapping video_id -> activation vector for model 1
        activation_vectors2: Dictionary mapping video_id -> activation vector for model 2
        common_videos: List of video IDs that exist in both models
    
    Returns:
        Tuple of (mean_distance, per_video_distances_dict)
    """
    distances = []
    per_video_distances = {}
    
    for video_id in common_videos:
        vec1 = activation_vectors1[video_id]
        vec2 = activation_vectors2[video_id]
        
        # Compute CLR values from activation vectors
        clr1 = compute_clr_values(vec1)
        clr2 = compute_clr_values(vec2)
        
        # Compute Euclidean distance
        distance = compute_clr_euclidean_distance(clr1, clr2)
        distances.append(distance)
        per_video_distances[video_id] = float(distance)
    
    mean_distance = np.mean(distances)
    return mean_distance, per_video_distances


def compute_clr_euclidean_distance(clr_vector1: np.ndarray, clr_vector2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two CLR vectors.
    
    D = sqrt(sum((clr_A - clr_B)^2))
    
    Args:
        clr_vector1: Array of CLR values for model 1
        clr_vector2: Array of CLR values for model 2
    
    Returns:
        Euclidean distance between the two CLR vectors
    
    Raises:
        ValueError: If the two vectors have different lengths
    """
    # Check that both vectors have the same length
    if len(clr_vector1) != len(clr_vector2):
        raise ValueError(f"CLR vectors must have the same length. Got {len(clr_vector1)} and {len(clr_vector2)}")
    
    # Compute Euclidean distance
    diff = clr_vector1 - clr_vector2
    distance = np.sqrt(np.sum(diff ** 2))
    
    return distance


def run_permutations(activation_vectors1: Dict[int, np.ndarray], activation_vectors2: Dict[int, np.ndarray],
                     common_videos: List[int], n_permutations: int) -> Tuple[np.ndarray, Dict[int, float], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Run permutations on activation vectors and compute permuted test statistics.
    
    For each permutation:
    - Randomly permute model labels for each frame independently (50% chance to swap)
    - Recompute CLR values from permuted activation vectors
    - Compute Euclidean distance for permuted CLR vectors
    - Compute mean distance across all videos
    
    Args:
        activation_vectors1: Dictionary mapping video_id -> activation vector for model 1
        activation_vectors2: Dictionary mapping video_id -> activation vector for model 2
        common_videos: List of video IDs that exist in both models
        n_permutations: Number of permutations to run
    
    Returns:
        Tuple of (array of permuted test statistics, example permutation per-video distances,
                  example permutation vectors for model 1, example permutation vectors for model 2)
        The example permutation is the first permutation (index 0)
    """
    permuted_statistics = []
    example_permutation_distances = None
    example_permuted_vectors1 = None
    example_permuted_vectors2 = None
    
    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 1000 == 0:
            print(f"  Completed {perm_idx + 1} / {n_permutations} permutations")
        
        # Create permuted activation vectors
        permuted_vectors1 = {}
        permuted_vectors2 = {}
        
        for video_id in common_videos:
            # Get original activation vectors
            vec1 = activation_vectors1[video_id].copy()
            vec2 = activation_vectors2[video_id].copy()
            
            # Permute model labels for each frame independently (50% chance to swap)
            # We permute the ACTIVATION vectors, not CLR values
            num_frames = len(vec1)
            swap_mask = np.random.random(num_frames) < 0.5
            
            # Swap activation values where mask is True
            vec1_permuted = np.where(swap_mask, vec2, vec1)
            vec2_permuted = np.where(swap_mask, vec1, vec2)
            
            permuted_vectors1[video_id] = vec1_permuted
            permuted_vectors2[video_id] = vec2_permuted
        
        # Compute mean CLR distance for permuted vectors
        permuted_statistic, per_video_distances = compute_mean_clr_distance(
            permuted_vectors1, permuted_vectors2, common_videos
        )
        permuted_statistics.append(permuted_statistic)
        
        # Save the first permutation as an example
        if perm_idx == 0:
            example_permutation_distances = per_video_distances
            example_permuted_vectors1 = permuted_vectors1.copy()
            example_permuted_vectors2 = permuted_vectors2.copy()
    
    return np.array(permuted_statistics), example_permutation_distances, example_permuted_vectors1, example_permuted_vectors2


def save_results_as_text(log_data: Dict, output_path: str):
    """
    Save log data in a readable text format.
    
    Args:
        log_data: Dictionary containing test results
        output_path: Path to save the text file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLR Permutation Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Test info
        test_info = log_data.get('test_info', {})
        f.write("TEST INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp: {test_info.get('timestamp', 'N/A')}\n")
        f.write(f"Model 1: {test_info.get('model1', 'N/A')}\n")
        f.write(f"Model 2: {test_info.get('model2', 'N/A')}\n")
        f.write(f"Number of permutations: {test_info.get('n_permutations', 'N/A')}\n")
        f.write(f"Random seed: {test_info.get('random_seed', 'N/A')}\n")
        f.write(f"Only fish frames: {test_info.get('only_fish', False)}\n")
        f.write(f"Number of videos: {test_info.get('n_videos', 'N/A')}\n")
        f.write(f"Common videos: {', '.join(map(str, test_info.get('common_videos', [])))}\n")
        f.write("\n")
        
        # Results
        results = log_data.get('results', {})
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        orig_stat = results.get('original_test_statistic', None)
        if orig_stat is not None and not np.isnan(orig_stat):
            f.write(f"Original test statistic (mean CLR distance): {orig_stat:.6f}\n")
        else:
            f.write("Original test statistic (mean CLR distance): N/A\n")
        p_val = results.get('p_value', None)
        if p_val is not None and not np.isnan(p_val):
            f.write(f"P-value: {p_val:.6f} ({p_val * 100:.4f}%)\n")
        else:
            f.write("P-value: N/A\n")
        f.write(f"Interpretation: {results.get('interpretation', 'N/A')}\n")
        f.write("\n")
        
        # Permuted statistics summary
        perm_summary = results.get('permuted_statistics_summary', {})
        f.write("PERMUTED STATISTICS SUMMARY\n")
        f.write("-" * 80 + "\n")
        def format_value(val, default='N/A'):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return f"{val:.6f}"
        
        f.write(f"Mean: {format_value(perm_summary.get('mean'))}\n")
        f.write(f"Std: {format_value(perm_summary.get('std'))}\n")
        f.write(f"Min: {format_value(perm_summary.get('min'))}\n")
        f.write(f"Max: {format_value(perm_summary.get('max'))}\n")
        f.write(f"Median: {format_value(perm_summary.get('median'))}\n")
        f.write(f"25th percentile: {format_value(perm_summary.get('q25'))}\n")
        f.write(f"75th percentile: {format_value(perm_summary.get('q75'))}\n")
        f.write("\n")
        
        # Original per-video distances
        per_video_distances = results.get('per_video_distances', {})
        f.write("ORIGINAL PER-VIDEO DISTANCES\n")
        f.write("-" * 80 + "\n")
        for video_id in sorted(per_video_distances.keys()):
            f.write(f"Video {video_id}: {per_video_distances[video_id]:.6f}\n")
        f.write("\n")
        
        # Example permutation
        example_perm = results.get('example_permutation', {})
        if example_perm:
            f.write("EXAMPLE PERMUTATION (First Permutation)\n")
            f.write("-" * 80 + "\n")
            mean_dist = example_perm.get('mean_distance', None)
            if mean_dist is not None:
                f.write(f"Mean distance: {mean_dist:.6f}\n")
            else:
                f.write("Mean distance: N/A\n")
            f.write("\n")
            f.write("Per-video distances:\n")
            example_distances = example_perm.get('per_video_distances', {})
            if example_distances:
                for video_id in sorted(example_distances.keys()):
                    f.write(f"Video {video_id}: {example_distances[video_id]:.6f}\n")
            else:
                f.write("No videos in example permutation\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")


def run_clr_permutation_test(output_data1: Dict, output_data2: Dict, n_permutations: int = 10000, 
                             random_seed: Optional[int] = None, log_file_path: Optional[str] = None,
                             results_txt_path: Optional[str] = None,
                             plot_output_dir: Optional[str] = None,
                             model_name1: str = "model1", model_name2: str = "model2",
                             frames_with_fish_data: Optional[Dict] = None, only_fish: bool = False) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Run permutation test on CLR-transformed activation vectors.
    
    For each video:
    1. Compute CLR values for both models
    2. Calculate Euclidean distance D between CLR vectors
    3. Compute test statistic T = mean(D) across all videos
    
    Then perform permutation test:
    - Randomly permute model labels for each frame independently (50% chance to swap)
    - Recalculate T after permutation
    - Repeat n_permutations times
    - Compare original T with distribution to get p-value
    
    Args:
        output_data1: Dictionary with video_id -> activation data for model 1
        output_data2: Dictionary with video_id -> activation data for model 2
        n_permutations: Number of permutations to run (default: 10000)
        random_seed: Random seed for reproducibility (optional)
        log_file_path: Path to save log file with test results (optional)
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        frames_with_fish_data: Dictionary with video_id -> frame annotation info (optional)
        only_fish: If True, only use frames with fish annotations (default: False)
    
    Returns:
        Tuple of (original_test_statistic, p_value, permuted_statistics_array, per_video_distances_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Find common videos
    common_videos = sorted(set(output_data1.keys()) & set(output_data2.keys()))
    
    # Exclude video 91
    if 91 in common_videos:
        common_videos = [v for v in common_videos if v != 91]
        print(f"Excluding video 91 from analysis")
    
    if len(common_videos) == 0:
        raise ValueError("No common videos found between the two models")
    
    print(f"Found {len(common_videos)} common videos for permutation test (excluding video 91)")
    
    if only_fish:
        if frames_with_fish_data is None:
            raise ValueError("--onlyfish flag requires frames_with_fish_data, but it was not provided or not found")
        print(f"Filtering to only use frames with fish annotations")
    
    # Compute original distances and test statistic
    original_distances = []
    per_video_distances = {}
    activation_vectors1 = {}
    activation_vectors2 = {}
    
    for video_id in common_videos:
        data1 = output_data1[video_id]
        data2 = output_data2[video_id]
        
        # Get unnormalized activation vectors (handle both formats)
        vec1 = get_unnormalized_activation_vector(data1)
        vec2 = get_unnormalized_activation_vector(data2)
        
        # Filter to only frames with fish if requested
        if only_fish and frames_with_fish_data and video_id in frames_with_fish_data:
            frame_indices_with_fish = frames_with_fish_data[video_id].get('frame_indices', [])
            if len(frame_indices_with_fish) > 0:
                # Filter vectors to only include frames with fish
                vec1 = vec1[frame_indices_with_fish]
                vec2 = vec2[frame_indices_with_fish]
            else:
                # Skip this video if no frames with fish
                print(f"Warning: Video {video_id} has no frames with fish annotations, skipping")
                continue
        elif only_fish:
            # Skip this video if frames_with_fish_data not available
            print(f"Warning: Video {video_id} not found in frames_with_fish_data, skipping")
            continue
        
        # Store original activation vectors (we'll permute these, not CLR values)
        activation_vectors1[video_id] = vec1
        activation_vectors2[video_id] = vec2
    
    # Update common_videos to reflect videos actually used (after filtering)
    common_videos = sorted(activation_vectors1.keys())
    
    if only_fish and len(common_videos) < len(set(output_data1.keys()) & set(output_data2.keys())):
        print(f"Note: {len(set(output_data1.keys()) & set(output_data2.keys())) - len(common_videos)} videos were skipped due to missing fish annotations")
    
    # Compute original test statistic (mean CLR distance across all videos)
    original_test_statistic, per_video_distances = compute_mean_clr_distance(
        activation_vectors1, activation_vectors2, common_videos
    )
    print(f"Original test statistic (mean CLR distance): {original_test_statistic:.6f}")
    
    # Run permutations
    print(f"Running {n_permutations} permutations...")
    permuted_statistics, example_permutation_distances, example_permuted_vectors1, example_permuted_vectors2 = run_permutations(
        activation_vectors1, activation_vectors2, common_videos, n_permutations
    )
    
    # Compute p-value: proportion of permuted statistics >= original statistic
    p_value = np.mean(permuted_statistics >= original_test_statistic)
    
    # Compute additional statistics
    permuted_mean = float(np.mean(permuted_statistics))
    permuted_std = float(np.std(permuted_statistics))
    permuted_min = float(np.min(permuted_statistics))
    permuted_max = float(np.max(permuted_statistics))
    permuted_median = float(np.median(permuted_statistics))
    permuted_q25 = float(np.percentile(permuted_statistics, 25))
    permuted_q75 = float(np.percentile(permuted_statistics, 75))
    
    print(f"\nPermutation test results:")
    print(f"  Original test statistic: {original_test_statistic:.6f}")
    print(f"  Mean of permuted statistics: {permuted_mean:.6f}")
    print(f"  Std of permuted statistics: {permuted_std:.6f}")
    print(f"  P-value: {p_value:.6f} ({p_value * 100:.4f}%)")
    
    if p_value < 0.001:
        interpretation = "Models are significantly different (p < 0.001)"
        print(f"  Interpretation: {interpretation}")
    elif p_value < 0.01:
        interpretation = "Models are significantly different (p < 0.01)"
        print(f"  Interpretation: {interpretation}")
    elif p_value < 0.05:
        interpretation = "Models are significantly different (p < 0.05)"
        print(f"  Interpretation: {interpretation}")
    else:
        interpretation = "No significant difference detected (p >= 0.05)"
        print(f"  Interpretation: {interpretation}")
    
    # Save log file if path is provided
    if log_file_path:
        log_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'model1': model_name1,
                'model2': model_name2,
                'n_permutations': n_permutations,
                'random_seed': random_seed,
                'only_fish': only_fish,
                'n_videos': len(common_videos),
                'common_videos': sorted(common_videos)
            },
            'results': {
                'original_test_statistic': float(original_test_statistic),
                'p_value': float(p_value),
                'interpretation': interpretation,
                'per_video_distances': per_video_distances,
                'permuted_statistics_summary': {
                    'mean': permuted_mean,
                    'std': permuted_std,
                    'min': permuted_min,
                    'max': permuted_max,
                    'median': permuted_median,
                    'q25': permuted_q25,
                    'q75': permuted_q75
                },
                'example_permutation': {
                    'per_video_distances': example_permutation_distances,
                    'mean_distance': float(np.mean(list(example_permutation_distances.values()))) if example_permutation_distances else None
                }
            }
        }
        
        # Ensure directory exists
        log_path_obj = Path(log_file_path)
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save log file (JSON format)
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nTest results saved to log file: {log_file_path}")
        
        # Save readable text version if path is provided
        if results_txt_path:
            txt_path_obj = Path(results_txt_path)
            txt_path_obj.parent.mkdir(parents=True, exist_ok=True)
            save_results_as_text(log_data, results_txt_path)
            print(f"Test results saved to text file: {results_txt_path}")
        
        # Create plots of permuted CLR values for first permutation if output directory is provided
        if plot_output_dir and example_permuted_vectors1 and example_permuted_vectors2:
            plot_output_path = Path(plot_output_dir)
            plot_output_path.mkdir(parents=True, exist_ok=True)
            print(f"\nCreating permuted CLR plots in {plot_output_path}...")
            
            for video_id in common_videos:
                vec1_permuted = example_permuted_vectors1[video_id]
                vec2_permuted = example_permuted_vectors2[video_id]
                
                # Get frame indices with fish for this video (if available)
                frames_with_fish = None
                if frames_with_fish_data and video_id in frames_with_fish_data:
                    frames_with_fish = frames_with_fish_data[video_id].get('frame_indices', None)
                
                # Create permuted CLR comparison plot
                permuted_clr_plot_path = plot_output_path / f"video_{video_id}_permuted_clr_comparison.png"
                plot_activation_vector_clr_comparison(
                    vec1_permuted, vec2_permuted, video_id,
                    0, 0,  # refiner_id not meaningful for permuted vectors
                    0.0, 0.0,  # score not meaningful for permuted vectors
                    None, None,  # category_id not meaningful
                    str(permuted_clr_plot_path), model_name1, model_name2, frames_with_fish
                )
                
                # Create permuted activation vector comparison plot (normalized to proportions)
                permuted_activation_plot_path = plot_output_path / f"video_{video_id}_permuted_activation_comparison.png"
                plot_activation_vector_comparison(
                    vec1_permuted, vec2_permuted, video_id,
                    0, 0,  # refiner_id not meaningful for permuted vectors
                    0.0, 0.0,  # score not meaningful for permuted vectors
                    None, None,  # category_id not meaningful
                    str(permuted_activation_plot_path), model_name1, model_name2, frames_with_fish
                )
            
            print(f"Saved permuted CLR and activation vector plots for {len(common_videos)} videos")
    
    return original_test_statistic, p_value, permuted_statistics, per_video_distances


def plot_activation_vector_comparison(activation_vector1: np.ndarray, activation_vector2: np.ndarray,
                                      video_id: int, refiner_id1: int, refiner_id2: int,
                                      score1: float, score2: float, category_id1: Optional[int],
                                      category_id2: Optional[int], output_path: str,
                                      model_name1: str = "model1", model_name2: str = "model2",
                                      frames_with_fish: Optional[List[int]] = None):
    """
    Plot two activation vectors (normalized to proportions) on the same line plot for comparison.
    
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
    # Normalize to proportions (sum to 1)
    sum1 = activation_vector1.sum()
    sum2 = activation_vector2.sum()
    if sum1 > 0:
        proportions1 = activation_vector1 / sum1
    else:
        proportions1 = np.full_like(activation_vector1, 1.0 / len(activation_vector1))
    
    if sum2 > 0:
        proportions2 = activation_vector2 / sum2
    else:
        proportions2 = np.full_like(activation_vector2, 1.0 / len(activation_vector2))
    
    num_frames = max(len(activation_vector1), len(activation_vector2))
    frame_numbers = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Add pale grey background for frames with fish
    if frames_with_fish is not None:
        for frame_idx in frames_with_fish:
            if 0 <= frame_idx < num_frames:
                ax.axvspan(frame_idx - 0.5, frame_idx + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
    
    # Determine colors based on model names
    color1 = 'red' if model_name1.lower() == 'silhouette' else 'steelblue'
    color2 = 'red' if model_name2.lower() == 'silhouette' else 'steelblue'
    
    # Capitalize model names for legend
    label1 = model_name1.capitalize()
    label2 = model_name2.capitalize()
    
    # Plot both models
    ax.plot(frame_numbers[:len(proportions1)], proportions1, 
            linewidth=2, color=color1, label=label1, zorder=2)
    ax.plot(frame_numbers[:len(proportions2)], proportions2, 
            linewidth=2, color=color2, label=label2, zorder=2)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Frame Importance Proportion', fontsize=12)
    
    # Create title
    title = f'Video {video_id} Permuted Frame Importance Proportions (First Permutation)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved permuted activation vector comparison plot to {output_path}")


def plot_activation_vector_clr_comparison(activation_vector1: np.ndarray, activation_vector2: np.ndarray,
                                         video_id: int, refiner_id1: int, refiner_id2: int,
                                         score1: float, score2: float, category_id1: Optional[int],
                                         category_id2: Optional[int], output_path: str,
                                         model_name1: str = "model1", model_name2: str = "model2",
                                         frames_with_fish: Optional[List[int]] = None):
    """
    Plot two activation vectors as CLR (centered log-ratio) transformed values for comparison.
    
    CLR transformation: clr_i = log(p_i / g(p)), where:
    - p_i is the proportion (normalized activation so it sums to 1)
    - g(p) is the geometric mean: [p_1 * p_2 * ... * p_k]^(1/k)
    
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
    # Compute CLR values using helper function
    clr_values1 = compute_clr_values(activation_vector1)
    clr_values2 = compute_clr_values(activation_vector2)
    
    num_frames = max(len(activation_vector1), len(activation_vector2))
    frame_numbers = np.arange(num_frames)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Add pale grey background for frames with fish
    if frames_with_fish is not None:
        for frame_idx in frames_with_fish:
            if 0 <= frame_idx < num_frames:
                ax.axvspan(frame_idx - 0.5, frame_idx + 0.5, color='grey', alpha=0.15, edgecolor='grey', linewidth=0, zorder=0)
    
    # Determine colors based on model names
    color1 = 'red' if model_name1.lower() == 'silhouette' else 'steelblue'
    color2 = 'red' if model_name2.lower() == 'silhouette' else 'steelblue'
    
    # Capitalize model names for legend
    label1 = model_name1.capitalize()
    label2 = model_name2.capitalize()
    
    # Plot both models
    ax.plot(frame_numbers[:len(clr_values1)], clr_values1, 
            linewidth=2, color=color1, label=label1, zorder=2)
    ax.plot(frame_numbers[:len(clr_values2)], clr_values2, 
            linewidth=2, color=color2, label=label2, zorder=2)
    
    # Add horizontal line at y=0 (since CLR values are centered around 0)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('CLR Value (log(p_i / g(p)))', fontsize=12)
    
    # Create title
    title = f'Video {video_id} CLR-Transformed Frame Importance Scores'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved CLR comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CLR permutation test on activation projection vectors from silhouette and camera models"
    )
    parser.add_argument(
        "--silhouette_json",
        type=str,
        default="/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_attn_extra/inference/activation_proj_top_predictions.json",
        help="Path to top_scoring_activation_proj.json (or activation_proj_top_predictions.json) for silhouette model"
    )
    parser.add_argument(
        "--camera_json",
        type=str,
        default="/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/attention/fold6_4443_attn_extra_fish_vid91/inference/activation_proj_top_predictions.json",
        help="Path to top_scoring_activation_proj.json (or activation_proj_top_predictions.json) for camera model"
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=10000,
        help="Number of permutations for permutation test (default: 10000)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for permutation tests (optional, for reproducibility)"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/home/simone/store/simone/dvis-model-outputs/top_fold_results/comparison/clr_permutation_test",
        help="Path to directory where log files will be saved (default: /home/simone/store/simone/dvis-model-outputs/top_fold_results/comparison/clr_permutation_test)"
    )
    parser.add_argument(
        "--onlyfish",
        action="store_true",
        help="Only use frames that contain fish annotations for CLR test"
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default=None,
        help="Path to val JSON file for frame annotation data (required if --onlyfish is used)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create CLR comparison plots for each video"
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default=None,
        help="Directory to save CLR comparison plots (default: same directory as silhouette JSON)"
    )
    
    args = parser.parse_args()
    
    # Validate flag combinations
    if args.onlyfish and not args.val_json:
        parser.error("--onlyfish requires --val_json flag to be enabled")
    
    # Load JSON files
    print(f"Loading silhouette model data from {args.silhouette_json}...")
    silhouette_data = load_activation_proj_json(args.silhouette_json)
    print(f"Loaded {len(silhouette_data)} videos from silhouette model")
    
    print(f"Loading camera model data from {args.camera_json}...")
    camera_data = load_activation_proj_json(args.camera_json)
    print(f"Loaded {len(camera_data)} videos from camera model")
    
    # Load frames with fish data if needed
    frames_with_fish_data = None
    if args.onlyfish or args.plot:
        if args.val_json:
            print(f"Loading frame annotation data from {args.val_json}...")
            frames_with_fish_data = analyze_frames_with_annotations(args.val_json)
            print(f"Loaded frame annotation data for {len(frames_with_fish_data)} videos")
        else:
            # Try to find val JSON in the same directory as silhouette JSON
            silhouette_dir = Path(args.silhouette_json).parent.parent
            val_json_path = find_val_json(str(silhouette_dir))
            if val_json_path:
                print(f"Found val JSON file: {val_json_path}")
                frames_with_fish_data = analyze_frames_with_annotations(val_json_path)
                print(f"Loaded frame annotation data for {len(frames_with_fish_data)} videos")
            else:
                if args.onlyfish:
                    parser.error("Could not find val JSON file. Please specify --val_json")
                else:
                    print("Warning: Could not find val JSON file. Plots will not show frames with fish annotations.")
    
    # Run permutation test
    print(f"\nRunning CLR permutation test...")
    print(f"  Model 1: silhouette")
    print(f"  Model 2: camera")
    print(f"  Number of permutations: {args.n_permutations}")
    if args.random_seed is not None:
        print(f"  Random seed: {args.random_seed}")
    if args.onlyfish:
        print(f"  Only using frames with fish annotations")
    
    # Determine log file paths from log_path argument
    log_path = Path(args.log_path)
    # If it's a file path (ends with .txt or .json), use it as the JSON file path
    # Otherwise, treat it as a directory and create file names there
    if log_path.suffix in ['.txt', '.json']:
        log_file_path = str(log_path)
        results_txt_path = str(log_path.parent / "clr_permutation_test_results.txt")
    else:
        # Treat as directory
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = str(log_path / "log.txt")
        results_txt_path = str(log_path / "clr_permutation_test_results.txt")
    
    # Determine plot output directory for permuted CLR plots (same as log_path)
    permuted_plots_dir = None
    if args.log_path:
        log_path_obj = Path(args.log_path)
        if log_path_obj.suffix in ['.txt', '.json']:
            permuted_plots_dir = str(log_path_obj.parent)
        else:
            permuted_plots_dir = str(log_path_obj)
    
    original_stat, p_value, permuted_stats, per_video_distances = run_clr_permutation_test(
        silhouette_data, camera_data,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        log_file_path=log_file_path,
        results_txt_path=results_txt_path,
        plot_output_dir=permuted_plots_dir,
        model_name1="silhouette",
        model_name2="camera",
        frames_with_fish_data=frames_with_fish_data,
        only_fish=args.onlyfish
    )
    
    print(f"\n=== CLR Permutation Test Summary ===")
    print(f"Test statistic (mean CLR distance): {original_stat:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Number of videos tested: {len(set(silhouette_data.keys()) & set(camera_data.keys()))}")
    print(f"Number of permutations: {args.n_permutations}")
    
    # Create CLR comparison plots if requested
    if args.plot:
        # Determine output directory
        if args.plot_output_dir:
            plot_output_dir = Path(args.plot_output_dir)
        else:
            # Default to same directory as silhouette JSON
            plot_output_dir = Path(args.silhouette_json).parent / "clr_comparison_plots"
        
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreating CLR comparison plots in {plot_output_dir}...")
        
        # Find videos that exist in both models
        common_videos = set(silhouette_data.keys()) & set(camera_data.keys())
        print(f"Found {len(common_videos)} videos in both models for comparison")
        
        for video_id in common_videos:
            data1 = silhouette_data[video_id]
            data2 = camera_data[video_id]
            
            # Get frame indices with fish for this video (if available)
            frames_with_fish = None
            if frames_with_fish_data and video_id in frames_with_fish_data:
                frames_with_fish = frames_with_fish_data[video_id].get('frame_indices', None)
            
            # Get unnormalized activation vectors (handle both formats)
            vec1_unnorm = get_unnormalized_activation_vector(data1)
            vec2_unnorm = get_unnormalized_activation_vector(data2)
            
            # Create CLR comparison plot
            clr_plot_path = plot_output_dir / f"video_{video_id}_activation_proj_comparison_clr.png"
            plot_activation_vector_clr_comparison(
                vec1_unnorm, vec2_unnorm, video_id,
                data1['refiner_id'], data2['refiner_id'],
                data1['score'], data2['score'],
                data1.get('category_id'), data2.get('category_id'),
                str(clr_plot_path), "silhouette", "camera", frames_with_fish
            )


if __name__ == "__main__":
    main()


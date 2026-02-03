#!/usr/bin/env python3
"""
Cumulative sum permutation test for comparing frame importance between two models.

This script:
1. Loads activation_proj data from two model directories
2. Computes number of frames needed to reach cumulative importance threshold for each video
3. Runs permutation test to compare differences between models
4. Generates plots and saves test results
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

# Import functions from summarize_activation_proj
from summarize_activation_proj import process_directory


def compute_frames_to_threshold(activation_vector: np.ndarray, threshold: float = 0.5) -> int:
    """
    Compute number of frames required to reach cumulative importance threshold.
    
    Steps:
    1. Convert to proportions (normalize to sum to 1)
    2. Sort by importance (highest to lowest)
    3. Compute cumulative sum
    4. Find number of frames needed to reach threshold
    
    Args:
        activation_vector: Array of shape (t,) - unnormalized frame importance weights
        threshold: Cumulative importance threshold (default: 0.5)
    
    Returns:
        Number of frames required to reach threshold
    """
    # Convert to proportions (normalize to sum to 1)
    sum_val = activation_vector.sum()
    if sum_val > 0:
        proportions = activation_vector / sum_val
    else:
        # If sum is zero, all frames are equally important
        proportions = np.full_like(activation_vector, 1.0 / len(activation_vector))
    
    # Sort by importance (highest to lowest)
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_proportions = proportions[sorted_indices]
    
    # Compute cumulative sum
    cumsum = np.cumsum(sorted_proportions)
    
    # Find number of frames needed to reach threshold
    # Find first index where cumsum >= threshold
    frames_needed = np.searchsorted(cumsum, threshold, side='left') + 1
    
    # Ensure at least 1 frame and at most all frames
    frames_needed = max(1, min(frames_needed, len(activation_vector)))
    
    return int(frames_needed)


def plot_cumsum_frames_histogram(per_video_frames1: Dict[int, int], per_video_frames2: Dict[int, int],
                                  model_name1: str, model_name2: str, threshold: float,
                                  output_path: str):
    """
    Plot overlapping histograms of frames required to reach cumulative sum threshold.
    
    Args:
        per_video_frames1: Dictionary mapping video_id -> number of frames for model 1
        per_video_frames2: Dictionary mapping video_id -> number of frames for model 2
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        threshold: Cumulative importance threshold used
        output_path: Path to save the plot
    """
    # Extract frame counts as arrays
    frames1 = np.array(list(per_video_frames1.values()))
    frames2 = np.array(list(per_video_frames2.values()))
    
    # Determine bin range (use same bins for both)
    all_frames = np.concatenate([frames1, frames2])
    min_frames = int(np.min(all_frames))
    max_frames = int(np.max(all_frames))
    
    # Create bins with width 2
    # Start from an even number (round down to nearest even)
    bin_start = (min_frames // 2) * 2
    # End at an even number (round up to nearest even, then add 2 for the end)
    bin_end = ((max_frames + 1) // 2) * 2 + 2
    # Create bin edges: [0, 2, 4, 6, ...] format
    bins = np.arange(bin_start, bin_end + 1, 2)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Determine colors based on model names
    color1 = 'red' if 'silhouette' in model_name1.lower() else 'steelblue'
    color2 = 'red' if 'silhouette' in model_name2.lower() else 'steelblue'
    
    # Capitalize model names for legend
    label1 = model_name1.capitalize()
    label2 = model_name2.capitalize()
    
    # Plot histograms with transparency so they overlap nicely
    ax.hist(frames1, bins=bins, alpha=0.6, color=color1, label=label1, edgecolor='black', linewidth=0.5)
    ax.hist(frames2, bins=bins, alpha=0.6, color=color2, label=label2, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    mean1 = np.mean(frames1)
    mean2 = np.mean(frames2)
    ax.axvline(mean1, color=color1, linestyle='--', linewidth=2, alpha=0.8, label=f'{label1} mean: {mean1:.1f}')
    ax.axvline(mean2, color=color2, linestyle='--', linewidth=2, alpha=0.8, label=f'{label2} mean: {mean2:.1f}')
    
    ax.set_xlabel('Number of Frames Required', fontsize=12)
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title(f'Distribution of Frames Required to Reach {threshold*100:.0f}% Cumulative Importance', 
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved cumulative sum histogram to {output_path}")


def plot_cumsum_difference_scatter(per_video_frames1: Dict[int, int], per_video_frames2: Dict[int, int],
                                   per_video_differences: Dict[int, float], model_name1: str, model_name2: str,
                                   threshold: float, output_path: str):
    """
    Plot scatter plot showing difference in frames required between two models for each video.
    
    Args:
        per_video_frames1: Dictionary mapping video_id -> number of frames for model 1
        per_video_frames2: Dictionary mapping video_id -> number of frames for model 2
        per_video_differences: Dictionary mapping video_id -> difference (model1 - model2)
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        threshold: Cumulative importance threshold used
        output_path: Path to save the plot
    """
    # Get video IDs and differences
    video_ids = sorted(per_video_differences.keys())
    differences = [per_video_differences[vid] for vid in video_ids]
    
    # Use sequential numbers (1 to n) instead of video IDs
    video_numbers = np.arange(1, len(video_ids) + 1)
    
    # Determine which model is silhouette and which is camera
    is_silhouette1 = 'silhouette' in model_name1.lower()
    is_silhouette2 = 'silhouette' in model_name2.lower()
    is_camera1 = 'camera' in model_name1.lower()
    is_camera2 = 'camera' in model_name2.lower()
    
    # Calculate difference: camera - silhouette
    # If model1 is camera, difference is already model1 - model2 (camera - silhouette)
    # If model2 is camera, we need to flip the sign
    if is_camera1:
        # model1 is camera, so differences are already camera - silhouette
        camera_minus_silhouette = differences
        ylabel = f'Frames (Camera - Silhouette)'
    elif is_camera2:
        # model2 is camera, so we need to flip: silhouette - camera -> camera - silhouette
        camera_minus_silhouette = [-d for d in differences]
        ylabel = f'Frames (Camera - Silhouette)'
    elif is_silhouette1:
        # model1 is silhouette, so differences are silhouette - camera, need to flip
        camera_minus_silhouette = [-d for d in differences]
        ylabel = f'Frames (Camera - Silhouette)'
    elif is_silhouette2:
        # model2 is silhouette, so differences are camera - silhouette already
        camera_minus_silhouette = differences
        ylabel = f'Frames (Camera - Silhouette)'
    else:
        # Neither is clearly identified, use model1 - model2
        camera_minus_silhouette = differences
        ylabel = f'Frames ({model_name1.capitalize()} - {model_name2.capitalize()})'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create scatter plot using sequential numbers
    ax.scatter(video_numbers, camera_minus_silhouette, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add horizontal dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No difference')
    
    # Add mean line
    mean_diff = np.mean(camera_minus_silhouette)
    ax.axhline(y=mean_diff, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Mean difference: {mean_diff:.2f}')
    
    ax.set_xlabel('Video Number', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Difference in Frames Required to Reach {threshold*100:.0f}% Cumulative Importance per Video', 
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved cumulative sum difference scatter plot to {output_path}")


def plot_cumsum_frames_boxplot(per_video_frames1: Dict[int, int], per_video_frames2: Dict[int, int],
                                model_name1: str, model_name2: str, threshold: float,
                                output_path: str):
    """
    Plot box plots showing distribution of frames required to reach cumulative sum threshold for both models.
    
    Args:
        per_video_frames1: Dictionary mapping video_id -> number of frames for model 1
        per_video_frames2: Dictionary mapping video_id -> number of frames for model 2
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        threshold: Cumulative importance threshold used
        output_path: Path to save the plot
    """
    # Extract frame counts as arrays
    frames1 = np.array(list(per_video_frames1.values()))
    frames2 = np.array(list(per_video_frames2.values()))
    
    # Capitalize model names for labels
    label1 = model_name1.capitalize()
    label2 = model_name2.capitalize()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create box plot data
    data = [frames1, frames2]
    labels = [label1, label2]
    
    # Determine colors based on model names
    colors = []
    for name in [model_name1, model_name2]:
        if 'silhouette' in name.lower():
            colors.append('red')
        else:
            colors.append('steelblue')
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Frames Required', fontsize=12)
    ax.set_title(f'Distribution of Frames Required to Reach {threshold*100:.0f}% Cumulative Importance', 
                 fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved cumulative sum frames box plot to {output_path}")


def plot_cumsum_difference_boxplot(per_video_frames1: Dict[int, int], per_video_frames2: Dict[int, int],
                                   per_video_differences: Dict[int, float], model_name1: str, model_name2: str,
                                   threshold: float, output_path: str):
    """
    Plot box plot showing distribution of differences (camera - silhouette) in frames required.
    
    Args:
        per_video_frames1: Dictionary mapping video_id -> number of frames for model 1
        per_video_frames2: Dictionary mapping video_id -> number of frames for model 2
        per_video_differences: Dictionary mapping video_id -> difference (model1 - model2)
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
        threshold: Cumulative importance threshold used
        output_path: Path to save the plot
    """
    # Get differences
    video_ids = sorted(per_video_differences.keys())
    differences = [per_video_differences[vid] for vid in video_ids]
    
    # Determine which model is silhouette and which is camera
    is_silhouette1 = 'silhouette' in model_name1.lower()
    is_silhouette2 = 'silhouette' in model_name2.lower()
    is_camera1 = 'camera' in model_name1.lower()
    is_camera2 = 'camera' in model_name2.lower()
    
    # Calculate difference: camera - silhouette
    if is_camera1:
        camera_minus_silhouette = differences
    elif is_camera2:
        camera_minus_silhouette = [-d for d in differences]
    elif is_silhouette1:
        camera_minus_silhouette = [-d for d in differences]
    elif is_silhouette2:
        camera_minus_silhouette = differences
    else:
        camera_minus_silhouette = differences
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create box plot
    bp = ax.boxplot([camera_minus_silhouette], labels=['Camera - Silhouette'], patch_artist=True, widths=0.6)
    
    # Color the box
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add horizontal dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No difference')
    
    ax.set_ylabel('Frames (Camera - Silhouette)', fontsize=12)
    ax.set_title(f'Distribution of Differences in Frames Required to Reach {threshold*100:.0f}% Cumulative Importance', 
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved cumulative sum difference box plot to {output_path}")


def run_cumsum_permutation_test(output_data1: Dict, output_data2: Dict, threshold: float = 0.5,
                                 n_permutations: int = 10000, random_seed: Optional[int] = None,
                                 log_file_path: Optional[str] = None, histogram_plot_path: Optional[str] = None,
                                 scatter_plot_path: Optional[str] = None, frames_boxplot_path: Optional[str] = None,
                                 difference_boxplot_path: Optional[str] = None, model_name1: str = "model1",
                                 model_name2: str = "model2") -> Tuple[float, float, np.ndarray, Dict]:
    """
    Run permutation test on cumulative sum threshold differences.
    
    For each video:
    1. Compute number of frames needed to reach threshold for each model
    2. Compute difference: d_v = n_v(A) - n_v(B)
    3. Compute test statistic: T_obs = mean(d_v)
    
    Permutation test:
    - For each video, independently flip sign of d_v with 50% probability
    - Compute T_perm = mean(permuted d_v)
    - Repeat many times
    - Two-sided p-value: proportion where |T_perm| >= |T_obs|
    
    Args:
        output_data1: Dictionary with video_id -> activation data for model 1
        output_data2: Dictionary with video_id -> activation data for model 2
        threshold: Cumulative importance threshold (default: 0.5)
        n_permutations: Number of permutations to run (default: 10000)
        random_seed: Random seed for reproducibility (optional)
        log_file_path: Path to save log file with test results (optional)
        histogram_plot_path: Path to save histogram plot (optional)
        scatter_plot_path: Path to save scatter plot (optional)
        frames_boxplot_path: Path to save frames box plot (optional)
        difference_boxplot_path: Path to save difference box plot (optional)
        model_name1: Name identifier for model 1
        model_name2: Name identifier for model 2
    
    Returns:
        Tuple of (original_test_statistic, p_value, permuted_statistics_array, per_video_differences_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Find common videos
    common_videos = sorted(set(output_data1.keys()) & set(output_data2.keys()))
    
    if len(common_videos) == 0:
        raise ValueError("No common videos found between the two models")
    
    print(f"Found {len(common_videos)} common videos for cumulative sum permutation test")
    print(f"Threshold: {threshold}")
    
    # Compute original differences and test statistic
    per_video_differences = {}
    per_video_frames1 = {}
    per_video_frames2 = {}
    
    for video_id in common_videos:
        data1 = output_data1[video_id]
        data2 = output_data2[video_id]
        
        # Get unnormalized activation vectors
        vec1 = np.array(data1['activation_vector_unnormalized'])
        vec2 = np.array(data2['activation_vector_unnormalized'])
        
        # Compute number of frames needed to reach threshold
        n1 = compute_frames_to_threshold(vec1, threshold)
        n2 = compute_frames_to_threshold(vec2, threshold)
        
        per_video_frames1[video_id] = n1
        per_video_frames2[video_id] = n2
        
        # Compute difference: d_v = n_v(A) - n_v(B)
        d_v = n1 - n2
        per_video_differences[video_id] = float(d_v)
    
    original_test_statistic = np.mean(list(per_video_differences.values()))
    print(f"Original test statistic (mean difference): {original_test_statistic:.6f}")
    print(f"  Model 1 ({model_name1}) mean frames: {np.mean(list(per_video_frames1.values())):.2f}")
    print(f"  Model 2 ({model_name2}) mean frames: {np.mean(list(per_video_frames2.values())):.2f}")
    
    # Run permutations
    print(f"Running {n_permutations} permutations...")
    permuted_statistics = []
    
    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 1000 == 0:
            print(f"  Completed {perm_idx + 1} / {n_permutations} permutations")
        
        # Permute: flip sign of d_v for each video independently with 50% probability
        permuted_differences = []
        for video_id in common_videos:
            d_v = per_video_differences[video_id]
            # Flip sign with 50% probability
            if np.random.random() < 0.5:
                permuted_d_v = -d_v
            else:
                permuted_d_v = d_v
            permuted_differences.append(permuted_d_v)
        
        permuted_statistic = np.mean(permuted_differences)
        permuted_statistics.append(permuted_statistic)
    
    permuted_statistics = np.array(permuted_statistics)
    
    # Compute two-sided p-value: proportion where |T_perm| >= |T_obs|
    abs_original = abs(original_test_statistic)
    abs_permuted = np.abs(permuted_statistics)
    p_value = np.mean(abs_permuted >= abs_original)
    
    # Compute additional statistics
    permuted_mean = float(np.mean(permuted_statistics))
    permuted_std = float(np.std(permuted_statistics))
    permuted_min = float(np.min(permuted_statistics))
    permuted_max = float(np.max(permuted_statistics))
    permuted_median = float(np.median(permuted_statistics))
    permuted_q25 = float(np.percentile(permuted_statistics, 25))
    permuted_q75 = float(np.percentile(permuted_statistics, 75))
    
    print(f"\nCumulative sum permutation test results:")
    print(f"  Original test statistic: {original_test_statistic:.6f}")
    print(f"  Mean of permuted statistics: {permuted_mean:.6f}")
    print(f"  Std of permuted statistics: {permuted_std:.6f}")
    print(f"  P-value (two-sided): {p_value:.6f} ({p_value * 100:.4f}%)")
    
    # Determine direction
    if original_test_statistic > 0:
        direction = f"Model 1 ({model_name1}) needs more frames than Model 2 ({model_name2})"
    elif original_test_statistic < 0:
        direction = f"Model 2 ({model_name2}) needs more frames than Model 1 ({model_name1})"
    else:
        direction = "Models need the same number of frames on average"
    
    if p_value < 0.001:
        interpretation = f"Significant difference detected (p < 0.001, two-sided). {direction}."
        print(f"  Interpretation: {interpretation}")
    elif p_value < 0.01:
        interpretation = f"Significant difference detected (p < 0.01, two-sided). {direction}."
        print(f"  Interpretation: {interpretation}")
    elif p_value < 0.05:
        interpretation = f"Significant difference detected (p < 0.05, two-sided). {direction}."
        print(f"  Interpretation: {interpretation}")
    else:
        interpretation = f"No significant difference detected (p >= 0.05, two-sided). {direction}."
        print(f"  Interpretation: {interpretation}")
    
    # Save log file if path is provided
    if log_file_path:
        log_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'model1': model_name1,
                'model2': model_name2,
                'threshold': threshold,
                'n_permutations': n_permutations,
                'random_seed': random_seed,
                'n_videos': len(common_videos),
                'common_videos': sorted(common_videos)
            },
            'results': {
                'original_test_statistic': float(original_test_statistic),
                'p_value': float(p_value),
                'interpretation': interpretation,
                'direction': direction,
                'per_video_differences': per_video_differences,
                'per_video_frames_model1': {k: int(v) for k, v in per_video_frames1.items()},
                'per_video_frames_model2': {k: int(v) for k, v in per_video_frames2.items()},
                'permuted_statistics_summary': {
                    'mean': permuted_mean,
                    'std': permuted_std,
                    'min': permuted_min,
                    'max': permuted_max,
                    'median': permuted_median,
                    'q25': permuted_q25,
                    'q75': permuted_q75
                }
            }
        }
        
        # Ensure directory exists
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save log file
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nTest results saved to log file: {log_file_path}")
    
    # Create histogram plot if path is provided
    if histogram_plot_path:
        plot_cumsum_frames_histogram(
            per_video_frames1, per_video_frames2,
            model_name1, model_name2, threshold,
            histogram_plot_path
        )
    
    # Create scatter plot if path is provided
    if scatter_plot_path:
        plot_cumsum_difference_scatter(
            per_video_frames1, per_video_frames2, per_video_differences,
            model_name1, model_name2, threshold,
            scatter_plot_path
        )
    
    # Create frames box plot if path is provided
    if frames_boxplot_path:
        plot_cumsum_frames_boxplot(
            per_video_frames1, per_video_frames2,
            model_name1, model_name2, threshold,
            frames_boxplot_path
        )
    
    # Create difference box plot if path is provided
    if difference_boxplot_path:
        plot_cumsum_difference_boxplot(
            per_video_frames1, per_video_frames2, per_video_differences,
            model_name1, model_name2, threshold,
            difference_boxplot_path
        )
    
    return original_test_statistic, p_value, permuted_statistics, per_video_differences


def main():
    parser = argparse.ArgumentParser(
        description="Run cumulative sum permutation test comparing two models"
    )
    parser.add_argument(
        "dir1",
        type=str,
        help="Base directory for model 1 (containing inference/ and attention_maps/ subdirectories)"
    )
    parser.add_argument(
        "dir2",
        type=str,
        help="Base directory for model 2 (containing inference/ and attention_maps/ subdirectories)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cumulative importance threshold (default: 0.5)"
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
        "--log_file",
        type=str,
        default=None,
        help="Path to save log file with test results (default: <dir1>/inference/cumsum_permutation_test_results.json)"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: <dir1>/inference/cumsum_test_plots/)"
    )
    
    args = parser.parse_args()
    
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
    
    model_name1 = extract_model_name(args.dir1)
    model_name2 = extract_model_name(args.dir2)
    
    # Process both directories
    print(f"Processing model 1 directory: {args.dir1}")
    output_data1, _ = process_directory(args.dir1, plot=False, model_name=model_name1)
    
    print(f"\nProcessing model 2 directory: {args.dir2}")
    output_data2, _ = process_directory(args.dir2, plot=False, model_name=model_name2)
    
    # Determine log file path
    if args.log_file:
        log_file_path = args.log_file
    else:
        base_path = Path(args.dir1)
        log_file_path = base_path / "inference" / "cumsum_permutation_test_results.json"
    
    # Determine plots directory
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        base_path = Path(args.dir1)
        plots_dir = base_path / "inference" / "cumsum_test_plots"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot paths
    histogram_plot_path = plots_dir / "cumsum_frames_histogram.png"
    scatter_plot_path = plots_dir / "cumsum_frames_difference_scatter.png"
    frames_boxplot_path = plots_dir / "cumsum_frames_boxplot.png"
    difference_boxplot_path = plots_dir / "cumsum_frames_difference_boxplot.png"
    
    # Run the test
    print(f"\nRunning cumulative sum permutation test...")
    print(f"  Model 1: {model_name1}")
    print(f"  Model 2: {model_name2}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Number of permutations: {args.n_permutations}")
    if args.random_seed is not None:
        print(f"  Random seed: {args.random_seed}")
    
    original_stat, p_value, permuted_stats, per_video_differences = run_cumsum_permutation_test(
        output_data1, output_data2,
        threshold=args.threshold,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        log_file_path=str(log_file_path),
        histogram_plot_path=str(histogram_plot_path),
        scatter_plot_path=str(scatter_plot_path),
        frames_boxplot_path=str(frames_boxplot_path),
        difference_boxplot_path=str(difference_boxplot_path),
        model_name1=model_name1,
        model_name2=model_name2
    )
    
    print(f"\n=== Cumulative Sum Permutation Test Summary ===")
    print(f"Test statistic (mean difference in frames to reach {args.threshold}): {original_stat:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Number of videos tested: {len(set(output_data1.keys()) & set(output_data2.keys()))}")
    print(f"Number of permutations: {args.n_permutations}")
    print(f"\nResults saved to: {log_file_path}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()


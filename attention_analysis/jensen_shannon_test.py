import json
import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import re
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def get_unnormalized_activation_vector(data: dict) -> np.ndarray:
    """
    Get unnormalized activation vector from data dictionary.
    
    For scrambled data, this function prefers the unscrambled version:
    - 'activation_vector_unnormalized' = unscrambled, unnormalized (preferred)
    - 'activation_vector_scrambled' = scrambled, unnormalized (fallback if unscrambled not available)
    
    The original activation vector summed to 1 (proportions). It was then min-max normalized
    to [0, 1] range. To get back to the original proportions that sum to 1:
    1. Reverse min-max normalization: x = x_norm * (max - min) + min
    2. Normalize to sum to 1 to restore proportions
    
    Priority order:
    1. 'activation_vector_unnormalized' (unscrambled, unnormalized) - preferred
    2. 'activation_vector_scrambled' (scrambled, unnormalized) - fallback for scrambled data
    3. 'activation_vector' with 'original_min'/'original_max' (unnormalize normalized vector)
    
    Args:
        data: Dictionary containing activation vector data
    
    Returns:
        Unnormalized activation vector as numpy array (sums to 1, proportions)
    """
    # Prefer unscrambled version if available
    if 'activation_vector_unnormalized' in data:
        return np.array(data['activation_vector_unnormalized'])
    # Fallback to scrambled version if unscrambled not available
    elif 'activation_vector_scrambled' in data:
        return np.array(data['activation_vector_scrambled'])
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
        raise ValueError("Data dictionary must contain one of: 'activation_vector_unnormalized', "
                        "'activation_vector_scrambled', or ('activation_vector', 'original_min', 'original_max')")

def compute_jsd_per_video(modelA, modelB):
    """
    Computes JSD for each pair of videos.
    
    modelA, modelB: lists of 1D numpy arrays (one per video)
    Returns: numpy array of JSD values of length N_videos
    """
    jsd_values = []
    for a, b in zip(modelA, modelB):
        # Normalize just in case (importance vectors should already sum to 1)
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        jsd = jensenshannon(a, b, base=2) ** 2  # square to convert JS distance to JS divergence
        jsd_values.append(jsd)
    
    return np.array(jsd_values)


def permutation_test(modelA, modelB, n_perm=10000, random_state=42, verbose=False):
    """
    Performs a permutation test on the mean JSD across videos.
    
    For each permutation, randomly permute model labels for each frame independently
    (50% chance to swap) within each video, then recompute JSD.
    
    H0: The two models are exchangeable (no systematic difference).
    
    Args:
        modelA, modelB: lists of 1D numpy arrays (one per video)
        n_perm: number of permutations
        random_state: random seed
        verbose: if True, print detailed debug information
    """
    rng = np.random.default_rng(random_state)
    n = len(modelA)

    # Compute observed JSD
    print("\nComputing observed JSD...")
    observed_jsd = compute_jsd_per_video(modelA, modelB)
    observed_mean = np.mean(observed_jsd)
    
    if verbose:
        print(f"  Observed JSD per video (first 5): {observed_jsd[:5]}")
        print(f"  Observed mean JSD: {observed_mean:.6f}")
        print(f"  Number of videos: {n}")

    # Permutation null distribution
    perm_means = np.zeros(n_perm)
    
    print(f"Running {n_perm} permutations (swapping frames independently within each video)...")
    for i in range(n_perm):
        permuted_A = []
        permuted_B = []
        frame_swap_info = []  # Track frame-level swapping for first video
        
        # For each video, swap frames independently
        for vid_idx, (a, b) in enumerate(zip(modelA, modelB)):
            # Permute model labels for each frame independently (50% chance to swap)
            num_frames = len(a)
            swap_mask = rng.random(num_frames) < 0.5
            
            # Swap activation values where mask is True
            vec1_permuted = np.where(swap_mask, b, a)
            vec2_permuted = np.where(swap_mask, a, b)
            
            # Renormalize to ensure they sum to 1 (probability distributions)
            sum1 = vec1_permuted.sum()
            sum2 = vec2_permuted.sum()
            if sum1 > 0:
                vec1_permuted = vec1_permuted / sum1
            else:
                vec1_permuted = np.full_like(vec1_permuted, 1.0 / num_frames)
            if sum2 > 0:
                vec2_permuted = vec2_permuted / sum2
            else:
                vec2_permuted = np.full_like(vec2_permuted, 1.0 / num_frames)
            
            permuted_A.append(vec1_permuted)
            permuted_B.append(vec2_permuted)
            
            # Save swap info for first video (for verbose output)
            if vid_idx == 0:
                frame_swap_info = swap_mask
        
        perm_jsd = compute_jsd_per_video(permuted_A, permuted_B)
        perm_means[i] = np.mean(perm_jsd)
        
        # Show detailed output for first permutation
        if verbose and i == 0:
            print(f"\n--- Detailed output for permutation iteration 0 ---")
            print(f"  First video frame swap mask (first 20 frames): {frame_swap_info[:20]}")
            print(f"  Number of frames swapped in first video: {np.sum(frame_swap_info)}/{len(frame_swap_info)}")
            print(f"  Permuted JSD per video (first 5): {perm_jsd[:5]}")
            print(f"  Permuted mean JSD: {perm_means[i]:.6f}")
            print(f"  Observed mean JSD: {observed_mean:.6f}")
            print(f"  Difference (permuted - observed): {perm_means[i] - observed_mean:.6f}")
        
        # Show progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i + 1}/{n_perm} permutations...")

    # One-sided p-value: is observed mean JSD > null?
    # Include the observed statistic in the null distribution (standard practice)
    all_statistics = np.concatenate([perm_means, [observed_mean]])
    p_value = np.mean(all_statistics >= observed_mean)
    
    if verbose:
        print(f"\n--- Permutation test summary ---")
        print(f"  Observed mean JSD: {observed_mean:.6f}")
        print(f"  Permutation null mean: {np.mean(perm_means):.6f}")
        print(f"  Permutation null std: {np.std(perm_means):.6f}")
        print(f"  Permutation null min: {np.min(perm_means):.6f}")
        print(f"  Permutation null max: {np.max(perm_means):.6f}")
        n_ge_observed = np.sum(perm_means >= observed_mean)
        print(f"  Number of permutations >= observed: {n_ge_observed}/{n_perm}")
        print(f"  P-value (including observed in null): {p_value:.6f} ({(n_ge_observed + 1)}/{n_perm + 1})")

    return {
        "observed_jsd": observed_jsd,
        "observed_mean_jsd": observed_mean,
        "perm_null_means": perm_means,
        "p_value": p_value
    }


def plot_permutation_histogram(perm_means: np.ndarray, observed_mean: float, p_value: float, 
                               output_path: str = "jsd_permutation_histogram.png"):
    """
    Plot histogram of permuted mean JSDs with observed value marked.
    
    Args:
        perm_means: Array of permuted mean JSD values
        observed_mean: Observed mean JSD value
        p_value: P-value from permutation test
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(perm_means, bins=50, alpha=0.7, color='steelblue', 
                               edgecolor='black', linewidth=0.5, label='Permuted JSD')
    
    # Add vertical line for observed value
    ax.axvline(observed_mean, color='red', linestyle='--', linewidth=2, 
               label='Observed JSD')
    
    ax.set_xlabel('Mean JSD', fontsize=24)
    ax.set_ylabel('Frequency', fontsize=24)
    ax.legend(loc='upper left', fontsize=14)
    ax.tick_params(axis='both', labelsize=20)
    
    # Add statistics text box just below the legend (upper left)
    stats_text = (f'Permuted mean: {np.mean(perm_means):.6f}\n'
                  f'Permuted std: {np.std(perm_means):.6f}\n'
                  f'Observed: {observed_mean:.6f}\n'
                  f'P-value: {p_value:.6f}')
    
    ax.text(0.015, 0.82, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=14, family='monospace')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved permutation histogram to {output_path}")


def plot_per_video_jsd(observed_jsd: np.ndarray, video_ids: list, output_path: str = "jsd_per_video_barplot.png"):
    """
    Plot bar plot of JSD per video, sorted by magnitude, with mean line.
    
    Args:
        observed_jsd: Array of JSD values (one per video)
        video_ids: List of video IDs corresponding to each JSD value
        output_path: Path to save the plot
    """
    # Sort by JSD magnitude
    sorted_indices = np.argsort(observed_jsd)
    sorted_jsd = observed_jsd[sorted_indices]
    sorted_video_ids = [video_ids[i] for i in sorted_indices]
    
    mean_jsd = np.mean(observed_jsd)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bar plot
    x_pos = np.arange(len(sorted_jsd))
    bars = ax.bar(x_pos, sorted_jsd, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at mean
    ax.axhline(mean_jsd, color='red', linestyle='--', linewidth=2, 
               label='Mean JSD')
    
    # Set x-axis labels (video IDs)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(vid) for vid in sorted_video_ids], rotation=45, ha='right', fontsize=14)
    
    ax.set_xlabel('Video ID (sorted by JSD magnitude)', fontsize=24)
    ax.set_ylabel('Jensen-Shannon Divergence', fontsize=24)
    ax.legend(loc='upper left', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=20)
    
    # Add statistics text box just below the legend (upper left)
    stats_text = (f'Mean JSD: {mean_jsd:.6f}\n'
                  f'Std JSD: {np.std(observed_jsd):.6f}\n'
                  f'Min JSD: {np.min(observed_jsd):.6f}\n'
                  f'Max JSD: {np.max(observed_jsd):.6f}')
    
    ax.text(0.01, 0.85, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=14, family='monospace')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved per-video JSD bar plot to {output_path}")


def extract_model_name(directory_path: str) -> str:
    """
    Extract model name from directory path (camera/silhouette, original/scrambled, with seed if present).
    
    Args:
        directory_path: Path to the model directory or JSON file
    
    Returns:
        Model name string (e.g., 'camera_original', 'camera_scrambled', 'camera_scrambled_seed1', 
        'silhouette_scrambled_seed2', etc.)
    """
    path_lower = directory_path.lower()
    is_scrambled = 'scrambled' in path_lower or 'scramble' in path_lower
    
    # Extract seed value if present (e.g., "seed1", "seed2", "seed10")
    seed_match = re.search(r'seed(\d+)', path_lower)
    seed_value = seed_match.group(1) if seed_match else None
    
    # Build model name
    if 'camera' in path_lower:
        base_name = 'camera_scrambled' if is_scrambled else 'camera_original'
    elif 'silhouette' in path_lower:
        base_name = 'silhouette_scrambled' if is_scrambled else 'silhouette_original'
    else:
        # Fallback: use last directory name
        return Path(directory_path).name
    
    # Append seed if present and scrambled
    if is_scrambled and seed_value:
        return f'{base_name}_seed{seed_value}'
    else:
        return base_name


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    silhouette_original_path = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_attn_extra/inference/activation_proj_top_predictions.json"
    silhouette_scrambled_path = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_seed31scramble/inference/activation_proj_top_predictions.json"
    camera_original_path = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/attention/fold6_4443_attn_extra_fish_vid91/activation_proj_top_predictions.json"
    camera_scrambled_path = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/attention/fold6_4443_seed75scramble/inference/activation_proj_top_predictions.json"
    model1_path = camera_original_path
    model2_path = camera_scrambled_path

    # Extract model names from paths
    model1_name = extract_model_name(model1_path)
    model2_name = extract_model_name(model2_path)
    
    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")

    # Load JSON files
    print(f"\nLoading {model1_name} data...")
    with open(model1_path, 'r') as f:
        model1_data = json.load(f)
    
    print(f"Loading {model2_name} data...")
    with open(model2_path, 'r') as f:
        model2_data = json.load(f)
    
    # Extract activation vectors for videos that exist in both models
    # The JSON structure is: {video_id: {'activation_vector': [...], ...}}
    modelA = []
    modelB = []
    common_video_ids = []
    
    print("Matching videos between models...")
    for video_id in model1_data.keys():
        # Skip video 91
        try:
            if int(video_id) == 91:
                continue
        except (ValueError, TypeError):
            if str(video_id) == "91":
                continue
        
        if video_id in model2_data:
            # Handle case where video_id might have a list of models (if multiple models processed)
            model1_entry = model1_data[video_id]
            model2_entry = model2_data[video_id]
            
            # If entry is a list, take the first one (or handle as needed)
            if isinstance(model1_entry, list):
                model1_entry = model1_entry[0]
            if isinstance(model2_entry, list):
                model2_entry = model2_entry[0]
            
            # Extract unnormalized activation vector (sums to 1, proportions)
            try:
                model1_vec = get_unnormalized_activation_vector(model1_entry)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not extract unnormalized activation vector for {model1_name} video {video_id}: {e}")
                continue
            
            try:
                model2_vec = get_unnormalized_activation_vector(model2_entry)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not extract unnormalized activation vector for {model2_name} video {video_id}: {e}")
                continue
            
            # Ensure vectors have the same length
            if len(model1_vec) != len(model2_vec):
                print(f"Warning: Video {video_id} has different lengths: {model1_name}={len(model1_vec)}, {model2_name}={len(model2_vec)}")
                # Use minimum length
                min_len = min(len(model1_vec), len(model2_vec))
                model1_vec = model1_vec[:min_len]
                model2_vec = model2_vec[:min_len]
            
            modelA.append(model1_vec)
            modelB.append(model2_vec)
            common_video_ids.append(video_id)
    
    print(f"Found {len(common_video_ids)} videos common to both models")
    
    if len(common_video_ids) == 0:
        print("Error: No common videos found between the two models!")
        exit(1)
    
    # Run permutation test
    print(f"\nRunning permutation test with {len(common_video_ids)} videos...")
    print(f"Model A = {model1_name}")
    print(f"Model B = {model2_name}")
    results = permutation_test(modelA, modelB, n_perm=10, verbose=True)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Number of videos: {len(common_video_ids)}")
    print(f"Observed mean JSD: {results['observed_mean_jsd']:.6f}")
    print(f"Permutation p-value: {results['p_value']:.6f}")
    print(f"Mean JSD per video: {np.mean(results['observed_jsd']):.6f}")
    print(f"Std JSD per video: {np.std(results['observed_jsd']):.6f}")
    print("="*60)
    
    # Create output directory based on model names
    import os
    comparison_dir = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/comparison/jensen_shannon"
    model_comparison_dir = f"{model1_name}_VS_{model2_name}"
    output_dir = os.path.join(comparison_dir, model_comparison_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")
    
    # Plot histogram
    histogram_path = os.path.join(output_dir, "jsd_permutation_histogram.png")
    plot_permutation_histogram(
        results['perm_null_means'],
        results['observed_mean_jsd'],
        results['p_value'],
        histogram_path
    )
    
    # Plot per-video JSD bar plot
    per_video_path = os.path.join(output_dir, "jsd_per_video_barplot.png")
    plot_per_video_jsd(
        results['observed_jsd'],
        common_video_ids,
        per_video_path
    )
    
    # Save results to JSON file
    results_data = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_path': model1_path,
            'model2_path': model2_path,
            'n_videos': len(common_video_ids),
            'n_permutations': 5000,
            'video_ids': [int(vid) if isinstance(vid, (int, str)) and str(vid).isdigit() else str(vid) for vid in common_video_ids]
        },
        'results': {
            'observed_mean_jsd': float(results['observed_mean_jsd']),
            'p_value': float(results['p_value']),
            'mean_jsd_per_video': float(np.mean(results['observed_jsd'])),
            'std_jsd_per_video': float(np.std(results['observed_jsd'])),
            'min_jsd_per_video': float(np.min(results['observed_jsd'])),
            'max_jsd_per_video': float(np.max(results['observed_jsd'])),
            'per_video_jsd': {
                str(vid): float(jsd) for vid, jsd in zip(common_video_ids, results['observed_jsd'])
            },
            'permutation_statistics': {
                'mean': float(np.mean(results['perm_null_means'])),
                'std': float(np.std(results['perm_null_means'])),
                'min': float(np.min(results['perm_null_means'])),
                'max': float(np.max(results['perm_null_means'])),
                'median': float(np.median(results['perm_null_means'])),
                'q25': float(np.percentile(results['perm_null_means'], 25)),
                'q75': float(np.percentile(results['perm_null_means'], 75))
            }
        }
    }
    
    results_json_path = os.path.join(output_dir, "jensen_shannon_test_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Saved test results to: {results_json_path}")

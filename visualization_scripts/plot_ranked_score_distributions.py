"""
Plot ranked score distributions comparing original evaluation vs scrambled evaluations.

This script loads scores from:
1. Original evaluation results.json
2. Multiple scrambled evaluation results.json files

It then plots the ranked score distributions overlaid for comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import defaultdict

def load_scores_from_json(json_path):
    """Load all prediction scores from a results.json file."""
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    scores = [pred['score'] for pred in predictions if 'score' in pred]
    return np.array(scores)

def count_videos(json_path):
    """Count the number of unique videos in a results.json file."""
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    video_ids = set(pred['video_id'] for pred in predictions if 'video_id' in pred)
    return len(video_ids)

def plot_ranked_score_distributions(original_path, scrambled_dir, output_path=None):
    """
    Plot ranked score distributions for original vs scrambled evaluations.
    
    Args:
        original_path: Path to original results.json
        scrambled_dir: Directory containing scrambled evaluation subdirectories
        output_path: Optional path to save the plot
    """
    # Load original scores
    print(f"Loading original scores from: {original_path}")
    original_scores = load_scores_from_json(original_path)
    num_videos = count_videos(original_path)
    print(f"  Found {len(original_scores)} predictions")
    print(f"  Found {num_videos} unique videos")
    
    # Load scrambled scores
    scrambled_dir = Path(scrambled_dir)
    scrambled_results = {}
    
    # Find all results.json files in scrambled subdirectories
    scrambled_json_files = sorted(glob.glob(str(scrambled_dir / "*/inference/results.json")))
    
    print(f"\nLoading scrambled scores from: {scrambled_dir}")
    for json_file in scrambled_json_files:
        seed_name = Path(json_file).parent.parent.name
        scores = load_scores_from_json(json_file)
        scrambled_results[seed_name] = scores
        print(f"  {seed_name}: {len(scores)} predictions")
    
    # Create figure with subplots for different visualization types
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ranked Score Distributions: Original vs Scrambled Evaluations', 
                 fontsize=16, fontweight='bold')
    
    # 1. Ranked scores (sorted descending) - Line plot
    ax1 = axes[0, 0]
    original_sorted = np.sort(original_scores)[::-1]  # Descending order
    ax1.plot(original_sorted, label='Original', linewidth=2, alpha=0.8, color='black')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scrambled_results)))
    for (seed_name, scores), color in zip(scrambled_results.items(), colors):
        sorted_scores = np.sort(scores)[::-1]
        ax1.plot(sorted_scores, label=f'Scrambled {seed_name}', 
                linewidth=1.5, alpha=0.6, color=color)
    
    # Add vertical line at number of videos
    ax1.axvline(x=num_videos, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Number of videos ({num_videos})')
    
    ax1.set_xlabel('Rank (sorted by score)', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Ranked Scores (Descending Order)', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Score histogram/distribution
    ax2 = axes[0, 1]
    ax2.hist(original_scores, bins=50, alpha=0.7, label='Original', 
            color='black', density=True, edgecolor='black', linewidth=1.5)
    
    for (seed_name, scores), color in zip(scrambled_results.items(), colors):
        ax2.hist(scores, bins=50, alpha=0.3, label=f'Scrambled {seed_name}',
                color=color, density=True, histtype='step', linewidth=1.5)
    
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Score Distribution (Histogram)', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution function (CDF)
    ax3 = axes[1, 0]
    sorted_orig = np.sort(original_scores)
    ax3.plot(sorted_orig, np.linspace(0, 1, len(sorted_orig)), 
            label='Original', linewidth=2, alpha=0.8, color='black')
    
    for (seed_name, scores), color in zip(scrambled_results.items(), colors):
        sorted_scores = np.sort(scores)
        ax3.plot(sorted_scores, np.linspace(0, 1, len(sorted_scores)),
                label=f'Scrambled {seed_name}', linewidth=1.5, alpha=0.6, color=color)
    
    ax3.set_xlabel('Score', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    box_data = [original_scores] + list(scrambled_results.values())
    box_labels = ['Original'] + [f'Scrambled\n{name}' for name in scrambled_results.keys()]
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('black')
    bp['boxes'][0].set_alpha(0.7)
    for i, (box, color) in enumerate(zip(bp['boxes'][1:], colors), 1):
        box.set_facecolor(color)
        box.set_alpha(0.5)
    
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Score Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.savefig('ranked_score_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: ranked_score_distributions.png")
    
    # Create zoomed-in plot focusing on the distribution drop-off
    fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(12, 8))
    
    # Determine zoom range - focus on top ranks where drop-off occurs
    zoom_range = 100
    
    original_sorted = np.sort(original_scores)[::-1]  # Descending order
    ax_zoom.plot(original_sorted[:zoom_range], label='Original', 
                linewidth=2.5, alpha=0.9, color='black', marker='o', markersize=3)
    
    for (seed_name, scores), color in zip(scrambled_results.items(), colors):
        sorted_scores = np.sort(scores)[::-1]
        ax_zoom.plot(sorted_scores[:zoom_range], label=f'Scrambled {seed_name}', 
                    linewidth=2, alpha=0.7, color=color, marker='s', markersize=2)
    
    # Add vertical line at number of videos
    ax_zoom.axvline(x=num_videos, color='red', linestyle='--', linewidth=2.5, 
                   alpha=0.8, label=f'Number of videos ({num_videos})', zorder=10)
    
    ax_zoom.set_xlabel('Rank (sorted by score)', fontsize=13, fontweight='bold')
    ax_zoom.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax_zoom.set_title(f'Ranked Scores - Zoomed View (Top {zoom_range} Ranks)', 
                     fontsize=14, fontweight='bold')
    ax_zoom.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_zoom.grid(True, alpha=0.3, linestyle='--')
    ax_zoom.set_xlim(0, zoom_range)
    
    # Set y-axis to focus on the score range in the zoomed area
    all_scores_zoom = np.concatenate([original_sorted[:zoom_range]] + 
                                     [np.sort(scores)[::-1][:zoom_range] 
                                      for scores in scrambled_results.values()])
    y_min = np.min(all_scores_zoom) * 0.95
    y_max = np.max(all_scores_zoom) * 1.05
    ax_zoom.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save zoomed plot
    if output_path:
        zoom_output = str(Path(output_path).with_name(Path(output_path).stem + '_zoomed.png'))
    else:
        zoom_output = 'ranked_score_distributions_zoomed.png'
    plt.savefig(zoom_output, dpi=300, bbox_inches='tight')
    print(f"Zoomed plot saved to: {zoom_output}")
    
    plt.close(fig_zoom)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nOriginal Evaluation:")
    print(f"  Mean score: {np.mean(original_scores):.6f}")
    print(f"  Median score: {np.median(original_scores):.6f}")
    print(f"  Std score: {np.std(original_scores):.6f}")
    print(f"  Min score: {np.min(original_scores):.6f}")
    print(f"  Max score: {np.max(original_scores):.6f}")
    print(f"  Number of predictions: {len(original_scores)}")
    
    print(f"\nScrambled Evaluations:")
    for seed_name, scores in scrambled_results.items():
        print(f"\n  {seed_name}:")
        print(f"    Mean score: {np.mean(scores):.6f}")
        print(f"    Median score: {np.median(scores):.6f}")
        print(f"    Std score: {np.std(scores):.6f}")
        print(f"    Min score: {np.min(scores):.6f}")
        print(f"    Max score: {np.max(scores):.6f}")
        print(f"    Number of predictions: {len(scores)}")
        print(f"    Difference from original (mean): {np.mean(scores) - np.mean(original_scores):+.6f}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot ranked score distributions comparing original vs scrambled evaluations'
    )
    parser.add_argument(
        '--original',
        type=str,
        default='/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/fold6/checkpoint_0004443/inference/results.json',
        help='Path to original results.json'
    )
    parser.add_argument(
        '--scrambled-dir',
        type=str,
        default='/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6',
        help='Directory containing scrambled evaluation subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: ranked_score_distributions.png)'
    )
    
    args = parser.parse_args()
    
    plot_ranked_score_distributions(args.original, args.scrambled_dir, args.output)


#!/usr/bin/env python3
"""
Script to create scatter plots comparing scrambled results with best model results.

Plots Category (species) on x-axis and Metric values on y-axis, with error bars
for scrambled results using standard error (SE) values.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_scrambled_data(scrambled_dir):
    """Load scrambled summary data with mean and SE values."""
    csv_path = Path(scrambled_dir) / "mask_metrics_category_summary.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def load_best_model_data(best_model_dir):
    """Load best model data."""
    csv_path = Path(best_model_dir) / "mask_metrics_category.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def create_combined_plot(scrambled_dir, best_model_dir, output_path=None):
    """Create a single combined plot with all metrics in different colors."""
    
    # Load data
    scrambled_df = load_scrambled_data(scrambled_dir)
    best_model_df = load_best_model_data(best_model_dir)
    
    # Metrics to plot
    metrics = ['ap_instance_per_cat', 'ap50_instance_per_cat', 'ap75_instance_per_cat']
    metric_labels = ['AP', 'AP50', 'AP75']
    metric_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    
    # Get categories
    categories = sorted(scrambled_df['category_name'].unique())
    
    # Check if we need to add "All" category
    # Compute "All" as mean across all categories
    all_category_data = {}
    for metric in metrics:
        # For scrambled: compute weighted mean and SE
        mean_col = f'{metric}_mean'
        se_col = f'{metric}_se'
        n_seeds_col = f'{metric}_n_seeds'
        
        if mean_col in scrambled_df.columns:
            means = scrambled_df[mean_col].dropna()
            ses = scrambled_df[se_col].dropna()
            
            if len(means) > 0:
                # Simple mean across categories (could be weighted by n_seeds if available)
                all_category_data[f'{metric}_mean'] = means.mean()
                # For SE of mean, use pooled standard error
                if len(ses) > 0:
                    all_category_data[f'{metric}_se'] = np.sqrt((ses**2).sum()) / len(ses)
                    # Get average n_seeds for t-distribution
                    if n_seeds_col in scrambled_df.columns:
                        n_seeds_values = scrambled_df[n_seeds_col].dropna()
                        if len(n_seeds_values) > 0:
                            avg_n_seeds = int(n_seeds_values.mean())
                            all_category_data[f'{metric}_n_seeds'] = avg_n_seeds
                        else:
                            all_category_data[f'{metric}_n_seeds'] = 10  # default
                    else:
                        all_category_data[f'{metric}_n_seeds'] = 10  # default
                else:
                    all_category_data[f'{metric}_se'] = 0.0
                    all_category_data[f'{metric}_n_seeds'] = 10
        
        # For best model: simple mean
        if metric in best_model_df.columns:
            values = best_model_df[metric].dropna()
            if len(values) > 0:
                all_category_data[metric] = values.mean()
    
    # Add "All" to categories if we have data
    if all_category_data:
        categories = list(categories) + ['All']
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Calculate x positions: for each category, we'll have 3 metrics grouped together
    n_categories = len(categories)
    n_metrics = len(metrics)
    width = 0.12  # Width of each group (adjusted for better spacing within category)
    spacing = 1.5  # Spacing between category groups (increased to separate categories more)
    
    # Prepare data for all metrics
    all_scrambled_values = {metric: [] for metric in metrics}
    all_scrambled_errors = {metric: [] for metric in metrics}
    all_best_model_values = {metric: [] for metric in metrics}
    
    for category in categories:
        for metric in metrics:
            if category == 'All':
                # Use computed "All" values
                scrambled_val = all_category_data.get(f'{metric}_mean', np.nan)
                se_val = all_category_data.get(f'{metric}_se', 0.0)
                n_seeds = all_category_data.get(f'{metric}_n_seeds', 10)
                best_model_val = all_category_data.get(metric, np.nan)
            else:
                # Get category-specific data
                scrambled_row = scrambled_df[scrambled_df['category_name'] == category]
                best_model_row = best_model_df[best_model_df['category_name'] == category]
                
                if len(scrambled_row) > 0:
                    scrambled_val = scrambled_row[f'{metric}_mean'].iloc[0]
                    se_val = scrambled_row[f'{metric}_se'].iloc[0]
                    n_seeds_col = f'{metric}_n_seeds'
                    if n_seeds_col in scrambled_row.columns:
                        n_seeds = int(scrambled_row[n_seeds_col].iloc[0])
                    else:
                        n_seeds = 10  # default
                else:
                    scrambled_val = np.nan
                    se_val = 0.0
                    n_seeds = 10
                
                if len(best_model_row) > 0:
                    best_model_val = best_model_row[metric].iloc[0]
                else:
                    best_model_val = np.nan
            
            # Calculate 95% confidence interval: SE * t-value
            if not np.isnan(scrambled_val) and se_val > 0 and n_seeds > 1:
                df = n_seeds - 1
                t_value = stats.t.ppf(0.975, df)  # 0.975 for 95% CI (two-tailed)
                ci_half_width = se_val * t_value
            else:
                ci_half_width = 0.0
            
            all_scrambled_values[metric].append(scrambled_val)
            all_scrambled_errors[metric].append(ci_half_width)
            all_best_model_values[metric].append(best_model_val)
    
    # Calculate x positions for each category group
    category_centers = np.arange(n_categories) * spacing
    
    # Plot each metric
    for idx, (metric, metric_label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        # Offset for each metric within a category group
        offset = (idx - (n_metrics - 1) / 2) * width
        x_positions = category_centers + offset
        
        # Scrambled data with error bars (no connecting lines)
        ax.errorbar(x_positions, all_scrambled_values[metric], yerr=all_scrambled_errors[metric],
                   fmt='o', capsize=3, capthick=1, elinewidth=1, markersize=6,
                   label=f'Scrambled {metric_label}', color=color, alpha=0.7)
        
        # Best model data
        ax.scatter(x_positions, all_best_model_values[metric], s=80, marker='s',
                  label=f'Original {metric_label}', color=color, alpha=0.7, zorder=5, edgecolors='darkred', linewidths=0.5)
    
    # Customize plot
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Metric Value (%)', fontsize=12)
    ax.set_title('Metrics Comparison: Scrambled vs Original', fontsize=14, fontweight='bold')
    ax.set_xticks(category_centers)
    
    # Create custom labels with "All" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'All':
            tick_labels.append(r'$\mathbf{All}$')  # Bold "All" using LaTeX
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)
    
    # Set y-axis limits
    all_values = []
    for metric in metrics:
        all_values.extend([v for v in all_scrambled_values[metric] + all_best_model_values[metric] if not np.isnan(v)])
    if all_values:
        y_min = max(0, min(all_values) - 10)
        y_max = min(100, max(all_values) + 10)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")
    
    plt.close()


def create_plot(scrambled_dir, best_model_dir, output_path=None):
    """Create scatter plot comparing scrambled and best model results."""
    
    # Load data
    scrambled_df = load_scrambled_data(scrambled_dir)
    best_model_df = load_best_model_data(best_model_dir)
    
    # Metrics to plot
    metrics = ['ap_instance_per_cat', 'ap50_instance_per_cat', 'ap75_instance_per_cat']
    metric_labels = ['AP', 'AP50', 'AP75']
    
    # Get categories
    categories = sorted(scrambled_df['category_name'].unique())
    
    # Check if we need to add "All" category
    # Compute "All" as mean across all categories
    all_category_data = {}
    for metric in metrics:
        # For scrambled: compute weighted mean and SE
        mean_col = f'{metric}_mean'
        se_col = f'{metric}_se'
        n_seeds_col = f'{metric}_n_seeds'
        
        if mean_col in scrambled_df.columns:
            means = scrambled_df[mean_col].dropna()
            ses = scrambled_df[se_col].dropna()
            
            if len(means) > 0:
                # Simple mean across categories (could be weighted by n_seeds if available)
                all_category_data[f'{metric}_mean'] = means.mean()
                # For SE of mean, use pooled standard error
                if len(ses) > 0:
                    all_category_data[f'{metric}_se'] = np.sqrt((ses**2).sum()) / len(ses)
                    # Get average n_seeds for t-distribution
                    if n_seeds_col in scrambled_df.columns:
                        n_seeds_values = scrambled_df[n_seeds_col].dropna()
                        if len(n_seeds_values) > 0:
                            avg_n_seeds = int(n_seeds_values.mean())
                            all_category_data[f'{metric}_n_seeds'] = avg_n_seeds
                        else:
                            all_category_data[f'{metric}_n_seeds'] = 10  # default
                    else:
                        all_category_data[f'{metric}_n_seeds'] = 10  # default
                else:
                    all_category_data[f'{metric}_se'] = 0.0
                    all_category_data[f'{metric}_n_seeds'] = 10
        
        # For best model: simple mean
        if metric in best_model_df.columns:
            values = best_model_df[metric].dropna()
            if len(values) > 0:
                all_category_data[metric] = values.mean()
    
    # Add "All" to categories if we have data
    if all_category_data:
        categories = list(categories) + ['All']
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Prepare data for plotting
        x_positions = []
        scrambled_values = []
        scrambled_errors = []
        best_model_values = []
        category_labels = []
        
        # Plot each category
        for cat_idx, category in enumerate(categories):
            x_positions.append(cat_idx)
            category_labels.append(category)
            
            if category == 'All':
                # Use computed "All" values
                scrambled_val = all_category_data.get(f'{metric}_mean', np.nan)
                se_val = all_category_data.get(f'{metric}_se', 0.0)
                n_seeds = all_category_data.get(f'{metric}_n_seeds', 10)
                best_model_val = all_category_data.get(metric, np.nan)
            else:
                # Get category-specific data
                scrambled_row = scrambled_df[scrambled_df['category_name'] == category]
                best_model_row = best_model_df[best_model_df['category_name'] == category]
                
                if len(scrambled_row) > 0:
                    scrambled_val = scrambled_row[f'{metric}_mean'].iloc[0]
                    se_val = scrambled_row[f'{metric}_se'].iloc[0]
                    n_seeds_col = f'{metric}_n_seeds'
                    if n_seeds_col in scrambled_row.columns:
                        n_seeds = int(scrambled_row[n_seeds_col].iloc[0])
                    else:
                        n_seeds = 10  # default
                else:
                    scrambled_val = np.nan
                    se_val = 0.0
                    n_seeds = 10
                
                if len(best_model_row) > 0:
                    best_model_val = best_model_row[metric].iloc[0]
                else:
                    best_model_val = np.nan
            
            # Calculate 95% confidence interval: SE * t-value
            # Degrees of freedom = n_seeds - 1
            if not np.isnan(scrambled_val) and se_val > 0 and n_seeds > 1:
                df = n_seeds - 1
                t_value = stats.t.ppf(0.975, df)  # 0.975 for 95% CI (two-tailed)
                ci_half_width = se_val * t_value
            else:
                ci_half_width = 0.0
            
            scrambled_values.append(scrambled_val)
            scrambled_errors.append(ci_half_width)
            best_model_values.append(best_model_val)
        
        # Create scatter plot
        # Scrambled data with error bars (95% CI)
        ax.errorbar(x_positions, scrambled_values, yerr=scrambled_errors,
                   fmt='o', capsize=5, capthick=1, elinewidth=1, markersize=8,
                   label='Scrambled', color='blue', alpha=0.7)
        
        # Best model data
        ax.scatter(x_positions, best_model_values, s=100, marker='s',
                  label='Original', color='red', alpha=0.7, zorder=5)
        
        # Customize plot
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel(f'{metric_label} (%)', fontsize=12)
        ax.set_title(f'{metric_label} by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(category_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        
        # Set y-axis to start from reasonable minimum
        all_values = [v for v in scrambled_values + best_model_values if not np.isnan(v)]
        if all_values:
            y_min = max(0, min(all_values) - 10)
            y_max = min(100, max(all_values) + 10)
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create scatter plots comparing scrambled results with best model results'
    )
    parser.add_argument(
        '--scrambled_dir',
        type=str,
        help='Directory containing scrambled summary (e.g., /path/to/scrambled_fold4/seeds_summary)'
    )
    parser.add_argument(
        '--best_model_dir',
        type=str,
        help='Directory containing best model inference results (e.g., /path/to/fold4/checkpoint_0006059/inference)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the plot (defaults to scrambled_dir/scrambled_vs_best_model_comparison.png)'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create a combined plot with all metrics in one plot (default: separate subplots)'
    )
    
    args = parser.parse_args()
    
    scrambled_dir = os.path.abspath(args.scrambled_dir)
    best_model_dir = os.path.abspath(args.best_model_dir)
    
    # Set default output path if not specified
    if args.output is None:
        if args.combined:
            output_path = os.path.join(scrambled_dir, "scrambled_vs_best_model_comparison_combined.png")
        else:
            output_path = os.path.join(scrambled_dir, "scrambled_vs_best_model_comparison.png")
    else:
        output_path = args.output
    
    print("=" * 60)
    print(f"Scrambled directory: {scrambled_dir}")
    print(f"Best model directory: {best_model_dir}")
    print(f"Output path: {output_path}")
    print(f"Plot style: {'Combined' if args.combined else 'Separate subplots'}")
    print("=" * 60)
    
    try:
        if args.combined:
            create_combined_plot(scrambled_dir, best_model_dir, output_path)
        else:
            create_plot(scrambled_dir, best_model_dir, output_path)
        print("\nDone!")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


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
    
    # Determine model type from directory name
    combined_path = str(scrambled_dir) + str(best_model_dir)
    if 'camera' in combined_path.lower():
        model_type = 'Camera'
    elif 'silhouette' in combined_path.lower():
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    # Load data
    scrambled_df = load_scrambled_data(scrambled_dir)
    best_model_df = load_best_model_data(best_model_dir)
    
    # Metrics to plot - only AP for silhouette, all metrics for camera
    if model_type == 'Silhouette':
        metrics = ['ap_instance_per_cat']
        metric_labels = ['AP']
        metric_colors = ['#1f77b4']  # Blue
    else:
        metrics = ['ap_instance_per_cat', 'ap50_instance_per_cat', 'ap75_instance_per_cat']
        metric_labels = ['AP', 'AP50', 'AP75']
        metric_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    
    # Get categories (keep original order from CSV)
    categories = scrambled_df['category_name'].unique().tolist()
    
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
    
    # Add "Mean" to categories if we have data
    if all_category_data:
        categories = list(categories) + ['Mean']
    
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
            if category == 'Mean':
                # Use computed "Mean" values
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
                   fmt='o', capsize=5, capthick=3, elinewidth=3, markersize=12,
                   label=f'Scrambled {metric_label}', color=color, alpha=0.7, 
                   markeredgecolor='red', markeredgewidth=1, zorder=6)
        
        # Best model data
        ax.scatter(x_positions, all_best_model_values[metric], s=150, marker='s',
                  label=f'Original {metric_label}', color=color, alpha=0.7, zorder=5, 
                  edgecolors='none', linewidths=0)
    
    # Customize plot
    ax.set_ylabel('Average Precision (AP)', fontsize=24)
    title_text = f'{model_type}'
    ax.set_title(title_text, fontsize=28, fontweight='bold', pad=20)
    ax.set_xticks(category_centers)
    
    # Create custom labels with "Mean" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'Mean':
            tick_labels.append(r'$\mathbf{Mean}$')  # Bold "Mean" using LaTeX
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=14)
    
    # Set y-axis limits
    all_values = []
    for metric in metrics:
        all_values.extend([v for v in all_scrambled_values[metric] + all_best_model_values[metric] if not np.isnan(v)])
    if all_values:
        y_min = max(0, min(all_values) - 10)
        y_max = 100
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
    
    # Determine model type from directory name
    combined_path = str(scrambled_dir) + str(best_model_dir)
    if 'camera' in combined_path.lower():
        model_type = 'Camera'
    elif 'silhouette' in combined_path.lower():
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    # Metrics to plot - only AP for silhouette, all metrics for camera
    if model_type == 'Silhouette':
        metrics = ['ap_instance_per_cat']
        metric_labels = ['AP']
    else:
        metrics = ['ap_instance_per_cat', 'ap50_instance_per_cat', 'ap75_instance_per_cat']
        metric_labels = ['AP', 'AP50', 'AP75']
    
    # Get categories (keep original order from CSV)
    categories = scrambled_df['category_name'].unique().tolist()
    
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
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]  # Make it iterable
    
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
                   fmt='o', capsize=5, capthick=3, elinewidth=3, markersize=12,
                   label='Scrambled', color='blue', alpha=0.7,
                   markeredgecolor='red', markeredgewidth=1, zorder=6)
        
        # Best model data
        ax.scatter(x_positions, best_model_values, s=150, marker='s',
                  label='Original', color='red', alpha=0.7, zorder=5,
                  edgecolors='none', linewidths=0)
        
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
            y_max = 100
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()


def create_ap75_plot(scrambled_dir, best_model_dir, output_path=None):
    """Create scatter plot comparing scrambled and best model results for AP75 only."""
    
    # Determine model type from directory name
    combined_path = str(scrambled_dir) + str(best_model_dir)
    if 'camera' in combined_path.lower():
        model_type = 'Camera'
    elif 'silhouette' in combined_path.lower():
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    # Load data
    scrambled_df = load_scrambled_data(scrambled_dir)
    best_model_df = load_best_model_data(best_model_dir)
    
    # Only AP75 metric
    metric = 'ap75_instance_per_cat'
    metric_label = 'AP75'
    color = '#ff7f0e'  # Orange color matching combined plot
    
    # Get categories (keep original order from CSV)
    categories = scrambled_df['category_name'].unique().tolist()
    
    # Check if we need to add "Mean" category
    # Compute "Mean" as mean across all categories
    all_category_data = {}
    
    # For scrambled: compute weighted mean and SE
    mean_col = f'{metric}_mean'
    se_col = f'{metric}_se'
    n_seeds_col = f'{metric}_n_seeds'
    
    if mean_col in scrambled_df.columns:
        means = scrambled_df[mean_col].dropna()
        ses = scrambled_df[se_col].dropna()
        
        if len(means) > 0:
            # Simple mean across categories
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
    
    # Add "Mean" to categories if we have data
    if all_category_data:
        categories = list(categories) + ['Mean']
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for plotting
    scrambled_values = []
    scrambled_errors = []
    best_model_values = []
    
    # Plot each category
    for category in categories:
        if category == 'Mean':
            # Use computed "Mean" values
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
        
        scrambled_values.append(scrambled_val)
        scrambled_errors.append(ci_half_width)
        best_model_values.append(best_model_val)
    
    # Calculate x positions for each category
    n_categories = len(categories)
    spacing = 1.5  # Spacing between category groups (matching combined plot)
    category_centers = np.arange(n_categories) * spacing
    
    # Scrambled data with error bars (matching combined plot style)
    ax.errorbar(category_centers, scrambled_values, yerr=scrambled_errors,
               fmt='o', capsize=5, capthick=3, elinewidth=3, markersize=12,
               label=f'Scrambled {metric_label}', color=color, alpha=0.7,
               markeredgecolor='red', markeredgewidth=1, zorder=6)
    
    # Best model data (matching combined plot style)
    ax.scatter(category_centers, best_model_values, s=150, marker='s',
              label=f'Original {metric_label}', color=color, alpha=0.7, zorder=5,
              edgecolors='none', linewidths=0)
    
    # Customize plot (matching combined plot style)
    ax.set_ylabel('Average Precision (AP)', fontsize=18)
    title_text = f'{model_type}'
    ax.set_title(title_text, fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(category_centers)
    
    # Create custom labels with "Mean" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'Mean':
            tick_labels.append(r'$\mathbf{Mean}$')  # Bold "Mean" using LaTeX
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=14)
    
    # Set y-axis limits
    all_values = [v for v in scrambled_values + best_model_values if not np.isnan(v)]
    if all_values:
        y_min = max(0, min(all_values) - 10)
        y_max = 100
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"AP75 plot saved to {output_path}")
    
    plt.close()


def find_other_model_dir(current_dir):
    """Find the corresponding directory for the other model type by substituting camera/silhouette."""
    current_dir_str = str(current_dir)
    current_dir_lower = current_dir_str.lower()
    
    if 'camera' in current_dir_lower:
        # Replace camera with silhouette (handle both lowercase and capitalized)
        other_dir = current_dir_str.replace('camera', 'silhouette').replace('Camera', 'Silhouette')
    elif 'silhouette' in current_dir_lower:
        # Replace silhouette with camera (handle both lowercase and capitalized)
        other_dir = current_dir_str.replace('silhouette', 'camera').replace('Silhouette', 'Camera')
    else:
        return None
    
    other_path = Path(other_dir)
    return other_path if other_path.exists() else None


def create_both_models_ap75_plot(scrambled_dir, best_model_dir, output_path=None):
    """Create AP plot comparing both camera and silhouette models."""
    
    # Determine which model type we're starting with
    combined_path = str(scrambled_dir) + str(best_model_dir)
    if 'camera' in combined_path.lower():
        primary_model = 'camera'
        primary_color = '#d62728'  # Red
        secondary_color = '#9467bd'  # Purple
    elif 'silhouette' in combined_path.lower():
        primary_model = 'silhouette'
        primary_color = '#9467bd'  # Purple
        secondary_color = '#d62728'  # Red
    else:
        raise ValueError("Could not determine model type from directory paths. Must contain 'camera' or 'silhouette'.")
    
    # Find the other model's directories
    other_scrambled_dir = find_other_model_dir(scrambled_dir)
    other_best_model_dir = find_other_model_dir(best_model_dir)
    
    if other_scrambled_dir is None or other_best_model_dir is None:
        raise FileNotFoundError(f"Could not find corresponding {('camera' if primary_model == 'silhouette' else 'silhouette')} model directories.")
    
    # Load data for both models
    primary_scrambled_df = load_scrambled_data(scrambled_dir)
    primary_best_model_df = load_best_model_data(best_model_dir)
    secondary_scrambled_df = load_scrambled_data(other_scrambled_dir)
    secondary_best_model_df = load_best_model_data(other_best_model_dir)
    
    # Only AP metric
    metric = 'ap_instance_per_cat'
    metric_label = 'AP'
    
    # Get categories (use primary model's categories as reference)
    categories = primary_scrambled_df['category_name'].unique().tolist()
    
    # Compute "Mean" category for both models
    def compute_mean_data(scrambled_df, best_model_df):
        all_category_data = {}
        mean_col = f'{metric}_mean'
        se_col = f'{metric}_se'
        n_seeds_col = f'{metric}_n_seeds'
        
        if mean_col in scrambled_df.columns:
            means = scrambled_df[mean_col].dropna()
            ses = scrambled_df[se_col].dropna()
            
            if len(means) > 0:
                all_category_data[f'{metric}_mean'] = means.mean()
                if len(ses) > 0:
                    all_category_data[f'{metric}_se'] = np.sqrt((ses**2).sum()) / len(ses)
                    if n_seeds_col in scrambled_df.columns:
                        n_seeds_values = scrambled_df[n_seeds_col].dropna()
                        if len(n_seeds_values) > 0:
                            avg_n_seeds = int(n_seeds_values.mean())
                            all_category_data[f'{metric}_n_seeds'] = avg_n_seeds
                        else:
                            all_category_data[f'{metric}_n_seeds'] = 10
                    else:
                        all_category_data[f'{metric}_n_seeds'] = 10
                else:
                    all_category_data[f'{metric}_se'] = 0.0
                    all_category_data[f'{metric}_n_seeds'] = 10
        
        if metric in best_model_df.columns:
            values = best_model_df[metric].dropna()
            if len(values) > 0:
                all_category_data[metric] = values.mean()
        
        return all_category_data
    
    primary_all_data = compute_mean_data(primary_scrambled_df, primary_best_model_df)
    secondary_all_data = compute_mean_data(secondary_scrambled_df, secondary_best_model_df)
    
    # Add "Mean" to categories if we have data
    if primary_all_data or secondary_all_data:
        categories = list(categories) + ['Mean']
    
    # Create single figure (wider to accommodate legend on the right)
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    # Prepare data for plotting
    def get_category_data(category, scrambled_df, best_model_df, all_data):
        if category == 'Mean':
            scrambled_val = all_data.get(f'{metric}_mean', np.nan)
            se_val = all_data.get(f'{metric}_se', 0.0)
            n_seeds = all_data.get(f'{metric}_n_seeds', 10)
            best_model_val = all_data.get(metric, np.nan)
        else:
            scrambled_row = scrambled_df[scrambled_df['category_name'] == category]
            best_model_row = best_model_df[best_model_df['category_name'] == category]
            
            if len(scrambled_row) > 0:
                scrambled_val = scrambled_row[f'{metric}_mean'].iloc[0]
                se_val = scrambled_row[f'{metric}_se'].iloc[0]
                n_seeds_col = f'{metric}_n_seeds'
                if n_seeds_col in scrambled_row.columns:
                    n_seeds = int(scrambled_row[n_seeds_col].iloc[0])
                else:
                    n_seeds = 10
            else:
                scrambled_val = np.nan
                se_val = 0.0
                n_seeds = 10
            
            if len(best_model_row) > 0:
                best_model_val = best_model_row[metric].iloc[0]
            else:
                best_model_val = np.nan
        
        # Calculate 95% confidence interval
        if not np.isnan(scrambled_val) and se_val > 0 and n_seeds > 1:
            df = n_seeds - 1
            t_value = stats.t.ppf(0.975, df)
            ci_half_width = se_val * t_value
        else:
            ci_half_width = 0.0
        
        return scrambled_val, ci_half_width, best_model_val
    
    # Calculate x positions for each category
    n_categories = len(categories)
    spacing = 1.5
    category_centers = np.arange(n_categories) * spacing
    width = 0.15  # Width for offsetting markers within each category
    
    # Plot both models
    for model_idx, (scrambled_df, best_model_df, all_data, color, model_name) in enumerate([
        (primary_scrambled_df, primary_best_model_df, primary_all_data, primary_color, 'Camera' if primary_model == 'camera' else 'Silhouette'),
        (secondary_scrambled_df, secondary_best_model_df, secondary_all_data, secondary_color, 'Camera' if primary_model == 'silhouette' else 'Silhouette')
    ]):
        offset = (model_idx - 0.5) * width
        x_positions = category_centers + offset
        
        scrambled_values = []
        scrambled_errors = []
        best_model_values = []
        
        for category in categories:
            scrambled_val, ci_half_width, best_model_val = get_category_data(
                category, scrambled_df, best_model_df, all_data
            )
            scrambled_values.append(scrambled_val)
            scrambled_errors.append(ci_half_width)
            best_model_values.append(best_model_val)
        
        # Scrambled data with error bars
        ax.errorbar(x_positions, scrambled_values, yerr=scrambled_errors,
                   fmt='o', capsize=5, capthick=3, elinewidth=3, markersize=12,
                   label=f'Scrambled {model_name} {metric_label}', color=color, alpha=0.7,
                   markeredgecolor='red', markeredgewidth=1, zorder=6)
        
        # Best model data
        ax.scatter(x_positions, best_model_values, s=150, marker='s',
                  label=f'Original {model_name} {metric_label}', color=color, alpha=0.7, zorder=5,
                  edgecolors='none', linewidths=0)
    
    # Customize plot
    ax.set_ylabel('Average Precision (AP)', fontsize=24)
    title_text = 'Camera vs Silhouette'
    ax.set_title(title_text, fontsize=28, fontweight='bold', pad=20)
    ax.set_xticks(category_centers)
    
    # Create custom labels with "Mean" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'Mean':
            tick_labels.append(r'$\mathbf{Mean}$')
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    
    # Set y-axis limits - collect all values from both models
    all_values = []
    for scrambled_df, best_model_df, all_data in [
        (primary_scrambled_df, primary_best_model_df, primary_all_data),
        (secondary_scrambled_df, secondary_best_model_df, secondary_all_data)
    ]:
        for category in categories:
            scrambled_val, _, best_model_val = get_category_data(
                category, scrambled_df, best_model_df, all_data
            )
            if not np.isnan(scrambled_val):
                all_values.append(scrambled_val)
            if not np.isnan(best_model_val):
                all_values.append(best_model_val)
    
    if all_values:
        y_min = max(0, min(all_values) - 10)
        y_max = 100
        ax.set_ylim(y_min, y_max)
    
    # Adjust layout to keep plot area the same size and add space for legend on the right
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Both models AP plot saved to {output_path}")
    
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
    parser.add_argument(
        '--both_models',
        action='store_true',
        help='Create a plot comparing both camera and silhouette models (AP only). Automatically finds the other model directory by substituting camera/silhouette in the path.'
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
    
    # Set default output path for AP75-only plot
    ap75_output_path = os.path.join(scrambled_dir, "scrambled_vs_best_model_comparison_ap75.png")
    
    # Set default output path for both models plot
    if args.both_models:
        if args.output:
            both_models_output_path = args.output
        else:
            both_models_output_path = os.path.join(scrambled_dir, "scrambled_vs_best_model_comparison_both_models_ap.png")
    
    print("=" * 60)
    print(f"Scrambled directory: {scrambled_dir}")
    print(f"Best model directory: {best_model_dir}")
    if args.both_models:
        print(f"Both models plot path: {both_models_output_path}")
    else:
        print(f"Output path: {output_path}")
        print(f"AP75-only plot path: {ap75_output_path}")
        print(f"Plot style: {'Combined' if args.combined else 'Separate subplots'}")
    print("=" * 60)
    
    try:
        if args.both_models:
            # Create the both models comparison plot
            create_both_models_ap75_plot(scrambled_dir, best_model_dir, both_models_output_path)
        else:
            # Always create the main comparison plot
            if args.combined:
                create_combined_plot(scrambled_dir, best_model_dir, output_path)
            else:
                create_plot(scrambled_dir, best_model_dir, output_path)
            
            # Always create the AP75-only plot
            create_ap75_plot(scrambled_dir, best_model_dir, ap75_output_path)
        
        print("\nDone!")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


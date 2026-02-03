#!/usr/bin/env python3
"""
Create a grouped bar plot from mask_metrics_category_summary.csv
Shows AP, AP75, and AP50 metrics as side-by-side bars by category.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import json
import glob
import re
import os
import sys


def create_grouped_bar_plot(csv_dir, output_path=None):
    """
    Create a grouped bar plot from mask_metrics_category_summary.csv
    
    Args:
        csv_dir: Directory containing mask_metrics_category_summary.csv
        output_path: Optional path to save the plot. If None, saves to csv_dir.
    """
    csv_dir = Path(csv_dir)
    csv_path = csv_dir / "mask_metrics_category_summary.csv"
    dataset_csv_path = csv_dir / "mask_metrics_dataset_summary.csv"
    
    # Determine model type from directory path
    dir_path_str = str(csv_dir).lower()
    if 'camera' in dir_path_str:
        model_type = 'Camera'
    elif 'silhouette' in dir_path_str:
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract the relevant columns
    categories = df['category_name'].values.tolist()
    ap_mean = df['ap_instance_per_cat_mean'].values
    ap_se = df['ap_instance_per_cat_se'].values
    ap_n_folds = df['ap_instance_per_cat_n_folds'].values
    
    ap75_mean = df['ap75_instance_per_cat_mean'].values
    ap75_se = df['ap75_instance_per_cat_se'].values
    ap75_n_folds = df['ap75_instance_per_cat_n_folds'].values
    
    ap50_mean = df['ap50_instance_per_cat_mean'].values
    ap50_se = df['ap50_instance_per_cat_se'].values
    ap50_n_folds = df['ap50_instance_per_cat_n_folds'].values
    
    # Read dataset summary CSV and add "Mean" category
    if dataset_csv_path.exists():
        dataset_df = pd.read_csv(dataset_csv_path)
        
        # Extract values for Mean category (dataset-level metrics)
        # Note: dataset summary has values in decimal form (0-1), need to convert to percentage (0-100)
        ap_row = dataset_df[dataset_df['metric_name'] == 'ap_instance_Aweighted']
        ap50_row = dataset_df[dataset_df['metric_name'] == 'ap50_instance_Aweighted']
        ap75_row = dataset_df[dataset_df['metric_name'] == 'ap75_instance_Aweighted']
        
        if len(ap_row) > 0 and len(ap50_row) > 0 and len(ap75_row) > 0:
            # Convert from decimal (0-1) to percentage (0-100)
            ap_mean_val = ap_row['mean'].iloc[0] * 100
            ap_se_val = ap_row['se'].iloc[0] * 100
            ap_n_val = int(ap_row['n_folds'].iloc[0])
            
            ap50_mean_val = ap50_row['mean'].iloc[0] * 100
            ap50_se_val = ap50_row['se'].iloc[0] * 100
            ap50_n_val = int(ap50_row['n_folds'].iloc[0])
            
            ap75_mean_val = ap75_row['mean'].iloc[0] * 100
            ap75_se_val = ap75_row['se'].iloc[0] * 100
            ap75_n_val = int(ap75_row['n_folds'].iloc[0])
            
            # Add Mean category to the arrays
            categories.append('Mean')
            ap_mean = np.append(ap_mean, ap_mean_val)
            ap_se = np.append(ap_se, ap_se_val)
            ap_n_folds = np.append(ap_n_folds, ap_n_val)
            
            ap75_mean = np.append(ap75_mean, ap75_mean_val)
            ap75_se = np.append(ap75_se, ap75_se_val)
            ap75_n_folds = np.append(ap75_n_folds, ap75_n_val)
            
            ap50_mean = np.append(ap50_mean, ap50_mean_val)
            ap50_se = np.append(ap50_se, ap50_se_val)
            ap50_n_folds = np.append(ap50_n_folds, ap50_n_val)
        else:
            print(f"Warning: Could not find required metrics in {dataset_csv_path}, skipping Mean category")
    else:
        print(f"Warning: Could not find {dataset_csv_path}, skipping Mean category")
    
    # Colors matching plot_scrambled_results.py
    # AP: Blue, AP50: Green, AP75: Orange
    colors = {
        'AP': '#1f77b4',
        'AP50': '#2ca02c',
        'AP75': '#ff7f0e'
    }
    
    # Create figure (squashed vertically)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the bar positions for grouped bars
    n_categories = len(categories)
    x_pos = np.arange(n_categories)
    width = 0.25  # Width of each bar
    
    # Calculate error bars (95% CI using t-distribution)
    def calculate_error(se_values, n_folds_values):
        errors = []
        for se, n in zip(se_values, n_folds_values):
            if not np.isnan(se) and se > 0 and n > 1:
                df = n - 1
                t_value = stats.t.ppf(0.975, df)  # 95% CI
                errors.append(se * t_value)
            else:
                errors.append(0.0)
        return np.array(errors)
    
    ap_error = calculate_error(ap_se, ap_n_folds)
    ap75_error = calculate_error(ap75_se, ap75_n_folds)
    ap50_error = calculate_error(ap50_se, ap50_n_folds)
    
    # Clip error bars so they don't exceed 100
    ap_error = np.minimum(ap_error, 100 - ap_mean)
    ap75_error = np.minimum(ap75_error, 100 - ap75_mean)
    ap50_error = np.minimum(ap50_error, 100 - ap50_mean)
    
    # Create grouped bars
    # Position bars side by side
    # Error bar styling via error_kw parameter
    error_kw = {'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5}
    
    # For silhouette models, only plot AP
    if model_type == 'Silhouette':
        bars1 = ax.bar(x_pos, ap_mean, width, yerr=ap_error, 
                       label='mAP', color=colors['AP'], alpha=0.5, 
                       error_kw=error_kw, ecolor='lightgrey')
    else:
        bars1 = ax.bar(x_pos - width, ap_mean, width, yerr=ap_error, 
                       label='mAP', color=colors['AP'], alpha=0.5, 
                       error_kw=error_kw, ecolor='lightgrey')
        bars2 = ax.bar(x_pos, ap75_mean, width, yerr=ap75_error,
                       label='mAP75', color=colors['AP75'], alpha=0.5,
                       error_kw=error_kw, ecolor='lightgrey')
        bars3 = ax.bar(x_pos + width, ap50_mean, width, yerr=ap50_error,
                       label='mAP50', color=colors['AP50'], alpha=0.5,
                       error_kw=error_kw, ecolor='lightgrey')
    
    # Customize the plot
    ax.set_ylabel('Average Precision (AP)', fontsize=15)
    ax.set_title(f'{model_type} Model Performance Metrics Across 6 Fold CV', fontsize=19, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    
    # Create custom labels with "Mean" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'Mean':
            tick_labels.append(r'$\mathbf{Mean}$')  # Bold "Mean" using LaTeX
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal reference lines for YTVIS 2021
    ax.axhline(y=64.5, color=colors['AP'], linestyle='--', linewidth=2, 
               label='YTVIS 2021 mAP', alpha=0.7)
    ax.axhline(y=72.4, color=colors['AP75'], linestyle='--', linewidth=2, 
               label='YTVIS 2021 mAP75', alpha=0.7)
    
    # Move legend outside plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    
    # Set y-axis limit
    if model_type == 'Silhouette':
        # For silhouette, don't force to 100, calculate based on data
        max_value = np.max(ap_mean + ap_error)
        y_max = min(100, max_value * 1.1)  # Add 10% padding, but cap at 100
        ax.set_ylim([0, y_max])
    else:
        max_value = min(100, max(
            np.max(ap_mean + ap_error),
            np.max(ap75_mean + ap75_error),
            np.max(ap50_mean + ap50_error)
        ))
        ax.set_ylim([0, 100])
    
    # Adjust layout to accommodate legend outside plot and move title up
    plt.tight_layout(rect=[0, 0, 0.85, 0.95], pad=2.0)
    
    # Save the plot
    if output_path is None:
        output_path = csv_dir / "mask_metrics_category_grouped.png"
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also show the plot
    plt.show()


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


def create_both_models_ap_plot(csv_dir, output_path=None):
    """
    Create an AP bar plot comparing both camera and silhouette models.
    
    Args:
        csv_dir: Directory containing mask_metrics_category_summary.csv (for one model)
        output_path: Optional path to save the plot. If None, saves to csv_dir.
    """
    csv_dir = Path(csv_dir)
    csv_path = csv_dir / "mask_metrics_category_summary.csv"
    dataset_csv_path = csv_dir / "mask_metrics_dataset_summary.csv"
    
    # Determine which model type we're starting with
    dir_path_str = str(csv_dir).lower()
    if 'camera' in dir_path_str:
        primary_model = 'Camera'
        primary_color = '#ff7f0e'  # Orange
        secondary_color = '#2ca02c'  # Green
    elif 'silhouette' in dir_path_str:
        primary_model = 'Silhouette'
        primary_color = '#2ca02c'  # Green
        secondary_color = '#ff7f0e'  # Orange
    else:
        raise ValueError("Could not determine model type from directory path. Must contain 'camera' or 'silhouette'.")
    
    # Find the other model's directory
    other_csv_dir = find_other_model_dir(csv_dir)
    if other_csv_dir is None:
        raise FileNotFoundError(f"Could not find corresponding {('Camera' if primary_model == 'Silhouette' else 'Silhouette')} model directory.")
    
    other_csv_path = other_csv_dir / "mask_metrics_category_summary.csv"
    other_dataset_csv_path = other_csv_dir / "mask_metrics_dataset_summary.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    if not other_csv_path.exists():
        raise FileNotFoundError(f"Could not find {other_csv_path}")
    
    # Read the CSVs for both models
    primary_df = pd.read_csv(csv_path)
    secondary_df = pd.read_csv(other_csv_path)
    
    # Extract AP data for primary model
    primary_categories = primary_df['category_name'].values.tolist()
    primary_ap_mean = primary_df['ap_instance_per_cat_mean'].values
    primary_ap_se = primary_df['ap_instance_per_cat_se'].values
    primary_ap_n_folds = primary_df['ap_instance_per_cat_n_folds'].values
    
    # Extract AP data for secondary model and align with primary categories
    # Create a dictionary for quick lookup
    secondary_dict = {}
    for idx, cat in enumerate(secondary_df['category_name'].values):
        secondary_dict[cat] = {
            'mean': secondary_df['ap_instance_per_cat_mean'].iloc[idx],
            'se': secondary_df['ap_instance_per_cat_se'].iloc[idx],
            'n_folds': secondary_df['ap_instance_per_cat_n_folds'].iloc[idx]
        }
    
    # Align secondary data with primary categories
    secondary_ap_mean = []
    secondary_ap_se = []
    secondary_ap_n_folds = []
    for cat in primary_categories:
        if cat in secondary_dict:
            secondary_ap_mean.append(secondary_dict[cat]['mean'])
            secondary_ap_se.append(secondary_dict[cat]['se'])
            secondary_ap_n_folds.append(secondary_dict[cat]['n_folds'])
        else:
            # If category not found, use NaN
            secondary_ap_mean.append(np.nan)
            secondary_ap_se.append(0.0)
            secondary_ap_n_folds.append(1)
    
    secondary_ap_mean = np.array(secondary_ap_mean)
    secondary_ap_se = np.array(secondary_ap_se)
    secondary_ap_n_folds = np.array(secondary_ap_n_folds)
    
    # Read dataset summary CSVs and add "Mean" category
    def add_mean_category(df, dataset_csv_path, categories, mean_vals, se_vals, n_folds_vals):
        if dataset_csv_path.exists():
            dataset_df = pd.read_csv(dataset_csv_path)
            ap_row = dataset_df[dataset_df['metric_name'] == 'ap_instance_Aweighted']
            
            if len(ap_row) > 0:
                ap_mean_val = ap_row['mean'].iloc[0] * 100  # Convert to percentage
                ap_se_val = ap_row['se'].iloc[0] * 100
                ap_n_val = int(ap_row['n_folds'].iloc[0])
                
                categories.append('Mean')
                mean_vals = np.append(mean_vals, ap_mean_val)
                se_vals = np.append(se_vals, ap_se_val)
                n_folds_vals = np.append(n_folds_vals, ap_n_val)
            else:
                print(f"Warning: Could not find AP metric in {dataset_csv_path}, skipping Mean category")
        else:
            print(f"Warning: Could not find {dataset_csv_path}, skipping Mean category")
        
        return categories, mean_vals, se_vals, n_folds_vals
    
    # Add Mean category for primary model
    primary_categories, primary_ap_mean, primary_ap_se, primary_ap_n_folds = add_mean_category(
        primary_df, dataset_csv_path, primary_categories.copy(), 
        primary_ap_mean.copy(), primary_ap_se.copy(), primary_ap_n_folds.copy()
    )
    
    # Add Mean category for secondary model
    # First, get the Mean value from secondary dataset
    secondary_mean_val = np.nan
    secondary_mean_se = 0.0
    secondary_mean_n = 1
    if other_dataset_csv_path.exists():
        other_dataset_df = pd.read_csv(other_dataset_csv_path)
        ap_row = other_dataset_df[other_dataset_df['metric_name'] == 'ap_instance_Aweighted']
        if len(ap_row) > 0:
            secondary_mean_val = ap_row['mean'].iloc[0] * 100
            secondary_mean_se = ap_row['se'].iloc[0] * 100
            secondary_mean_n = int(ap_row['n_folds'].iloc[0])
    
    # Append Mean to secondary arrays
    secondary_ap_mean = np.append(secondary_ap_mean, secondary_mean_val)
    secondary_ap_se = np.append(secondary_ap_se, secondary_mean_se)
    secondary_ap_n_folds = np.append(secondary_ap_n_folds, secondary_mean_n)
    
    # Use primary model's categories as reference (should be the same for both)
    categories = primary_categories
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the bar positions for grouped bars
    n_categories = len(categories)
    x_pos = np.arange(n_categories)
    width = 0.35  # Width of each bar
    
    # Calculate error bars (95% CI using t-distribution)
    def calculate_error(se_values, n_folds_values):
        errors = []
        for se, n in zip(se_values, n_folds_values):
            if not np.isnan(se) and se > 0 and n > 1:
                df = n - 1
                t_value = stats.t.ppf(0.975, df)  # 95% CI
                errors.append(se * t_value)
            else:
                errors.append(0.0)
        return np.array(errors)
    
    primary_ap_error = calculate_error(primary_ap_se, primary_ap_n_folds)
    secondary_ap_error = calculate_error(secondary_ap_se, secondary_ap_n_folds)
    
    # Clip error bars so they don't exceed 100
    primary_ap_error = np.minimum(primary_ap_error, 100 - primary_ap_mean)
    secondary_ap_error = np.minimum(secondary_ap_error, 100 - secondary_ap_mean)
    
    # Error bar styling
    error_kw = {'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5}
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width/2, primary_ap_mean, width, yerr=primary_ap_error,
                   label=f'{primary_model} mAP', color=primary_color, alpha=0.7,
                   error_kw=error_kw, ecolor='lightgrey')
    bars2 = ax.bar(x_pos + width/2, secondary_ap_mean, width, yerr=secondary_ap_error,
                   label=f"{'Camera' if primary_model == 'Silhouette' else 'Silhouette'} mAP", 
                   color=secondary_color, alpha=0.7,
                   error_kw=error_kw, ecolor='lightgrey')
    
    # Add horizontal reference line for YTVIS 2021 AP
    ax.axhline(y=64.5, color='gray', linestyle='--', linewidth=2,
               label='YTVIS 2021 mAP', alpha=0.7)
    
    # Customize the plot
    ax.set_ylabel('Average Precision (AP)', fontsize=18)
    ax.set_title('Camera vs Silhouette Model AP Performance Across 6 Fold CV', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    
    # Create custom labels with "Mean" in bold
    tick_labels = []
    for cat in categories:
        if cat == 'Mean':
            tick_labels.append(r'$\mathbf{Mean}$')  # Bold "Mean" using LaTeX
        else:
            tick_labels.append(cat)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='lower left', fontsize=14)
    
    # Set y-axis limit capped at 100
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = csv_dir / "mask_metrics_category_both_models_ap.png"
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Both models AP plot saved to {output_path}")
    
    plt.close()


def create_per_fold_plot(csv_dir, output_path=None):
    """
    Create a scatter plot showing AP, AP50, and AP75 for each fold.
    
    Args:
        csv_dir: Directory containing mask_metrics_dataset_combined.csv
        output_path: Optional path to save the plot. If None, saves to csv_dir.
    """
    csv_dir = Path(csv_dir)
    combined_csv_path = csv_dir / "mask_metrics_dataset_combined.csv"
    
    # Determine model type from directory path
    dir_path_str = str(csv_dir).lower()
    if 'camera' in dir_path_str:
        model_type = 'Camera'
    elif 'silhouette' in dir_path_str:
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    if not combined_csv_path.exists():
        raise FileNotFoundError(f"Could not find {combined_csv_path}")
    
    # Read the combined CSV
    df = pd.read_csv(combined_csv_path)
    
    # Extract AP, AP50, AP75 values for each fold
    # The CSV has metric_name column with values like 'ap_instance_Aweighted', etc.
    # and value column with the actual metric values (in decimal 0-1 format)
    
    folds = sorted(df['fold_number'].unique())
    metrics_data = {
        'AP': [],
        'AP50': [],
        'AP75': []
    }
    
    for fold in folds:
        fold_data = df[df['fold_number'] == fold]
        
        # Extract AP
        ap_row = fold_data[fold_data['metric_name'] == 'ap_instance_Aweighted']
        if len(ap_row) > 0:
            ap_val = ap_row['value'].iloc[0] * 100  # Convert to percentage
            metrics_data['AP'].append(ap_val)
        else:
            metrics_data['AP'].append(np.nan)
        
        # Extract AP50
        ap50_row = fold_data[fold_data['metric_name'] == 'ap50_instance_Aweighted']
        if len(ap50_row) > 0:
            ap50_val = ap50_row['value'].iloc[0] * 100  # Convert to percentage
            metrics_data['AP50'].append(ap50_val)
        else:
            metrics_data['AP50'].append(np.nan)
        
        # Extract AP75
        ap75_row = fold_data[fold_data['metric_name'] == 'ap75_instance_Aweighted']
        if len(ap75_row) > 0:
            ap75_val = ap75_row['value'].iloc[0] * 100  # Convert to percentage
            metrics_data['AP75'].append(ap75_val)
        else:
            metrics_data['AP75'].append(np.nan)
    
    # Colors matching other plots
    colors = {
        'AP': '#1f77b4',      # Blue
        'AP50': '#2ca02c',    # Green
        'AP75': '#ff7f0e'     # Orange
    }
    
    # Markers for each metric
    markers = {
        'AP': 'o',      # Circle
        'AP50': 's',    # Square
        'AP75': '^'     # Triangle
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert folds to numeric positions for plotting
    x_pos = np.array(folds)
    
    # For silhouette models, only plot AP
    metrics_to_plot = ['AP'] if model_type == 'Silhouette' else ['AP', 'AP75', 'AP50']
    
    # Plot each metric with scatter points and connecting lines
    for metric in metrics_to_plot:
        values = np.array(metrics_data[metric])
        # Filter out NaN values for line plotting
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            x_valid = x_pos[valid_mask]
            y_valid = values[valid_mask]
            
            # Plot line connecting the points (no label to avoid duplicates)
            ax.plot(x_valid, y_valid, color=colors[metric], 
                   linewidth=2, alpha=0.6, linestyle='-')
            
            # Plot scatter points
            # Use mAP, mAP75, mAP50 in labels
            metric_label = 'mAP' if metric == 'AP' else ('mAP75' if metric == 'AP75' else 'mAP50')
            # Set zorder so triangle (AP75) appears on top of square (AP50)
            zorder_map = {'AP': 5, 'AP50': 6, 'AP75': 7}
            ax.scatter(x_valid, y_valid, color=colors[metric], 
                      marker=markers[metric], s=150, alpha=0.8,
                      edgecolors='white', linewidths=2, zorder=zorder_map[metric],
                      label=metric_label)
    
    # Customize the plot
    ax.set_xlabel('Fold', fontsize=20)
    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=20)
    ax.set_title(f'{model_type}', 
                 fontsize=21, fontweight='bold', pad=20)
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', alpha=0.2, linestyle=':')
    
    # Add horizontal reference lines for YTVIS 2021
    ax.axhline(y=64.5, color=colors['AP'], linestyle='--', linewidth=2, 
               label='YTVIS 2021 mAP', alpha=0.7)
    ax.axhline(y=72.4, color=colors['AP75'], linestyle='--', linewidth=2, 
               label='YTVIS 2021 mAP75', alpha=0.7)
    
    # Legend - larger text and keep inside plot
    # Place legend on left for Camera model, right for others
    legend_loc = 'upper left' if model_type == 'Camera' else 'upper right'
    ax.legend(loc=legend_loc, fontsize=16, frameon=True)
    
    # Calculate y-axis limits based on data
    all_values = []
    for metric in metrics_to_plot:
        values = np.array(metrics_data[metric])
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            all_values.extend(valid_values)
    
    if len(all_values) > 0:
        y_min = min(all_values)
        y_max = max(all_values)
        # Add padding: 5% below minimum
        y_range = y_max - y_min
        y_padding_bottom = max(y_range * 0.05, 2)  # At least 2 units of padding
        y_axis_min = max(0, y_min - y_padding_bottom)  # Don't go below 0
        # For silhouette models, don't force to 100
        if model_type == 'Silhouette':
            # Add small padding (5% or at least 2 units), but cap at 100
            y_padding_top = max(y_range * 0.05, 2)
            y_axis_max = min(100, y_max + y_padding_top)
        else:
            y_axis_max = 100  # Always go to 100 for non-silhouette
        ax.set_ylim([y_axis_min, y_axis_max])
    else:
        # Fallback if no valid data
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = csv_dir / "mask_metrics_per_fold.png"
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-fold plot saved to {output_path}")
    
    plt.close()


def find_inference_dir(fold_dir):
    """Find the inference directory within a fold directory."""
    # Look for inference directories in checkpoint subdirectories
    checkpoint_dirs = glob.glob(os.path.join(fold_dir, "checkpoint_*/inference"))
    if checkpoint_dirs:
        # Return the first one found (assuming one checkpoint per fold)
        return checkpoint_dirs[0]
    return None

def is_valid_fold_dir(dir_path):
    """Check if directory is a valid fold directory (fold followed by digits only)."""
    if not dir_path.is_dir():
        return False
    name = dir_path.name
    # Match pattern: "fold" followed by one or more digits, nothing else
    return bool(re.match(r'^fold\d+$', name))

def create_confusion_matrix_from_all_folds(base_dir, output_path=None):
    """
    Create a confusion matrix by combining predictions from all folds.
    
    Args:
        base_dir: Base directory containing fold directories (e.g., /path/to/camera)
        output_path: Optional path to save the plot. If None, saves to base_dir/folds_summary.
    """
    base_path = Path(base_dir)
    
    # Determine model type from directory path
    dir_path_str = str(base_path).lower()
    if 'camera' in dir_path_str:
        model_type = 'Camera'
    elif 'silhouette' in dir_path_str:
        model_type = 'Silhouette'
    else:
        model_type = 'Model'  # Default if neither found
    
    # Find all fold directories
    fold_dirs = sorted([d for d in base_path.iterdir() if is_valid_fold_dir(d)])
    
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No valid fold directories found in {base_path}")
    
    print(f"Found {len(fold_dirs)} fold directories")
    
    # Import confusion matrix functions
    # Add the script directory to path to import confusion_mat_plot
    script_dir = Path(__file__).parent
    import sys
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    try:
        from confusion_mat_plot import simple_confusion_matrix, plot_confusion_matrix
    except ImportError as e:
        raise ImportError(f"Could not import confusion matrix functions from confusion_mat_plot.py: {e}")
    
    # Collect all predictions and ground truth from all folds
    all_predictions = []
    all_ground_truth_annotations = []
    ground_truth_categories = None
    
    for fold_dir in fold_dirs:
        fold_num = int(re.match(r'^fold(\d+)$', fold_dir.name).group(1))
        print(f"Processing fold {fold_num}...")
        
        # Find inference directory
        inference_dir = find_inference_dir(fold_dir)
        if inference_dir is None:
            print(f"  Warning: Could not find inference directory in {fold_dir}, skipping")
            continue
        
        # Load results.json
        results_json_path = Path(inference_dir) / "results.json"
        if not results_json_path.exists():
            print(f"  Warning: Could not find {results_json_path}, skipping")
            continue
        
        with open(results_json_path, 'r') as f:
            fold_predictions = json.load(f)
        
        # Add fold number to each prediction for tracking (optional)
        for pred in fold_predictions:
            pred['fold_number'] = fold_num
        
        all_predictions.extend(fold_predictions)
        print(f"  Loaded {len(fold_predictions)} predictions from fold {fold_num}")
        
        # Load ground truth for this fold
        # The validation JSON is in the checkpoint directory (parent of inference directory)
        checkpoint_dir = Path(inference_dir).parent
        # Look for val_fold*_all_frames*.json in the checkpoint directory
        val_json_pattern = str(checkpoint_dir / "val_fold*_all_frames*.json")
        matches = glob.glob(val_json_pattern)
        
        if matches:
            val_json_path = Path(matches[0])
            with open(val_json_path, 'r') as f:
                fold_ground_truth = json.load(f)
            
            # Combine annotations from this fold
            all_ground_truth_annotations.extend(fold_ground_truth['annotations'])
            
            # Store categories from first fold (should be same across all folds)
            if ground_truth_categories is None:
                ground_truth_categories = fold_ground_truth['categories']
            
            print(f"  Loaded ground truth from {val_json_path} ({len(fold_ground_truth['annotations'])} annotations)")
        else:
            print(f"  Warning: Could not find validation JSON in {checkpoint_dir}")
    
    # Create combined ground truth structure
    if ground_truth_categories is None or len(all_ground_truth_annotations) == 0:
        raise FileNotFoundError("Could not find ground truth JSON files for any fold")
    
    ground_truth = {
        'categories': ground_truth_categories,
        'annotations': all_ground_truth_annotations
    }
    
    print(f"\nTotal ground truth annotations across all folds: {len(all_ground_truth_annotations)}")
    
    if len(all_predictions) == 0:
        raise ValueError("No predictions found in any fold")
    
    print(f"\nTotal predictions across all folds: {len(all_predictions)}")
    
    # Compute confusion matrix using simple method: takes top scoring prediction per video
    # This matches the behavior in mask_metrics.py
    print("\nComputing confusion matrix (using top scoring prediction per video)...")
    # Use confidence_threshold=0.0 to include all predictions, then take top per video
    cm, class_names, metrics = simple_confusion_matrix(all_predictions, ground_truth, confidence_threshold=0.0)
    
    # Determine output path
    if output_path is None:
        folds_summary_dir = base_path / "folds_summary"
        folds_summary_dir.mkdir(parents=True, exist_ok=True)
        output_path = folds_summary_dir / "confusion_matrix_all_folds.png"
    else:
        output_path = Path(output_path)
    
    # Plot confusion matrix with larger text
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
    
    # Increase font sizes for all text elements
    # Title
    ax.set_title(f"{model_type} Model Confusion Matrix (All Folds Combined)\n(Method: Highest Scoring Prediction per Video)", 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Axis labels
    ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax.set_ylabel('True', fontsize=18, fontweight='bold')
    
    # Tick labels
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=16)
    
    # Increase font size for text in confusion matrix cells
    for text in ax.texts:
        text.set_fontsize(16)
        text.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nConfusion matrix saved to {output_path}")
    
    # Print metrics summary
    print("\nPer-Class Classification Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-" * 80)
    
    total_tp = total_fp = total_fn = 0
    for class_name in class_names:
        if class_name in metrics:
            m = metrics[class_name]
            print(f"{class_name:<15} {m['precision']:<10.3f} {m['recall']:<10.3f} "
                  f"{m['f1']:<10.3f} {m['tp']:<5} {m['fp']:<5} {m['fn']:<5}")
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    print("-" * 80)
    print(f"{'OVERALL':<15} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {total_tp:<5} {total_fp:<5} {total_fn:<5}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Create a grouped bar plot from mask_metrics_category_summary.csv'
    )
    parser.add_argument(
        'csv_dir',
        type=str,
        help='Directory containing mask_metrics_category_summary.csv (or base directory for --cm_plot)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: csv_dir/mask_metrics_category_grouped.png)'
    )
    parser.add_argument(
        '--both_models',
        action='store_true',
        help='Create a plot comparing both camera and silhouette models (AP only). Automatically finds the other model directory by substituting camera/silhouette in the path.'
    )
    parser.add_argument(
        '--per_fold',
        action='store_true',
        help='Create a plot showing AP, AP50, and AP75 for each fold. Requires mask_metrics_dataset_combined.csv in the csv_dir.'
    )
    parser.add_argument(
        '--cm_plot',
        action='store_true',
        help='Create a confusion matrix plot by combining predictions from all folds. Requires base directory with fold directories (e.g., /path/to/camera).'
    )
    
    args = parser.parse_args()
    
    if args.cm_plot:
        if args.output:
            create_confusion_matrix_from_all_folds(args.csv_dir, args.output)
        else:
            create_confusion_matrix_from_all_folds(args.csv_dir)
    elif args.per_fold:
        if args.output:
            create_per_fold_plot(args.csv_dir, args.output)
        else:
            create_per_fold_plot(args.csv_dir)
    elif args.both_models:
        if args.output:
            create_both_models_ap_plot(args.csv_dir, args.output)
        else:
            create_both_models_ap_plot(args.csv_dir)
    else:
        create_grouped_bar_plot(args.csv_dir, args.output)


if __name__ == '__main__':
    main()


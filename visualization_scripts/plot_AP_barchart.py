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
    
    bars1 = ax.bar(x_pos - width, ap_mean, width, yerr=ap_error, 
                   label='AP', color=colors['AP'], alpha=0.5, 
                   error_kw=error_kw, ecolor='lightgrey')
    bars2 = ax.bar(x_pos, ap75_mean, width, yerr=ap75_error,
                   label='AP75', color=colors['AP75'], alpha=0.5,
                   error_kw=error_kw, ecolor='lightgrey')
    bars3 = ax.bar(x_pos + width, ap50_mean, width, yerr=ap50_error,
                   label='AP50', color=colors['AP50'], alpha=0.5,
                   error_kw=error_kw, ecolor='lightgrey')
    
    # Customize the plot
    ax.set_ylabel('AP Score', fontsize=15)
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
               label='YTVIS 2021 AP', alpha=0.7)
    ax.axhline(y=72.4, color=colors['AP75'], linestyle='--', linewidth=2, 
               label='YTVIS 2021 AP75', alpha=0.7)
    
    # Move legend outside plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    
    # Set y-axis limit capped at 100
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


def create_both_models_ap75_plot(csv_dir, output_path=None):
    """
    Create an AP75 bar plot comparing both camera and silhouette models.
    
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
    
    # Extract AP75 data for primary model
    primary_categories = primary_df['category_name'].values.tolist()
    primary_ap75_mean = primary_df['ap75_instance_per_cat_mean'].values
    primary_ap75_se = primary_df['ap75_instance_per_cat_se'].values
    primary_ap75_n_folds = primary_df['ap75_instance_per_cat_n_folds'].values
    
    # Extract AP75 data for secondary model and align with primary categories
    # Create a dictionary for quick lookup
    secondary_dict = {}
    for idx, cat in enumerate(secondary_df['category_name'].values):
        secondary_dict[cat] = {
            'mean': secondary_df['ap75_instance_per_cat_mean'].iloc[idx],
            'se': secondary_df['ap75_instance_per_cat_se'].iloc[idx],
            'n_folds': secondary_df['ap75_instance_per_cat_n_folds'].iloc[idx]
        }
    
    # Align secondary data with primary categories
    secondary_ap75_mean = []
    secondary_ap75_se = []
    secondary_ap75_n_folds = []
    for cat in primary_categories:
        if cat in secondary_dict:
            secondary_ap75_mean.append(secondary_dict[cat]['mean'])
            secondary_ap75_se.append(secondary_dict[cat]['se'])
            secondary_ap75_n_folds.append(secondary_dict[cat]['n_folds'])
        else:
            # If category not found, use NaN
            secondary_ap75_mean.append(np.nan)
            secondary_ap75_se.append(0.0)
            secondary_ap75_n_folds.append(1)
    
    secondary_ap75_mean = np.array(secondary_ap75_mean)
    secondary_ap75_se = np.array(secondary_ap75_se)
    secondary_ap75_n_folds = np.array(secondary_ap75_n_folds)
    
    # Read dataset summary CSVs and add "Mean" category
    def add_mean_category(df, dataset_csv_path, categories, mean_vals, se_vals, n_folds_vals):
        if dataset_csv_path.exists():
            dataset_df = pd.read_csv(dataset_csv_path)
            ap75_row = dataset_df[dataset_df['metric_name'] == 'ap75_instance_Aweighted']
            
            if len(ap75_row) > 0:
                ap75_mean_val = ap75_row['mean'].iloc[0] * 100  # Convert to percentage
                ap75_se_val = ap75_row['se'].iloc[0] * 100
                ap75_n_val = int(ap75_row['n_folds'].iloc[0])
                
                categories.append('Mean')
                mean_vals = np.append(mean_vals, ap75_mean_val)
                se_vals = np.append(se_vals, ap75_se_val)
                n_folds_vals = np.append(n_folds_vals, ap75_n_val)
            else:
                print(f"Warning: Could not find AP75 metric in {dataset_csv_path}, skipping Mean category")
        else:
            print(f"Warning: Could not find {dataset_csv_path}, skipping Mean category")
        
        return categories, mean_vals, se_vals, n_folds_vals
    
    # Add Mean category for primary model
    primary_categories, primary_ap75_mean, primary_ap75_se, primary_ap75_n_folds = add_mean_category(
        primary_df, dataset_csv_path, primary_categories.copy(), 
        primary_ap75_mean.copy(), primary_ap75_se.copy(), primary_ap75_n_folds.copy()
    )
    
    # Add Mean category for secondary model
    # First, get the Mean value from secondary dataset
    secondary_mean_val = np.nan
    secondary_mean_se = 0.0
    secondary_mean_n = 1
    if other_dataset_csv_path.exists():
        other_dataset_df = pd.read_csv(other_dataset_csv_path)
        ap75_row = other_dataset_df[other_dataset_df['metric_name'] == 'ap75_instance_Aweighted']
        if len(ap75_row) > 0:
            secondary_mean_val = ap75_row['mean'].iloc[0] * 100
            secondary_mean_se = ap75_row['se'].iloc[0] * 100
            secondary_mean_n = int(ap75_row['n_folds'].iloc[0])
    
    # Append Mean to secondary arrays
    secondary_ap75_mean = np.append(secondary_ap75_mean, secondary_mean_val)
    secondary_ap75_se = np.append(secondary_ap75_se, secondary_mean_se)
    secondary_ap75_n_folds = np.append(secondary_ap75_n_folds, secondary_mean_n)
    
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
    
    primary_ap75_error = calculate_error(primary_ap75_se, primary_ap75_n_folds)
    secondary_ap75_error = calculate_error(secondary_ap75_se, secondary_ap75_n_folds)
    
    # Clip error bars so they don't exceed 100
    primary_ap75_error = np.minimum(primary_ap75_error, 100 - primary_ap75_mean)
    secondary_ap75_error = np.minimum(secondary_ap75_error, 100 - secondary_ap75_mean)
    
    # Error bar styling
    error_kw = {'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5}
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width/2, primary_ap75_mean, width, yerr=primary_ap75_error,
                   label=f'{primary_model} AP75', color=primary_color, alpha=0.7,
                   error_kw=error_kw, ecolor='lightgrey')
    bars2 = ax.bar(x_pos + width/2, secondary_ap75_mean, width, yerr=secondary_ap75_error,
                   label=f"{'Camera' if primary_model == 'Silhouette' else 'Silhouette'} AP75", 
                   color=secondary_color, alpha=0.7,
                   error_kw=error_kw, ecolor='lightgrey')
    
    # Add horizontal reference line for YTVIS 2021 AP75
    ax.axhline(y=72.4, color='gray', linestyle='--', linewidth=2,
               label='YTVIS 2021 AP75', alpha=0.7)
    
    # Customize the plot
    ax.set_ylabel('AP Score', fontsize=18)
    ax.set_title('Camera vs Silhouette Model AP75 Performance Across 6 Fold CV', 
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
        output_path = csv_dir / "mask_metrics_category_both_models_ap75.png"
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Both models AP75 plot saved to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create a grouped bar plot from mask_metrics_category_summary.csv'
    )
    parser.add_argument(
        'csv_dir',
        type=str,
        help='Directory containing mask_metrics_category_summary.csv'
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
        help='Create a plot comparing both camera and silhouette models (AP75 only). Automatically finds the other model directory by substituting camera/silhouette in the path.'
    )
    
    args = parser.parse_args()
    
    if args.both_models:
        if args.output:
            create_both_models_ap75_plot(args.csv_dir, args.output)
        else:
            create_both_models_ap75_plot(args.csv_dir)
    else:
        create_grouped_bar_plot(args.csv_dir, args.output)


if __name__ == '__main__':
    main()


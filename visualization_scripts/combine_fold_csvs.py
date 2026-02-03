#!/usr/bin/env python3
"""
Script to combine CSV files from multiple folds and create summary statistics.

For each CSV type, creates a combined CSV with fold_number column.
For mask_metrics_dataset.csv and mask_metrics_category.csv, also creates summary CSVs
with mean, standard deviation, and standard error for each metric.
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from glob import glob

def find_inference_dir(fold_dir):
    """Find the inference directory within a fold directory."""
    # Look for inference directories in checkpoint subdirectories
    checkpoint_dirs = glob(os.path.join(fold_dir, "checkpoint_*/inference"))
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

def combine_csvs(base_dir, output_dir):
    """Combine CSV files from all folds."""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV file names to process
    csv_files = [
        "mask_metrics_dataset.csv",
        "mask_metrics_category.csv",
        "mask_metrics_frame.csv",
        "mask_metrics_video.csv",
        "mask_metrics.csv"
    ]
    
    # Find all fold directories (only directories matching "fold" followed by digits)
    fold_dirs = sorted([d for d in base_path.iterdir() if is_valid_fold_dir(d)])
    
    print(f"Found {len(fold_dirs)} fold directories")
    
    # Process each CSV type
    for csv_name in csv_files:
        print(f"\nProcessing {csv_name}...")
        combined_data = []
        
        for fold_dir in fold_dirs:
            # Extract fold number (should be safe since we filtered for valid fold dirs)
            match = re.match(r'^fold(\d+)$', fold_dir.name)
            if match:
                fold_num = int(match.group(1))
            else:
                print(f"  Warning: Could not extract fold number from {fold_dir.name}, skipping")
                continue
            
            # Find inference directory
            inference_dir = find_inference_dir(fold_dir)
            if inference_dir is None:
                print(f"  Warning: Could not find inference directory in {fold_dir}")
                continue
            
            csv_path = Path(inference_dir) / csv_name
            if not csv_path.exists():
                print(f"  Warning: {csv_path} does not exist")
                continue
            
            # Read CSV
            try:
                df = pd.read_csv(csv_path)
                # Add fold_number column
                df.insert(0, "fold_number", fold_num)
                combined_data.append(df)
                print(f"  Added fold {fold_num}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue
        
        if combined_data:
            # Combine all dataframes
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Save combined CSV
            output_filename = csv_name.replace(".csv", "_combined.csv")
            output_file = output_path / output_filename
            combined_df.to_csv(output_file, index=False)
            print(f"  Saved combined CSV: {output_file} ({len(combined_df)} rows)")
        else:
            print(f"  No data found for {csv_name}")

def create_summary_stats(base_dir, output_dir):
    """Create summary statistics for dataset and category CSVs."""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all fold directories (only directories matching "fold" followed by digits)
    fold_dirs = sorted([d for d in base_path.iterdir() if is_valid_fold_dir(d)])
    
    # Process mask_metrics_dataset.csv
    print("\nCreating summary for mask_metrics_dataset.csv...")
    dataset_data = []
    
    for fold_dir in fold_dirs:
        # Extract fold number (should be safe since we filtered for valid fold dirs)
        match = re.match(r'^fold(\d+)$', fold_dir.name)
        if match:
            fold_num = int(match.group(1))
        else:
            print(f"  Warning: Could not extract fold number from {fold_dir.name}, skipping")
            continue
        inference_dir = find_inference_dir(fold_dir)
        if inference_dir is None:
            continue
        
        csv_path = Path(inference_dir) / "mask_metrics_dataset.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Transpose to have metrics as columns
                df_dict = dict(zip(df['metric_name'], df['value']))
                df_dict['fold_number'] = fold_num
                dataset_data.append(df_dict)
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue
    
    if dataset_data:
        dataset_df = pd.DataFrame(dataset_data)
        # Get numeric columns (exclude fold_number)
        numeric_cols = [col for col in dataset_df.columns if col != 'fold_number']
        
        # Calculate statistics
        summary_data = []
        for col in numeric_cols:
            values = dataset_df[col].values
            # Filter out any invalid values (like -1.0 or NaN)
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values, ddof=1)  # Sample standard deviation
                se_val = std_val / np.sqrt(len(valid_values))  # Standard error
                
                summary_data.append({
                    'metric_name': col,
                    'mean': mean_val,
                    'sd': std_val,
                    'se': se_val,
                    'n_folds': len(valid_values)
                })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = output_path / "mask_metrics_dataset_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"  Saved summary CSV: {output_file}")
    
    # Process mask_metrics_category.csv
    print("\nCreating summary for mask_metrics_category.csv...")
    category_data = []
    
    for fold_dir in fold_dirs:
        # Extract fold number (should be safe since we filtered for valid fold dirs)
        match = re.match(r'^fold(\d+)$', fold_dir.name)
        if match:
            fold_num = int(match.group(1))
        else:
            print(f"  Warning: Could not extract fold number from {fold_dir.name}, skipping")
            continue
        inference_dir = find_inference_dir(fold_dir)
        if inference_dir is None:
            continue
        
        csv_path = Path(inference_dir) / "mask_metrics_category.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df['fold_number'] = fold_num
                category_data.append(df)
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue
    
    if category_data:
        category_df = pd.concat(category_data, ignore_index=True)
        
        # Group by category_id and category_name, then calculate statistics
        # Get numeric columns (exclude category_id, category_name, fold_number, and any text columns)
        exclude_cols = ['category_id', 'category_name', 'fold_number']
        numeric_cols = [col for col in category_df.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(category_df[col])]
        
        summary_rows = []
        for (cat_id, cat_name), group in category_df.groupby(['category_id', 'category_name']):
            row = {
                'category_id': cat_id,
                'category_name': cat_name
            }
            
            for col in numeric_cols:
                values = group[col].values
                # Filter out invalid values (NaN, -1.0, empty strings)
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values, ddof=1)
                    se_val = std_val / np.sqrt(len(valid_values))
                    
                    row[f'{col}_mean'] = mean_val
                    row[f'{col}_sd'] = std_val
                    row[f'{col}_se'] = se_val
                    row[f'{col}_n_folds'] = len(valid_values)
                else:
                    row[f'{col}_mean'] = np.nan
                    row[f'{col}_sd'] = np.nan
                    row[f'{col}_se'] = np.nan
                    row[f'{col}_n_folds'] = 0
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        output_file = output_path / "mask_metrics_category_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"  Saved summary CSV: {output_file}")

def main():
    # Accept base directory as command-line argument, or use default
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera"
    
    # Create output directory in the same parent directory
    output_dir = os.path.join(base_dir, "folds_summary")
    
    print("=" * 60)
    print(f"Processing directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Combining CSV files from all folds")
    print("=" * 60)
    combine_csvs(base_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Creating summary statistics")
    print("=" * 60)
    create_summary_stats(base_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Script to combine CSV files from multiple seed directories and create summary statistics.

For each CSV type, creates a combined CSV with seed_number column.
For mask_metrics_dataset.csv and mask_metrics_category.csv, also creates summary CSVs
with mean, standard deviation, and standard error for each metric.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def find_seed_directories(base_dir):
    """Find all seed directories matching the pattern eval_*_seed*"""
    base_path = Path(base_dir)
    seed_dirs = []
    
    # Look for directories matching the pattern eval_*_seed* (exclude seeds_summary)
    for item in base_path.iterdir():
        if item.is_dir() and 'seed' in item.name and item.name != 'seeds_summary':
            seed_dirs.append(item)
    
    # Sort by seed number for consistent output
    seed_dirs.sort(key=lambda x: int(x.name.split('seed')[-1]) if x.name.split('seed')[-1].isdigit() else 999)
    
    return seed_dirs


def extract_seed_number(seed_dir_name):
    """Extract seed number from directory name (e.g., eval_6059_all_frames_seed1 -> 1)"""
    try:
        seed_part = seed_dir_name.split('seed')[-1]
        return int(seed_part)
    except (ValueError, IndexError):
        return None


def combine_csvs(base_dir, output_dir):
    """Combine CSV files from all seed directories."""
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
    
    # Find all seed directories
    seed_dirs = find_seed_directories(base_path)
    
    print(f"Found {len(seed_dirs)} seed directories")
    
    # Process each CSV type
    for csv_name in csv_files:
        print(f"\nProcessing {csv_name}...")
        combined_data = []
        
        for seed_dir in seed_dirs:
            # Extract seed number
            seed_num = extract_seed_number(seed_dir.name)
            if seed_num is None:
                print(f"  Warning: Could not extract seed number from {seed_dir.name}")
                continue
            
            # Find inference directory (directly in seed directory)
            inference_dir = seed_dir / "inference"
            if not inference_dir.exists():
                print(f"  Warning: Could not find inference directory in {seed_dir}")
                continue
            
            csv_path = inference_dir / csv_name
            if not csv_path.exists():
                print(f"  Warning: {csv_path} does not exist")
                continue
            
            # Read CSV
            try:
                df = pd.read_csv(csv_path)
                # Add seed_number column
                df.insert(0, "seed_number", seed_num)
                combined_data.append(df)
                print(f"  Added seed {seed_num}: {len(df)} rows")
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
    
    # Find all seed directories
    seed_dirs = find_seed_directories(base_path)
    
    # Process mask_metrics_dataset.csv
    print("\nCreating summary for mask_metrics_dataset.csv...")
    dataset_data = []
    
    for seed_dir in seed_dirs:
        seed_num = extract_seed_number(seed_dir.name)
        if seed_num is None:
            continue
        
        inference_dir = seed_dir / "inference"
        if not inference_dir.exists():
            continue
        
        csv_path = inference_dir / "mask_metrics_dataset.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Transpose to have metrics as columns
                df_dict = dict(zip(df['metric_name'], df['value']))
                df_dict['seed_number'] = seed_num
                dataset_data.append(df_dict)
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue
    
    if dataset_data:
        dataset_df = pd.DataFrame(dataset_data)
        # Get numeric columns (exclude seed_number)
        numeric_cols = [col for col in dataset_df.columns if col != 'seed_number']
        
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
                    'n_seeds': len(valid_values)
                })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = output_path / "mask_metrics_dataset_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"  Saved summary CSV: {output_file}")
    
    # Process mask_metrics_category.csv
    print("\nCreating summary for mask_metrics_category.csv...")
    category_data = []
    
    for seed_dir in seed_dirs:
        seed_num = extract_seed_number(seed_dir.name)
        if seed_num is None:
            continue
        
        inference_dir = seed_dir / "inference"
        if not inference_dir.exists():
            continue
        
        csv_path = inference_dir / "mask_metrics_category.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df['seed_number'] = seed_num
                category_data.append(df)
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
                continue
    
    if category_data:
        category_df = pd.concat(category_data, ignore_index=True)
        
        # Group by category_id and category_name, then calculate statistics
        # Get numeric columns (exclude category_id, category_name, seed_number, and any text columns)
        exclude_cols = ['category_id', 'category_name', 'seed_number']
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
                    row[f'{col}_n_seeds'] = len(valid_values)
                else:
                    row[f'{col}_mean'] = np.nan
                    row[f'{col}_sd'] = np.nan
                    row[f'{col}_se'] = np.nan
                    row[f'{col}_n_seeds'] = 0
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        output_file = output_path / "mask_metrics_category_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"  Saved summary CSV: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine CSV files from multiple seed directories and create summary statistics'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Base directory containing seed subdirectories (e.g., /path/to/scrambled_fold4)'
    )
    
    args = parser.parse_args()
    base_dir = os.path.abspath(args.directory)
    
    # Create output directory in the same directory
    output_dir = os.path.join(base_dir, "seeds_summary")
    
    print("=" * 60)
    print(f"Processing directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Combining CSV files from all seeds")
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

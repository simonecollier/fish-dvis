#!/usr/bin/env python3
"""
Batch extract metrics from multiple model directories for hyperparameter tuning.
This script processes multiple model directories and creates a comprehensive
spreadsheet with all metrics for hyperparameter analysis.

Usage:
    # Process all models in a directory
    python batch_extract_hyperparameter_metrics.py --models-dir /path/to/models --output-csv results.csv
    
    # Process specific models
    python batch_extract_hyperparameter_metrics.py --model-list model1,model2,model3 --output-csv results.csv
    
    # Process models with pattern matching
    python batch_extract_hyperparameter_metrics.py --model-pattern "dvis_*" --output-csv results.csv
"""

import os
import glob
import argparse
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
import sys

# Add the current directory to the path to import the extraction script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_hyperparameter_metrics import create_hyperparameter_row

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_model_directories(models_dir: str, model_pattern: str = None) -> List[str]:
    """
    Find all model directories in the specified directory.
    
    Args:
        models_dir: Directory containing model directories
        model_pattern: Optional pattern to filter model names
        
    Returns:
        List of model directory paths
    """
    model_dirs = []
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory does not exist: {models_dir}")
        return model_dirs
    
    # Get all subdirectories
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Check if it looks like a model directory
            if (os.path.exists(os.path.join(item_path, 'metrics.json')) or
                os.path.exists(os.path.join(item_path, 'checkpoint_evaluations')) or
                os.path.exists(os.path.join(item_path, 'comprehensive_metrics_summary.csv')) or
                os.path.exists(os.path.join(item_path, 'mask_metrics_summary.csv'))):
                
                # Apply pattern filter if specified
                if model_pattern:
                    if glob.fnmatch.fnmatch(item, model_pattern):
                        model_dirs.append(item_path)
                else:
                    model_dirs.append(item_path)
    
    return sorted(model_dirs)

def process_model_list(model_list: List[str]) -> List[str]:
    """
    Process a list of model paths and validate they exist.
    
    Args:
        model_list: List of model directory paths
        
    Returns:
        List of valid model directory paths
    """
    valid_models = []
    
    for model_path in model_list:
        if os.path.exists(model_path):
            valid_models.append(model_path)
        else:
            logger.warning(f"Model directory does not exist: {model_path}")
    
    return valid_models

def extract_metrics_from_models(model_dirs: List[str], include_analysis: bool = True) -> pd.DataFrame:
    """
    Extract metrics from multiple model directories.
    
    Args:
        model_dirs: List of model directory paths
        include_analysis: Whether to include analysis metrics
        
    Returns:
        DataFrame with metrics from all models
    """
    all_rows = []
    
    for i, model_dir in enumerate(model_dirs):
        logger.info(f"Processing {i+1}/{len(model_dirs)}: {os.path.basename(model_dir)}")
        
        try:
            row = create_hyperparameter_row(model_dir, include_analysis)
            all_rows.append(row)
            logger.info(f"✓ Successfully extracted metrics from {os.path.basename(model_dir)}")
        except Exception as e:
            logger.error(f"✗ Failed to extract metrics from {os.path.basename(model_dir)}: {e}")
            # Add a row with basic info and error
            error_row = {
                'model_name': os.path.basename(model_dir),
                'model_path': model_dir,
                'extraction_error': str(e),
                'extraction_timestamp': pd.Timestamp.now().isoformat()
            }
            all_rows.append(error_row)
    
    if not all_rows:
        logger.error("No metrics extracted from any models")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Sort by model name for consistency
    if 'model_name' in df.columns:
        df = df.sort_values('model_name')
    
    return df

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics that are useful for hyperparameter analysis.
    
    Args:
        df: DataFrame with extracted metrics
        
    Returns:
        DataFrame with additional derived metrics
    """
    # Add efficiency metrics
    if 'best_map_50' in df.columns and 'total_iterations' in df.columns:
        df['map_50_per_iteration'] = df['best_map_50'] / df['total_iterations']
    
    if 'best_mean_iou' in df.columns and 'total_iterations' in df.columns:
        df['iou_per_iteration'] = df['best_mean_iou'] / df['total_iterations']
    
    # Add performance ranking
    if 'best_map_50' in df.columns:
        df['map_50_rank'] = df['best_map_50'].rank(ascending=False)
    
    if 'best_mean_iou' in df.columns:
        df['iou_rank'] = df['best_mean_iou'].rank(ascending=False)
    
    # Add composite score (weighted average of key metrics)
    if all(col in df.columns for col in ['best_map_50', 'best_mean_iou']):
        df['composite_score'] = (0.6 * df['best_map_50'] + 0.4 * df['best_mean_iou'])
        df['composite_rank'] = df['composite_score'].rank(ascending=False)
    
    # Add training efficiency
    if 'total_training_time' in df.columns and 'best_map_50' in df.columns:
        df['map_50_per_hour'] = df['best_map_50'] / (df['total_training_time'] / 3600)
    
    return df

def create_summary_report(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary report of the hyperparameter analysis.
    
    Args:
        df: DataFrame with all metrics
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'hyperparameter_analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total models analyzed: {len(df)}\n")
        f.write(f"Analysis timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Best performing models
        f.write("TOP PERFORMING MODELS\n")
        f.write("-" * 40 + "\n")
        
        if 'best_map_50' in df.columns:
            top_map_models = df.nlargest(5, 'best_map_50')[['model_name', 'best_map_50', 'learning_rate', 'batch_size', 'num_frames']]
            f.write("Top 5 models by mAP@0.5:\n")
            for _, row in top_map_models.iterrows():
                f.write(f"  {row['model_name']}: {row['best_map_50']:.4f} "
                       f"(lr={row['learning_rate']}, bs={row['batch_size']}, frames={row['num_frames']})\n")
            f.write("\n")
        
        if 'best_mean_iou' in df.columns:
            top_iou_models = df.nlargest(5, 'best_mean_iou')[['model_name', 'best_mean_iou', 'learning_rate', 'batch_size', 'num_frames']]
            f.write("Top 5 models by Mean IoU:\n")
            for _, row in top_iou_models.iterrows():
                f.write(f"  {row['model_name']}: {row['best_mean_iou']:.4f} "
                       f"(lr={row['learning_rate']}, bs={row['batch_size']}, frames={row['num_frames']})\n")
            f.write("\n")
        
        # Hyperparameter analysis
        f.write("HYPERPARAMETER ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Learning rate analysis
        if 'learning_rate' in df.columns and 'best_map_50' in df.columns:
            lr_analysis = df.groupby('learning_rate')['best_map_50'].agg(['mean', 'std', 'count']).round(4)
            f.write("Learning rate impact on mAP@0.5:\n")
            f.write(lr_analysis.to_string())
            f.write("\n\n")
        
        # Batch size analysis
        if 'batch_size' in df.columns and 'best_map_50' in df.columns:
            bs_analysis = df.groupby('batch_size')['best_map_50'].agg(['mean', 'std', 'count']).round(4)
            f.write("Batch size impact on mAP@0.5:\n")
            f.write(bs_analysis.to_string())
            f.write("\n\n")
        
        # Frame count analysis
        if 'num_frames' in df.columns and 'best_map_50' in df.columns:
            frame_analysis = df.groupby('num_frames')['best_map_50'].agg(['mean', 'std', 'count']).round(4)
            f.write("Frame count impact on mAP@0.5:\n")
            f.write(frame_analysis.to_string())
            f.write("\n\n")
        
        # Training stability analysis
        f.write("TRAINING STABILITY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        if 'overfitting_detected' in df.columns:
            overfitting_count = df['overfitting_detected'].value_counts()
            f.write(f"Overfitting detected: {overfitting_count.get(True, 0)} models\n")
            f.write(f"No overfitting: {overfitting_count.get(False, 0)} models\n\n")
        
        if 'training_stability' in df.columns:
            stability_count = df['training_stability'].value_counts()
            f.write("Training stability distribution:\n")
            for stability, count in stability_count.items():
                f.write(f"  {stability}: {count} models\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Find best hyperparameter combinations
        if all(col in df.columns for col in ['learning_rate', 'batch_size', 'num_frames', 'best_map_50']):
            best_models = df.nlargest(3, 'best_map_50')
            f.write("Best hyperparameter combinations:\n")
            for i, (_, row) in enumerate(best_models.iterrows(), 1):
                f.write(f"  {i}. {row['model_name']}: "
                       f"lr={row['learning_rate']}, bs={row['batch_size']}, frames={row['num_frames']} "
                       f"(mAP@0.5={row['best_map_50']:.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated by batch_extract_hyperparameter_metrics.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Batch extract metrics for hyperparameter tuning')
    parser.add_argument('--models-dir', type=str,
                       help='Directory containing model directories')
    parser.add_argument('--model-list', type=str,
                       help='Comma-separated list of model directory paths')
    parser.add_argument('--model-pattern', type=str,
                       help='Pattern to match model directory names (e.g., "dvis_*")')
    parser.add_argument('--output-csv', type=str, default='hyperparameter_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for reports')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing CSV file instead of overwriting')
    parser.add_argument('--update-existing', action='store_true',
                       help='Update existing rows if models already exist in CSV')
    parser.add_argument('--include-analysis', action='store_true', default=True,
                       help='Include analysis metrics (overfitting, convergence, etc.)')
    parser.add_argument('--exclude-analysis', action='store_true',
                       help='Exclude analysis metrics')
    parser.add_argument('--add-derived-metrics', action='store_true', default=True,
                       help='Add derived metrics (efficiency, rankings, etc.)')
    
    args = parser.parse_args()
    
    if args.exclude_analysis:
        args.include_analysis = False
    
    # Find model directories
    model_dirs = []
    
    if args.models_dir:
        logger.info(f"Searching for models in: {args.models_dir}")
        model_dirs = find_model_directories(args.models_dir, args.model_pattern)
    elif args.model_list:
        logger.info("Processing specified model list")
        model_list = [path.strip() for path in args.model_list.split(',')]
        model_dirs = process_model_list(model_list)
    else:
        logger.error("Must specify either --models-dir or --model-list")
        return
    
    if not model_dirs:
        logger.error("No valid model directories found")
        return
    
    logger.info(f"Found {len(model_dirs)} model directories to process")
    
    # Handle existing CSV file
    existing_df = None
    output_path = os.path.join(args.output_dir, args.output_csv)
    
    if args.append or args.update_existing:
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                logger.info(f"Found existing CSV with {len(existing_df)} rows")
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")
                existing_df = None
    
    # Extract metrics
    logger.info("Extracting metrics from all models...")
    new_df = extract_metrics_from_models(model_dirs, args.include_analysis)
    
    if new_df.empty:
        logger.error("No metrics extracted")
        return
    
    # Handle appending or updating
    if args.update_existing and existing_df is not None:
        # Update existing models and add new ones
        final_df = existing_df.copy()
        updated_count = 0
        added_count = 0
        
        for _, new_row in new_df.iterrows():
            model_name = new_row['model_name']
            existing_mask = final_df['model_name'] == model_name
            
            if existing_mask.any():
                # Update existing row
                final_df.loc[existing_mask] = new_row
                updated_count += 1
                logger.info(f"Updated existing row for model: {model_name}")
            else:
                # Add new row
                final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
                added_count += 1
                logger.info(f"Added new row for model: {model_name}")
        
        logger.info(f"Updated {updated_count} existing models, added {added_count} new models")
        
    elif args.append and existing_df is not None:
        # Always append new models
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        logger.info(f"Appended {len(new_df)} new models to existing CSV")
        
    else:
        # Create new file
        final_df = new_df
        logger.info(f"Creating new CSV file with {len(new_df)} models")
    
    # Add derived metrics
    if args.add_derived_metrics:
        logger.info("Adding derived metrics...")
        final_df = add_derived_metrics(final_df)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    logger.info(f"Metrics saved to: {output_path}")
    
    # Create summary report
    logger.info("Creating summary report...")
    create_summary_report(final_df, args.output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(model_dirs)} models")
    logger.info(f"Total models in CSV: {len(final_df)}")
    logger.info(f"Successful extractions: {len(final_df[final_df['extraction_error'].isna()])}")
    logger.info(f"Failed extractions: {len(final_df[final_df['extraction_error'].notna()])}")
    logger.info(f"Output files:")
    logger.info(f"  - CSV: {output_path}")
    logger.info(f"  - Summary: {os.path.join(args.output_dir, 'hyperparameter_analysis_summary.txt')}")
    
    # Show top performers
    if 'best_map_50' in final_df.columns:
        logger.info("\nTop 3 models by mAP@0.5:")
        top_models = final_df.nlargest(3, 'best_map_50')
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            logger.info(f"  {i}. {row['model_name']}: {row['best_map_50']:.4f}")
    
    # Show new models if appending
    if args.append and existing_df is not None and 'best_map_50' in final_df.columns:
        new_models = final_df[~final_df['model_name'].isin(existing_df['model_name'])]
        if len(new_models) > 0:
            logger.info(f"\nNew models added:")
            for _, row in new_models.iterrows():
                if 'best_map_50' in row and not pd.isna(row['best_map_50']):
                    logger.info(f"  - {row['model_name']}: mAP@0.5 = {row['best_map_50']:.4f}")

if __name__ == "__main__":
    main()

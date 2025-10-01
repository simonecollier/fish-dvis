#!/usr/bin/env python3
"""
Extract metrics from checkpoint analysis results for hyperparameter tuning spreadsheets.
This script extracts all available metrics and analysis results from the comprehensive
checkpoint analysis and formats them for easy import into spreadsheets.

Usage:
    python extract_hyperparameter_metrics.py --model-dir /path/to/model --output-csv results.csv
    
    # Extract only best epoch metrics
    python extract_hyperparameter_metrics.py --model-dir /path/to/model --best-epoch-only
    
    # Include all analysis metrics
    python extract_hyperparameter_metrics.py --model-dir /path/to/model --include-analysis
"""

import os
import json
import re
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_config_parameters(model_dir: str) -> Dict[str, Any]:
    """
    Extract configuration parameters from model directory name and config files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary of configuration parameters
    """
    config = {
        'model_name': os.path.basename(model_dir),
        'learning_rate': None,
        'batch_size': None,
        'num_frames': None,
        'model_type': None,
        'optimizer': None,
        'scheduler': None,
        'data_augmentation': None,
        'loss_weights': None,
        'backbone': None,
        'neck': None,
        'head': None,
    }
    
    model_name = config['model_name'].lower()
    
    # Extract learning rate
    lr_patterns = [
        r'lr_?(\d+\.?\d*)',
        r'learning_rate_?(\d+\.?\d*)',
        r'(\d+\.?\d*)lr',
        r'(\d+\.?\d*)e-?(\d+)',  # Scientific notation
    ]
    
    for pattern in lr_patterns:
        match = re.search(pattern, model_name)
        if match:
            if 'e-' in pattern:
                # Handle scientific notation
                base = float(match.group(1))
                exp = int(match.group(2))
                config['learning_rate'] = base * (10 ** -exp)
            else:
                config['learning_rate'] = float(match.group(1))
            break
    
    # Extract batch size
    bs_patterns = [r'bs_?(\d+)', r'batch_?(\d+)', r'(\d+)bs']
    for pattern in bs_patterns:
        match = re.search(pattern, model_name)
        if match:
            config['batch_size'] = int(match.group(1))
            break
    
    # Extract frame count
    frame_patterns = [r'(\d+)f', r'frames_?(\d+)', r'(\d+)frames']
    for pattern in frame_patterns:
        match = re.search(pattern, model_name)
        if match:
            config['num_frames'] = int(match.group(1))
            break
    
    # Extract model type
    if 'unmasked' in model_name:
        config['model_type'] = 'unmasked'
    elif 'masked' in model_name:
        config['model_type'] = 'masked'
    elif 'dvis' in model_name:
        config['model_type'] = 'dvis'
    elif 'yolo' in model_name:
        config['model_type'] = 'yolo'
    
    # Extract optimizer
    if 'adam' in model_name:
        config['optimizer'] = 'adam'
    elif 'sgd' in model_name:
        config['optimizer'] = 'sgd'
    elif 'adamw' in model_name:
        config['optimizer'] = 'adamw'
    
    # Extract scheduler
    if 'cosine' in model_name:
        config['scheduler'] = 'cosine'
    elif 'step' in model_name:
        config['scheduler'] = 'step'
    elif 'poly' in model_name:
        config['scheduler'] = 'polynomial'
    
    # Extract backbone
    if 'resnet' in model_name:
        config['backbone'] = 'resnet'
    elif 'swin' in model_name:
        config['backbone'] = 'swin'
    elif 'vit' in model_name:
        config['backbone'] = 'vit'
    
    # Try to read config file if it exists
    config_file = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Extract additional parameters
            if 'SOLVER' in yaml_config:
                solver = yaml_config['SOLVER']
                if 'BASE_LR' in solver and config['learning_rate'] is None:
                    config['learning_rate'] = solver['BASE_LR']
                if 'IMS_PER_BATCH' in solver and config['batch_size'] is None:
                    config['batch_size'] = solver['IMS_PER_BATCH']
                if 'OPTIMIZER' in solver and config['optimizer'] is None:
                    config['optimizer'] = solver['OPTIMIZER']
                    
            if 'MODEL' in yaml_config:
                model_config = yaml_config['MODEL']
                if 'BACKBONE' in model_config and config['backbone'] is None:
                    config['backbone'] = model_config['BACKBONE']
                    
        except Exception as e:
            logger.warning(f"Could not read config file: {e}")
    
    return config

def extract_basic_metrics(model_dir: str) -> Dict[str, Any]:
    """
    Extract basic performance metrics from the comprehensive summary CSV.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary of basic metrics
    """
    metrics = {}
    
    # Try to find comprehensive summary CSV
    summary_csv_path = os.path.join(model_dir, 'comprehensive_metrics_summary.csv')
    if not os.path.exists(summary_csv_path):
        summary_csv_path = os.path.join(model_dir, 'mask_metrics_summary.csv')
    
    if os.path.exists(summary_csv_path):
        try:
            df = pd.read_csv(summary_csv_path)
            
            if not df.empty:
                # Best performance metrics
                for metric in ['mean_iou', 'mean_boundary_f', 'map_10', 'map_25', 'map_50', 'map_75', 'map_95']:
                    if metric in df.columns:
                        best_idx = df[metric].idxmax()
                        best_value = df.loc[best_idx, metric]
                        best_iter = df.loc[best_idx, 'iteration']
                        
                        metrics[f'best_{metric}'] = best_value
                        metrics[f'best_{metric}_iteration'] = best_iter
                
                # Final performance metrics
                final_row = df.iloc[-1]
                for metric in ['mean_iou', 'mean_boundary_f', 'map_10', 'map_25', 'map_50', 'map_75', 'map_95']:
                    if metric in df.columns:
                        metrics[f'final_{metric}'] = final_row[metric]
                
                # Training statistics
                metrics['total_iterations'] = df['iteration'].max()
                metrics['num_checkpoints'] = len(df)
                metrics['iteration_range'] = df['iteration'].max() - df['iteration'].min()
                
                # Performance improvement
                if 'map_50' in df.columns:
                    first_map = df['map_50'].iloc[0]
                    best_map = df['map_50'].max()
                    if first_map > 0:
                        metrics['map_50_improvement_pct'] = ((best_map - first_map) / first_map) * 100
                    else:
                        metrics['map_50_improvement_pct'] = float('inf') if best_map > 0 else 0
                
                # Performance stability (standard deviation)
                for metric in ['mean_iou', 'map_50']:
                    if metric in df.columns:
                        metrics[f'{metric}_std'] = df[metric].std()
                        metrics[f'{metric}_mean'] = df[metric].mean()
                        metrics[f'{metric}_min'] = df[metric].min()
                        metrics[f'{metric}_max'] = df[metric].max()
                
        except Exception as e:
            logger.warning(f"Could not read summary CSV: {e}")
    
    return metrics

def extract_analysis_metrics(model_dir: str) -> Dict[str, Any]:
    """
    Extract analysis metrics from the performance report.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary of analysis metrics
    """
    analysis = {}
    
    report_path = os.path.join(model_dir, 'model_performance_report.txt')
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Extract overfitting analysis
            if 'OVERFITTING DETECTED' in content:
                analysis['overfitting_detected'] = True
                if 'Severity: severe' in content:
                    analysis['overfitting_severity'] = 'severe'
                elif 'Severity: moderate' in content:
                    analysis['overfitting_severity'] = 'moderate'
                else:
                    analysis['overfitting_severity'] = 'mild'
            else:
                analysis['overfitting_detected'] = False
                analysis['overfitting_severity'] = 'none'
            
            # Extract convergence analysis
            convergence_match = re.search(r'Convergence ratio: ([\d.]+)', content)
            if convergence_match:
                analysis['convergence_ratio'] = float(convergence_match.group(1))
            
            if 'Model appears to have converged well' in content:
                analysis['converged'] = True
            else:
                analysis['converged'] = False
            
            # Extract training dynamics
            stability_match = re.search(r'Stability: (\w+)', content)
            if stability_match:
                analysis['training_stability'] = stability_match.group(1)
            
            # Extract performance by training phase
            early_match = re.search(r'Early performance: ([\d.]+)', content)
            mid_match = re.search(r'Mid performance: ([\d.]+)', content)
            late_match = re.search(r'Late performance: ([\d.]+)', content)
            
            if early_match:
                analysis['early_performance'] = float(early_match.group(1))
            if mid_match:
                analysis['mid_performance'] = float(mid_match.group(1))
            if late_match:
                analysis['late_performance'] = float(late_match.group(1))
            
            # Extract model characteristics
            model_type_match = re.search(r'Model type: (\w+)', content)
            if model_type_match:
                analysis['model_type'] = model_type_match.group(1)
            
            lr_match = re.search(r'Learning rate: ([\d.]+)', content)
            if lr_match:
                analysis['learning_rate'] = float(lr_match.group(1))
            
            frame_match = re.search(r'Frame count: (\d+)', content)
            if frame_match:
                analysis['frame_count'] = int(frame_match.group(1))
            
        except Exception as e:
            logger.warning(f"Could not read performance report: {e}")
    
    return analysis

def extract_training_metrics(model_dir: str) -> Dict[str, Any]:
    """
    Extract training metrics from metrics.json if available.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary of training metrics
    """
    training = {}
    
    metrics_json_path = os.path.join(model_dir, 'metrics.json')
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Extract final training metrics
            if isinstance(metrics_data, list) and len(metrics_data) > 0:
                final_metrics = metrics_data[-1]
                
                # Common training metrics
                for key in ['total_loss', 'loss_ce', 'loss_mask', 'loss_dice', 'loss_bbox', 'loss_giou']:
                    if key in final_metrics:
                        training[f'final_{key}'] = final_metrics[key]
                
                # Learning rate
                if 'lr' in final_metrics:
                    training['final_learning_rate'] = final_metrics['lr']
                
                # Training time
                if 'time' in final_metrics:
                    training['total_training_time'] = final_metrics['time']
                
                # Extract best metrics during training
                if len(metrics_data) > 1:
                    # Find best loss
                    losses = [m.get('total_loss', float('inf')) for m in metrics_data if 'total_loss' in m]
                    if losses:
                        training['best_total_loss'] = min(losses)
                        training['worst_total_loss'] = max(losses)
                    
                    # Calculate training stability
                    if 'total_loss' in final_metrics:
                        loss_values = [m.get('total_loss', 0) for m in metrics_data if 'total_loss' in m]
                        if len(loss_values) > 1:
                            training['loss_std'] = np.std(loss_values)
                            training['loss_mean'] = np.mean(loss_values)
            
        except Exception as e:
            logger.warning(f"Could not read metrics.json: {e}")
    
    return training

def create_hyperparameter_row(model_dir: str, include_analysis: bool = True) -> Dict[str, Any]:
    """
    Create a single row of data for hyperparameter tuning spreadsheet.
    
    Args:
        model_dir: Path to model directory
        include_analysis: Whether to include analysis metrics
        
    Returns:
        Dictionary containing all metrics for one model
    """
    row = {}
    
    # Extract configuration parameters
    config = extract_config_parameters(model_dir)
    row.update(config)
    
    # Extract basic performance metrics
    metrics = extract_basic_metrics(model_dir)
    row.update(metrics)
    
    # Extract training metrics
    training = extract_training_metrics(model_dir)
    row.update(training)
    
    # Extract analysis metrics if requested
    if include_analysis:
        analysis = extract_analysis_metrics(model_dir)
        row.update(analysis)
    
    # Add metadata
    row['model_path'] = model_dir
    row['extraction_timestamp'] = pd.Timestamp.now().isoformat()
    
    return row

def main():
    parser = argparse.ArgumentParser(description='Extract metrics for hyperparameter tuning')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory')
    parser.add_argument('--output-csv', type=str, default='hyperparameter_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing CSV file instead of overwriting')
    parser.add_argument('--update-existing', action='store_true',
                       help='Update existing row if model already exists in CSV')
    parser.add_argument('--best-epoch-only', action='store_true',
                       help='Only include metrics from the best epoch')
    parser.add_argument('--include-analysis', action='store_true', default=True,
                       help='Include analysis metrics (overfitting, convergence, etc.)')
    parser.add_argument('--exclude-analysis', action='store_true',
                       help='Exclude analysis metrics')
    
    args = parser.parse_args()
    
    if args.exclude_analysis:
        args.include_analysis = False
    
    # Create the data row
    logger.info(f"Extracting metrics from: {args.model_dir}")
    row = create_hyperparameter_row(args.model_dir, args.include_analysis)
    
    # Handle existing CSV file
    existing_df = None
    if args.append or args.update_existing:
        if os.path.exists(args.output_csv):
            try:
                existing_df = pd.read_csv(args.output_csv)
                logger.info(f"Found existing CSV with {len(existing_df)} rows")
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")
                existing_df = None
    
    # Convert to DataFrame
    new_df = pd.DataFrame([row])
    
    # Handle appending or updating
    if args.update_existing and existing_df is not None:
        # Check if model already exists
        model_name = row['model_name']
        existing_mask = existing_df['model_name'] == model_name
        
        if existing_mask.any():
            # Update existing row
            existing_df.loc[existing_mask] = new_df.iloc[0]
            logger.info(f"Updated existing row for model: {model_name}")
            final_df = existing_df
        else:
            # Append new row
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            logger.info(f"Added new row for model: {model_name}")
    elif args.append and existing_df is not None:
        # Always append
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        logger.info(f"Appended new row for model: {row['model_name']}")
    else:
        # Create new file
        final_df = new_df
        logger.info(f"Creating new CSV file")
    
    # Save to CSV
    final_df.to_csv(args.output_csv, index=False)
    logger.info(f"Metrics saved to: {args.output_csv}")
    
    # Print summary
    logger.info("\nExtracted Metrics Summary:")
    logger.info("=" * 50)
    
    # Configuration
    logger.info("Configuration:")
    for key in ['model_name', 'learning_rate', 'batch_size', 'num_frames', 'model_type', 'optimizer']:
        if key in row and row[key] is not None:
            logger.info(f"  {key}: {row[key]}")
    
    # Best performance
    logger.info("\nBest Performance:")
    for key in ['best_mean_iou', 'best_map_50', 'best_map_75']:
        if key in row and row[key] is not None:
            logger.info(f"  {key}: {row[key]:.4f}")
    
    # Training stats
    logger.info("\nTraining Statistics:")
    for key in ['total_iterations', 'num_checkpoints', 'convergence_ratio']:
        if key in row and row[key] is not None:
            logger.info(f"  {key}: {row[key]}")
    
    if args.include_analysis:
        logger.info("\nAnalysis:")
        for key in ['overfitting_detected', 'training_stability', 'converged']:
            if key in row and row[key] is not None:
                logger.info(f"  {key}: {row[key]}")
    
    # Show comparison with existing models if available
    if existing_df is not None and 'best_map_50' in existing_df.columns and 'best_map_50' in row:
        current_map = row['best_map_50']
        if not pd.isna(current_map):
            existing_models = existing_df[existing_df['model_name'] != row['model_name']]
            if len(existing_models) > 0 and 'best_map_50' in existing_models.columns:
                best_existing = existing_models['best_map_50'].max()
                if not pd.isna(best_existing):
                    improvement = current_map - best_existing
                    logger.info(f"\nComparison with existing models:")
                    logger.info(f"  Current model mAP@0.5: {current_map:.4f}")
                    logger.info(f"  Best existing mAP@0.5: {best_existing:.4f}")
                    if improvement > 0:
                        logger.info(f"  ✓ Improvement: +{improvement:.4f}")
                    else:
                        logger.info(f"  ✗ Decrease: {improvement:.4f}")
                    
                    # Show ranking
                    all_models = pd.concat([existing_models, new_df], ignore_index=True)
                    ranking = all_models['best_map_50'].rank(ascending=False)
                    current_rank = ranking.iloc[-1]
                    logger.info(f"  Current rank: {current_rank:.0f}/{len(all_models)}")

if __name__ == "__main__":
    main()

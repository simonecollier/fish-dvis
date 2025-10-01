#!/usr/bin/env python3
"""
Comprehensive checkpoint analysis script that combines mask metrics generation and advanced plotting.
This script will:
1. Optionally run mask metrics analysis for each checkpoint
2. Create comprehensive performance plots and analysis
3. Generate trend analysis and best checkpoint identification
4. Provide detailed insights about model performance

Usage:
    # Basic analysis of existing results
    python analyze_checkpoint_results.py --model-dir /path/to/model
    
    # Run mask metrics and then analyze
    python analyze_checkpoint_results.py --model-dir /path/to/model --run-mask-metrics
    
    # Fast mode for quick analysis
    python analyze_checkpoint_results.py --model-dir /path/to/model --run-mask-metrics
    
    # Basic summary only (no advanced analysis)
    python analyze_checkpoint_results.py --model-dir /path/to/model --analysis-level basic
"""

import os
import json
import glob
import subprocess
import argparse
import re
import yaml
from typing import List, Dict
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_training_loss import plot_training_loss
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Note: Global caching removed due to multiprocessing incompatibility

def find_checkpoint_results(base_dir: str, evaluation_dir: str = None) -> List[str]:
    """
    Find all checkpoint result directories.
    
    Args:
        base_dir: Base directory containing checkpoint results
        evaluation_dir: Specific evaluation directory (for multi-run structure)
        
    Returns:
        List of checkpoint result directories
    """
    checkpoint_dirs = []
    
    # If evaluation_dir is provided, use it directly
    if evaluation_dir:
        if os.path.exists(evaluation_dir):
            base_dir = evaluation_dir
        else:
            logger.error(f"Evaluation directory not found: {evaluation_dir}")
            return []
    
    # Look for checkpoint_evaluations directory if not using specific evaluation_dir
    if not evaluation_dir:
        checkpoint_eval_dir = os.path.join(base_dir, "checkpoint_evaluations")
        if os.path.exists(checkpoint_eval_dir):
            base_dir = checkpoint_eval_dir
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint_'):
            # Check if it has inference results
            inference_dir = os.path.join(item_path, 'inference')
            if os.path.exists(inference_dir):
                checkpoint_dirs.append(item_path)
    
    # Sort by iteration number
    def extract_iteration(dir_path):
        dir_name = os.path.basename(dir_path)
        match = re.search(r'checkpoint_(\d+)', dir_name)
        return int(match.group(1)) if match else 0
    
    checkpoint_dirs.sort(key=extract_iteration)
    return checkpoint_dirs

def cleanup_existing_metrics(checkpoint_dir: str) -> None:
    """
    Clean up all existing metric files for a checkpoint.
    This ensures that when recomputing metrics (especially when switching between fast and full mode),
    old metric files don't interfere with the new analysis.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
    """
    inference_dir = os.path.join(checkpoint_dir, "inference")
    if not os.path.exists(inference_dir):
        return
    
    # List of all metric files that should be cleaned up
    metric_files = [
        "mask_metrics.csv",              # Combined metrics (backward compatibility)
        "mask_metrics_frame.csv",        # Frame-level metrics
        "mask_metrics_video.csv",        # Video-level metrics  
        "mask_metrics_category.csv",     # Category-level metrics
        "mask_metrics_dataset.csv",      # Dataset-level metrics
        "confusion_matrix.png",          # Confusion matrix plot
        "AP_per_category.png"            # AP per category plot
    ]
    
    files_removed = []
    for filename in metric_files:
        file_path = os.path.join(inference_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                files_removed.append(filename)
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
    
    if files_removed:
        logger.info(f"Cleaned up existing metrics for {os.path.basename(checkpoint_dir)}: {', '.join(files_removed)}")

def detect_stride_and_get_val_json(model_dir: str, default_val_json: str, config_file: str = None) -> str:
    """
    Detect the sampling frame stride from config and return the appropriate val.json file.
    
    Args:
        model_dir: Path to the model directory
        default_val_json: Default validation JSON file path
        config_file: Optional custom config file path
        
    Returns:
        Path to the appropriate validation JSON file
    """
    if config_file:
        config_path = config_file
    else:
        config_path = os.path.join(model_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}. Using default val.json")
        return default_val_json
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract sampling frame stride
        stride = config.get('INPUT', {}).get('SAMPLING_FRAME_STRIDE', 1)
        
        if stride > 1:
            # Check if stride-specific val.json exists in model directory
            stride_val_json = os.path.join(model_dir, f'val_stride{stride}.json')
            if os.path.exists(stride_val_json):
                logger.info(f"Detected stride {stride}, using {stride_val_json}")
                return stride_val_json
            else:
                # Check in data directory
                data_dir = os.path.dirname(default_val_json)
                stride_val_json = os.path.join(data_dir, f'val_stride{stride}.json')
                if os.path.exists(stride_val_json):
                    logger.info(f"Detected stride {stride}, using {stride_val_json}")
                    return stride_val_json
                else:
                    logger.warning(f"Stride {stride} detected but val_stride{stride}.json not found. Using default val.json")
                    logger.warning(f"This may cause alignment issues between predictions and ground truth!")
                    return default_val_json
        else:
            logger.info(f"Stride {stride} detected, using default val.json")
            return default_val_json
            
    except Exception as e:
        logger.warning(f"Could not read config file {config_path}: {e}. Using default val.json")
        return default_val_json

def run_mask_metrics_analysis(checkpoint_dir: str, val_json: str, confidence_threshold: float = 0.0, force_recompute: bool = False) -> bool:
    """
    Run mask metrics analysis for a single checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
        val_json: Path to validation JSON file
        confidence_threshold: Minimum confidence threshold for predictions
        force_recompute: Force recomputation even if CSV exists
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    results_json = os.path.join(checkpoint_dir, "inference", "results.json")
    csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
    cm_plot_path = os.path.join(checkpoint_dir, "inference", "confusion_matrix.png")
    
    if not os.path.exists(results_json):
        logger.warning(f"Results JSON not found: {results_json}")
        return False
    
    # Clean up existing metric files if forcing recomputation
    if force_recompute:
        cleanup_existing_metrics(checkpoint_dir)
    
    # Skip if mask metrics already exist and not forcing recomputation
    if os.path.exists(csv_path) and not force_recompute:
        logger.info(f"Mask metrics already exist for: {os.path.basename(checkpoint_dir)}")
        return True
    
    logger.info(f"Running mask metrics analysis for: {os.path.basename(checkpoint_dir)}")
    
    try:
        # Import mask_metrics functions directly to avoid subprocess overhead
        from mask_metrics import main as mask_metrics_main
        
        # Call the main function directly (silent mode for parallel processing)
        mask_metrics_main(
            results_json=results_json,
            val_json=val_json,
            csv_path=csv_path,
            cm_plot_path=cm_plot_path,
            confidence_threshold=confidence_threshold,
            verbose=False
        )
        
        logger.info(f"Mask metrics analysis completed for {os.path.basename(checkpoint_dir)}")
        return True
        
    except Exception as e:
        logger.error(f"Exception during mask metrics analysis: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to subprocess if direct import fails
        try:
            logger.info("Falling back to subprocess method...")
            cmd = [
                "python", "/home/simone/fish-dvis/visualization_scripts/mask_metrics.py",
                "--results-json", results_json,
                "--val-json", val_json,
                "--csv-path", csv_path,
                "--cm-plot-path", cm_plot_path,
                "--confidence-threshold", str(confidence_threshold)
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env['DETECTRON2_DATASETS'] = '/data'
            env['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'
            
            # Run analysis
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info(f"Mask metrics analysis completed for {os.path.basename(checkpoint_dir)}")
                return True
            else:
                logger.error(f"Mask metrics analysis failed for {os.path.basename(checkpoint_dir)}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Mask metrics analysis timed out for {os.path.basename(checkpoint_dir)}")
            return False
        except Exception as e2:
            logger.error(f"Exception during subprocess fallback: {e2}")
            return False

def process_checkpoint_wrapper(args_tuple):
    """
    Wrapper function for processing a single checkpoint.
    This function needs to be at module level for pickling.
    
    Args:
        args_tuple: Tuple of (checkpoint_dir, val_json, confidence_threshold, force_recompute)
        
    Returns:
        Tuple of (checkpoint_dir, success_status)
    """
    checkpoint_dir, val_json, confidence_threshold, force_recompute = args_tuple
    return checkpoint_dir, run_mask_metrics_analysis(checkpoint_dir, val_json, confidence_threshold, force_recompute)

def run_mask_metrics_analysis_parallel(checkpoint_dirs: List[str], val_json: str, confidence_threshold: float = 0.0, 
                                     max_workers: int = None, force_recompute: bool = False) -> Dict[str, bool]:
    """
    Run mask metrics analysis for multiple checkpoints in parallel.
    
    Args:
        checkpoint_dirs: List of checkpoint directories to process
        val_json: Path to validation JSON file
        confidence_threshold: Minimum confidence threshold for predictions
        max_workers: Maximum number of parallel workers (default: CPU count)
        force_recompute: Force recomputation even if CSV exists
        
    Returns:
        Dictionary mapping checkpoint_dir to success status
    """
    if max_workers is None:
        # Use fewer workers to avoid I/O contention and memory issues
        # Each process loads large validation JSON and results files
        max_workers = min(max(1, mp.cpu_count() // 2), len(checkpoint_dirs), 8)
    
    logger.info(f"Running parallel mask metrics analysis with {max_workers} workers...")
    
    # Filter checkpoints that need processing
    checkpoints_to_process = []
    for checkpoint_dir in checkpoint_dirs:
        csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
        if force_recompute or not os.path.exists(csv_path):
            checkpoints_to_process.append(checkpoint_dir)
        else:
            logger.info(f"Skipping {os.path.basename(checkpoint_dir)} - CSV already exists")
    
    if not checkpoints_to_process:
        logger.info("No checkpoints need processing")
        return {checkpoint_dir: True for checkpoint_dir in checkpoint_dirs}
    
    logger.info(f"Processing {len(checkpoints_to_process)} checkpoints in parallel...")
    
    # Prepare arguments for each checkpoint
    args_list = [
        (checkpoint_dir, val_json, confidence_threshold, force_recompute)
        for checkpoint_dir in checkpoints_to_process
    ]
    
    # Process checkpoints in parallel
    results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_checkpoint = {
            executor.submit(process_checkpoint_wrapper, args): checkpoint_dir 
            for args, checkpoint_dir in zip(args_list, checkpoints_to_process)
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_checkpoint):
            checkpoint_dir = future_to_checkpoint[future]
            try:
                checkpoint_dir, success = future.result()
                results[checkpoint_dir] = success
                completed += 1
                
                status = "SUCCESS" if success else "FAILED"
                logger.info(f"Completed {completed}/{len(checkpoints_to_process)}: {os.path.basename(checkpoint_dir)} - {status}")
                
            except Exception as e:
                logger.error(f"Exception processing {os.path.basename(checkpoint_dir)}: {e}")
                results[checkpoint_dir] = False
                completed += 1
    
    # Add results for checkpoints that were skipped
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir not in results:
            results[checkpoint_dir] = True  # Skipped = success
    
    elapsed_time = time.time() - start_time
    successful = sum(1 for success in results.values() if success)
    
    logger.info(f"Parallel processing completed in {elapsed_time:.1f}s")
    logger.info(f"Successfully processed {successful}/{len(checkpoint_dirs)} checkpoints")
    
    return results

def collect_summary_data(base_dir: str, evaluation_dir: str = None) -> pd.DataFrame:
    """
    Collect summary data from all checkpoint CSV files.
    
    Args:
        base_dir: Base directory containing all checkpoint results
        
    Returns:
        DataFrame with summary metrics for all checkpoints
    """
    checkpoint_dirs = find_checkpoint_results(base_dir, evaluation_dir)
    
    if not checkpoint_dirs:
        logger.error("No checkpoint result directories found!")
        return pd.DataFrame()
    
    # Collect data from all checkpoint CSV files
    summary_data = []
    
    for checkpoint_dir in checkpoint_dirs:
        # Try new separate CSV structure first
        dataset_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_dataset.csv")
        frame_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_frame.csv")
        video_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_video.csv")
        
        # Extract iteration number
        dir_name = os.path.basename(checkpoint_dir)
        match = re.search(r'checkpoint_(\d+)', dir_name)
        iteration = int(match.group(1)) if match else 0
        
        if os.path.exists(dataset_csv_path):
            # Use new separate CSV structure
            try:
                df_dataset = pd.read_csv(dataset_csv_path)
                
                # Convert dataset metrics to dictionary
                metrics_dict = {}
                for _, row in df_dataset.iterrows():
                    metrics_dict[row['metric_name']] = row['value']
                
                # Also get mean IoU from frame CSV if available
                mean_iou = None
                if os.path.exists(frame_csv_path):
                    try:
                        df_frame = pd.read_csv(frame_csv_path)
                        if len(df_frame) > 0:
                            mean_iou = df_frame['frame_IoU'].mean()
                    except Exception as e:
                        logger.warning(f"Could not read frame CSV from {checkpoint_dir}: {e}")
                
                # Get per-video metrics from video CSV if available
                mean_video_ap = None
                mean_video_ap50 = None
                if os.path.exists(video_csv_path):
                    try:
                        df_video = pd.read_csv(video_csv_path)
                        if len(df_video) > 0:
                            # Check if per-video metrics columns exist
                            if 'video_AP' in df_video.columns:
                                mean_video_ap = df_video['video_AP'].mean()
                            if 'video_AP50' in df_video.columns:
                                mean_video_ap50 = df_video['video_AP50'].mean()
                    except Exception as e:
                        logger.warning(f"Could not read video CSV from {checkpoint_dir}: {e}")
                
                # Note: All results now use the simplified metrics format

                summary_data.append({
                    'iteration': iteration,
                    'mean_iou': mean_iou,
                    'mean_boundary_f': metrics_dict.get('dataset_boundary_Fmeasure', None),
                    'mean_video_ap': mean_video_ap,
                    'mean_video_ap50': mean_video_ap50,
                    'ap10_track': metrics_dict.get('ap10_track', None),
                    'ap25_track': metrics_dict.get('ap25_track', None),
                    'ap50_track': metrics_dict.get('ap50_track', None),
                    'ap75_track': metrics_dict.get('ap75_track', None),
                    'ap95_track': metrics_dict.get('ap95_track', None),
                    # Area-weighted track metrics
                    'ap50_track_Aweighted': metrics_dict.get('ap50_track_Aweighted', None),
                    'ap50_track_small': metrics_dict.get('ap50_track_small', None),
                    'ap50_track_medium': metrics_dict.get('ap50_track_medium', None),
                    'ap50_track_large': metrics_dict.get('ap50_track_large', None),
                    # Standard COCO metrics
                    'ap_instance_Aweighted': metrics_dict.get('ap_instance_Aweighted', None),
                    'ap50_instance_Aweighted': metrics_dict.get('ap50_instance_Aweighted', None),
                    'ap75_instance_Aweighted': metrics_dict.get('ap75_instance_Aweighted', None),
                    'aps_instance_Aweighted': metrics_dict.get('aps_instance_Aweighted', None),
                    'apm_instance_Aweighted': metrics_dict.get('apm_instance_Aweighted', None),
                    'apl_instance_Aweighted': metrics_dict.get('apl_instance_Aweighted', None),
                    'ar1_instance': metrics_dict.get('ar1_instance', None),
                    'ar10_instance': metrics_dict.get('ar10_instance', None),
                    # Per-category instance-level metrics (not available in dataset CSV)
                    'ap_instance_per_cat': None,
                    'ap50_instance_per_cat': None,
                    'ap75_instance_per_cat': None,
                    'aps_instance_per_cat': None,
                    'apm_instance_per_cat': None,
                    'apl_instance_per_cat': None,
                    'ar1_instance_per_cat': None,
                    'ar10_instance_per_cat': None,
                    # Temporal consistency metrics
                    'track_completeness': metrics_dict.get('track_completeness', None),
                    'temporal_iou_stability': metrics_dict.get('temporal_iou_stability', None),
                    'track_fragmentation': metrics_dict.get('track_fragmentation', None),
                    'mean_track_length': metrics_dict.get('mean_track_length', None),
                    # Standard tracking metrics
                    'MOTA': metrics_dict.get('MOTA', None),
                    'MOTP': metrics_dict.get('MOTP', None),
                    'IDF1': metrics_dict.get('IDF1', None),
                    'HOTA': metrics_dict.get('HOTA', None),
                    'DetA': metrics_dict.get('DetA', None),
                    'AssA': metrics_dict.get('AssA', None),
                })
            except Exception as e:
                logger.warning(f"Could not read dataset CSV from {checkpoint_dir}: {e}")
        
        else:
            # Fallback to old combined CSV structure
            csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Get metrics (take first row since all rows have same dataset-level metrics)
                    if len(df) > 0:
                        row = df.iloc[0]
                        summary_data.append({
                            'iteration': iteration,
                            'mean_iou': row.get('dataset_IoU', None),
                            'mean_boundary_f': row.get('dataset_boundary_Fmeasure', None),
                            'mean_video_ap': row.get('video_AP', None),  # This will be None for old format
                            'mean_video_ap50': row.get('video_AP50', None),  # This will be None for old format
                            'ap10_track': row.get('ap10_track', None),
                            'ap25_track': row.get('ap25_track', None),
                            'ap50_track': row.get('ap50_track', None),
                            'ap75_track': row.get('ap75_track', None),
                            'ap95_track': row.get('ap95_track', None),
                            # Area-weighted track metrics
                            'ap50_track_Aweighted': row.get('ap50_track_Aweighted', None),
                            'ap50_track_small': row.get('ap50_track_small', None),
                            'ap50_track_medium': row.get('ap50_track_medium', None),
                            'ap50_track_large': row.get('ap50_track_large', None),
                            # Standard COCO metrics
                            'ap_instance_Aweighted': row.get('ap_instance_Aweighted', None),
                            'ap50_instance_Aweighted': row.get('ap50_instance_Aweighted', None),
                            'ap75_instance_Aweighted': row.get('ap75_instance_Aweighted', None),
                            'aps_instance_Aweighted': row.get('aps_instance_Aweighted', None),
                            'apm_instance_Aweighted': row.get('apm_instance_Aweighted', None),
                            'apl_instance_Aweighted': row.get('apl_instance_Aweighted', None),
                            'ar1_instance': row.get('ar1_instance', None),
                            'ar10_instance': row.get('ar10_instance', None),
                            # Per-category instance-level metrics
                            'ap_instance_per_cat': row.get('ap_instance_per_cat', None),
                            'ap50_instance_per_cat': row.get('ap50_instance_per_cat', None),
                            'ap75_instance_per_cat': row.get('ap75_instance_per_cat', None),
                            'aps_instance_per_cat': row.get('aps_instance_per_cat', None),
                            'apm_instance_per_cat': row.get('apm_instance_per_cat', None),
                            'apl_instance_per_cat': row.get('apl_instance_per_cat', None),
                            'ar1_instance_per_cat': row.get('ar1_instance_per_cat', None),
                            'ar10_instance_per_cat': row.get('ar10_instance_per_cat', None),
                            # Temporal consistency metrics
                            'track_completeness': row.get('track_completeness', None),
                            'temporal_iou_stability': row.get('temporal_iou_stability', None),
                            'track_fragmentation': row.get('track_fragmentation', None),
                            'mean_track_length': row.get('mean_track_length', None),
                            # Standard tracking metrics
                            'MOTA': row.get('MOTA', None),
                            'MOTP': row.get('MOTP', None),
                            'IDF1': row.get('IDF1', None),
                            'HOTA': row.get('HOTA', None),
                            'DetA': row.get('DetA', None),
                            'AssA': row.get('AssA', None),
                        })
                except Exception as e:
                    logger.warning(f"Could not read CSV from {checkpoint_dir}: {e}")
    
    if not summary_data:
        logger.warning("No summary data found")
        return pd.DataFrame()
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('iteration')
    
    return df_summary

def collect_per_species_data(base_dir: str, evaluation_dir: str = None) -> pd.DataFrame:
    """
    Collect per-species data from all checkpoint CSV files.
    
    Args:
        base_dir: Base directory containing all checkpoint results
        
    Returns:
        DataFrame with per-species metrics for all checkpoints
    """
    checkpoint_dirs = find_checkpoint_results(base_dir, evaluation_dir)
    
    if not checkpoint_dirs:
        logger.error("No checkpoint result directories found!")
        return pd.DataFrame()
    
    # Collect per-species data from all checkpoint CSV files
    per_species_data = []
    
    for checkpoint_dir in checkpoint_dirs:
        # Try new separate CSV structure first
        category_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_category.csv")
        frame_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_frame.csv")
        
        # Extract iteration number
        dir_name = os.path.basename(checkpoint_dir)
        match = re.search(r'checkpoint_(\d+)', dir_name)
        iteration = int(match.group(1)) if match else 0
        
        if os.path.exists(category_csv_path):
            # Use new separate CSV structure
            try:
                df_category = pd.read_csv(category_csv_path)
                
                # Get per-species metrics from category CSV
                for _, row in df_category.iterrows():
                    per_species_data.append({
                        'iteration': iteration,
                        'category_id': row.get('category_id'),
                        'category_name': row.get('category_name'),
                        'iou': None,  # Not available in category CSV
                        'boundary_f': None,  # Not available in category CSV
                        'ap_10': row.get('ap10_track_per_cat', None),  # Track-level metrics (removed)
                        'ap_25': row.get('ap25_track_per_cat', None),  # Track-level metrics (removed)
                        'ap_50': row.get('ap50_track_per_cat', None),  # Track-level metrics (removed)
                        'ap_75': row.get('ap75_track_per_cat', None),  # Track-level metrics (removed)
                        'ap_95': row.get('ap95_track_per_cat', None),  # Track-level metrics (removed)
                        # Area-weighted track metrics per species (not available in category CSV)
                        'ap50_track_Aweighted': None,
                        'ap50_track_small': None,
                        'ap50_track_medium': None,
                        'ap50_track_large': None,
                        # COCO metrics per species (not available in category CSV)
                        'ap_instance_Aweighted': None,
                        'ap50_instance_Aweighted': None,
                        'ap75_instance_Aweighted': None,
                        'aps_instance_Aweighted': None,
                        'apm_instance_Aweighted': None,
                        'apl_instance_Aweighted': None,
                        'ar1_instance': None,
                        'ar10_instance': None,
                        # Per-category instance-level metrics per species
                        'ap_instance_per_cat': row.get('ap_instance_per_cat', None),
                        'ap50_instance_per_cat': row.get('ap50_instance_per_cat', None),
                        'ap75_instance_per_cat': row.get('ap75_instance_per_cat', None),
                        'aps_instance_per_cat': row.get('aps_instance_per_cat', None),
                        'apm_instance_per_cat': row.get('apm_instance_per_cat', None),
                        'apl_instance_per_cat': row.get('apl_instance_per_cat', None),
                        'ar1_instance_per_cat': row.get('ar1_instance_per_cat', None),
                        'ar10_instance_per_cat': row.get('ar10_instance_per_cat', None),
                        # Standard tracking metrics per species (not available in category CSV)
                        'MOTA': None,
                        'MOTP': None,
                        'IDF1': None,
                        'HOTA': None,
                        'DetA': None,
                        'AssA': None,
                    })
            except Exception as e:
                logger.warning(f"Could not read category CSV from {checkpoint_dir}: {e}")
        
        else:
            # Fallback to old combined CSV structure
            csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Get per-species metrics (skip the dataset-level row)
                    for _, row in df.iterrows():
                        # Skip dataset-level metrics (these have NaN category_id)
                        if pd.isna(row.get('category_id')):
                            continue
                        
                        per_species_data.append({
                            'iteration': iteration,
                            'category_id': row.get('category_id'),
                            'category_name': row.get('gt_category_name'),
                            'iou': row.get('IoU', None),
                            'boundary_f': row.get('boundary_Fmeasure', None),
                            'ap_10': row.get('AP@0.1', None),
                            'ap_25': row.get('AP@0.25', None),
                            'ap_50': row.get('AP@0.5', None),
                            'ap_75': row.get('AP@0.75', None),
                            'ap_95': row.get('AP@0.95', None),
                            # Area-weighted track metrics per species
                            'ap50_track_Aweighted': row.get('ap50_track_Aweighted', None),
                            'ap50_track_small': row.get('ap50_track_small', None),
                            'ap50_track_medium': row.get('ap50_track_medium', None),
                            'ap50_track_large': row.get('ap50_track_large', None),
                            # COCO metrics per species
                            'ap_instance_Aweighted': row.get('ap_instance_Aweighted', None),
                            'ap50_instance_Aweighted': row.get('ap50_instance_Aweighted', None),
                            'ap75_instance_Aweighted': row.get('ap75_instance_Aweighted', None),
                            'aps_instance_Aweighted': row.get('aps_instance_Aweighted', None),
                            'apm_instance_Aweighted': row.get('apm_instance_Aweighted', None),
                            'apl_instance_Aweighted': row.get('apl_instance_Aweighted', None),
                            'ar1_instance': row.get('ar1_instance', None),
                            'ar10_instance': row.get('ar10_instance', None),
                            # Per-category instance-level metrics per species
                            'ap_instance_per_cat': row.get('ap_instance_per_cat', None),
                            'ap50_instance_per_cat': row.get('ap50_instance_per_cat', None),
                            'ap75_instance_per_cat': row.get('ap75_instance_per_cat', None),
                            'aps_instance_per_cat': row.get('aps_instance_per_cat', None),
                            'apm_instance_per_cat': row.get('apm_instance_per_cat', None),
                            'apl_instance_per_cat': row.get('apl_instance_per_cat', None),
                            'ar1_instance_per_cat': row.get('ar1_instance_per_cat', None),
                            'ar10_instance_per_cat': row.get('ar10_instance_per_cat', None),
                            # Standard tracking metrics per species
                            'MOTA': row.get('MOTA', None),
                            'MOTP': row.get('MOTP', None),
                            'IDF1': row.get('IDF1', None),
                            'HOTA': row.get('HOTA', None),
                            'DetA': row.get('DetA', None),
                            'AssA': row.get('AssA', None),
                        })
                except Exception as e:
                    logger.warning(f"Could not read CSV from {checkpoint_dir}: {e}")
    
    if not per_species_data:
        logger.warning("No per-species data found")
        return pd.DataFrame()
    
    # Create per-species DataFrame
    df_per_species = pd.DataFrame(per_species_data)
    df_per_species = df_per_species.sort_values(['iteration', 'category_id'])
    
    return df_per_species

def create_basic_summary_plot(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create basic summary plot (similar to run_mask_metrics_for_checkpoints.py).
    Adapts to available metrics - shows only COCO metrics in fast mode.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to plot")
        return
    
    # Check what data is available
    has_track_data = 'ap50_track' in df_summary.columns and df_summary['ap50_track'].notna().any()
    
    if has_track_data:
        # Full mode: include all metrics
        metrics = ['mean_iou', 'mean_boundary_f', 'ap25_track', 'ap50_track', 'ap75_track', 'ap95_track', 
                   'ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted', 
                   'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
        titles = ['Mean IoU', 'Mean Boundary F-measure', 'AP25 (Track)', 'AP50 (Track)', 'AP75 (Track)', 'AP95 (Track)',
                  'AP (Instance, A-weighted)', 'AP50 (Instance, A-weighted)', 'AP75 (Instance, A-weighted)', 
                  'APs (Instance, A-weighted)', 'APm (Instance, A-weighted)', 'APl (Instance, A-weighted)']
        fig_title = 'Model Performance Across Checkpoints (All Metrics)'
        fig_size = (20, 15)
        subplot_shape = (3, 4)
    else:
        # Simplified mode: only COCO instance metrics
        metrics = ['ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted', 
                   'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
        titles = ['AP (Instance, A-weighted)', 'AP50 (Instance, A-weighted)', 'AP75 (Instance, A-weighted)', 
                  'APs (Instance, A-weighted)', 'APm (Instance, A-weighted)', 'APl (Instance, A-weighted)']
        fig_title = 'Model Performance Across Checkpoints (COCO Metrics Only)'
        fig_size = (15, 10)
        subplot_shape = (2, 3)
    
    # Filter metrics to only those that exist in the dataframe and have data
    available_metrics = []
    available_titles = []
    for metric, title in zip(metrics, titles):
        if metric in df_summary.columns and df_summary[metric].notna().any():
            available_metrics.append(metric)
            available_titles.append(title)
    
    if not available_metrics:
        logger.warning("No valid metrics found for basic summary plot")
        return
    
    # Adjust subplot layout based on available metrics
    n_metrics = len(available_metrics)
    if n_metrics <= 6:
        if n_metrics <= 3:
            subplot_shape = (1, n_metrics) if n_metrics > 1 else (1, 2)
            fig_size = (5 * n_metrics, 5)
        else:
            subplot_shape = (2, 3)
            fig_size = (15, 10)
    
    # Create plots
    fig, axes = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=fig_size)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif subplot_shape[0] == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(available_metrics, available_titles)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter out None values
        valid_data = df_summary[df_summary[metric].notna()]
        
        if len(valid_data) > 0:
            ax.plot(valid_data['iteration'], valid_data[metric], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['iteration'], valid_data[metric], 1)
                p = np.poly1d(z)
                ax.plot(valid_data['iteration'], p(valid_data['iteration']), "--", alpha=0.8,
                       label=f'Trend: {z[0]:.2e}x + {z[1]:.3f}')
                ax.legend()
        else:
            ax.set_title(f'{title} (No Data)')
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'mask_metrics_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary CSV
    summary_csv_path = os.path.join(output_dir, 'mask_metrics_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    logger.info(f"Basic summary plot saved to: {summary_plot_path}")
    logger.info(f"Summary CSV saved to: {summary_csv_path}")
    
    # Print summary
    logger.info("\nAll Metrics Summary:")
    logger.info("=" * 80)
    for _, row in df_summary.iterrows():
        mean_iou = row.get('mean_iou', 'N/A')
        ap50_track = row.get('ap50_track', 'N/A')
        ap50_instance = row.get('ap50_instance_Aweighted', 'N/A')
        ap_instance = row.get('ap_instance_Aweighted', 'N/A')
        
        # Format values properly
        iou_str = f"{mean_iou:.4f}" if isinstance(mean_iou, (int, float)) else str(mean_iou)
        map_str = f"{ap50_track:.4f}" if isinstance(ap50_track, (int, float)) else str(ap50_track)
        ap50_str = f"{ap50_instance:.4f}" if isinstance(ap50_instance, (int, float)) else str(ap50_instance)
        ap_str = f"{ap_instance:.4f}" if isinstance(ap_instance, (int, float)) else str(ap_instance)
        
        logger.info(f"Iteration {row['iteration']}: "
                   f"IoU={iou_str}, "
                   f"mAP@0.5(Video)={map_str}, "
                   f"AP50(COCO)={ap50_str}, "
                   f"AP(COCO)={ap_str}")
        
        # Print temporal metrics if available
        track_completeness = row.get('track_completeness', 'N/A')
        temporal_stability = row.get('temporal_iou_stability', 'N/A')
        if isinstance(track_completeness, (int, float)) and isinstance(temporal_stability, (int, float)):
            logger.info(f"  Temporal: Completeness={track_completeness:.3f}, Stability={temporal_stability:.3f}")

def create_ap_per_category_plots_for_all_checkpoints(base_dir: str, evaluation_dir: str = None) -> None:
    """
    Create AP per category plots for all checkpoints.
    
    Args:
        base_dir: Base directory containing all checkpoint results
        evaluation_dir: Specific evaluation directory (for multi-run structure)
    """
    checkpoint_dirs = find_checkpoint_results(base_dir, evaluation_dir)
    
    if not checkpoint_dirs:
        logger.warning("No checkpoint directories found for AP per category plots")
        return
    
    logger.info(f"Creating AP per category plots for {len(checkpoint_dirs)} checkpoints...")
    
    for checkpoint_dir in checkpoint_dirs:
        # Try new separate CSV structure first
        category_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_category.csv")
        plot_path = os.path.join(checkpoint_dir, "inference", "AP_per_category.png")
        
        if os.path.exists(category_csv_path):
            try:
                plot_ap_per_category_from_csv(category_csv_path, plot_path)
            except Exception as e:
                logger.warning(f"Failed to create AP per category plot for {os.path.basename(checkpoint_dir)}: {e}")
        else:
            # Fallback to old combined CSV structure
            csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    plot_ap_per_category_from_csv(csv_path, plot_path)
                except Exception as e:
                    logger.warning(f"Failed to create AP per category plot for {os.path.basename(checkpoint_dir)}: {e}")
            else:
                logger.warning(f"No CSV file found for {os.path.basename(checkpoint_dir)}")

def create_comprehensive_analysis(df_summary: pd.DataFrame, output_dir: str, df_per_species: pd.DataFrame = None, base_dir: str = None, evaluation_dir: str = None, config_file: str = None) -> None:
    """
    Create comprehensive analysis (similar to plot_existing_checkpoint_results.py).
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
        df_per_species: DataFrame with per-species metrics (optional)
    """
    if df_summary.empty:
        logger.warning("No data to analyze")
        return
    
    # Check if we have track-level data available
    has_track_data = df_summary['ap50_track'].notna().any()
    
    # Create comprehensive plots (adapt based on available data)
    create_performance_plots(df_summary, output_dir)
    create_coco_comparison_plots(df_summary, output_dir)
    
    # Only create track-dependent plots if we have track data
    if has_track_data:
        create_comprehensive_comparison_plots(df_summary, output_dir)
    else:
        logger.info("No track-level data available - skipping comprehensive comparison plots")
    
    # Create temporal consistency plots (if data available)
    if df_summary['track_completeness'].notna().any():
        create_temporal_consistency_plots(df_summary, output_dir)
    else:
        logger.info("No temporal consistency data available - skipping temporal plots")
    
    create_trend_analysis(df_summary, output_dir)
    create_best_checkpoint_analysis(df_summary, output_dir)
    
    # Create tracking metrics plots (if data available)
    if df_summary['MOTA'].notna().any():
        create_tracking_metrics_plots(df_summary, output_dir)
    else:
        logger.info("No tracking metrics data available - skipping tracking plots")
    
    # Create area-weighted comparison plots (if data available)
    if df_summary['ap50_track_Aweighted'].notna().any():
        create_area_weighted_comparison_plots(df_summary, output_dir)
    else:
        logger.info("No area-weighted data available - skipping area-weighted plots")
    
    # Create per-species plots if data is available
    if df_per_species is not None and not df_per_species.empty:
        logger.info("Creating per-species performance plots...")
        # Per-species performance plots removed - not useful metrics
        create_per_category_comparison_plots(df_per_species, output_dir)
    
    # Create AP per category plots for all checkpoints
    logger.info("Creating AP per category plots for all checkpoints...")
    create_ap_per_category_plots_for_all_checkpoints(base_dir, evaluation_dir)
    
    # Generate training loss plots if metrics.json exists
    model_dir = os.path.dirname(output_dir) if output_dir.endswith('checkpoint_evaluations') else output_dir
    metrics_json_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_json_path):
        logger.info("Generating training loss plots...")
        plot_training_loss(model_dir)
    else:
        logger.info("No metrics.json found - skipping training loss plots")
    
    # Create comprehensive model performance report
    logger.info("Generating model performance report...")
    create_model_performance_report(df_summary, output_dir, model_dir, df_per_species, config_file)
    
    # Save comprehensive summary CSV
    summary_csv_path = os.path.join(output_dir, 'comprehensive_metrics_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    logger.info(f"Comprehensive analysis saved to: {output_dir}")
    logger.info(f"Summary CSV saved to: {summary_csv_path}")
    logger.info(f"Model performance report saved to: {output_dir}/model_performance_report.txt")

def create_performance_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create performance comparison plots.
    Adapts to available metrics - shows only COCO metrics in fast mode.
    """
    # Check what data is available
    has_track_data = 'ap50_track' in df_summary.columns and df_summary['ap50_track'].notna().any()
    
    # Define metrics based on available data
    if has_track_data:
        # Full mode: include all metrics
        metrics = ['mean_iou', 'mean_boundary_f', 'ap25_track', 'ap50_track', 'ap75_track', 'ap95_track', 
                   'ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted', 
                   'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
        titles = ['Mean IoU', 'Mean Boundary F-measure', 'AP25 (Track)', 'AP50 (Track)', 'AP75 (Track)', 'AP95 (Track)',
                  'AP (Instance, A-weighted)', 'AP50 (Instance, A-weighted)', 'AP75 (Instance, A-weighted)', 
                  'APs (Instance, A-weighted)', 'APm (Instance, A-weighted)', 'APl (Instance, A-weighted)']
        fig_title = 'Model Performance Across Checkpoints (All Metrics)'
        fig_size = (20, 15)
        subplot_shape = (3, 4)
    else:
        # Simplified mode: only COCO instance metrics
        metrics = ['ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted', 
                   'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
        titles = ['AP (Instance, A-weighted)', 'AP50 (Instance, A-weighted)', 'AP75 (Instance, A-weighted)', 
                  'APs (Instance, A-weighted)', 'APm (Instance, A-weighted)', 'APl (Instance, A-weighted)']
        fig_title = 'Model Performance Across Checkpoints (COCO Metrics Only)'
        fig_size = (15, 10)
        subplot_shape = (2, 3)
    
    # Filter metrics to only those that exist in the dataframe and have data
    available_metrics = []
    available_titles = []
    for metric, title in zip(metrics, titles):
        if metric in df_summary.columns and df_summary[metric].notna().any():
            available_metrics.append(metric)
            available_titles.append(title)
    
    if not available_metrics:
        logger.warning("No valid metrics found for performance plots")
        return
    
    # Adjust subplot layout based on available metrics
    n_metrics = len(available_metrics)
    if n_metrics <= 6:
        if n_metrics <= 3:
            subplot_shape = (1, n_metrics) if n_metrics > 1 else (1, 2)
            fig_size = (5 * n_metrics, 5)
        else:
            subplot_shape = (2, 3)
            fig_size = (15, 10)
    
    # Create plots
    fig, axes = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=fig_size)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif subplot_shape[0] == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(available_metrics, available_titles)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter out None values
        valid_data = df_summary[df_summary[metric].notna()]
        
        if len(valid_data) > 0:
            ax.plot(valid_data['iteration'], valid_data[metric], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['iteration'], valid_data[metric], 1)
                p = np.poly1d(z)
                ax.plot(valid_data['iteration'], p(valid_data['iteration']), "--", alpha=0.8,
                       label=f'Trend: {z[0]:.2e}x + {z[1]:.3f}')
                ax.legend()
        else:
            ax.set_title(f'{title} (No Data)')
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    performance_plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_coco_comparison_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create COCO-specific comparison plots.
    Adapts to available data - shows only instance metrics in fast mode.
    """
    # Check what data is available
    has_track_data = 'ap50_track' in df_summary.columns and df_summary['ap50_track'].notna().any()
    has_instance_data = 'ap50_instance_Aweighted' in df_summary.columns and df_summary['ap50_instance_Aweighted'].notna().any()
    
    if not has_instance_data:
        logger.warning("No instance-level metrics available for COCO comparison plots")
        return
    
    if has_track_data:
        # Full mode: compare track vs instance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Track-level vs Instance-level Metrics Comparison', fontsize=16, fontweight='bold')
    else:
        # Fast mode: show only instance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Instance-level COCO Metrics (Fast Mode)', fontsize=16, fontweight='bold')
    
    # AP comparison
    ax1 = axes[0, 0]
    if has_track_data and 'ap50_track' in df_summary.columns and 'ap50_instance_Aweighted' in df_summary.columns:
        # Full mode: compare track vs instance
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data['ap50_track'], 'o-', 
                    color='blue', linewidth=2, markersize=6, label='AP50 (Track)')
            ax1.plot(valid_data['iteration'], valid_data['ap50_instance_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP50 (Instance, A-weighted)')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('AP50')
            ax1.set_title('AP50 Comparison (Track vs Instance)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    elif 'ap50_instance_Aweighted' in df_summary.columns:
        # Fast mode: show only instance metrics
        valid_data = df_summary[df_summary['ap50_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data['ap50_instance_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP50 (Instance, A-weighted)')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('AP50')
            ax1.set_title('AP50 (Instance-level, Area-weighted)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No AP50 data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('AP50 (No Data)')
    
    # AP75 comparison
    ax2 = axes[0, 1]
    if has_track_data and 'ap75_track' in df_summary.columns and 'ap75_instance_Aweighted' in df_summary.columns:
        # Full mode: compare track vs instance
        valid_data = df_summary[df_summary['ap75_track'].notna() & df_summary['ap75_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax2.plot(valid_data['iteration'], valid_data['ap75_track'], 'o-', 
                    color='blue', linewidth=2, markersize=6, label='AP75 (Track)')
            ax2.plot(valid_data['iteration'], valid_data['ap75_instance_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP75 (Instance, A-weighted)')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('AP75')
            ax2.set_title('AP75 Comparison (Track vs Instance)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    elif 'ap75_instance_Aweighted' in df_summary.columns:
        # Fast mode: show only instance metrics
        valid_data = df_summary[df_summary['ap75_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax2.plot(valid_data['iteration'], valid_data['ap75_instance_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP75 (Instance, A-weighted)')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('AP75')
            ax2.set_title('AP75 (Instance-level, Area-weighted)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No AP75 data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('AP75 (No Data)')
    
    # Area-based AP comparison
    ax3 = axes[1, 0]
    area_metrics = ['aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
    colors = ['green', 'orange', 'purple']
    labels = ['APs (Small, Instance, A-weighted)', 'APm (Medium, Instance, A-weighted)', 'APl (Large, Instance, A-weighted)']
    
    for metric, color, label in zip(area_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax3.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('AP')
    ax3.set_title('COCO Area-Based AP (Instance-level, Area-weighted)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Recall comparison
    ax4 = axes[1, 1]
    recall_metrics = ['ar1_instance', 'ar10_instance']
    colors = ['cyan', 'magenta']
    labels = ['AR1 (Instance)', 'AR10 (Instance)']
    
    for metric, color, label in zip(recall_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax4.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('AR')
    ax4.set_title('COCO Average Recall (Instance-level)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    coco_plot_path = os.path.join(output_dir, 'coco_comparison.png')
    plt.savefig(coco_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_consistency_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create temporal consistency analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Temporal Consistency Analysis', fontsize=16, fontweight='bold')
    
    # Track completeness
    ax1 = axes[0, 0]
    if 'track_completeness' in df_summary.columns:
        valid_data = df_summary[df_summary['track_completeness'].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data['track_completeness'], 'o-', 
                    color='blue', linewidth=2, markersize=6)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Track Completeness')
            ax1.set_title('Track Completeness (Higher is Better)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
    
    # Temporal IoU stability
    ax2 = axes[0, 1]
    if 'temporal_iou_stability' in df_summary.columns:
        valid_data = df_summary[df_summary['temporal_iou_stability'].notna()]
        if len(valid_data) > 0:
            ax2.plot(valid_data['iteration'], valid_data['temporal_iou_stability'], 'o-', 
                    color='green', linewidth=2, markersize=6)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Temporal IoU Stability')
            ax2.set_title('Temporal IoU Stability (Higher is Better)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
    
    # Track fragmentation
    ax3 = axes[1, 0]
    if 'track_fragmentation' in df_summary.columns:
        valid_data = df_summary[df_summary['track_fragmentation'].notna()]
        if len(valid_data) > 0:
            ax3.plot(valid_data['iteration'], valid_data['track_fragmentation'], 'o-', 
                    color='red', linewidth=2, markersize=6)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Track Fragmentation')
            ax3.set_title('Track Fragmentation (Lower is Better)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
    
    # Mean track length
    ax4 = axes[1, 1]
    if 'mean_track_length' in df_summary.columns:
        valid_data = df_summary[df_summary['mean_track_length'].notna()]
        if len(valid_data) > 0:
            ax4.plot(valid_data['iteration'], valid_data['mean_track_length'], 'o-', 
                    color='purple', linewidth=2, markersize=6)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Mean Track Length (frames)')
            ax4.set_title('Mean Track Length')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    temporal_plot_path = os.path.join(output_dir, 'temporal_consistency.png')
    plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_trend_analysis(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create trend analysis plots.
    """
    # Determine which metrics are available
    available_metrics = []
    metric_info = [
        ('ap50_track', 'AP50 (Track)', '#1f77b4'),
        ('mean_iou', 'Mean IoU', '#ff7f0e'),
        ('ap50_instance_Aweighted', 'AP50 (Instance, A-weighted)', '#2ca02c'),
        ('ap_instance_Aweighted', 'AP (Instance, A-weighted)', '#d62728')
    ]
    
    for metric, label, color in metric_info:
        if metric in df_summary.columns and df_summary[metric].notna().any():
            available_metrics.append((metric, label, color))
    
    if not available_metrics:
        logger.info("No trend data available - skipping trend analysis")
        return
    
    # Determine number of subplots based on available data
    has_track_data = 'ap50_track' in df_summary.columns and df_summary['ap50_track'].notna().any()
    
    if has_track_data:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None
    
    fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Key metrics over time
    for metric, label, color in available_metrics:
        valid_data = df_summary[df_summary[metric].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                    color=color, linewidth=2, markersize=6, label=label)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Key Performance Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance improvement (only if track data available)
    if ax2 is not None and has_track_data:
        valid_data = df_summary[df_summary['ap50_track'].notna()]
        if len(valid_data) > 1:
            # Calculate improvement from first to last checkpoint
            first_val = valid_data.iloc[0]['ap50_track']
            last_val = valid_data.iloc[-1]['ap50_track']
            
            # Avoid division by zero
            if first_val > 0:
                improvement = ((last_val - first_val) / first_val) * 100
            else:
                improvement = float('inf') if last_val > 0 else 0
            
            # Plot improvement over time
            improvements = []
            for i in range(len(valid_data)):
                if i == 0:
                    improvements.append(0)
                else:
                    if first_val > 0:
                        current_improvement = ((valid_data.iloc[i]['ap50_track'] - first_val) / first_val) * 100
                    else:
                        current_improvement = float('inf') if valid_data.iloc[i]['ap50_track'] > 0 else 0
                    improvements.append(current_improvement)
            
            ax2.plot(valid_data['iteration'], improvements, 'o-', 
                    color='green', linewidth=2, markersize=6)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title(f'Performance Improvement\n(Total: {improvement:.1f}%)')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    trend_plot_path = os.path.join(output_dir, 'trend_analysis.png')
    plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_per_species_performance_plots(df_per_species: pd.DataFrame, output_dir: str) -> None:
    """
    Create per-species performance plots showing key metrics by species.
    
    Args:
        df_per_species: DataFrame with per-species metrics
        output_dir: Directory to save plots
    """
    if df_per_species.empty:
        logger.warning("No per-species data to plot")
        return
    
    # Get unique species
    species_list = df_per_species['category_name'].unique()
    if len(species_list) == 0:
        logger.warning("No species found in data")
        return
    
    # Create color palette for species
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_list)))
    color_dict = dict(zip(species_list, colors))
    
    # Create plots for key metrics
    key_metrics = [
        ('ap_50', 'AP@0.5 (Video)', 'AP@0.5'),
        ('ap50_instance_Aweighted', 'AP50 (Instance, A-weighted)', 'AP50'),
        ('iou', 'IoU', 'IoU'),
        ('ap_instance_Aweighted', 'AP (Instance, A-weighted)', 'AP'),
        ('aps_instance_Aweighted', 'APs (Instance, A-weighted)', 'APs'),
        ('IDF1', 'IDF1', 'IDF1')
    ]
    
    # Check if any species have valid data for any metric
    has_valid_data = False
    for species in species_list:
        species_data = df_per_species[df_per_species['category_name'] == species]
        for metric, _, _ in key_metrics:
            if metric in species_data.columns and species_data[metric].notna().any():
                has_valid_data = True
                break
        if has_valid_data:
            break
    
    if not has_valid_data:
        logger.warning("No valid per-species data found for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Per-Species Performance Metrics Across Checkpoints', fontsize=16, fontweight='bold')
    
    for idx, (metric, title, short_title) in enumerate(key_metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot each species
        for species in species_list:
            species_data = df_per_species[df_per_species['category_name'] == species]
            if len(species_data) > 0 and metric in species_data.columns:
                valid_data = species_data[species_data[metric].notna()]
                if len(valid_data) > 0:
                    ax.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                           color=color_dict[species], linewidth=2, markersize=4, 
                           label=species, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Species')
        # Only add legend if there are plots
        if ax.get_legend_handles_labels()[0]:  # Check if there are any legend handles
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for better visualization
        if metric in ['iou', 'boundary_f']:
            ax.set_ylim(0, 1)
        elif 'ap' in metric.lower():
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    per_species_plot_path = os.path.join(output_dir, 'per_species_performance.png')
    plt.savefig(per_species_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary table of best performance per species
    # Per-species summary table removed - not useful metrics
    
    logger.info(f"Per-species performance plots saved to: {per_species_plot_path}")

def create_per_species_summary_table(df_per_species: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary table showing best performance per species.
    
    Args:
        df_per_species: DataFrame with per-species metrics
        output_dir: Directory to save the table
    """
    if df_per_species.empty:
        return
    
    # Get unique species
    species_list = df_per_species['category_name'].unique()
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, len(species_list) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data - only show available metrics
    table_data = []
    metrics_to_show = [
        ('ap50_instance_Aweighted', 'AP50 (Instance, A-weighted)'),
        ('ap_instance_Aweighted', 'AP (Instance, A-weighted)'),
        ('aps_instance_Aweighted', 'APs (Instance, A-weighted)'),
        ('apm_instance_Aweighted', 'APm (Instance, A-weighted)'),
        ('apl_instance_Aweighted', 'APl (Instance, A-weighted)'),
        ('ar1_instance', 'AR1 (Instance)'),
        ('ar10_instance', 'AR10 (Instance)')
    ]
    
    # Filter to only show metrics that exist in the data
    available_metrics = []
    for metric, label in metrics_to_show:
        if metric in df_per_species.columns:
            available_metrics.append((metric, label))
    
    if not available_metrics:
        logger.info("No per-species metrics available - skipping per-species summary table")
        return
    
    for species in species_list:
        species_data = df_per_species[df_per_species['category_name'] == species]
        if len(species_data) == 0:
            continue
        
        row_data = [species]
        for metric, metric_name in available_metrics:
            if metric in species_data.columns:
                valid_data = species_data[species_data[metric].notna()]
                if len(valid_data) > 0:
                    best_idx = valid_data[metric].idxmax()
                    best_value = valid_data.loc[best_idx, metric]
                    best_iter = valid_data.loc[best_idx, 'iteration']
                    row_data.append(f"{best_value:.3f} ({best_iter:.0f})")
                else:
                    row_data.append("N/A")
            else:
                row_data.append("N/A")
        
        table_data.append(row_data)
    
    # Check if we have any data to display
    if not table_data:
        logger.warning("No per-species data available for summary table")
        plt.close(fig)
        return
    
    # Create table
    headers = ['Species'] + [metric_name for _, metric_name in available_metrics]
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Best Performance per Species (Value at Best Iteration)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_table_path = os.path.join(output_dir, 'per_species_summary_table.png')
    plt.savefig(summary_table_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also save as CSV
    summary_csv_path = os.path.join(output_dir, 'per_species_summary.csv')
    summary_df = pd.DataFrame(table_data, columns=headers)
    summary_df.to_csv(summary_csv_path, index=False)
    
    logger.info(f"Per-species summary table saved to: {summary_table_path}")
    logger.info(f"Per-species summary CSV saved to: {summary_csv_path}")

def create_tracking_metrics_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create tracking-specific performance plots showing all tracking metrics.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to plot")
        return
    
    # Create tracking metrics plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Tracking Performance Metrics Across Checkpoints', fontsize=16, fontweight='bold')
    
    # Custom temporal metrics
    custom_metrics = [
        ('track_completeness', 'Track Completeness', 'Fraction of frames with valid predictions'),
        ('temporal_iou_stability', 'Temporal IoU Stability', '1 - std(IoU) across frames'),
        ('track_fragmentation', 'Track Fragmentation', 'Number of gaps in tracks (lower is better)')
    ]
    
    # Standard tracking metrics
    standard_metrics = [
        ('MOTA', 'MOTA', 'Multiple Object Tracking Accuracy'),
        ('IDF1', 'IDF1', 'ID F1-Score (Identity Consistency)'),
        ('HOTA', 'HOTA', 'Higher Order Tracking Accuracy')
    ]
    
    # Plot custom temporal metrics
    for i, (metric, title, description) in enumerate(custom_metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        valid_data = df_summary[df_summary[metric].notna()]
        if len(valid_data) > 0:
            ax.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                   color='blue', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(f'{title}\n{description}')
            ax.grid(True, alpha=0.3)
            
            # Set appropriate y-axis limits
            if metric == 'track_fragmentation':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 1)
    
    # Plot standard tracking metrics
    for i, (metric, title, description) in enumerate(standard_metrics):
        row = 1  # Second row
        col = i
        ax = axes[row, col]
        
        valid_data = df_summary[df_summary[metric].notna()]
        if len(valid_data) > 0:
            ax.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                   color='red', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(f'{title}\n{description}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    tracking_plot_path = os.path.join(output_dir, 'tracking_metrics.png')
    plt.savefig(tracking_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed tracking analysis table
    create_tracking_analysis_table(df_summary, output_dir)
    
    logger.info(f"Tracking metrics plots saved to: {tracking_plot_path}")

def create_tracking_analysis_table(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create a detailed tracking analysis table.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save the table
    """
    if df_summary.empty:
        return
    
    # Find best checkpoints for different tracking metrics
    tracking_metrics = {
        'MOTA': 'MOTA (Multiple Object Tracking Accuracy)',
        'IDF1': 'IDF1 (ID F1-Score)',
        'HOTA': 'HOTA (Higher Order Tracking Accuracy)',
        'track_completeness': 'Track Completeness',
        'temporal_iou_stability': 'Temporal IoU Stability',
        'DetA': 'DetA (Detection Accuracy)',
        'AssA': 'AssA (Association Accuracy)'
    }
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(14, len(tracking_metrics) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for metric, metric_name in tracking_metrics.items():
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                best_idx = valid_data[metric].idxmax()
                best_value = valid_data.loc[best_idx, metric]
                best_iter = valid_data.loc[best_idx, 'iteration']
                worst_idx = valid_data[metric].idxmin()
                worst_value = valid_data.loc[worst_idx, metric]
                worst_iter = valid_data.loc[worst_idx, 'iteration']
                avg_value = valid_data[metric].mean()
                
                table_data.append([
                    metric_name,
                    f"{best_value:.4f} ({best_iter:.0f})",
                    f"{worst_value:.4f} ({worst_iter:.0f})",
                    f"{avg_value:.4f}"
                ])
    
    # Create table
    headers = ['Tracking Metric', 'Best (Iteration)', 'Worst (Iteration)', 'Average']
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Tracking Metrics Analysis (Best, Worst, and Average Performance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    tracking_table_path = os.path.join(output_dir, 'tracking_analysis_table.png')
    plt.savefig(tracking_table_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also save as CSV
    tracking_csv_path = os.path.join(output_dir, 'tracking_analysis.csv')
    tracking_df = pd.DataFrame(table_data, columns=headers)
    tracking_df.to_csv(tracking_csv_path, index=False)
    
    logger.info(f"Tracking analysis table saved to: {tracking_table_path}")
    logger.info(f"Tracking analysis CSV saved to: {tracking_csv_path}")

def create_area_weighted_comparison_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create plots comparing regular mAP@0.5 vs area-weighted mAP@0.5.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AP50 (Track): Unweighted vs Area-Weighted Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Regular vs Area-weighted AP50 (Track)
    ax1 = axes[0, 0]
    if 'ap50_track' in df_summary.columns and 'ap50_track_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_track_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data['ap50_track'], 'o-', 
                    color='blue', linewidth=2, markersize=6, label='AP50 (Track, Unweighted)')
            ax1.plot(valid_data['iteration'], valid_data['ap50_track_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP50 (Track, Area-Weighted)')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('AP50')
            ax1.set_title('AP50 (Track): Unweighted vs Area-Weighted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area breakdown
    ax2 = axes[0, 1]
    area_metrics = ['ap50_track_small', 'ap50_track_medium', 'ap50_track_large']
    colors = ['green', 'orange', 'purple']
    labels = ['Small (< 128²)', 'Medium (128²-256²)', 'Large (> 256²)']
    
    for metric, color, label in zip(area_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax2.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('mAP@0.5')
    ax2.set_title('mAP@0.5 by Object Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference between regular and area-weighted
    ax3 = axes[1, 0]
    if 'ap50_track' in df_summary.columns and 'ap50_track_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_track_Aweighted'].notna()]
        if len(valid_data) > 0:
            difference = valid_data['ap50_track_Aweighted'] - valid_data['ap50_track']
            ax3.plot(valid_data['iteration'], difference, 'o-', 
                    color='cyan', linewidth=2, markersize=6)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Difference (Area-Weighted - Unweighted)')
            ax3.set_title('Difference: AP50 (Track, A-weighted) - AP50 (Track, Unweighted)')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['iteration'], difference, 1)
                p = np.poly1d(z)
                ax3.plot(valid_data['iteration'], p(valid_data['iteration']), "--", alpha=0.8,
                       label=f'Trend: {z[0]:.2e}x + {z[1]:.3f}')
                ax3.legend()
    
    # Plot 4: Ratio between area-weighted and regular
    ax4 = axes[1, 1]
    if 'ap50_track' in df_summary.columns and 'ap50_track_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_track_Aweighted'].notna()]
        if len(valid_data) > 0:
            # Avoid division by zero
            ratio = np.where(valid_data['ap50_track'] > 0, 
                           valid_data['ap50_track_Aweighted'] / valid_data['ap50_track'], 
                           np.nan)
            valid_ratio = ~np.isnan(ratio)
            if np.any(valid_ratio):
                ax4.plot(valid_data['iteration'][valid_ratio], ratio[valid_ratio], 'o-', 
                        color='magenta', linewidth=2, markersize=6)
                ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Ratio (Area-Weighted / Unweighted)')
                ax4.set_title('Performance Ratio: AP50 (Track, A-weighted) / AP50 (Track, Unweighted)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    area_comparison_plot_path = os.path.join(output_dir, 'area_weighted_comparison.png')
    plt.savefig(area_comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    create_area_weighted_summary_table(df_summary, output_dir)
    
    logger.info(f"Area-weighted comparison plots saved to: {area_comparison_plot_path}")

def create_area_weighted_summary_table(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary table for area-weighted metrics.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save the table
    """
    if df_summary.empty:
        return
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    
    # Best performance for each metric
    metrics_to_show = [
        ('ap50_track', 'AP50 (Track, Unweighted)'),
        ('ap50_track_Aweighted', 'AP50 (Track, Area-Weighted)'),
        ('ap50_track_small', 'AP50 (Track, Small Objects)'),
        ('ap50_track_medium', 'AP50 (Track, Medium Objects)'),
        ('ap50_track_large', 'AP50 (Track, Large Objects)')
    ]
    
    for metric, metric_name in metrics_to_show:
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                best_idx = valid_data[metric].idxmax()
                best_value = valid_data.loc[best_idx, metric]
                best_iter = valid_data.loc[best_idx, 'iteration']
                avg_value = valid_data[metric].mean()
                
                table_data.append([
                    metric_name,
                    f"{best_value:.4f} ({best_iter:.0f})",
                    f"{avg_value:.4f}"
                ])
    
    # Create table
    headers = ['Metric', 'Best (Iteration)', 'Average']
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('AP50 (Track) Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_table_path = os.path.join(output_dir, 'area_weighted_summary_table.png')
    plt.savefig(summary_table_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also save as CSV
    summary_csv_path = os.path.join(output_dir, 'area_weighted_summary.csv')
    summary_df = pd.DataFrame(table_data, columns=headers)
    summary_df.to_csv(summary_csv_path, index=False)
    
    logger.info(f"Area-weighted summary table saved to: {summary_table_path}")
    logger.info(f"Area-weighted summary CSV saved to: {summary_csv_path}")

def create_best_checkpoint_analysis(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create analysis of best performing checkpoints.
    Adapts to available metrics - prioritizes COCO metrics when track metrics are unavailable.
    """
    # Check what data is available
    has_track_data = 'ap50_track' in df_summary.columns and df_summary['ap50_track'].notna().any()
    
    # Define metrics to analyze based on available data
    if has_track_data:
        # Full mode: include both track and instance metrics
        metrics_to_analyze = [
            'ap50_track', 'ap75_track', 'mean_iou',
            'ap50_instance_Aweighted', 'ap_instance_Aweighted', 'ap75_instance_Aweighted', 
            'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted'
        ]
        logger.info("Creating best checkpoint analysis with full metrics (track + instance)")
    else:
        # Simplified mode: prioritize COCO instance metrics
        metrics_to_analyze = [
            'ap50_instance_Aweighted', 'ap_instance_Aweighted', 'ap75_instance_Aweighted', 
            'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted'
        ]
        logger.info("Creating best checkpoint analysis with COCO metrics only")
        logger.info("Note: Track-level metrics not available - analysis limited to instance-level COCO metrics")
    
    # Find best checkpoints for different metrics
    best_checkpoints = {}
    
    for metric in metrics_to_analyze:
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                best_idx = valid_data[metric].idxmax()
                best_checkpoints[metric] = {
                    'iteration': valid_data.loc[best_idx, 'iteration'],
                    'value': valid_data.loc[best_idx, metric]
                }
    
    # Create summary table
    if best_checkpoints:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        for metric, info in best_checkpoints.items():
            metric_name = {
                'ap50_track': 'AP50 (Track)',
                'mean_iou': 'Mean IoU',
                'ap75_track': 'AP75 (Track)',
                'ap50_instance_Aweighted': 'AP50 (Instance, A-weighted)',
                'ap_instance_Aweighted': 'AP (Instance, A-weighted)',
                'ap75_instance_Aweighted': 'AP75 (Instance, A-weighted)',
                'aps_instance_Aweighted': 'APs (Instance, A-weighted)',
                'apm_instance_Aweighted': 'APm (Instance, A-weighted)',
                'apl_instance_Aweighted': 'APl (Instance, A-weighted)'
            }.get(metric, metric)
            table_data.append([metric_name, info['iteration'], f"{info['value']:.4f}"])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Best Iteration', 'Best Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Set title based on available metrics
        if has_track_data:
            title = 'Best Performing Checkpoints (Full Analysis)'
        else:
            title = 'Best Performing Checkpoints (COCO Metrics Only)'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        best_checkpoints_path = os.path.join(output_dir, 'best_checkpoints.png')
        plt.savefig(best_checkpoints_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        logger.info("\nBest Performing Checkpoints:")
        logger.info("=" * 50)
        for metric, info in best_checkpoints.items():
            metric_name = {
                'ap50_track': 'AP50 (Track)',
                'mean_iou': 'Mean IoU',
                'ap75_track': 'AP75 (Track)',
                'ap50_instance_Aweighted': 'AP50 (Instance, A-weighted)',
                'ap_instance_Aweighted': 'AP (Instance, A-weighted)',
                'ap75_instance_Aweighted': 'AP75 (Instance, A-weighted)',
                'aps_instance_Aweighted': 'APs (Instance, A-weighted)',
                'apm_instance_Aweighted': 'APm (Instance, A-weighted)',
                'apl_instance_Aweighted': 'APl (Instance, A-weighted)',
                'ap50_track_per_cat': 'AP50 (Track, Per-cat)',  # Track-level metrics (removed)
                'ap50_instance_per_cat': 'AP50 (Instance, Per-cat)'
            }.get(metric, metric)
            logger.info(f"{metric_name}: Iteration {info['iteration']} (Value: {info['value']:.4f})")
        
        logger.info(f"Best checkpoint analysis saved to: {best_checkpoints_path}")
    else:
        logger.warning("No valid metrics found for best checkpoint analysis")
        logger.info("This may happen if:")
        logger.info("  - All checkpoints were processed in fast mode but no COCO metrics are available")
        logger.info("  - No checkpoints have been successfully analyzed yet")
        logger.info("  - All metric values are NaN or missing")

def analyze_overfitting(df_summary: pd.DataFrame) -> dict:
    """
    Analyze signs of overfitting in the training curve.
    Robust to missing data (fast mode compatibility).
    """
    analysis = {
        'overfitting_detected': False,
        'severity': 'none',
        'indicators': [],
        'recommendations': []
    }
    
    if len(df_summary) < 3:
        return analysis
    
    # Choose the best available metric for overfitting analysis
    primary_metric = None
    metric_name = None
    
    # Priority order: mean_iou > ap50_instance_Aweighted > ap_instance_Aweighted
    if 'mean_iou' in df_summary.columns and df_summary['mean_iou'].notna().any():
        primary_metric = 'mean_iou'
        metric_name = 'Mean IoU'
    elif 'ap50_instance_Aweighted' in df_summary.columns and df_summary['ap50_instance_Aweighted'].notna().any():
        primary_metric = 'ap50_instance_Aweighted'
        metric_name = 'AP50 (Instance)'
    elif 'ap_instance_Aweighted' in df_summary.columns and df_summary['ap_instance_Aweighted'].notna().any():
        primary_metric = 'ap_instance_Aweighted'
        metric_name = 'AP (Instance)'
    
    if primary_metric is None:
        analysis['indicators'].append("No suitable metrics available for overfitting analysis")
        return analysis
    
    # Get valid data for the chosen metric
    valid_data = df_summary[df_summary[primary_metric].notna()]
    if len(valid_data) < 3:
        analysis['indicators'].append(f"Insufficient {metric_name} data for overfitting analysis (only {len(valid_data)} valid points)")
        return analysis
    
    # Calculate moving averages to smooth the curves
    window = min(5, len(valid_data) // 3)
    if window > 1:
        valid_data = valid_data.copy()
        valid_data[f'{primary_metric}_smooth'] = valid_data[primary_metric].rolling(window=window, center=True).mean()
    
    # Find peaks and trends
    max_idx = valid_data[primary_metric].idxmax()
    
    # Check for overfitting indicators
    indicators = []
    
    # 1. Performance degradation after peak
    if max_idx < valid_data.index[-1]:  # Peak is not at the end
        final_value = valid_data[primary_metric].iloc[-1]
        peak_value = valid_data.loc[max_idx, primary_metric]
        
        if pd.notna(final_value) and pd.notna(peak_value) and peak_value > 0:
            degradation = peak_value - final_value
            degradation_pct = (degradation / peak_value) * 100
            
            if degradation > 0.1:
                indicators.append(f"Significant {metric_name} degradation: {degradation:.4f} ({degradation_pct:.1f}%)")
                analysis['severity'] = 'severe'
                analysis['overfitting_detected'] = True
            elif degradation > 0.05:
                indicators.append(f"Moderate {metric_name} degradation: {degradation:.4f} ({degradation_pct:.1f}%)")
                analysis['severity'] = 'moderate'
                analysis['overfitting_detected'] = True
            elif degradation > 0.01:
                indicators.append(f"Minimal {metric_name} degradation: {degradation:.4f} ({degradation_pct:.1f}%)")
            else:
                indicators.append(f"No significant {metric_name} degradation detected")
        else:
            indicators.append(f"Unable to assess {metric_name} degradation (invalid values)")
    else:
        indicators.append(f"{metric_name} peak at final checkpoint - no degradation detected")
    
    # 2. High variance in later iterations
    if len(valid_data) > 10:
        early_values = valid_data[primary_metric].iloc[:len(valid_data)//3]
        late_values = valid_data[primary_metric].iloc[-len(valid_data)//3:]
        
        early_std = early_values.std()
        late_std = late_values.std()
        
        if pd.notna(early_std) and pd.notna(late_std) and early_std > 0:
            if late_std > early_std * 1.5:
                indicators.append(f"High variance in later iterations (early: {early_std:.4f}, late: {late_std:.4f})")
                analysis['overfitting_detected'] = True
        else:
            indicators.append("Unable to assess variance (insufficient variation in data)")
    
    # 3. Check for oscillation
    if len(valid_data) > 5:
        recent_values = valid_data[primary_metric].iloc[-5:].dropna()
        if len(recent_values) > 1:
            try:
                oscillations = np.abs(np.diff(recent_values.values))
                avg_oscillation = np.mean(oscillations)
                
                if pd.notna(avg_oscillation) and avg_oscillation > 0.1:
                    indicators.append(f"High oscillation: {avg_oscillation:.4f} (learning rate may be too high)")
                    analysis['overfitting_detected'] = True
            except (TypeError, ValueError):
                indicators.append("Unable to assess oscillation (invalid data)")
    
    analysis['indicators'] = indicators
    return analysis

def analyze_convergence(df_summary: pd.DataFrame) -> dict:
    """
    Analyze model convergence.
    Robust to missing data (fast mode compatibility).
    """
    analysis = {
        'converged': False,
        'convergence_ratio': 0.0,
        'recommendations': []
    }
    
    if len(df_summary) < 2:
        return analysis
    
    # Choose the best available metric for convergence analysis
    primary_metric = None
    metric_name = None
    
    # Priority order: mean_iou > ap50_instance_Aweighted > ap_instance_Aweighted
    if 'mean_iou' in df_summary.columns and df_summary['mean_iou'].notna().any():
        primary_metric = 'mean_iou'
        metric_name = 'Mean IoU'
    elif 'ap50_instance_Aweighted' in df_summary.columns and df_summary['ap50_instance_Aweighted'].notna().any():
        primary_metric = 'ap50_instance_Aweighted'
        metric_name = 'AP50 (Instance)'
    elif 'ap_instance_Aweighted' in df_summary.columns and df_summary['ap_instance_Aweighted'].notna().any():
        primary_metric = 'ap_instance_Aweighted'
        metric_name = 'AP (Instance)'
    
    if primary_metric is None:
        analysis['recommendations'].append("No suitable metrics available for convergence analysis")
        return analysis
    
    # Get valid data for the chosen metric
    valid_data = df_summary[df_summary[primary_metric].notna()]
    if len(valid_data) < 2:
        analysis['recommendations'].append(f"Insufficient {metric_name} data for convergence analysis (only {len(valid_data)} valid points)")
        return analysis
    
    final_performance = valid_data[primary_metric].iloc[-1]
    max_performance = valid_data[primary_metric].max()
    
    if pd.notna(final_performance) and pd.notna(max_performance) and max_performance > 0:
        convergence_ratio = final_performance / max_performance
        analysis['convergence_ratio'] = convergence_ratio
        
        if convergence_ratio < 0.8:
            analysis['recommendations'].append(f"Model may not have converged - consider training longer (final/max {metric_name}: {convergence_ratio:.3f})")
        elif convergence_ratio > 0.95:
            analysis['converged'] = True
            analysis['recommendations'].append(f"Model appears to have converged well (final/max {metric_name}: {convergence_ratio:.3f})")
        else:
            analysis['recommendations'].append(f"Model shows moderate convergence (final/max {metric_name}: {convergence_ratio:.3f})")
    else:
        analysis['recommendations'].append(f"Unable to assess convergence (invalid {metric_name} values)")
    
    return analysis

def analyze_training_dynamics(df_summary: pd.DataFrame) -> dict:
    """
    Analyze training dynamics and stability.
    Robust to missing data (fast mode compatibility).
    """
    analysis = {
        'stability': 'unknown',
        'learning_rate_analysis': {},
        'recommendations': []
    }
    
    if len(df_summary) < 3:
        return analysis
    
    # Choose the best available metric for training dynamics analysis
    primary_metric = None
    metric_name = None
    
    # Priority order: mean_iou > ap50_instance_Aweighted > ap_instance_Aweighted
    if 'mean_iou' in df_summary.columns and df_summary['mean_iou'].notna().any():
        primary_metric = 'mean_iou'
        metric_name = 'Mean IoU'
    elif 'ap50_instance_Aweighted' in df_summary.columns and df_summary['ap50_instance_Aweighted'].notna().any():
        primary_metric = 'ap50_instance_Aweighted'
        metric_name = 'AP50 (Instance)'
    elif 'ap_instance_Aweighted' in df_summary.columns and df_summary['ap_instance_Aweighted'].notna().any():
        primary_metric = 'ap_instance_Aweighted'
        metric_name = 'AP (Instance)'
    
    if primary_metric is None:
        analysis['recommendations'].append("No suitable metrics available for training dynamics analysis")
        return analysis
    
    # Get valid data for the chosen metric
    valid_data = df_summary[df_summary[primary_metric].notna()]
    if len(valid_data) < 3:
        analysis['recommendations'].append(f"Insufficient {metric_name} data for training dynamics analysis (only {len(valid_data)} valid points)")
        return analysis
    
    # Analyze learning rate impact
    third = len(valid_data) // 3
    early_values = valid_data[primary_metric].iloc[:third]
    mid_values = valid_data[primary_metric].iloc[third:2*third]
    late_values = valid_data[primary_metric].iloc[2*third:]
    
    early_performance = early_values.mean() if len(early_values) > 0 else None
    mid_performance = mid_values.mean() if len(mid_values) > 0 else None
    late_performance = late_values.mean() if len(late_values) > 0 else None
    
    analysis['learning_rate_analysis'] = {
        'early': early_performance if pd.notna(early_performance) else 0.0,
        'mid': mid_performance if pd.notna(mid_performance) else 0.0,
        'late': late_performance if pd.notna(late_performance) else 0.0
    }
    
    # Stability analysis
    std_performance = valid_data[primary_metric].std()
    if pd.notna(std_performance):
        if std_performance < 0.05:
            analysis['stability'] = 'stable'
        elif std_performance < 0.1:
            analysis['stability'] = 'moderate'
        else:
            analysis['stability'] = 'unstable'
    else:
        analysis['stability'] = 'unknown'
    
    # Learning rate recommendations
    if (pd.notna(late_performance) and pd.notna(mid_performance) and 
        mid_performance > 0 and late_performance < mid_performance * 0.9):
        analysis['recommendations'].append(f"Consider reducing learning rate (late {metric_name} declined)")
    elif (pd.notna(late_performance) and pd.notna(mid_performance) and 
          mid_performance > 0 and late_performance > mid_performance * 1.1):
        analysis['recommendations'].append(f"Learning rate might be too conservative (late {metric_name} still improving)")
    elif pd.notna(late_performance) and pd.notna(mid_performance):
        analysis['recommendations'].append(f"Training dynamics appear stable ({metric_name})")
    else:
        analysis['recommendations'].append(f"Unable to assess training dynamics (invalid {metric_name} values)")
    
    return analysis

def analyze_model_specific_characteristics(model_dir: str, config_file: str = None) -> dict:
    """
    Analyze model-specific characteristics based on directory name and configuration.
    """
    analysis = {
        'model_type': 'unknown',
        'learning_rate': 'unknown',
        'frame_count': 'unknown',
        'config_analysis': {},
        'recommendations': []
    }
    
    model_name = os.path.basename(model_dir)
    
    # Load config file for detailed analysis
    if config_file:
        config_path = config_file
    else:
        config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract key parameters
            analysis['config_analysis'] = {
                'learning_rate': config.get('SOLVER', {}).get('BASE_LR', 'unknown'),
                'batch_size': config.get('SOLVER', {}).get('IMS_PER_BATCH', 'unknown'),
                'max_iterations': config.get('SOLVER', {}).get('MAX_ITER', 'unknown'),
                'training_resolution': config.get('INPUT', {}).get('MIN_SIZE_TRAIN', 'unknown'),
                'test_resolution': config.get('INPUT', {}).get('MIN_SIZE_TEST', 'unknown'),
                'frame_sampling': config.get('INPUT', {}).get('SAMPLING_FRAME_NUM', 'unknown'),
                'window_size': config.get('MODEL', {}).get('MASK_FORMER', {}).get('TEST', {}).get('WINDOW_SIZE', 'unknown')
            }
            
            # Update analysis based on actual config values
            analysis['learning_rate'] = analysis['config_analysis']['learning_rate']
            analysis['frame_count'] = analysis['config_analysis']['frame_sampling']
            
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Detect model type
    if 'unmasked' in model_name.lower():
        analysis['model_type'] = 'unmasked'
        analysis['recommendations'].extend([
            "Unmasked model detected - consider using masked training data if available",
            "Consider adjusting loss weights for better mask learning",
            "Consider increasing data augmentation"
        ])
    
    # Learning rate recommendations based on actual config
    if analysis['learning_rate'] == 0.0001:
        analysis['recommendations'].extend([
            "Learning rate 0.0001 detected - consider trying 0.00005 for more stable training",
            "Consider using learning rate scheduling"
        ])
    elif analysis['learning_rate'] == 0.00001:
        analysis['recommendations'].append("Learning rate 0.00001 detected - may be too conservative")
    
    # Frame count recommendations
    if analysis['frame_count'] == 15:
        analysis['recommendations'].extend([
            "15-frame model detected - consider testing with different frame numbers (5, 10, 20)",
            "Consider adjusting temporal consistency loss"
        ])
    elif analysis['frame_count'] == 7:
        analysis['recommendations'].extend([
            "7-frame model detected - consider increasing for better temporal consistency",
            "Consider testing with 15 frames for comparison"
        ])
    
    # Resolution recommendations
    if analysis['config_analysis'].get('training_resolution'):
        if isinstance(analysis['config_analysis']['training_resolution'], (list, tuple)):
            min_res = min(analysis['config_analysis']['training_resolution'])
            max_res = max(analysis['config_analysis']['training_resolution'])
            if max_res > 800:
                analysis['recommendations'].append("High resolution training detected - monitor memory usage")
            elif min_res < 400:
                analysis['recommendations'].append("Low resolution training - consider higher resolution for better accuracy")
    
    return analysis

def create_model_performance_report(df_summary: pd.DataFrame, output_dir: str, model_dir: str, df_per_species: pd.DataFrame = None, config_file: str = None) -> None:
    """
    Create comprehensive model performance report.
    """
    report_path = os.path.join(output_dir, 'model_performance_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Metrics Description Section
        f.write("METRICS DESCRIPTION\n")
        f.write("-" * 40 + "\n")
        f.write("This report includes three types of metrics:\n\n")
        
        f.write("1. VIDEO-LEVEL METRICS (Custom Implementation):\n")
        f.write("   - Evaluation Level: Video tracks across multiple frames\n")
        f.write("   - Matching Strategy: Hungarian algorithm for optimal track assignment\n")
        f.write("   - IoU Computation: Average IoU across all frames in a video\n")
        f.write("   - Interpolation: 101-point interpolation for AP calculation\n")
        f.write("   - Confidence Filtering: Uses ALL predictions (no threshold filtering)\n")
        f.write("   - Class Verification: Predictions only match GT of same category\n\n")
        
        f.write("   Metrics:\n")
        f.write("   - Mean IoU: Average Intersection over Union across all frame predictions\n")
        f.write("   - Mean Boundary F-measure: Boundary precision/recall using 2-pixel threshold\n")
        f.write("   - mAP@0.X: Mean Average Precision at IoU threshold X (video-level)\n")
        f.write("   - AP@0.X: Average Precision at IoU threshold X per category (video-level)\n\n")
        
        f.write("2. FRAME-LEVEL COCO METRICS (DVIS-DAQ Evaluation):\n")
        f.write("   - Evaluation Level: Individual frames (standard COCO protocol)\n")
        f.write("   - Matching Strategy: Greedy matching per frame within same category\n")
        f.write("   - IoU Computation: Frame-wise IoU (no temporal averaging)\n")
        f.write("   - Class Verification: ✅ PREDICTIONS ONLY MATCH GT OF SAME CATEGORY\n")
        f.write("   - Interpolation: Standard COCO interpolation method (101-point)\n")
        f.write("   - Confidence Filtering: Uses ALL model predictions (no threshold filtering)\n")
        f.write("   - Implementation: Direct DVIS-DAQ OVISeval with default parameters\n")
        f.write("   - Grouping: Predictions and GT grouped by (video_id, category_id)\n\n")
        
        f.write("   Metrics:\n")
        f.write("   - AP: Average Precision at IoU=0.50:0.95 (primary COCO metric)\n")
        f.write("   - AP50: Average Precision at IoU=0.50\n")
        f.write("   - AP75: Average Precision at IoU=0.75\n")
        f.write("   - APs: AP for small objects (< 32² pixels)\n")
        f.write("   - APm: AP for medium objects (32² - 96² pixels)\n")
        f.write("   - APl: AP for large objects (> 96² pixels)\n")
        f.write("   - AR1: Average Recall with maxDets=1\n")
        f.write("   - AR10: Average Recall with maxDets=10\n")
        f.write("   - Per-Category: All above metrics computed per species\n\n")
        
        f.write("3. TEMPORAL CONSISTENCY METRICS (Video-Specific):\n")
        f.write("   - Evaluation Level: Video tracks with temporal analysis\n")
        f.write("   - Matching Strategy: Hungarian algorithm for track assignment\n")
        f.write("   - Analysis: Temporal consistency across frames\n")
        f.write("   - Confidence Filtering: Uses ALL predictions (no threshold filtering)\n")
        f.write("   - Class Verification: Predictions only match GT of same category\n\n")
        
        f.write("   Custom Metrics:\n")
        f.write("   - Track Completeness: Fraction of frames where both GT and prediction exist\n")
        f.write("   - Temporal IoU Stability: 1 - std(IoU) across frames (higher = more stable)\n")
        f.write("   - Track Fragmentation: Number of gaps in prediction tracks (lower = better)\n")
        f.write("   - Mean Track Length: Average number of frames per track\n\n")
        
        f.write("4. STANDARD TRACKING METRICS (Industry Standard):\n")
        f.write("   - Evaluation Level: Video tracks with standard tracking evaluation\n")
        f.write("   - Matching Strategy: Hungarian algorithm for track assignment\n")
        f.write("   - Analysis: Standard tracking metrics used in tracking papers\n")
        f.write("   - Confidence Filtering: Uses ALL predictions (no threshold filtering)\n")
        f.write("   - Class Verification: Predictions only match GT of same category\n\n")
        
        f.write("   Standard Metrics:\n")
        f.write("   - MOTA (Multiple Object Tracking Accuracy): Overall tracking accuracy\n")
        f.write("     Formula: MOTA = 1 - (FN + FP + IDSW) / GT\n")
        f.write("     Components: False Negatives, False Positives, Identity Switches\n")
        f.write("   - MOTP (Multiple Object Tracking Precision): Average IoU of matched predictions\n")
        f.write("   - IDF1 (ID F1-Score): Identity consistency F1-score\n")
        f.write("     Measures: How well identities are maintained across frames\n")
        f.write("   - HOTA (Higher Order Tracking Accuracy): Comprehensive tracking metric\n")
        f.write("     Combines: Detection accuracy (DetA) and association accuracy (AssA)\n")
        f.write("   - DetA (Detection Accuracy): Component of HOTA measuring detection quality\n")
        f.write("   - AssA (Association Accuracy): Component of HOTA measuring association quality\n\n")
        
        f.write("NOTE: Video-level metrics provide more sophisticated temporal analysis,\n")
        f.write("while frame-level COCO metrics enable standard comparison with other methods.\n")
        f.write("Temporal consistency metrics specifically analyze video-specific behaviors.\n\n")
        
        f.write("EVALUATION METHOD COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write("Video-Level vs Frame-Level Evaluation:\n\n")
        
        f.write("Video-Level (Custom):\n")
        f.write("  ✓ Better for understanding temporal performance\n")
        f.write("  ✓ Considers track-level consistency\n")
        f.write("  ✓ More appropriate for video instance segmentation\n")
        f.write("  ✗ Not directly comparable to other papers\n")
        f.write("  ✗ More complex implementation\n\n")
        
        f.write("Frame-Level (DVIS-DAQ COCO):\n")
        f.write("  ✓ Standard evaluation protocol (exact DVIS-DAQ implementation)\n")
        f.write("  ✓ Direct comparison with other methods\n")
        f.write("  ✓ Simpler interpretation\n")
        f.write("  ✓ Proper class verification (useCats=1)\n")
        f.write("  ✓ Uses all predictions (no artificial filtering)\n")
        f.write("  ✗ Ignores temporal relationships\n")
        f.write("  ✗ May not reflect real video performance\n\n")
        
        f.write("PERFORMANCE DIAGNOSTICS:\n")
        f.write("-" * 40 + "\n")
        f.write("Understanding AP vs mAP Discrepancies:\n\n")
        f.write("High AP50 (Frame-level) + Low mAP@0.5 (Video-level) = Tracking Problem:\n")
        f.write("  ✓ Model segments objects well in individual frames\n")
        f.write("  ✗ Model struggles with consistent tracking across frames\n")
        f.write("  → Focus on temporal consistency and track continuity\n\n")
        
        f.write("Low AP50 (Frame-level) + High mAP@0.5 (Video-level) = Segmentation Problem:\n")
        f.write("  ✗ Model segments objects poorly in individual frames\n")
        f.write("  ✓ Model tracks consistently but with poor segmentation\n")
        f.write("  → Focus on mask quality and frame-level accuracy\n\n")
        
        f.write("Both Low = Fundamental Model Issues:\n")
        f.write("  ✗ Model struggles with both segmentation and tracking\n")
        f.write("  → Consider architecture changes, more training data, or different approach\n\n")
        
        f.write("Both High = Good Performance:\n")
        f.write("  ✓ Model performs well on both tasks\n")
        f.write("  → Consider fine-tuning for specific use case\n\n")
        
        f.write("TRACKING IMPROVEMENT STRATEGIES:\n")
        f.write("-" * 40 + "\n")
        f.write("When AP50 > mAP@0.5 (tracking problem detected):\n\n")
        f.write("1. Temporal Consistency Loss:\n")
        f.write("   - Increase temporal consistency loss weight\n")
        f.write("   - Add temporal smoothness regularization\n")
        f.write("   - Use longer video sequences during training\n\n")
        
        f.write("2. Frame Sampling:\n")
        f.write("   - Increase SAMPLING_FRAME_NUM (try 20-30 frames)\n")
        f.write("   - Reduce SAMPLING_FRAME_STRIDE (use every frame)\n")
        f.write("   - Use temporal data augmentation\n\n")
        
        f.write("3. Model Architecture:\n")
        f.write("   - Increase temporal attention layers\n")
        f.write("   - Add temporal memory mechanisms\n")
        f.write("   - Use 3D convolutions for temporal modeling\n\n")
        
        f.write("4. Training Strategy:\n")
        f.write("   - Use curriculum learning (start with short sequences)\n")
        f.write("   - Add temporal consistency validation\n")
        f.write("   - Use temporal data augmentation\n\n")
        
        f.write("SEGMENTATION IMPROVEMENT STRATEGIES:\n")
        f.write("-" * 40 + "\n")
        f.write("When mAP@0.5 > AP50 (segmentation problem detected):\n\n")
        f.write("1. Mask Quality:\n")
        f.write("   - Increase mask loss weight\n")
        f.write("   - Add boundary loss\n")
        f.write("   - Use higher resolution training\n\n")
        
        f.write("2. Data Quality:\n")
        f.write("   - Improve annotation quality\n")
        f.write("   - Add more diverse training data\n")
        f.write("   - Use data augmentation for masks\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("- Use DVIS-DAQ COCO metrics for paper reporting and method comparison\n")
        f.write("- Use video-level metrics for model understanding and optimization\n")
        f.write("- Use temporal consistency metrics for video-specific analysis\n")
        f.write("- All metrics now use ALL predictions (no confidence thresholding)\n")
        f.write("- DVIS-DAQ evaluation ensures proper class verification\n")
        f.write("- Monitor AP vs mAP ratio to diagnose tracking vs segmentation issues\n\n")
        
        f.write("TECHNICAL DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write("IoU Calculation:\n")
        f.write("  IoU = intersection_area / union_area\n")
        f.write("  - Video-level: Average IoU across all frames in a track\n")
        f.write("  - Frame-level: IoU computed independently per frame\n\n")
        
        f.write("AP Calculation:\n")
        f.write("  - Sort predictions by confidence score\n")
        f.write("  - Match to ground truth using IoU threshold\n")
        f.write("  - Compute precision-recall curve\n")
        f.write("  - Interpolate AP using specified method\n\n")
        
        f.write("Matching Strategies:\n")
        f.write("  - Hungarian Algorithm: Optimal assignment minimizing total cost (video-level)\n")
        f.write("  - Greedy Matching: Match each prediction to best available GT (frame-level COCO)\n")
        f.write("  - Cost Function: 1 - IoU (for Hungarian) or IoU threshold (for greedy)\n")
        f.write("  - Class Verification: ✅ COCO metrics ONLY match predictions to GT of same category\n")
        f.write("  - DVIS-DAQ Implementation: useCats=1, groups by (video_id, category_id)\n\n")
        
        f.write("Area Categories:\n")
        f.write("  - Small: < 32² = 1,024 pixels\n")
        f.write("  - Medium: 32² to 96² = 1,024 to 9,216 pixels\n")
        f.write("  - Large: > 96² = 9,216 pixels\n\n")
        
        f.write("Temporal Metrics:\n")
        f.write("  - Track Completeness: valid_frames / total_frames\n")
        f.write("  - IoU Stability: 1 - std(IoU_values)\n")
        f.write("  - Fragmentation: gaps / (total_frames - 1)\n")
        f.write("  - Track Length: average frames per track\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total checkpoints analyzed: {len(df_summary)}\n")
        f.write(f"Iteration range: {df_summary['iteration'].min():.0f} - {df_summary['iteration'].max():.0f}\n")
        f.write(f"Training duration: {df_summary['iteration'].max() - df_summary['iteration'].min():.0f} iterations\n\n")
        
        # Best performance
        f.write("BEST PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        
        # Video and COCO metrics
        for metric in ['mean_iou', 'ap50_track', 'ap75_track', 'ap50_instance_Aweighted', 'ap_instance_Aweighted', 'ap75_instance_Aweighted', 'aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted', 'ap50_track_per_cat', 'ap50_instance_per_cat']:  # Note: track-level metrics removed
            if metric in df_summary.columns:
                valid_data = df_summary[df_summary[metric].notna()]
                if len(valid_data) > 0:
                    best_idx = valid_data[metric].idxmax()
                    best_iter = valid_data.loc[best_idx, 'iteration']
                    best_value = valid_data.loc[best_idx, metric]
                    metric_name = {
                        'mean_iou': 'Mean IoU',
                        'ap50_track': 'AP50 (Track)',
                        'ap75_track': 'AP75 (Track)',
                        'ap50_instance_Aweighted': 'AP50 (Instance, A-weighted)',
                        'ap_instance_Aweighted': 'AP (Instance, A-weighted)',
                        'ap75_instance_Aweighted': 'AP75 (Instance, A-weighted)',
                        'aps_instance_Aweighted': 'APs (Instance, A-weighted)',
                        'apm_instance_Aweighted': 'APm (Instance, A-weighted)',
                        'apl_instance_Aweighted': 'APl (Instance, A-weighted)'
                    }.get(metric, metric)
                    f.write(f"{metric_name}: {best_value:.4f} at iteration {best_iter:.0f}\n")
                else:
                    metric_name = {
                        'mean_iou': 'Mean IoU',
                        'ap50_track': 'AP50 (Track)',
                        'ap75_track': 'AP75 (Track)',
                        'ap50_instance_Aweighted': 'AP50 (Instance, A-weighted)',
                        'ap_instance_Aweighted': 'AP (Instance, A-weighted)',
                        'ap75_instance_Aweighted': 'AP75 (Instance, A-weighted)',
                        'aps_instance_Aweighted': 'APs (Instance, A-weighted)',
                        'apm_instance_Aweighted': 'APm (Instance, A-weighted)',
                        'apl_instance_Aweighted': 'APl (Instance, A-weighted)'
                    }.get(metric, metric)
                    f.write(f"{metric_name}: No valid data available\n")
        
        # Tracking metrics
        f.write("\nTracking Metrics:\n")
        for metric in ['MOTA', 'IDF1', 'HOTA', 'track_completeness', 'temporal_iou_stability']:
            if metric in df_summary.columns:
                valid_data = df_summary[df_summary[metric].notna()]
                if len(valid_data) > 0:
                    best_idx = valid_data[metric].idxmax()
                    best_iter = valid_data.loc[best_idx, 'iteration']
                    best_value = valid_data.loc[best_idx, metric]
                    metric_name = {
                        'MOTA': 'MOTA',
                        'IDF1': 'IDF1',
                        'HOTA': 'HOTA',
                        'track_completeness': 'Track Completeness',
                        'temporal_iou_stability': 'Temporal IoU Stability'
                    }.get(metric, metric)
                    f.write(f"{metric_name}: {best_value:.4f} at iteration {best_iter:.0f}\n")
                else:
                    metric_name = {
                        'MOTA': 'MOTA',
                        'IDF1': 'IDF1',
                        'HOTA': 'HOTA',
                        'track_completeness': 'Track Completeness',
                        'temporal_iou_stability': 'Temporal IoU Stability'
                    }.get(metric, metric)
                    f.write(f"{metric_name}: No valid data available\n")
        
        # Track fragmentation (lower is better)
        if 'track_fragmentation' in df_summary.columns:
            valid_data = df_summary[df_summary['track_fragmentation'].notna()]
            if len(valid_data) > 0:
                best_idx = valid_data['track_fragmentation'].idxmin()
                best_iter = valid_data.loc[best_idx, 'iteration']
                best_value = valid_data.loc[best_idx, 'track_fragmentation']
                f.write(f"Track Fragmentation (Best): {best_value:.4f} at iteration {best_iter:.0f}\n")
            else:
                f.write(f"Track Fragmentation (Best): No valid data available\n")
        
        f.write("\n")
        
        # Per-species analysis
        if df_per_species is not None and not df_per_species.empty:
            f.write("PER-SPECIES ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Get unique species
            species_list = df_per_species['category_name'].unique()
            f.write(f"Total species analyzed: {len(species_list)}\n")
            f.write(f"Species: {', '.join(species_list)}\n\n")
            
            # Best performance per species
            f.write("Best Performance per Species:\n")
            key_metrics = ['ap_50', 'ap50_instance_per_cat', 'iou', 'ap_instance_per_cat', 'MOTA', 'IDF1', 'HOTA']
            metric_names = ['AP@0.5 (Video)', 'AP50 (Instance, Per-cat)', 'IoU', 'AP (Instance, Per-cat)', 'MOTA', 'IDF1', 'HOTA']
            
            for species in species_list:
                species_data = df_per_species[df_per_species['category_name'] == species]
                if len(species_data) == 0:
                    continue
                
                f.write(f"\n{species}:\n")
                for metric, metric_name in zip(key_metrics, metric_names):
                    if metric in species_data.columns:
                        valid_data = species_data[species_data[metric].notna()]
                        if len(valid_data) > 0:
                            best_idx = valid_data[metric].idxmax()
                            best_value = valid_data.loc[best_idx, metric]
                            best_iter = valid_data.loc[best_idx, 'iteration']
                            f.write(f"  {metric_name}: {best_value:.4f} at iteration {best_iter:.0f}\n")
            
            # Species performance ranking
            f.write("\nSpecies Performance Ranking (by AP@0.5 Video):\n")
            species_ranking = []
            for species in species_list:
                species_data = df_per_species[df_per_species['category_name'] == species]
                if len(species_data) > 0 and 'ap_50' in species_data.columns:
                    valid_data = species_data[species_data['ap_50'].notna()]
                    if len(valid_data) > 0:
                        best_ap50 = valid_data['ap_50'].max()
                        species_ranking.append((species, best_ap50))
            
            # Sort by performance
            species_ranking.sort(key=lambda x: x[1], reverse=True)
            for i, (species, ap50) in enumerate(species_ranking, 1):
                f.write(f"  {i}. {species}: {ap50:.4f}\n")
            
            # Performance gaps analysis
            if len(species_ranking) > 1:
                best_performance = species_ranking[0][1]
                worst_performance = species_ranking[-1][1]
                performance_gap = best_performance - worst_performance
                f.write(f"\nPerformance Analysis:\n")
                f.write(f"  Best performing species: {species_ranking[0][0]} ({best_performance:.4f})\n")
                f.write(f"  Worst performing species: {species_ranking[-1][0]} ({worst_performance:.4f})\n")
                f.write(f"  Performance gap: {performance_gap:.4f}\n")
                
                if performance_gap > 0.3:
                    f.write(f"  ⚠️  Large performance gap detected - consider species-specific training\n")
                elif performance_gap > 0.1:
                    f.write(f"  💡 Moderate performance gap - some species may need attention\n")
                else:
                    f.write(f"  ✅ Good balance across species\n")
            
            f.write("\n")
        
        # Overfitting analysis
        overfitting_analysis = analyze_overfitting(df_summary)
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if overfitting_analysis['overfitting_detected']:
            f.write(f"⚠️  OVERFITTING DETECTED (Severity: {overfitting_analysis['severity']})\n")
        else:
            f.write("✅ No significant overfitting detected\n")
        
        for indicator in overfitting_analysis['indicators']:
            f.write(f"• {indicator}\n")
        f.write("\n")
        
        # Convergence analysis
        convergence_analysis = analyze_convergence(df_summary)
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Convergence ratio: {convergence_analysis['convergence_ratio']:.2f}\n")
        if convergence_analysis['converged']:
            f.write("✅ Model appears to have converged well\n")
        else:
            f.write("⚠️  Model may not have converged\n")
        
        for rec in convergence_analysis['recommendations']:
            f.write(f"• {rec}\n")
        f.write("\n")
        
        # Training dynamics
        dynamics_analysis = analyze_training_dynamics(df_summary)
        f.write("TRAINING DYNAMICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Stability: {dynamics_analysis['stability']}\n")
        
        # Handle case where learning_rate_analysis might be empty
        if 'learning_rate_analysis' in dynamics_analysis and dynamics_analysis['learning_rate_analysis']:
            lr_analysis = dynamics_analysis['learning_rate_analysis']
            f.write(f"Early performance: {lr_analysis.get('early', 0.0):.4f}\n")
            f.write(f"Mid performance: {lr_analysis.get('mid', 0.0):.4f}\n")
            f.write(f"Late performance: {lr_analysis.get('late', 0.0):.4f}\n")
        else:
            f.write("Early performance: N/A (insufficient data)\n")
            f.write("Mid performance: N/A (insufficient data)\n")
            f.write("Late performance: N/A (insufficient data)\n")
        
        for rec in dynamics_analysis['recommendations']:
            f.write(f"• {rec}\n")
        f.write("\n")
        
        # Model-specific analysis
        model_analysis = analyze_model_specific_characteristics(model_dir, config_file)
        f.write("MODEL-SPECIFIC ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model type: {model_analysis['model_type']}\n")
        f.write(f"Learning rate: {model_analysis['learning_rate']}\n")
        f.write(f"Frame count: {model_analysis['frame_count']}\n")
        
        for rec in model_analysis['recommendations']:
            f.write(f"• {rec}\n")
        f.write("\n")
        
        # Training duration recommendations
        f.write("TRAINING DURATION RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        total_iterations = df_summary['iteration'].max()
        if total_iterations < 5000:
            f.write("💡 Consider training for more iterations\n")
        elif total_iterations > 15000:
            f.write("💡 Training may be excessive - consider early stopping\n")
        else:
            f.write("✅ Training duration appears appropriate\n")
        f.write("\n")
        
        # Summary and recommendations
        f.write("SUMMARY AND RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        all_recommendations = []
        all_recommendations.extend(overfitting_analysis['recommendations'])
        all_recommendations.extend(convergence_analysis['recommendations'])
        all_recommendations.extend(dynamics_analysis['recommendations'])
        all_recommendations.extend(model_analysis['recommendations'])
        
        for i, rec in enumerate(all_recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("QUICK REFERENCE GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("For Paper Reporting: Focus on DVIS-DAQ COCO metrics (AP, AP50, AP75)\n")
        f.write("For Model Analysis: Consider video-level metrics and temporal consistency\n")
        f.write("For Hyperparameter Tuning: Monitor both COCO and video metrics\n")
        f.write("For Debugging: Use temporal consistency metrics to identify issues\n\n")
        
        f.write("Key Metrics to Watch:\n")
        f.write("- AP50 (DVIS-DAQ COCO): Primary metric for comparison with other methods\n")
        f.write("- AP (DVIS-DAQ COCO): Overall performance metric (IoU=0.50:0.95)\n")
        f.write("- mAP@0.5 (Video): Best indicator of video-level performance\n")
        f.write("- Track Completeness: Indicates if model maintains tracks over time\n")
        f.write("- Temporal IoU Stability: Shows consistency of predictions\n\n")
        
        f.write("Performance Diagnosis:\n")
        f.write("- AP50 vs mAP@0.5 ratio: >1.0 = tracking problem, <1.0 = segmentation problem\n")
        f.write("- Temporal metrics: Track completeness <0.5 = tracking issues\n")
        f.write("- IoU stability: <0.7 = temporal consistency problems\n\n")
        
        f.write("Important Notes:\n")
        f.write("- DVIS-DAQ evaluation uses ALL predictions (no confidence filtering)\n")
        f.write("- Class verification is enforced (predictions only match same category)\n")
        f.write("- Results are identical to DVIS-DAQ built-in evaluation\n")
        f.write("- All metrics are now consistent in their prediction usage\n\n")
        
        f.write("Report generated by analyze_checkpoint_results.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Model performance report saved to: {report_path}")

def create_comprehensive_comparison_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive comparison plots showing all metric types and their relationships.
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to plot")
        return
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Metric Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Track vs Instance AP50 (Dataset-level)
    ax1 = axes[0, 0]
    if 'ap50_track' in df_summary.columns and 'ap50_instance_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data['ap50_track'], 'o-', 
                    color='blue', linewidth=2, markersize=6, label='AP50 (Track)')
            ax1.plot(valid_data['iteration'], valid_data['ap50_instance_Aweighted'], 's-', 
                    color='red', linewidth=2, markersize=6, label='AP50 (Instance, A-weighted)')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('AP50')
            ax1.set_title('AP50: Track vs Instance (Dataset-level)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Track vs Instance AP50 (Per-category)
    ax2 = axes[0, 1]
    # Note: Track-level metrics removed, only instance-level available
    if 'ap50_instance_per_cat' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_instance_per_cat'].notna()]
        if len(valid_data) > 0:
            ax2.plot(valid_data['iteration'], valid_data['ap50_instance_per_cat'], 's-', 
                    color='orange', linewidth=2, markersize=6, label='AP50 (Instance, Per-cat)')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('AP50')
            ax2.set_title('AP50: Track vs Instance (Per-category)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Area-weighted vs Unweighted (Track-level)
    ax3 = axes[0, 2]
    if 'ap50_track' in df_summary.columns and 'ap50_track_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_track_Aweighted'].notna()]
        if len(valid_data) > 0:
            ax3.plot(valid_data['iteration'], valid_data['ap50_track'], 'o-', 
                    color='purple', linewidth=2, markersize=6, label='AP50 (Track, Unweighted)')
            ax3.plot(valid_data['iteration'], valid_data['ap50_track_Aweighted'], 's-', 
                    color='brown', linewidth=2, markersize=6, label='AP50 (Track, A-weighted)')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('AP50')
            ax3.set_title('AP50: Unweighted vs Area-weighted (Track)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Area breakdown (Track-level)
    ax4 = axes[1, 0]
    area_metrics = ['ap50_track_small', 'ap50_track_medium', 'ap50_track_large']
    colors = ['cyan', 'magenta', 'yellow']
    labels = ['Small', 'Medium', 'Large']
    
    for metric, color, label in zip(area_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax4.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=f'Track {label}')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('AP50')
    ax4.set_title('AP50 by Object Size (Track-level)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Area breakdown (Instance-level)
    ax5 = axes[1, 1]
    area_metrics = ['aps_instance_Aweighted', 'apm_instance_Aweighted', 'apl_instance_Aweighted']
    colors = ['cyan', 'magenta', 'yellow']
    labels = ['Small', 'Medium', 'Large']
    
    for metric, color, label in zip(area_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax5.plot(valid_data['iteration'], valid_data[metric], 's-', 
                        color=color, linewidth=2, markersize=6, label=f'Instance {label}')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('AP')
    ax5.set_title('AP by Object Size (Instance-level)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: IoU thresholds comparison (Track-level)
    ax6 = axes[1, 2]
    iou_metrics = ['ap10_track', 'ap25_track', 'ap50_track', 'ap75_track', 'ap95_track']
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    labels = ['AP@0.1', 'AP@0.25', 'AP@0.5', 'AP@0.75', 'AP@0.95']
    
    for metric, color, label in zip(iou_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax6.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('AP')
    ax6.set_title('AP by IoU Threshold (Track-level)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Tracking metrics comparison
    ax7 = axes[2, 0]
    tracking_metrics = ['MOTA', 'IDF1', 'HOTA']
    colors = ['blue', 'green', 'red']
    labels = ['MOTA', 'IDF1', 'HOTA']
    
    for metric, color, label in zip(tracking_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax7.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Score')
    ax7.set_title('Standard Tracking Metrics')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Temporal consistency metrics
    ax8 = axes[2, 1]
    temporal_metrics = ['track_completeness', 'temporal_iou_stability']
    colors = ['blue', 'green']
    labels = ['Track Completeness', 'Temporal IoU Stability']
    
    for metric, color, label in zip(temporal_metrics, colors, labels):
        if metric in df_summary.columns:
            valid_data = df_summary[df_summary[metric].notna()]
            if len(valid_data) > 0:
                ax8.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                        color=color, linewidth=2, markersize=6, label=label)
    
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Score')
    ax8.set_title('Temporal Consistency Metrics')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Performance ratio analysis
    ax9 = axes[2, 2]
    if 'ap50_track' in df_summary.columns and 'ap50_instance_Aweighted' in df_summary.columns:
        valid_data = df_summary[df_summary['ap50_track'].notna() & df_summary['ap50_instance_Aweighted'].notna()]
        if len(valid_data) > 0:
            # Calculate performance ratio
            ratio = valid_data['ap50_track'] / valid_data['ap50_instance_Aweighted']
            ratio = ratio.replace([np.inf, -np.inf], np.nan)  # Handle division by zero
            
            ax9.plot(valid_data['iteration'], ratio, 'o-', 
                    color='purple', linewidth=2, markersize=6)
            ax9.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
            ax9.set_xlabel('Iteration')
            ax9.set_ylabel('Ratio (Track/Instance)')
            ax9.set_title('Performance Ratio: Track vs Instance AP50')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            # Add interpretation
            mean_ratio = ratio.mean()
            if mean_ratio > 1.1:
                ax9.text(0.05, 0.95, 'Track > Instance\n(Tracking advantage)', 
                        transform=ax9.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            elif mean_ratio < 0.9:
                ax9.text(0.05, 0.95, 'Instance > Track\n(Segmentation advantage)', 
                        transform=ax9.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            else:
                ax9.text(0.05, 0.95, 'Balanced Performance', 
                        transform=ax9.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Comprehensive comparison plots saved to: {comparison_plot_path}")

def create_per_category_comparison_plots(df_per_species: pd.DataFrame, output_dir: str) -> None:
    """
    Create per-category comparison plots showing instance performance for each species.
    Only shows plots when relevant data is available.
    
    Args:
        df_per_species: DataFrame with per-species metrics
        output_dir: Directory to save plots
    """
    if df_per_species.empty:
        logger.warning("No per-species data to plot")
        return
    
    # Get unique species
    species_list = df_per_species['category_name'].unique()
    if len(species_list) == 0:
        logger.warning("No species found in data")
        return
    
    # Check what data is available
    has_track_data = 'ap_50' in df_per_species.columns and df_per_species['ap_50'].notna().any()
    has_instance_data = 'ap50_instance_per_cat' in df_per_species.columns and df_per_species['ap50_instance_per_cat'].notna().any()
    has_area_weighted_data = 'ap50_instance_Aweighted' in df_per_species.columns and df_per_species['ap50_instance_Aweighted'].notna().any()
    
    if not has_instance_data:
        logger.info("No per-category instance data available - skipping per-category comparison plots")
        return
    
    # Create color palette for species
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_list)))
    color_dict = dict(zip(species_list, colors))
    
    # Determine layout based on available data
    if has_track_data and has_area_weighted_data:
        # All data available - 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        title = 'Per-Category Track vs Instance Performance Comparison'
    elif has_track_data or has_area_weighted_data:
        # Some data available - 2x1 layout
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        ax1, ax2 = axes[0], axes[1]
        ax3, ax4 = None, None
        title = 'Per-Category Instance Performance Comparison'
    else:
        # Only basic instance data - 1x1 layout
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax2, ax3, ax4 = None, None, None
        title = 'Per-Category Instance Performance Comparison'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Instance AP50 by species (always show if instance data available)
    for species in species_list:
        species_data = df_per_species[df_per_species['category_name'] == species]
        if len(species_data) > 0:
            # Instance-level AP50
            if 'ap50_instance_per_cat' in species_data.columns:
                valid_data = species_data[species_data['ap50_instance_per_cat'].notna()]
                if len(valid_data) > 0:
                    ax1.plot(valid_data['iteration'], valid_data['ap50_instance_per_cat'], 'o-', 
                           color=color_dict[species], linewidth=2, markersize=4, 
                           label=species, alpha=0.8)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('AP50')
    ax1.set_title('AP50 Instance Performance by Species')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area-weighted instance performance (only if available)
    if ax2 is not None and has_area_weighted_data:
        for species in species_list:
            species_data = df_per_species[df_per_species['category_name'] == species]
            if len(species_data) > 0:
                # Instance-level area-weighted
                if 'ap50_instance_Aweighted' in species_data.columns:
                    valid_data = species_data[species_data['ap50_instance_Aweighted'].notna()]
                    if len(valid_data) > 0:
                        ax2.plot(valid_data['iteration'], valid_data['ap50_instance_Aweighted'], 's--', 
                               color=color_dict[species], linewidth=2, markersize=4, 
                               label=species, alpha=0.8)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('AP50')
        ax2.set_title('AP50 Instance Performance (Area-weighted) by Species')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best performance summary by species (only instance data)
    if ax3 is not None:
        species_summary = []
        for species in species_list:
            species_data = df_per_species[df_per_species['category_name'] == species]
            if len(species_data) > 0:
                instance_best = 0
                
                if 'ap50_instance_per_cat' in species_data.columns:
                    valid_data = species_data[species_data['ap50_instance_per_cat'].notna()]
                    if len(valid_data) > 0:
                        instance_best = valid_data['ap50_instance_per_cat'].max()
                
                if instance_best > 0:
                    species_summary.append((species, instance_best))
        
        if species_summary:
            species_names = [s[0] for s in species_summary]
            instance_scores = [s[1] for s in species_summary]
            
            x = np.arange(len(species_names))
            width = 0.5
            
            ax3.bar(x, instance_scores, width, label='Instance AP50 (Best)', alpha=0.8, color='steelblue')
            
            ax3.set_xlabel('Species')
            ax3.set_ylabel('AP50')
            ax3.set_title('Best Instance AP50 Performance by Species')
            ax3.set_xticks(x)
            ax3.set_xticklabels(species_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    per_category_plot_path = os.path.join(output_dir, 'per_category_comparison.png')
    plt.savefig(per_category_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    create_per_category_summary_table(df_per_species, output_dir)
    
    logger.info(f"Per-category comparison plots saved to: {per_category_plot_path}")

def plot_ap_per_category_from_csv(csv_path: str, output_path: str = "AP_per_category.png") -> None:
    """
    Reads mask_metrics_category.csv and plots the AP@0.1, AP@0.25, AP@0.5, AP@0.75, AP@0.95 for each category in the dataset.
    The x-axis is the category, the y-axis is AP, and each AP level is a different colored point.
    
    Args:
        csv_path: Path to the mask_metrics_category.csv file
        output_path: Path to save the plot
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Use the new column names that match the renamed metrics
        # Look for instance-level AP columns (new format)
        ap_columns = ["ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat"]
        
        # Check which columns actually exist in the CSV
        available_columns = [col for col in ap_columns if col in df.columns]
        if not available_columns:
            # Fallback to old track-level column names if new ones don't exist
            ap_columns = ["ap10_track_per_cat", "ap25_track_per_cat", "ap50_track_per_cat", "ap75_track_per_cat", "ap95_track_per_cat"]
            available_columns = [col for col in ap_columns if col in df.columns]
        if not available_columns:
            # Fallback to even older column names
            ap_columns = ["AP@0.1", "AP@0.25", "AP@0.5", "AP@0.75", "AP@0.95"]
            available_columns = [col for col in ap_columns if col in df.columns]
        
        if not available_columns:
            logger.warning(f"No AP columns found in {csv_path}. Available columns: {list(df.columns)}")
            return
        
        if df.empty:
            logger.warning(f"No per-category data found in {csv_path}")
            return
        
        # Use category_name for grouping
        grouped = df.groupby('category_name')[available_columns].first().reset_index()

        plt.figure(figsize=(12, 6))
        markers = ['o', 's', '^', 'D', 'v']
        colors = ['tab:purple', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        
        for i, ap_col in enumerate(available_columns):
            plt.scatter(grouped['category_name'], grouped[ap_col], 
                       label=ap_col, marker=markers[i], color=colors[i], s=80)
        
        plt.xlabel('Category')
        plt.ylabel('Average Precision (AP)')
        plt.title('AP per Category at Different IoU Thresholds')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.legend(title='IoU Threshold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"AP per category plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating AP per category plot: {e}")

def create_per_category_summary_table(df_per_species: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary table showing AP50_instance metrics per category.
    Shows two columns:
    1. Best per-category AP50_instance across all checkpoints (each species can have different best checkpoint)
    2. Per-category AP50_instance values for the best overall checkpoint (same checkpoint for all species)
    
    Args:
        df_per_species: DataFrame with per-species metrics
        output_dir: Directory to save the table
    """
    if df_per_species.empty:
        return
    
    # Check if we have the required metric
    if 'ap50_instance_per_cat' not in df_per_species.columns:
        logger.info("No per-category AP50_instance metrics available - skipping per-category summary table")
        return
    
    # Get unique species
    species_list = df_per_species['category_name'].unique()
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, len(species_list) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    
    # Find the best overall checkpoint (highest average AP50_instance_per_cat across all species)
    overall_best_checkpoint = None
    best_overall_score = -1
    
    for species in species_list:
        species_data = df_per_species[df_per_species['category_name'] == species]
        if len(species_data) == 0:
            continue
        
        valid_data = species_data[species_data['ap50_instance_per_cat'].notna()]
        if len(valid_data) > 0:
            # Find best checkpoint for this species
            best_idx = valid_data['ap50_instance_per_cat'].idxmax()
            best_value = valid_data.loc[best_idx, 'ap50_instance_per_cat']
            best_iter = valid_data.loc[best_idx, 'iteration']
            
            # Check if this checkpoint gives the best overall score
            checkpoint_data = df_per_species[df_per_species['iteration'] == best_iter]
            checkpoint_scores = checkpoint_data[checkpoint_data['ap50_instance_per_cat'].notna()]['ap50_instance_per_cat']
            if len(checkpoint_scores) > 0:
                avg_score = checkpoint_scores.mean()
                if avg_score > best_overall_score:
                    best_overall_score = avg_score
                    overall_best_checkpoint = best_iter
    
    # If no overall best checkpoint found, use the checkpoint with highest individual score
    if overall_best_checkpoint is None:
        all_valid_data = df_per_species[df_per_species['ap50_instance_per_cat'].notna()]
        if len(all_valid_data) > 0:
            best_idx = all_valid_data['ap50_instance_per_cat'].idxmax()
            overall_best_checkpoint = all_valid_data.loc[best_idx, 'iteration']
    
    # Build table data
    for species in species_list:
        species_data = df_per_species[df_per_species['category_name'] == species]
        if len(species_data) == 0:
            continue
        
        row_data = [species]
        
        # Column 1: Best per-category AP50_instance across all checkpoints
        valid_data = species_data[species_data['ap50_instance_per_cat'].notna()]
        if len(valid_data) > 0:
            best_idx = valid_data['ap50_instance_per_cat'].idxmax()
            best_value = valid_data.loc[best_idx, 'ap50_instance_per_cat']
            best_iter = valid_data.loc[best_idx, 'iteration']
            row_data.append(f"{best_value:.3f} ({best_iter:.0f})")
        else:
            row_data.append("N/A")
        
        # Column 2: Per-category AP50_instance for best overall checkpoint
        if overall_best_checkpoint is not None:
            checkpoint_data = species_data[species_data['iteration'] == overall_best_checkpoint]
            if len(checkpoint_data) > 0 and checkpoint_data['ap50_instance_per_cat'].notna().any():
                checkpoint_value = checkpoint_data['ap50_instance_per_cat'].iloc[0]
                row_data.append(f"{checkpoint_value:.3f} ({overall_best_checkpoint:.0f})")
            else:
                row_data.append("N/A")
        else:
            row_data.append("N/A")
        
        table_data.append(row_data)
    
    if not table_data:
        logger.info("No per-category data available for summary table")
        plt.close(fig)
        return
    
    # Create table
    headers = ['Species', 'Best Per-Category AP50', 'AP50 at Best Overall Checkpoint']
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center')
    
    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Per-Category AP50 Instance Metrics Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save table
    per_category_table_path = os.path.join(output_dir, 'per_category_summary_table.png')
    plt.savefig(per_category_table_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save CSV
    per_category_csv_path = os.path.join(output_dir, 'per_category_summary.csv')
    df_table = pd.DataFrame(table_data, columns=headers)
    df_table.to_csv(per_category_csv_path, index=False)
    
    logger.info(f"Per-category summary table saved to: {per_category_table_path}")
    logger.info(f"Per-category summary CSV saved to: {per_category_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive checkpoint analysis')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing checkpoint evaluations')
    parser.add_argument('--evaluation-dir', type=str, default=None,
                       help='Specific evaluation directory (for multi-run structure)')
    parser.add_argument('--val-json', type=str, default=None,
                       help='Path to validation JSON file (defaults to val.json in model directory)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file (defaults to model_dir/config.yaml)')
    parser.add_argument('--run-mask-metrics', action='store_true',
                       help='Force run mask metrics analysis for all checkpoints (overwrites existing)')
    parser.add_argument('--skip-mask-metrics', action='store_true',
                       help='Skip mask metrics analysis and only create plots from existing data')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Confidence threshold for filtering predictions (default: 0.0 = no threshold)')
    parser.add_argument('--analysis-level', choices=['basic', 'comprehensive'], default='comprehensive',
                       help='Level of analysis to perform')
    parser.add_argument('--parallel', action='store_true',
                       help='Run mask metrics analysis in parallel')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: CPU count - 2)')
    
    args = parser.parse_args()
    
    # Set default val.json to model directory if not provided
    # Handle val.json path
    if args.val_json is None:
        # No val.json specified, use automatic detection based on config
        default_val_json = os.path.join(args.model_dir, 'val.json')
        actual_val_json = detect_stride_and_get_val_json(args.model_dir, default_val_json, args.config_file)
    else:
        # User explicitly specified val.json, use it directly
        actual_val_json = args.val_json
        logger.info(f"Using user-specified val.json: {actual_val_json}")
    
    # Find all checkpoint result directories
    checkpoint_dirs = find_checkpoint_results(args.model_dir, args.evaluation_dir)
    logger.info(f"Found {len(checkpoint_dirs)} checkpoint result directories")
    
    if not checkpoint_dirs:
        logger.error("No checkpoint result directories found!")
        return
    
    # Check if mask metrics need to be computed
    need_mask_metrics = False
    if args.run_mask_metrics:
        # Force recomputation
        need_mask_metrics = True
        logger.info("Force running mask metrics analysis for all checkpoints...")
    elif not args.skip_mask_metrics:
        # Check if any checkpoints are missing mask metrics
        missing_metrics = []
        for checkpoint_dir in checkpoint_dirs:
            csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
            if not os.path.exists(csv_path):
                missing_metrics.append(checkpoint_dir)
        
        if missing_metrics:
            need_mask_metrics = True
            logger.info(f"Found {len(missing_metrics)} checkpoints missing mask metrics. Computing automatically...")
        else:
            logger.info("All checkpoints already have mask metrics. Skipping computation.")
    
    # Run mask metrics analysis if needed
    if need_mask_metrics:
        if args.parallel:
            # Use parallel processing
            results = run_mask_metrics_analysis_parallel(
                checkpoint_dirs, 
                actual_val_json, 
                args.confidence_threshold, 
                args.max_workers, 
                args.run_mask_metrics
            )
            successful_analyses = sum(1 for success in results.values() if success)
        else:
            # Use sequential processing
            successful_analyses = 0
            
            for i, checkpoint_dir in enumerate(checkpoint_dirs):
                logger.info(f"Processing {i+1}/{len(checkpoint_dirs)}: {os.path.basename(checkpoint_dir)}")
                
                success = run_mask_metrics_analysis(checkpoint_dir, actual_val_json, args.confidence_threshold, args.run_mask_metrics)
                if success:
                    successful_analyses += 1
        
        logger.info(f"Successfully analyzed {successful_analyses}/{len(checkpoint_dirs)} checkpoints")
    
    # Determine output directory for plots
    if args.evaluation_dir:
        output_dir = args.evaluation_dir
    else:
        output_dir = args.model_dir
    
    # Collect summary data
    logger.info("Collecting summary data...")
    df_summary = collect_summary_data(args.model_dir, args.evaluation_dir)
    
    if df_summary.empty:
        logger.error("No summary data found. Make sure mask metrics analysis has been run.")
        return
    
    # Collect per-species data
    logger.info("Collecting per-species data...")
    df_per_species = collect_per_species_data(args.model_dir, args.evaluation_dir)
    
    # Create analysis based on level
    if args.analysis_level == 'comprehensive':
        logger.info("Creating comprehensive analysis...")
        create_comprehensive_analysis(df_summary, output_dir, df_per_species, args.model_dir, args.evaluation_dir, args.config_file)
    else:
        logger.info("Creating basic summary...")
        create_basic_summary_plot(df_summary, output_dir)
    
    logger.info("Analysis complete! Check the generated plots and CSV files.")

if __name__ == "__main__":
    main() 
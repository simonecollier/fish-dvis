#!/usr/bin/env python3
"""
Bootstrap evaluation script for checkpoint evaluations.

This script performs bootstrap sampling (with replacement) on videos from the validation set
to estimate confidence intervals for AP, AP50, and AP75 metrics.
"""

import json
import argparse
import random
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import contextlib
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from dvis_daq_eval import compute_dvis_daq_metrics


def load_json(json_path):
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data, json_path):
    """Save data to JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f)


def extract_fold_number(model_dir):
    """Extract fold number from model directory name (e.g., 'model_silhouette_fold3' -> '3')."""
    model_dir_name = Path(model_dir).name
    if 'fold' in model_dir_name.lower():
        parts = model_dir_name.split('_')
        for part in parts:
            if part.lower().startswith('fold'):
                fold_num = part[4:]  # Remove 'fold' prefix
                return fold_num
    return None


def extract_data_type(model_dir):
    """Extract data type from model directory name (e.g., 'model_silhouette_fold3' -> 'silhouette')."""
    model_dir_name = Path(model_dir).name
    parts = model_dir_name.split('_')
    for part in parts:
        if part.lower() in ['camera', 'silhouette']:
            return part.lower()
    return None


def merge_val_jsons(val_jsons):
    """Merge multiple validation JSON files into one, ensuring unique video IDs."""
    if not val_jsons:
        return None
    
    merged = {
        'videos': [],
        'annotations': [],
        'categories': val_jsons[0].get('categories', []),
        'info': val_jsons[0].get('info', {}),
        'licenses': val_jsons[0].get('licenses', [])
    }
    
    seen_video_ids = set()
    
    for val_json in val_jsons:
        for video in val_json.get('videos', []):
            if video['id'] not in seen_video_ids:
                merged['videos'].append(video)
                seen_video_ids.add(video['id'])
        
        for ann in val_json.get('annotations', []):
            if ann.get('video_id') in seen_video_ids:
                merged['annotations'].append(ann)
    
    return merged


def merge_predictions(predictions_list):
    """Merge multiple prediction lists, keeping all predictions."""
    merged = []
    for preds in predictions_list:
        merged.extend(preds)
    return merged


def read_actual_metrics(checkpoint_eval_dir):
    """
    Read actual model metrics from mask_metrics_dataset.csv.
    
    Args:
        checkpoint_eval_dir: Path to checkpoint evaluation directory
    
    Returns:
        Dictionary with 'AP', 'AP50', 'AP75' values or None if file not found
    """
    csv_path = checkpoint_eval_dir / "inference" / "mask_metrics_dataset.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        metrics = {}
        for metric_name in ['ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted']:
            row = df[df['metric_name'] == metric_name]
            if not row.empty:
                value = row.iloc[0]['value']
                # Map to our metric names
                if metric_name == 'ap_instance_Aweighted':
                    metrics['AP'] = value
                elif metric_name == 'ap50_instance_Aweighted':
                    metrics['AP50'] = value
                elif metric_name == 'ap75_instance_Aweighted':
                    metrics['AP75'] = value
        return metrics if len(metrics) == 3 else None
    except Exception as e:
        print(f"Warning: Could not read metrics from {csv_path}: {e}")
        return None


def validate_checkpoint_directories(checkpoint_eval_dirs, require_plotting_files=False):
    """
    Validate that all required files exist in checkpoint directories before processing.
    
    Args:
        checkpoint_eval_dirs: List of checkpoint evaluation directory paths
        require_plotting_files: If True, also validate that CSV files exist for plotting
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    for checkpoint_eval_dir in checkpoint_eval_dirs:
        checkpoint_eval_dir = Path(checkpoint_eval_dir)
        fold_num = None
        
        # Check if directory exists
        if not checkpoint_eval_dir.exists():
            errors.append(f"Checkpoint directory does not exist: {checkpoint_eval_dir}")
            continue
        
        # Find model directory and extract fold number
        model_dir = checkpoint_eval_dir.parent.parent
        fold_num = extract_fold_number(model_dir)
        
        if not fold_num:
            errors.append(f"Could not extract fold number from model directory: {model_dir}")
            continue
        
        # Check results.json
        results_json_path = checkpoint_eval_dir / "inference" / "results.json"
        if not results_json_path.exists():
            errors.append(f"Fold {fold_num}: results.json not found at {results_json_path}")
        else:
            # Try to load and validate it's a list
            try:
                with open(results_json_path, 'r') as f:
                    results = json.load(f)
                if not isinstance(results, list):
                    errors.append(f"Fold {fold_num}: results.json is not a list")
            except Exception as e:
                errors.append(f"Fold {fold_num}: Could not read/parse results.json: {e}")
        
        # Check val JSON
        val_json_path = find_val_json(model_dir, fold_num)
        if not val_json_path.exists():
            errors.append(f"Fold {fold_num}: Validation JSON not found at {val_json_path}")
        else:
            # Try to load and validate structure
            try:
                with open(val_json_path, 'r') as f:
                    val_data = json.load(f)
                if 'videos' not in val_data:
                    errors.append(f"Fold {fold_num}: Validation JSON missing 'videos' key")
                if 'annotations' not in val_data:
                    errors.append(f"Fold {fold_num}: Validation JSON missing 'annotations' key")
            except Exception as e:
                errors.append(f"Fold {fold_num}: Could not read/parse validation JSON: {e}")
        
        # Check CSV file for plotting (if required)
        if require_plotting_files:
            csv_path = checkpoint_eval_dir / "inference" / "mask_metrics_dataset.csv"
            if not csv_path.exists():
                warnings.append(f"Fold {fold_num}: mask_metrics_dataset.csv not found at {csv_path} (plotting will be skipped)")
            else:
                # Try to read and validate required metrics exist
                try:
                    df = pd.read_csv(csv_path)
                    required_metrics = ['ap_instance_Aweighted', 'ap50_instance_Aweighted', 'ap75_instance_Aweighted']
                    missing_metrics = []
                    for metric_name in required_metrics:
                        if df[df['metric_name'] == metric_name].empty:
                            missing_metrics.append(metric_name)
                    if missing_metrics:
                        warnings.append(f"Fold {fold_num}: Missing metrics in CSV: {missing_metrics}")
                except Exception as e:
                    warnings.append(f"Fold {fold_num}: Could not read/parse mask_metrics_dataset.csv: {e}")
    
    # Check if we have exactly 6 folds when multiple directories provided
    if len(checkpoint_eval_dirs) > 1:
        fold_numbers = []
        for checkpoint_eval_dir in checkpoint_eval_dirs:
            model_dir = Path(checkpoint_eval_dir).parent.parent
            fold_num = extract_fold_number(model_dir)
            if fold_num:
                try:
                    fold_numbers.append(int(fold_num))
                except ValueError:
                    pass
        
        if len(fold_numbers) != len(set(fold_numbers)):
            errors.append("Duplicate fold numbers detected")
        
        if len(checkpoint_eval_dirs) == 6 and len(set(fold_numbers)) != 6:
            errors.append(f"Expected 6 unique folds, but found {len(set(fold_numbers))} unique fold numbers: {sorted(set(fold_numbers))}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings


def plot_bootstrap_results(all_fold_results, combined_results, output_path, checkpoint_eval_dirs=None, actual_metrics=None):
    """
    Create a scatter plot with error bars showing bootstrap results for all folds and pooled.
    
    Args:
        all_fold_results: Dictionary mapping fold numbers to bootstrap results
        combined_results: Dictionary with combined bootstrap results
        output_path: Path to save the plot
        checkpoint_eval_dirs: Optional list of checkpoint evaluation directories to read actual metrics
        actual_metrics: Optional dictionary mapping fold numbers to actual metrics (if None, will try to read from checkpoint_eval_dirs)
    """
    # Prepare data
    fold_nums = sorted([int(k) for k in all_fold_results.keys()])
    metrics = ['AP', 'AP50', 'AP75']
    colors = {'AP': '#1f77b4', 'AP50': '#ff7f0e', 'AP75': '#2ca02c'}  # Blue, Orange, Green
    metric_labels = {'AP': 'AP', 'AP50': 'AP50', 'AP75': 'AP75'}
    
    # Read actual metrics if not provided
    if actual_metrics is None and checkpoint_eval_dirs is not None:
        actual_metrics = {}
        for checkpoint_eval_dir in checkpoint_eval_dirs:
            # Find fold number
            model_dir = checkpoint_eval_dir.parent.parent
            fold_num_str = extract_fold_number(model_dir)
            if fold_num_str:
                try:
                    fold_num = int(fold_num_str)
                    metrics_dict = read_actual_metrics(checkpoint_eval_dir)
                    if metrics_dict:
                        actual_metrics[fold_num] = metrics_dict
                except ValueError:
                    pass
        
        # Calculate pooled average
        if actual_metrics:
            pooled_metrics = {}
            for metric in metrics:
                values = [actual_metrics[fn][metric] for fn in fold_nums if fn in actual_metrics and metric in actual_metrics[fn]]
                if values:
                    pooled_metrics[metric] = np.mean(values)
            if pooled_metrics:
                actual_metrics['Pooled'] = pooled_metrics
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # X positions: one for each fold + pooled
    x_positions = list(range(len(fold_nums) + 1))
    x_labels = [f'Fold {fn}' for fn in fold_nums] + ['Pooled']
    
    # Offset for each metric to avoid overlapping error bars
    offsets = {'AP': -0.15, 'AP50': 0.0, 'AP75': 0.15}
    
    # Marker shapes for actual model outputs (different for each fold)
    marker_shapes = {
        1: 's',  # square
        2: '^',  # triangle up
        3: 'D',  # diamond
        4: 'v',  # triangle down
        5: '<',  # triangle left
        6: '>',  # triangle right
        'Pooled': '*'  # star
    }
    
    # Plot each metric
    for metric in metrics:
        means = []
        lower_bounds = []
        upper_bounds = []
        
        # Get values for each fold
        for fold_num in fold_nums:
            fold_results = all_fold_results[str(fold_num)]
            metric_data = fold_results[metric]
            means.append(metric_data['mean'])
            lower_bounds.append(metric_data['mean'] - metric_data['q025'])
            upper_bounds.append(metric_data['q975'] - metric_data['mean'])
        
        # Get values for pooled
        means.append(combined_results[metric]['mean'])
        lower_bounds.append(combined_results[metric]['mean'] - combined_results[metric]['q025'])
        upper_bounds.append(combined_results[metric]['q975'] - combined_results[metric]['mean'])
        
        # Adjust x positions with offset
        x_plot = [x + offsets[metric] for x in x_positions]
        
        # Plot scatter points with error bars
        ax.errorbar(
            x_plot, means,
            yerr=[lower_bounds, upper_bounds],
            fmt='o',
            color=colors[metric],
            label=metric_labels[metric],
            capsize=5,
            capthick=2,
            markersize=8,
            elinewidth=2,
            alpha=0.8
        )
    
    # Plot actual model output points with different shapes
    if actual_metrics:
        for metric in metrics:
            # Plot actual values for each fold
            for fold_num in fold_nums:
                if fold_num in actual_metrics and metric in actual_metrics[fold_num]:
                    x_pos = fold_nums.index(fold_num) + offsets[metric]
                    value = actual_metrics[fold_num][metric]
                    marker = marker_shapes.get(fold_num, 'X')
                    
                    ax.scatter(
                        x_pos, value,
                        marker=marker,
                        color=colors[metric],
                        s=150,  # Larger size for actual values
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.9,
                        zorder=5  # Plot on top
                    )
            
            # Plot actual value for pooled
            if 'Pooled' in actual_metrics and metric in actual_metrics['Pooled']:
                x_pos = len(fold_nums) + offsets[metric]
                value = actual_metrics['Pooled'][metric]
                marker = marker_shapes.get('Pooled', '*')
                
                ax.scatter(
                    x_pos, value,
                    marker=marker,
                    color=colors[metric],
                    s=150,  # Larger size for actual values
                    edgecolors='black',
                    linewidths=1.5,
                    alpha=0.9,
                    zorder=5  # Plot on top
                )
    
    # Set labels and title
    ax.set_xlabel('Fold / Pooled', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Bootstrap Evaluation Results: AP, AP50, AP75 across Folds and Pooled', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bootstrap plot saved to: {output_path}")
    plt.close()


def find_val_json(model_dir, fold_num):
    """Find the validation JSON file in the model directory."""
    model_dir = Path(model_dir)
    
    # Try val_foldX.json first
    if fold_num:
        val_json_path = model_dir / f"val_fold{fold_num}.json"
        if val_json_path.exists():
            return val_json_path
    
    # Try other common names
    possible_names = ["val.json", "validation.json"]
    for name in possible_names:
        val_json_path = model_dir / name
        if val_json_path.exists():
            return val_json_path
    
    # Search for any val*.json file
    val_files = list(model_dir.glob("val*.json"))
    if val_files:
        return val_files[0]
    
    raise FileNotFoundError(f"Could not find validation JSON file in {model_dir}")


def get_video_ids_from_val_json(val_json):
    """Extract all video IDs from the validation JSON."""
    video_ids = [video['id'] for video in val_json.get('videos', [])]
    return video_ids


def filter_predictions_by_video_ids(predictions, video_ids):
    """Filter predictions to only include those for the specified video IDs."""
    video_ids_set = set(video_ids)
    return [pred for pred in predictions if pred.get('video_id') in video_ids_set]


def filter_val_json_by_video_ids(val_json, video_ids):
    """Filter validation JSON to only include specified video IDs and their annotations."""
    video_ids_set = set(video_ids)
    
    # Filter videos
    filtered_videos = [v for v in val_json.get('videos', []) if v['id'] in video_ids_set]
    
    # Filter annotations
    filtered_annotations = [
        ann for ann in val_json.get('annotations', [])
        if ann.get('video_id') in video_ids_set
    ]
    
    # Create filtered JSON structure
    filtered_json = {
        'videos': filtered_videos,
        'annotations': filtered_annotations,
        'categories': val_json.get('categories', []),
        'info': val_json.get('info', {}),
        'licenses': val_json.get('licenses', [])
    }
    
    return filtered_json


def bootstrap_sample_video_ids(video_ids, n_samples=None, seed=None):
    """Create a bootstrap sample of video IDs (with replacement)."""
    if n_samples is None:
        n_samples = len(video_ids)
    if seed is not None:
        rng = random.Random(seed)
        return rng.choices(video_ids, k=n_samples)
    return random.choices(video_ids, k=n_samples)


def process_single_bootstrap(args_tuple):
    """
    Process a single bootstrap sample. This function is designed to be called in parallel.
    
    Args:
        args_tuple: Tuple of (i, video_ids, n_videos, predictions, val_json, temp_dir_str, fold_num, seed)
    
    Returns:
        Tuple of (i, ap, ap50, ap75, error_message)
    """
    i, video_ids, n_videos, predictions, val_json, temp_dir_str, fold_num, seed = args_tuple
    
    try:
        # Convert temp_dir_str back to Path
        temp_dir = Path(temp_dir_str)
        
        # Set seed for this bootstrap sample (seed + i to ensure different samples)
        rng = random.Random(seed + i)
        np.random.seed(seed + i)
        
        # Sample video IDs with replacement
        sampled_video_ids = bootstrap_sample_video_ids(video_ids, n_samples=n_videos, seed=seed + i)
        
        # Filter predictions and validation JSON
        filtered_predictions = filter_predictions_by_video_ids(predictions, sampled_video_ids)
        filtered_val_json = filter_val_json_by_video_ids(val_json, sampled_video_ids)
        
        # Save filtered validation JSON to temporary file
        if fold_num == "Pooled":
            temp_val_json_path = temp_dir / f"val_bootstrap_combined_{i}.json"
            unique_dataset_name = f"ytvis_fishway_val_bootstrap_combined_{i}"
        else:
            temp_val_json_path = temp_dir / f"val_bootstrap_{i}.json"
            unique_dataset_name = f"ytvis_fishway_val_bootstrap_{i}"
        
        save_json(filtered_val_json, temp_val_json_path)
        
        # Compute metrics with unique dataset name for each bootstrap sample
        # Note: We need to import compute_dvis_daq_metrics here for multiprocessing
        # Use skip_folder_rename=True since folder is managed at script level
        from dvis_daq_eval import compute_dvis_daq_metrics as compute_metrics
        
        # Suppress verbose output from compute_dvis_daq_metrics
        with suppress_stdout_stderr():
            metrics = compute_metrics(
                filtered_predictions,
                str(temp_val_json_path),
                dataset_name=unique_dataset_name,
                skip_folder_rename=True  # Folder is already renamed at script level
            )
        
        if metrics:
            return (i, metrics.get('AP', np.nan), metrics.get('AP50', np.nan), 
                    metrics.get('AP75', np.nan), None)
        else:
            return (i, np.nan, np.nan, np.nan, "Metrics returned None")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return (i, np.nan, np.nan, np.nan, f"{str(e)}\n{error_trace}")


def setup_pycocotools_folder():
    """
    Rename pycocotools folder at the start of bootstrap evaluation.
    This is done once at script level to avoid race conditions in parallel processing.
    
    Returns:
        bool: True if folder was already renamed, False if we renamed it
    """
    dvis_daq_path = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_Plus/data_video/datasets'
    pycocotools_path = os.path.join(dvis_daq_path, 'pycocotools')
    pycocotools_backup_path = os.path.join(dvis_daq_path, 'pycocotools_backup')
    
    # Check if pycocotools folder exists
    if not os.path.exists(pycocotools_path):
        raise FileNotFoundError(f"pycocotools folder not found at {pycocotools_path}")
    
    # Check if already renamed (backup exists)
    if os.path.exists(pycocotools_backup_path):
        print("Note: pycocotools folder appears to already be renamed. Will restore it at the end.")
        return True  # Already renamed
    
    # Rename pycocotools to pycocotools_backup
    print("Renaming pycocotools folder for bootstrap evaluation...")
    shutil.move(pycocotools_path, pycocotools_backup_path)
    print("✓ pycocotools folder renamed successfully")
    return False  # We renamed it


def restore_pycocotools_folder(was_already_renamed=False):
    """
    Restore pycocotools folder at the end of bootstrap evaluation.
    
    Args:
        was_already_renamed: If True, don't restore (it was already renamed before we started)
    """
    if was_already_renamed:
        print("Note: pycocotools folder was already renamed before script start. Not restoring.")
        return
    
    dvis_daq_path = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_Plus/data_video/datasets'
    pycocotools_path = os.path.join(dvis_daq_path, 'pycocotools')
    pycocotools_backup_path = os.path.join(dvis_daq_path, 'pycocotools_backup')
    
    try:
        if os.path.exists(pycocotools_backup_path):
            if os.path.exists(pycocotools_path):
                shutil.rmtree(pycocotools_path)
            shutil.move(pycocotools_backup_path, pycocotools_path)
            print("✓ pycocotools folder restored successfully")
        else:
            print("Warning: pycocotools_backup folder not found, cannot restore")
    except Exception as e:
        print(f"Warning: Failed to restore pycocotools folder: {e}")


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_bootstrap_evaluation(
    checkpoint_eval_dir,
    n_bootstrap_samples=500,
    seed=42,
    fold_num=None,
    show_progress=True,
    n_jobs=1
):
    """
    Run bootstrap evaluation on a checkpoint evaluation directory.
    
    Args:
        checkpoint_eval_dir: Path to checkpoint evaluation directory
        n_bootstrap_samples: Number of bootstrap samples to generate
        seed: Random seed for reproducibility
        fold_num: Optional fold number to display in progress (if None, will be extracted)
        show_progress: Whether to show progress messages
    
    Returns:
        Dictionary with bootstrap statistics
    """
    random.seed(seed)
    np.random.seed(seed)
    
    checkpoint_eval_dir = Path(checkpoint_eval_dir)
    
    # Find results.json in inference folder
    results_json_path = checkpoint_eval_dir / "inference" / "results.json"
    if not results_json_path.exists():
        raise FileNotFoundError(f"Results JSON not found at {results_json_path}")
    
    # Find model directory (parent of checkpoint_evaluations)
    model_dir = checkpoint_eval_dir.parent.parent
    
    # Extract fold number and find val JSON
    if fold_num is None:
        fold_num = extract_fold_number(model_dir)
    val_json_path = find_val_json(model_dir, fold_num)
    
    if show_progress:
        print(f"Loading results from: {results_json_path}")
        print(f"Loading validation data from: {val_json_path}")
        print(f"Fold number: {fold_num}")
    
    # Load data
    predictions = load_json(results_json_path)
    val_json = load_json(val_json_path)
    
    # Get all video IDs from validation set
    video_ids = get_video_ids_from_val_json(val_json)
    n_videos = len(video_ids)
    
    if show_progress:
        print(f"Found {n_videos} videos in validation set")
        print(f"Running {n_bootstrap_samples} bootstrap samples...")
    
    # Store bootstrap results (initialize with NaN to maintain order)
    bootstrap_aps = [np.nan] * n_bootstrap_samples
    bootstrap_ap50s = [np.nan] * n_bootstrap_samples
    bootstrap_ap75s = [np.nan] * n_bootstrap_samples
    
    # Create temporary directory for filtered JSON files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        if n_jobs > 1:
            # Parallel processing
            if show_progress:
                print(f"Using {n_jobs} parallel workers...")
            
            # Prepare arguments for parallel processing
            # Convert temp_dir to string for pickling
            bootstrap_args = [
                (i, video_ids, n_videos, predictions, val_json, str(temp_dir), fold_num, seed)
                for i in range(n_bootstrap_samples)
            ]
            
            completed = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(process_single_bootstrap, args): args[0]
                    for args in bootstrap_args
                }
                
                # Process completed tasks
                for future in as_completed(future_to_index):
                    i, ap, ap50, ap75, error_msg = future.result()
                    bootstrap_aps[i] = ap
                    bootstrap_ap50s[i] = ap50
                    bootstrap_ap75s[i] = ap75
                    
                    completed += 1
                    if show_progress:
                        print(f"Fold {fold_num}, Bootstrap {completed}/{n_bootstrap_samples}", end='\r')
                    
                    if error_msg and show_progress:
                        print(f"\nWarning: Error computing metrics for Fold {fold_num}, bootstrap sample {i+1}: {error_msg[:200]}")  # Truncate long errors
        else:
            # Sequential processing (original code)
            for i in range(n_bootstrap_samples):
                if show_progress:
                    print(f"Fold {fold_num}, Bootstrap {i+1}/{n_bootstrap_samples}", end='\r')
                
                # Sample video IDs with replacement
                sampled_video_ids = bootstrap_sample_video_ids(video_ids, n_samples=n_videos)
                
                # Filter predictions and validation JSON
                filtered_predictions = filter_predictions_by_video_ids(predictions, sampled_video_ids)
                filtered_val_json = filter_val_json_by_video_ids(val_json, sampled_video_ids)
                
                # Save filtered validation JSON to temporary file
                temp_val_json_path = temp_dir / f"val_bootstrap_{i}.json"
                save_json(filtered_val_json, temp_val_json_path)
                
                # Compute metrics with unique dataset name for each bootstrap sample
                try:
                    unique_dataset_name = f"ytvis_fishway_val_bootstrap_{i}"
                    # Suppress verbose output from compute_dvis_daq_metrics
                    # Use skip_folder_rename=True since folder is managed at script level
                    with suppress_stdout_stderr():
                        metrics = compute_dvis_daq_metrics(
                            filtered_predictions,
                            str(temp_val_json_path),
                            dataset_name=unique_dataset_name,
                            skip_folder_rename=True
                        )
                    
                    if metrics:
                        bootstrap_aps[i] = metrics.get('AP', np.nan)
                        bootstrap_ap50s[i] = metrics.get('AP50', np.nan)
                        bootstrap_ap75s[i] = metrics.get('AP75', np.nan)
                    else:
                        bootstrap_aps[i] = np.nan
                        bootstrap_ap50s[i] = np.nan
                        bootstrap_ap75s[i] = np.nan
                except Exception as e:
                    if show_progress:
                        print(f"\nWarning: Error computing metrics for Fold {fold_num}, bootstrap sample {i+1}: {e}")
                    bootstrap_aps[i] = np.nan
                    bootstrap_ap50s[i] = np.nan
                    bootstrap_ap75s[i] = np.nan
    
    if show_progress:
        print(f"\nFold {fold_num} completed: {n_bootstrap_samples} bootstrap samples processed")
    
    # Compute statistics
    bootstrap_aps = np.array(bootstrap_aps)
    bootstrap_ap50s = np.array(bootstrap_ap50s)
    bootstrap_ap75s = np.array(bootstrap_ap75s)
    
    # Remove NaN values for statistics
    valid_aps = bootstrap_aps[~np.isnan(bootstrap_aps)]
    valid_ap50s = bootstrap_ap50s[~np.isnan(bootstrap_ap50s)]
    valid_ap75s = bootstrap_ap75s[~np.isnan(bootstrap_ap75s)]
    
    results = {
        'n_bootstrap_samples': n_bootstrap_samples,
        'n_videos': n_videos,
        'n_valid_samples': {
            'AP': len(valid_aps),
            'AP50': len(valid_ap50s),
            'AP75': len(valid_ap75s)
        },
        'AP': {
            'mean': float(np.mean(valid_aps)) if len(valid_aps) > 0 else np.nan,
            'std': float(np.std(valid_aps, ddof=1)) if len(valid_aps) > 0 else np.nan,
            'median': float(np.median(valid_aps)) if len(valid_aps) > 0 else np.nan,
            'q025': float(np.percentile(valid_aps, 2.5)) if len(valid_aps) > 0 else np.nan,
            'q975': float(np.percentile(valid_aps, 97.5)) if len(valid_aps) > 0 else np.nan,
            'all_values': bootstrap_aps.tolist()
        },
        'AP50': {
            'mean': float(np.mean(valid_ap50s)) if len(valid_ap50s) > 0 else np.nan,
            'std': float(np.std(valid_ap50s, ddof=1)) if len(valid_ap50s) > 0 else np.nan,
            'median': float(np.median(valid_ap50s)) if len(valid_ap50s) > 0 else np.nan,
            'q025': float(np.percentile(valid_ap50s, 2.5)) if len(valid_ap50s) > 0 else np.nan,
            'q975': float(np.percentile(valid_ap50s, 97.5)) if len(valid_ap50s) > 0 else np.nan,
            'all_values': bootstrap_ap50s.tolist()
        },
        'AP75': {
            'mean': float(np.mean(valid_ap75s)) if len(valid_ap75s) > 0 else np.nan,
            'std': float(np.std(valid_ap75s, ddof=1)) if len(valid_ap75s) > 0 else np.nan,
            'median': float(np.median(valid_ap75s)) if len(valid_ap75s) > 0 else np.nan,
            'q025': float(np.percentile(valid_ap75s, 2.5)) if len(valid_ap75s) > 0 else np.nan,
            'q975': float(np.percentile(valid_ap75s, 97.5)) if len(valid_ap75s) > 0 else np.nan,
            'all_values': bootstrap_ap75s.tolist()
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation for checkpoint evaluations"
    )
    parser.add_argument(
        "checkpoint_eval_dirs",
        type=str,
        nargs='+',
        help="Path(s) to checkpoint evaluation directory(ies) (e.g., .../checkpoint_evaluations/checkpoint_0003433). Can provide 1-6 directories."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of bootstrap samples (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (only used for single directory mode, default: bootstrap_results.json in checkpoint_eval_dir)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel workers for bootstrap processing (default: 1, use -1 for all available CPUs)"
    )
    
    args = parser.parse_args()
    
    # Determine number of jobs
    if args.n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = max(1, args.n_jobs)
    
    checkpoint_eval_dirs = [Path(d) for d in args.checkpoint_eval_dirs]
    
    # Setup pycocotools folder renaming (once at script level to avoid race conditions)
    print("Setting up pycocotools folder for evaluation...")
    was_already_renamed = setup_pycocotools_folder()
    print()
    
    try:
        # Validate all checkpoint directories before processing
        print("Validating checkpoint directories and required files...")
        print("="*80)
        
        # Require plotting files only if we have 6 folds (for combined bootstrap)
        require_plotting = len(checkpoint_eval_dirs) == 6
        is_valid, errors, warnings = validate_checkpoint_directories(
            checkpoint_eval_dirs, 
            require_plotting_files=require_plotting
        )
        
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
            print()
        
        if errors:
            print("Errors found - cannot proceed:")
            for error in errors:
                print(f"  ✗ {error}")
            print("\nPlease fix the above errors before running bootstrap evaluation.")
            sys.exit(1)
        
        if warnings and not require_plotting:
            print("Note: Some plotting files are missing, but bootstrap evaluation will proceed.")
            print("      Plotting will be skipped if CSV files are not available.")
            print()
        
        print("✓ All required files validated successfully!")
        print("="*80)
        print()
        
        # If multiple directories provided, process all folds
        if len(checkpoint_eval_dirs) > 1:
            if len(checkpoint_eval_dirs) != 6:
                print(f"Warning: Expected 6 folds, but got {len(checkpoint_eval_dirs)}. Will still process all provided folds.")
            
            # Extract data type from first model directory
            first_model_dir = checkpoint_eval_dirs[0].parent.parent
            data_type = extract_data_type(first_model_dir)
            
            if not data_type:
                print("Warning: Could not extract data type from model directory. Using 'unknown'.")
                data_type = 'unknown'
            
            print(f"Processing {len(checkpoint_eval_dirs)} folds for data type: {data_type}")
            print("="*80)
            
            # Process each fold individually
            all_fold_results = {}
            all_val_jsons = []
            all_predictions = []
            
            for checkpoint_eval_dir in checkpoint_eval_dirs:
                # Find model directory
                model_dir = checkpoint_eval_dir.parent.parent
                fold_num = extract_fold_number(model_dir)
                
                print(f"\nProcessing Fold {fold_num}...")
                print("-" * 80)
                
                # Run bootstrap for this fold
                fold_results = run_bootstrap_evaluation(
                    checkpoint_eval_dir,
                    n_bootstrap_samples=args.n_samples,
                    seed=args.seed,
                    fold_num=fold_num,
                    show_progress=True,
                    n_jobs=n_jobs
                )
                
                # Save in original checkpoint directory
                fold_output_path = checkpoint_eval_dir / "bootstrap_results.json"
                save_json(fold_results, fold_output_path)
                print(f"Fold {fold_num} bootstrap results saved to: {fold_output_path}")
                
                all_fold_results[fold_num] = fold_results
                
                # Load data for combined bootstrap
                results_json_path = checkpoint_eval_dir / "inference" / "results.json"
                val_json_path = find_val_json(model_dir, fold_num)
                
                predictions = load_json(results_json_path)
                val_json = load_json(val_json_path)
                
                all_predictions.append(predictions)
                all_val_jsons.append(val_json)
            
            # Run combined bootstrap if we have all 6 folds
            if len(checkpoint_eval_dirs) == 6:
                print("\n" + "="*80)
                print("RUNNING COMBINED BOOTSTRAP (All 6 Folds)")
                print("="*80)
                
                # Set seed for combined bootstrap
                random.seed(args.seed)
                np.random.seed(args.seed)
                
                # Merge all validation sets
                merged_val_json = merge_val_jsons(all_val_jsons)
                merged_predictions = merge_predictions(all_predictions)
                
                # Get all video IDs from merged validation set
                all_video_ids = get_video_ids_from_val_json(merged_val_json)
                n_all_videos = len(all_video_ids)
                
                print(f"Combined validation set: {n_all_videos} videos")
                print(f"Running {args.n_samples} bootstrap samples on combined set...")
                
                # Store bootstrap results (initialize with NaN to maintain order)
                bootstrap_aps = [np.nan] * args.n_samples
                bootstrap_ap50s = [np.nan] * args.n_samples
                bootstrap_ap75s = [np.nan] * args.n_samples
                
                # Create temporary directory for filtered JSON files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    
                    if n_jobs > 1:
                        # Parallel processing for pooled bootstrap
                        print(f"Using {n_jobs} parallel workers for pooled bootstrap...")
                        
                        # Prepare arguments for parallel processing
                        # Convert temp_dir to string for pickling
                        bootstrap_args = [
                            (i, all_video_ids, n_all_videos, merged_predictions, merged_val_json, str(temp_dir), "Pooled", args.seed)
                            for i in range(args.n_samples)
                        ]
                        
                        completed = 0
                        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                            # Submit all tasks
                            future_to_index = {
                                executor.submit(process_single_bootstrap, args): args[0]
                                for args in bootstrap_args
                            }
                            
                            # Process completed tasks
                            for future in as_completed(future_to_index):
                                i, ap, ap50, ap75, error_msg = future.result()
                                bootstrap_aps[i] = ap
                                bootstrap_ap50s[i] = ap50
                                bootstrap_ap75s[i] = ap75
                                
                                completed += 1
                                print(f"Pooled, Bootstrap {completed}/{args.n_samples}", end='\r')
                                
                                if error_msg:
                                    print(f"\nWarning: Error computing metrics for Pooled, bootstrap sample {i+1}: {error_msg[:200]}")  # Truncate long errors
                    else:
                        # Sequential processing
                        for i in range(args.n_samples):
                            print(f"Pooled, Bootstrap {i+1}/{args.n_samples}", end='\r')
                            
                            # Sample video IDs with replacement
                            sampled_video_ids = bootstrap_sample_video_ids(all_video_ids, n_samples=n_all_videos)
                            
                            # Filter predictions and validation JSON
                            filtered_predictions = filter_predictions_by_video_ids(merged_predictions, sampled_video_ids)
                            filtered_val_json = filter_val_json_by_video_ids(merged_val_json, sampled_video_ids)
                            
                            # Save filtered validation JSON to temporary file
                            temp_val_json_path = temp_dir / f"val_bootstrap_combined_{i}.json"
                            save_json(filtered_val_json, temp_val_json_path)
                            
                            # Compute metrics
                            try:
                                unique_dataset_name = f"ytvis_fishway_val_bootstrap_combined_{i}"
                                # Suppress verbose output from compute_dvis_daq_metrics
                                # Use skip_folder_rename=True since folder is managed at script level
                                with suppress_stdout_stderr():
                                    metrics = compute_dvis_daq_metrics(
                                        filtered_predictions,
                                        str(temp_val_json_path),
                                        dataset_name=unique_dataset_name,
                                        skip_folder_rename=True
                                    )
                                
                                if metrics:
                                    bootstrap_aps[i] = metrics.get('AP', np.nan)
                                    bootstrap_ap50s[i] = metrics.get('AP50', np.nan)
                                    bootstrap_ap75s[i] = metrics.get('AP75', np.nan)
                                else:
                                    bootstrap_aps[i] = np.nan
                                    bootstrap_ap50s[i] = np.nan
                                    bootstrap_ap75s[i] = np.nan
                            except Exception as e:
                                print(f"\nWarning: Error computing metrics for Pooled, bootstrap sample {i+1}: {e}")
                                bootstrap_aps[i] = np.nan
                                bootstrap_ap50s[i] = np.nan
                                bootstrap_ap75s[i] = np.nan
                
                print(f"\nPooled bootstrap completed: {args.n_samples} bootstrap samples processed")
                
                # Compute statistics for combined bootstrap
                bootstrap_aps = np.array(bootstrap_aps)
                bootstrap_ap50s = np.array(bootstrap_ap50s)
                bootstrap_ap75s = np.array(bootstrap_ap75s)
                
                valid_aps = bootstrap_aps[~np.isnan(bootstrap_aps)]
                valid_ap50s = bootstrap_ap50s[~np.isnan(bootstrap_ap50s)]
                valid_ap75s = bootstrap_ap75s[~np.isnan(bootstrap_ap75s)]
                
                combined_results = {
                    'n_bootstrap_samples': args.n_samples,
                    'n_videos': n_all_videos,
                    'n_valid_samples': {
                        'AP': len(valid_aps),
                        'AP50': len(valid_ap50s),
                        'AP75': len(valid_ap75s)
                    },
                    'AP': {
                        'mean': float(np.mean(valid_aps)) if len(valid_aps) > 0 else np.nan,
                        'std': float(np.std(valid_aps, ddof=1)) if len(valid_aps) > 0 else np.nan,
                        'median': float(np.median(valid_aps)) if len(valid_aps) > 0 else np.nan,
                        'q025': float(np.percentile(valid_aps, 2.5)) if len(valid_aps) > 0 else np.nan,
                        'q975': float(np.percentile(valid_aps, 97.5)) if len(valid_aps) > 0 else np.nan,
                        'all_values': bootstrap_aps.tolist()
                    },
                    'AP50': {
                        'mean': float(np.mean(valid_ap50s)) if len(valid_ap50s) > 0 else np.nan,
                        'std': float(np.std(valid_ap50s, ddof=1)) if len(valid_ap50s) > 0 else np.nan,
                        'median': float(np.median(valid_ap50s)) if len(valid_ap50s) > 0 else np.nan,
                        'q025': float(np.percentile(valid_ap50s, 2.5)) if len(valid_ap50s) > 0 else np.nan,
                        'q975': float(np.percentile(valid_ap50s, 97.5)) if len(valid_ap50s) > 0 else np.nan,
                        'all_values': bootstrap_ap50s.tolist()
                    },
                    'AP75': {
                        'mean': float(np.mean(valid_ap75s)) if len(valid_ap75s) > 0 else np.nan,
                        'std': float(np.std(valid_ap75s, ddof=1)) if len(valid_ap75s) > 0 else np.nan,
                        'median': float(np.median(valid_ap75s)) if len(valid_ap75s) > 0 else np.nan,
                        'q025': float(np.percentile(valid_ap75s, 2.5)) if len(valid_ap75s) > 0 else np.nan,
                        'q975': float(np.percentile(valid_ap75s, 97.5)) if len(valid_ap75s) > 0 else np.nan,
                        'all_values': bootstrap_ap75s.tolist()
                    }
                }
                
                # Save all results to bootstrap directory
                bootstrap_base_dir = Path("/home/simone/store/simone/bootstrap") / data_type
                bootstrap_base_dir.mkdir(parents=True, exist_ok=True)
                
                # Save individual fold results
                for fold_num, fold_results in all_fold_results.items():
                    fold_output = bootstrap_base_dir / f"fold_{fold_num}_bootstrap_results.json"
                    save_json(fold_results, fold_output)
                    print(f"Saved fold {fold_num} results to: {fold_output}")
                
                # Save combined results
                combined_output = bootstrap_base_dir / "combined_bootstrap_results.json"
                save_json(combined_results, combined_output)
                print(f"\nCombined bootstrap results saved to: {combined_output}")
                
                # Create and save plot
                plot_output = bootstrap_base_dir / "bootstrap_results_plot.png"
                plot_bootstrap_results(all_fold_results, combined_results, plot_output, 
                                     checkpoint_eval_dirs=checkpoint_eval_dirs)
                
                # Print combined summary
                print("\n" + "="*80)
                print("COMBINED BOOTSTRAP EVALUATION SUMMARY")
                print("="*80)
                print(f"Number of bootstrap samples: {combined_results['n_bootstrap_samples']}")
                print(f"Number of videos per sample: {combined_results['n_videos']}")
                print()
                print("AP (Average Precision):")
                print(f"  Mean: {combined_results['AP']['mean']:.4f}")
                print(f"  Std:  {combined_results['AP']['std']:.4f}")
                print(f"  95% CI: [{combined_results['AP']['q025']:.4f}, {combined_results['AP']['q975']:.4f}]")
                print()
                print("AP50 (Average Precision @ IoU=0.50):")
                print(f"  Mean: {combined_results['AP50']['mean']:.4f}")
                print(f"  Std:  {combined_results['AP50']['std']:.4f}")
                print(f"  95% CI: [{combined_results['AP50']['q025']:.4f}, {combined_results['AP50']['q975']:.4f}]")
                print()
                print("AP75 (Average Precision @ IoU=0.75):")
                print(f"  Mean: {combined_results['AP75']['mean']:.4f}")
                print(f"  Std:  {combined_results['AP75']['std']:.4f}")
                print(f"  95% CI: [{combined_results['AP75']['q025']:.4f}, {combined_results['AP75']['q975']:.4f}]")
                print("="*80)
        
        else:
            # Single directory mode (original behavior)
            checkpoint_eval_dir = checkpoint_eval_dirs[0]
            
            # Run bootstrap evaluation
            results = run_bootstrap_evaluation(
                checkpoint_eval_dir,
                n_bootstrap_samples=args.n_samples,
                seed=args.seed,
                n_jobs=n_jobs
            )
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = checkpoint_eval_dir / "bootstrap_results.json"
            
            # Save results
            save_json(results, output_path)
            print(f"\nBootstrap results saved to: {output_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("BOOTSTRAP EVALUATION SUMMARY")
            print("="*80)
            print(f"Number of bootstrap samples: {results['n_bootstrap_samples']}")
            print(f"Number of videos per sample: {results['n_videos']}")
            print()
            print("AP (Average Precision):")
            print(f"  Mean: {results['AP']['mean']:.4f}")
            print(f"  Std:  {results['AP']['std']:.4f}")
            print(f"  95% CI: [{results['AP']['q025']:.4f}, {results['AP']['q975']:.4f}]")
            print()
            print("AP50 (Average Precision @ IoU=0.50):")
            print(f"  Mean: {results['AP50']['mean']:.4f}")
            print(f"  Std:  {results['AP50']['std']:.4f}")
            print(f"  95% CI: [{results['AP50']['q025']:.4f}, {results['AP50']['q975']:.4f}]")
            print()
            print("AP75 (Average Precision @ IoU=0.75):")
            print(f"  Mean: {results['AP75']['mean']:.4f}")
            print(f"  Std:  {results['AP75']['std']:.4f}")
            print(f"  95% CI: [{results['AP75']['q025']:.4f}, {results['AP75']['q975']:.4f}]")
            print("="*80)
    
    finally:
        # Always restore pycocotools folder at the end
        print()
        print("Restoring pycocotools folder...")
        restore_pycocotools_folder(was_already_renamed)


if __name__ == "__main__":
    main()

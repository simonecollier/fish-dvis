#!/usr/bin/env python3
"""
Single checkpoint analysis script that provides detailed analysis of one checkpoint.

This script will:
1. Run mask metrics analysis for the checkpoint if needed
2. Create detailed performance analysis and visualizations
3. Generate comprehensive reports for the specific checkpoint
4. Provide insights about the model's performance at this point

Usage:
    # Basic analysis of existing results
    python analyze_single_checkpoint.py --checkpoint-dir /path/to/checkpoint_1000
    
    # Run mask metrics and then analyze
    python analyze_single_checkpoint.py --checkpoint-dir /path/to/checkpoint_1000 --run-mask-metrics
    
    # Fast mode for quick analysis
    python analyze_single_checkpoint.py --checkpoint-dir /path/to/checkpoint_1000 --run-mask-metrics --fast-mode
"""

import os
import json
import subprocess
import argparse
import re
import yaml
from typing import Dict, Optional
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_mask_metrics_analysis(checkpoint_dir: str, val_json: str, confidence_threshold: float = 0.05, 
                            fast_mode: bool = False, force_recompute: bool = False) -> bool:
    """
    Run mask metrics analysis for a single checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
        val_json: Path to validation JSON file
        confidence_threshold: Minimum confidence threshold for predictions
        fast_mode: Run in fast mode (skip expensive operations)
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
    
    # Skip if mask metrics already exist and not forcing recomputation
    if os.path.exists(csv_path) and not force_recompute:
        logger.info(f"Mask metrics already exist for: {os.path.basename(checkpoint_dir)}")
        return True
    
    logger.info(f"Running mask metrics analysis for: {os.path.basename(checkpoint_dir)}")
    
    try:
        # Run the mask metrics script
        cmd = [
            "python", "/home/simone/fish-dvis/visualization_scripts/mask_metrics.py",
            "--results-json", results_json,
            "--val-json", val_json,
            "--csv-path", csv_path,
            "--cm-plot-path", cm_plot_path,
            "--confidence-threshold", str(confidence_threshold)
        ]
        
        if fast_mode:
            cmd.append("--fast-mode")
        
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
    except Exception as e:
        logger.error(f"Exception during mask metrics analysis: {e}")
        return False

def load_checkpoint_data(checkpoint_dir: str) -> Dict:
    """
    Load all available data for a single checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
        
    Returns:
        Dictionary containing all loaded data
    """
    data = {
        'dataset_metrics': None,
        'frame_metrics': None,
        'category_metrics': None,
        'confusion_matrix': None,
        'results_json': None
    }
    
    # Try new separate CSV structure first
    dataset_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_dataset.csv")
    frame_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_frame.csv")
    category_csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics_category.csv")
    
    # Load dataset-level metrics
    if os.path.exists(dataset_csv_path):
        try:
            data['dataset_metrics'] = pd.read_csv(dataset_csv_path)
            logger.info("Loaded dataset-level metrics")
        except Exception as e:
            logger.warning(f"Could not load dataset CSV: {e}")
    
    # Load frame-level metrics
    if os.path.exists(frame_csv_path):
        try:
            data['frame_metrics'] = pd.read_csv(frame_csv_path)
            logger.info("Loaded frame-level metrics")
        except Exception as e:
            logger.warning(f"Could not load frame CSV: {e}")
    
    # Load category-level metrics
    if os.path.exists(category_csv_path):
        try:
            data['category_metrics'] = pd.read_csv(category_csv_path)
            logger.info("Loaded category-level metrics")
        except Exception as e:
            logger.warning(f"Could not load category CSV: {e}")
    
    # Fallback to old combined CSV structure
    if data['dataset_metrics'] is None:
        csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Separate dataset and category metrics
                dataset_rows = df[df['category_id'].isna()]
                category_rows = df[df['category_id'].notna()]
                
                if not dataset_rows.empty:
                    data['dataset_metrics'] = dataset_rows
                if not category_rows.empty:
                    data['category_metrics'] = category_rows
                
                logger.info("Loaded combined CSV structure")
            except Exception as e:
                logger.warning(f"Could not load combined CSV: {e}")
    
    # Load results JSON
    results_json_path = os.path.join(checkpoint_dir, "inference", "results.json")
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                data['results_json'] = json.load(f)
            logger.info("Loaded results JSON")
        except Exception as e:
            logger.warning(f"Could not load results JSON: {e}")
    
    return data

def extract_checkpoint_info(checkpoint_dir: str) -> Dict:
    """
    Extract information about the checkpoint from directory name and config.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
        
    Returns:
        Dictionary with checkpoint information
    """
    info = {
        'iteration': None,
        'model_name': None,
        'config': None,
        'training_info': {}
    }
    
    # Extract iteration from directory name
    dir_name = os.path.basename(checkpoint_dir)
    match = re.search(r'checkpoint_(\d+)', dir_name)
    if match:
        info['iteration'] = int(match.group(1))
    
    # Try to find model directory
    model_dir = os.path.dirname(checkpoint_dir)
    if 'checkpoint_evaluations' in model_dir:
        model_dir = os.path.dirname(model_dir)
    
    info['model_name'] = os.path.basename(model_dir)
    
    # Load config file
    config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                info['config'] = yaml.safe_load(f)
            
            # Extract key training parameters
            if info['config']:
                solver = info['config'].get('SOLVER', {})
                input_config = info['config'].get('INPUT', {})
                model_config = info['config'].get('MODEL', {})
                
                info['training_info'] = {
                    'learning_rate': solver.get('BASE_LR', 'unknown'),
                    'batch_size': solver.get('IMS_PER_BATCH', 'unknown'),
                    'max_iterations': solver.get('MAX_ITER', 'unknown'),
                    'training_resolution': input_config.get('MIN_SIZE_TRAIN', 'unknown'),
                    'test_resolution': input_config.get('MIN_SIZE_TEST', 'unknown'),
                    'frame_sampling': input_config.get('SAMPLING_FRAME_NUM', 'unknown'),
                    'window_size': model_config.get('MASK_FORMER', {}).get('TEST', {}).get('WINDOW_SIZE', 'unknown')
                }
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    return info

def create_performance_summary_plots(data: Dict, output_dir: str, checkpoint_info: Dict) -> None:
    """
    Create comprehensive performance summary plots for the checkpoint.
    
    Args:
        data: Dictionary containing all checkpoint data
        output_dir: Directory to save plots
        checkpoint_info: Dictionary with checkpoint information
    """
    if not data['dataset_metrics'] is not None and not data['category_metrics'] is not None:
        logger.warning("No data available for plotting")
        return
    
    # Create comprehensive performance dashboard
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'Checkpoint {checkpoint_info["iteration"]} Performance Analysis\n{checkpoint_info["model_name"]}', 
                 fontsize=16, fontweight='bold')
    
    # Set up grid layout
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Overall metrics summary
    ax1 = fig.add_subplot(gs[0, :])
    if data['dataset_metrics'] is not None:
        metrics_to_plot = []
        labels = []
        
        # Extract key metrics
        for _, row in data['dataset_metrics'].iterrows():
            metric_name = row['metric_name']
            value = row['value']
            
            if 'ap50' in metric_name.lower() or 'iou' in metric_name.lower() or 'boundary' in metric_name.lower():
                metrics_to_plot.append(value)
                labels.append(metric_name.replace('_', ' ').title())
        
        if metrics_to_plot:
            bars = ax1.bar(range(len(metrics_to_plot)), metrics_to_plot, color='skyblue', alpha=0.7)
            ax1.set_xticks(range(len(metrics_to_plot)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel('Score')
            ax1.set_title('Key Performance Metrics')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_to_plot):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Per-category AP50 performance
    ax2 = fig.add_subplot(gs[1, 0])
    if data['category_metrics'] is not None:
        categories = data['category_metrics']['category_name'].tolist()
        ap50_values = data['category_metrics']['ap50_track_per_cat'].tolist()
        
        bars = ax2.bar(range(len(categories)), ap50_values, color='lightgreen', alpha=0.7)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.set_ylabel('AP50')
        ax2.set_title('AP50 by Category')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, ap50_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: IoU thresholds comparison
    ax3 = fig.add_subplot(gs[1, 1])
    if data['category_metrics'] is not None:
        # Get average AP across categories for different thresholds
        ap_columns = ['ap10_track_per_cat', 'ap25_track_per_cat', 'ap50_track_per_cat', 
                     'ap75_track_per_cat', 'ap95_track_per_cat']
        thresholds = ['0.1', '0.25', '0.5', '0.75', '0.95']
        
        avg_ap_values = []
        for col in ap_columns:
            if col in data['category_metrics'].columns:
                avg_ap = data['category_metrics'][col].mean()
                avg_ap_values.append(avg_ap)
            else:
                avg_ap_values.append(0)
        
        ax3.plot(thresholds, avg_ap_values, 'o-', linewidth=2, markersize=8, color='red')
        ax3.set_xlabel('IoU Threshold')
        ax3.set_ylabel('Average AP')
        ax3.set_title('AP vs IoU Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (threshold, value) in enumerate(zip(thresholds, avg_ap_values)):
            ax3.annotate(f'{value:.3f}', (i, value), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 4: Frame-level IoU distribution
    ax4 = fig.add_subplot(gs[1, 2])
    if data['frame_metrics'] is not None and 'frame_IoU' in data['frame_metrics'].columns:
        iou_values = data['frame_metrics']['frame_IoU'].dropna()
        if len(iou_values) > 0:
            ax4.hist(iou_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_xlabel('Frame IoU')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Frame-level IoU Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            mean_iou = iou_values.mean()
            median_iou = iou_values.median()
            ax4.axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
            ax4.axvline(median_iou, color='green', linestyle='--', label=f'Median: {median_iou:.3f}')
            ax4.legend()
    
    # Plot 5: Per-category detailed metrics
    ax5 = fig.add_subplot(gs[2, :])
    if data['category_metrics'] is not None:
        # Create a heatmap of metrics by category
        metrics_for_heatmap = ['ap10_track_per_cat', 'ap25_track_per_cat', 'ap50_track_per_cat', 
                              'ap75_track_per_cat', 'ap95_track_per_cat']
        
        heatmap_data = []
        categories = []
        
        for _, row in data['category_metrics'].iterrows():
            category = row['category_name']
            categories.append(category)
            row_data = []
            for metric in metrics_for_heatmap:
                if metric in row:
                    row_data.append(row[metric])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, 
                                    index=categories,
                                    columns=['AP@0.1', 'AP@0.25', 'AP@0.5', 'AP@0.75', 'AP@0.95'])
            
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5)
            ax5.set_title('Per-Category AP Performance Heatmap')
            ax5.set_xlabel('IoU Threshold')
            ax5.set_ylabel('Category')
    
    # Plot 6: Training configuration summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    if checkpoint_info['training_info']:
        config_text = "Training Configuration:\n\n"
        for key, value in checkpoint_info['training_info'].items():
            config_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        ax6.text(0.1, 0.9, config_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Plot 7: Performance comparison with benchmarks
    ax7 = fig.add_subplot(gs[4, :])
    if data['dataset_metrics'] is not None:
        # Extract key metrics for comparison
        key_metrics = {}
        for _, row in data['dataset_metrics'].iterrows():
            metric_name = row['metric_name']
            value = row['value']
            
            if any(keyword in metric_name.lower() for keyword in ['ap50', 'iou', 'boundary']):
                key_metrics[metric_name] = value
        
        if key_metrics:
            metric_names = list(key_metrics.keys())
            metric_values = list(key_metrics.values())
            
            bars = ax7.bar(range(len(metric_names)), metric_values, 
                          color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(metric_names)])
            ax7.set_xticks(range(len(metric_names)))
            ax7.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], 
                               rotation=45, ha='right')
            ax7.set_ylabel('Score')
            ax7.set_title('Performance Metrics Summary')
            ax7.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 8: Model architecture and training info
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    
    model_info = f"Model: {checkpoint_info['model_name']}\n"
    model_info += f"Iteration: {checkpoint_info['iteration']}\n"
    model_info += f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if data['dataset_metrics'] is not None:
        model_info += "Available Metrics:\n"
        for _, row in data['dataset_metrics'].iterrows():
            model_info += f"  • {row['metric_name']}: {row['value']:.4f}\n"
    
    ax8.text(0.1, 0.9, model_info, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'checkpoint_performance_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Performance summary plots saved to: {summary_plot_path}")

def create_detailed_analysis_report(data: Dict, output_dir: str, checkpoint_info: Dict) -> None:
    """
    Create a detailed analysis report for the checkpoint.
    
    Args:
        data: Dictionary containing all checkpoint data
        output_dir: Directory to save the report
        checkpoint_info: Dictionary with checkpoint information
    """
    report_path = os.path.join(output_dir, 'checkpoint_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CHECKPOINT {checkpoint_info['iteration']} ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic information
        f.write("BASIC INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {checkpoint_info['model_name']}\n")
        f.write(f"Iteration: {checkpoint_info['iteration']}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training configuration
        if checkpoint_info['training_info']:
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            for key, value in checkpoint_info['training_info'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
        
        # Dataset-level metrics
        if data['dataset_metrics'] is not None:
            f.write("DATASET-LEVEL METRICS\n")
            f.write("-" * 40 + "\n")
            for _, row in data['dataset_metrics'].iterrows():
                metric_name = row['metric_name'].replace('_', ' ').title()
                value = row['value']
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
        
        # Category-level analysis
        if data['category_metrics'] is not None:
            f.write("CATEGORY-LEVEL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Best and worst performing categories
            if 'ap50_track_per_cat' in data['category_metrics'].columns:
                ap50_data = data['category_metrics'][['category_name', 'ap50_track_per_cat']].dropna()
                if len(ap50_data) > 0:
                    best_category = ap50_data.loc[ap50_data['ap50_track_per_cat'].idxmax()]
                    worst_category = ap50_data.loc[ap50_data['ap50_track_per_cat'].idxmin()]
                    
                    f.write(f"Best performing category: {best_category['category_name']} (AP50: {best_category['ap50_track_per_cat']:.4f})\n")
                    f.write(f"Worst performing category: {worst_category['category_name']} (AP50: {worst_category['ap50_track_per_cat']:.4f})\n")
                    f.write(f"Performance gap: {best_category['ap50_track_per_cat'] - worst_category['ap50_track_per_cat']:.4f}\n\n")
            
            # Detailed per-category metrics
            f.write("Detailed Per-Category Metrics:\n")
            for _, row in data['category_metrics'].iterrows():
                category = row['category_name']
                f.write(f"\n{category}:\n")
                
                for col in data['category_metrics'].columns:
                    if col != 'category_name' and col != 'category_id':
                        value = row[col]
                        if pd.notna(value):
                            metric_name = col.replace('_', ' ').title()
                            f.write(f"  {metric_name}: {value:.4f}\n")
        
        # Frame-level analysis
        if data['frame_metrics'] is not None:
            f.write("\nFRAME-LEVEL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            if 'frame_IoU' in data['frame_metrics'].columns:
                iou_values = data['frame_metrics']['frame_IoU'].dropna()
                if len(iou_values) > 0:
                    f.write(f"Total frames analyzed: {len(iou_values)}\n")
                    f.write(f"Mean frame IoU: {iou_values.mean():.4f}\n")
                    f.write(f"Median frame IoU: {iou_values.median():.4f}\n")
                    f.write(f"Std frame IoU: {iou_values.std():.4f}\n")
                    f.write(f"Min frame IoU: {iou_values.min():.4f}\n")
                    f.write(f"Max frame IoU: {iou_values.max():.4f}\n")
                    
                    # IoU distribution analysis
                    high_iou = (iou_values > 0.8).sum()
                    medium_iou = ((iou_values > 0.5) & (iou_values <= 0.8)).sum()
                    low_iou = (iou_values <= 0.5).sum()
                    
                    f.write(f"\nIoU Distribution:\n")
                    f.write(f"  High IoU (>0.8): {high_iou} frames ({high_iou/len(iou_values)*100:.1f}%)\n")
                    f.write(f"  Medium IoU (0.5-0.8): {medium_iou} frames ({medium_iou/len(iou_values)*100:.1f}%)\n")
                    f.write(f"  Low IoU (≤0.5): {low_iou} frames ({low_iou/len(iou_values)*100:.1f}%)\n")
        
        # Performance insights
        f.write("\nPERFORMANCE INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        if data['dataset_metrics'] is not None:
            # Extract key metrics for analysis
            key_metrics = {}
            for _, row in data['dataset_metrics'].iterrows():
                key_metrics[row['metric_name']] = row['value']
            
            # Analyze performance
            if 'ap50_track' in key_metrics:
                ap50 = key_metrics['ap50_track']
                if ap50 > 0.8:
                    f.write("✅ Excellent AP50 performance (>0.8)\n")
                elif ap50 > 0.6:
                    f.write("✅ Good AP50 performance (0.6-0.8)\n")
                elif ap50 > 0.4:
                    f.write("⚠️  Moderate AP50 performance (0.4-0.6)\n")
                else:
                    f.write("❌ Poor AP50 performance (<0.4)\n")
            
            if 'mean_iou' in key_metrics:
                mean_iou = key_metrics['mean_iou']
                if mean_iou > 0.7:
                    f.write("✅ Excellent IoU performance (>0.7)\n")
                elif mean_iou > 0.5:
                    f.write("✅ Good IoU performance (0.5-0.7)\n")
                elif mean_iou > 0.3:
                    f.write("⚠️  Moderate IoU performance (0.3-0.5)\n")
                else:
                    f.write("❌ Poor IoU performance (<0.3)\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if data['category_metrics'] is not None and 'ap50_track_per_cat' in data['category_metrics'].columns:
            ap50_data = data['category_metrics'][['category_name', 'ap50_track_per_cat']].dropna()
            if len(ap50_data) > 0:
                worst_performing = ap50_data.loc[ap50_data['ap50_track_per_cat'].idxmin()]
                if worst_performing['ap50_track_per_cat'] < 0.3:
                    f.write(f"• Focus on improving performance for {worst_performing['category_name']}\n")
                    f.write("• Consider adding more training data for this category\n")
                    f.write("• Review annotation quality for this category\n")
                
                performance_gap = ap50_data['ap50_track_per_cat'].max() - ap50_data['ap50_track_per_cat'].min()
                if performance_gap > 0.3:
                    f.write("• Large performance gap between categories detected\n")
                    f.write("• Consider category-balanced training or loss weighting\n")
        
        if data['frame_metrics'] is not None and 'frame_IoU' in data['frame_metrics'].columns:
            iou_values = data['frame_metrics']['frame_IoU'].dropna()
            if len(iou_values) > 0:
                low_iou_frames = (iou_values <= 0.5).sum()
                if low_iou_frames > len(iou_values) * 0.3:
                    f.write("• High proportion of low-IoU frames detected\n")
                    f.write("• Consider improving mask quality or resolution\n")
                    f.write("• Review data augmentation strategies\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated by analyze_single_checkpoint.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Detailed analysis report saved to: {report_path}")

def create_confusion_matrix_analysis(data: Dict, output_dir: str) -> None:
    """
    Create confusion matrix analysis if available.
    
    Args:
        data: Dictionary containing all checkpoint data
        output_dir: Directory to save confusion matrix analysis
    """
    confusion_matrix_path = os.path.join(os.path.dirname(output_dir), "inference", "confusion_matrix.png")
    
    if os.path.exists(confusion_matrix_path):
        logger.info(f"Confusion matrix found at: {confusion_matrix_path}")
        
        # Copy confusion matrix to output directory
        import shutil
        output_cm_path = os.path.join(output_dir, "confusion_matrix.png")
        shutil.copy2(confusion_matrix_path, output_cm_path)
        logger.info(f"Confusion matrix copied to: {output_cm_path}")
    else:
        logger.info("No confusion matrix found")

def main():
    parser = argparse.ArgumentParser(description='Single checkpoint analysis')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to the checkpoint directory to analyze')
    parser.add_argument('--val-json', type=str, default='/data/fishway_ytvis/val.json',
                       help='Path to validation JSON file')
    parser.add_argument('--run-mask-metrics', action='store_true',
                       help='Force run mask metrics analysis (overwrites existing)')
    parser.add_argument('--confidence-threshold', type=float, default=0.05,
                       help='Confidence threshold for filtering predictions')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Run in fast mode (skip expensive operations)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis (default: checkpoint directory)')
    
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = args.checkpoint_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract checkpoint information
    logger.info("Extracting checkpoint information...")
    checkpoint_info = extract_checkpoint_info(args.checkpoint_dir)
    
    # Run mask metrics analysis if needed
    if args.run_mask_metrics:
        logger.info("Running mask metrics analysis...")
        success = run_mask_metrics_analysis(args.checkpoint_dir, args.val_json, 
                                          args.confidence_threshold, args.fast_mode, True)
        if not success:
            logger.error("Mask metrics analysis failed")
            return
    
    # Load checkpoint data
    logger.info("Loading checkpoint data...")
    data = load_checkpoint_data(args.checkpoint_dir)
    
    if data['dataset_metrics'] is None and data['category_metrics'] is None:
        logger.error("No data found. Make sure mask metrics analysis has been run.")
        return
    
    # Create analysis
    logger.info("Creating performance summary plots...")
    create_performance_summary_plots(data, output_dir, checkpoint_info)
    
    logger.info("Creating detailed analysis report...")
    create_detailed_analysis_report(data, output_dir, checkpoint_info)
    
    logger.info("Creating confusion matrix analysis...")
    create_confusion_matrix_analysis(data, output_dir)
    
    logger.info(f"Single checkpoint analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

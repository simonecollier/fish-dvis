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
    python analyze_checkpoint_results.py --model-dir /path/to/model --run-mask-metrics --fast-mode
    
    # Basic summary only (no advanced analysis)
    python analyze_checkpoint_results.py --model-dir /path/to/model --analysis-level basic
"""

import os
import json
import glob
import subprocess
import argparse
import re
from typing import List, Dict
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_training_loss import plot_training_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_checkpoint_results(base_dir: str) -> List[str]:
    """
    Find all checkpoint result directories.
    
    Args:
        base_dir: Base directory containing checkpoint results
        
    Returns:
        List of checkpoint result directories
    """
    checkpoint_dirs = []
    
    # Look for checkpoint_evaluations directory
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

def run_mask_metrics_analysis(checkpoint_dir: str, val_json: str, confidence_threshold: float = 0.05, fast_mode: bool = False) -> bool:
    """
    Run mask metrics analysis for a single checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint results
        val_json: Path to validation JSON file
        confidence_threshold: Minimum confidence threshold for predictions
        fast_mode: Run in fast mode (skip expensive operations)
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    results_json = os.path.join(checkpoint_dir, "inference", "results.json")
    csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
    cm_plot_path = os.path.join(checkpoint_dir, "inference", "confusion_matrix.png")
    ap_plot_path = os.path.join(checkpoint_dir, "inference", "AP_per_category.png")
    
    if not os.path.exists(results_json):
        logger.warning(f"Results JSON not found: {results_json}")
        return False
    
    # Skip if mask metrics already exist
    if os.path.exists(csv_path):
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
            "--ap-plot-path", ap_plot_path,
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

def collect_summary_data(base_dir: str) -> pd.DataFrame:
    """
    Collect summary data from all checkpoint CSV files.
    
    Args:
        base_dir: Base directory containing all checkpoint results
        
    Returns:
        DataFrame with summary metrics for all checkpoints
    """
    checkpoint_dirs = find_checkpoint_results(base_dir)
    
    if not checkpoint_dirs:
        logger.error("No checkpoint result directories found!")
        return pd.DataFrame()
    
    # Collect data from all checkpoint CSV files
    summary_data = []
    
    for checkpoint_dir in checkpoint_dirs:
        csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Extract iteration number
                dir_name = os.path.basename(checkpoint_dir)
                match = re.search(r'checkpoint_(\d+)', dir_name)
                iteration = int(match.group(1)) if match else 0
                
                # Get metrics (take first row since all rows have same dataset-level metrics)
                if len(df) > 0:
                    row = df.iloc[0]
                    summary_data.append({
                        'iteration': iteration,
                        'mean_iou': row.get('dataset_IoU', None),
                        'mean_boundary_f': row.get('dataset_boundary_Fmeasure', None),
                        'map_10': row.get('mAP@0.1', None),
                        'map_25': row.get('mAP@0.25', None),
                        'map_50': row.get('mAP@0.5', None),
                        'map_75': row.get('mAP@0.75', None),
                        'map_95': row.get('mAP@0.95', None),
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

def create_basic_summary_plot(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create basic summary plot (similar to run_mask_metrics_for_checkpoints.py).
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to plot")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Across Checkpoints (Mask Metrics)', fontsize=16, fontweight='bold')
    
    metrics = ['mean_iou', 'mean_boundary_f', 'map_25', 'map_50', 'map_75', 'map_95']
    titles = ['Mean IoU', 'Mean Boundary F-measure', 'mAP@0.25', 'mAP@0.5', 'mAP@0.75', 'mAP@0.95']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
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
    logger.info("\nMask Metrics Summary:")
    logger.info("=" * 80)
    for _, row in df_summary.iterrows():
        mean_iou = row.get('mean_iou', 'N/A')
        map_50 = row.get('map_50', 'N/A')
        
        # Format values properly
        iou_str = f"{mean_iou:.4f}" if isinstance(mean_iou, (int, float)) else str(mean_iou)
        map_str = f"{map_50:.4f}" if isinstance(map_50, (int, float)) else str(map_50)
        
        logger.info(f"Iteration {row['iteration']}: "
                   f"IoU={iou_str}, "
                   f"mAP@0.5={map_str}")

def create_comprehensive_analysis(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive analysis (similar to plot_existing_checkpoint_results.py).
    
    Args:
        df_summary: DataFrame with summary metrics
        output_dir: Directory to save plots
    """
    if df_summary.empty:
        logger.warning("No data to analyze")
        return
    
    # Create comprehensive plots
    create_performance_plots(df_summary, output_dir)
    create_trend_analysis(df_summary, output_dir)
    create_best_checkpoint_analysis(df_summary, output_dir)
    
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
    create_model_performance_report(df_summary, output_dir, model_dir)
    
    # Save comprehensive summary CSV
    summary_csv_path = os.path.join(output_dir, 'comprehensive_metrics_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    logger.info(f"Comprehensive analysis saved to: {output_dir}")
    logger.info(f"Summary CSV saved to: {summary_csv_path}")
    logger.info(f"Model performance report saved to: {output_dir}/model_performance_report.txt")

def create_performance_plots(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create performance comparison plots.
    """
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Across Checkpoints', fontsize=16, fontweight='bold')
    
    metrics = ['mean_iou', 'mean_boundary_f', 'map_25', 'map_50', 'map_75', 'map_95']
    titles = ['Mean IoU', 'Mean Boundary F-measure', 'mAP@0.25', 'mAP@0.5', 'mAP@0.75', 'mAP@0.95']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
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
    
    plt.tight_layout()
    performance_plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_trend_analysis(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create trend analysis plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Key metrics over time
    ax1 = axes[0]
    key_metrics = ['map_50', 'mean_iou']
    colors = ['#1f77b4', '#ff7f0e']
    labels = ['mAP@0.5', 'Mean IoU']
    
    for metric, color, label in zip(key_metrics, colors, labels):
        valid_data = df_summary[df_summary[metric].notna()]
        if len(valid_data) > 0:
            ax1.plot(valid_data['iteration'], valid_data[metric], 'o-', 
                    color=color, linewidth=2, markersize=6, label=label)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Key Performance Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance improvement
    ax2 = axes[1]
    if 'map_50' in df_summary.columns:
        valid_data = df_summary[df_summary['map_50'].notna()]
        if len(valid_data) > 1:
            # Calculate improvement from first to last checkpoint
            first_val = valid_data.iloc[0]['map_50']
            last_val = valid_data.iloc[-1]['map_50']
            
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
                        current_improvement = ((valid_data.iloc[i]['map_50'] - first_val) / first_val) * 100
                    else:
                        current_improvement = float('inf') if valid_data.iloc[i]['map_50'] > 0 else 0
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

def create_best_checkpoint_analysis(df_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create analysis of best performing checkpoints.
    """
    # Find best checkpoints for different metrics
    best_checkpoints = {}
    
    for metric in ['map_50', 'mean_iou', 'map_75']:
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
                'map_50': 'mAP@0.5',
                'mean_iou': 'Mean IoU',
                'map_75': 'mAP@0.75'
            }.get(metric, metric)
            table_data.append([metric_name, info['iteration'], f"{info['value']:.4f}"])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Best Iteration', 'Best Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Best Performing Checkpoints', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        best_checkpoints_path = os.path.join(output_dir, 'best_checkpoints.png')
        plt.savefig(best_checkpoints_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        logger.info("\nBest Performing Checkpoints:")
        logger.info("=" * 50)
        for metric, info in best_checkpoints.items():
            metric_name = {
                'map_50': 'mAP@0.5',
                'mean_iou': 'Mean IoU',
                'map_75': 'mAP@0.75'
            }.get(metric, metric)
            logger.info(f"{metric_name}: Iteration {info['iteration']} (Value: {info['value']:.4f})")

def analyze_overfitting(df_summary: pd.DataFrame) -> dict:
    """
    Analyze signs of overfitting in the training curve.
    """
    analysis = {
        'overfitting_detected': False,
        'severity': 'none',
        'indicators': [],
        'recommendations': []
    }
    
    if len(df_summary) < 3:
        return analysis
    
    # Calculate moving averages to smooth the curves
    window = min(5, len(df_summary) // 3)
    if window > 1:
        df_summary['mean_iou_smooth'] = df_summary['mean_iou'].rolling(window=window, center=True).mean()
        df_summary['map_50_smooth'] = df_summary['map_50'].rolling(window=window, center=True).mean()
    
    # Find peaks and trends
    max_iou_idx = df_summary['mean_iou'].idxmax()
    max_map_idx = df_summary['map_50'].idxmax()
    
    # Check for overfitting indicators
    indicators = []
    
    # 1. Performance degradation after peak
    if max_iou_idx < len(df_summary) - 1:
        final_iou = df_summary['mean_iou'].iloc[-1]
        peak_iou = df_summary.loc[max_iou_idx, 'mean_iou']
        iou_degradation = peak_iou - final_iou
        degradation_pct = (iou_degradation / peak_iou) * 100
        
        if iou_degradation > 0.1:
            indicators.append(f"Significant IoU degradation: {iou_degradation:.4f} ({degradation_pct:.1f}%)")
            analysis['severity'] = 'severe'
            analysis['overfitting_detected'] = True
        elif iou_degradation > 0.05:
            indicators.append(f"Moderate IoU degradation: {iou_degradation:.4f} ({degradation_pct:.1f}%)")
            analysis['severity'] = 'moderate'
            analysis['overfitting_detected'] = True
        else:
            indicators.append(f"Minimal IoU degradation: {iou_degradation:.4f} ({degradation_pct:.1f}%)")
    
    # 2. High variance in later iterations
    if len(df_summary) > 10:
        early_std = df_summary['mean_iou'].iloc[:len(df_summary)//3].std()
        late_std = df_summary['mean_iou'].iloc[-len(df_summary)//3:].std()
        
        if late_std > early_std * 1.5:
            indicators.append(f"High variance in later iterations (early: {early_std:.4f}, late: {late_std:.4f})")
            analysis['overfitting_detected'] = True
    
    # 3. Check for oscillation
    if len(df_summary) > 5:
        recent_values = df_summary['mean_iou'].iloc[-5:]
        oscillations = np.abs(np.diff(recent_values))
        avg_oscillation = np.mean(oscillations)
        
        if avg_oscillation > 0.1:
            indicators.append(f"High oscillation: {avg_oscillation:.4f} (learning rate may be too high)")
            analysis['overfitting_detected'] = True
    
    analysis['indicators'] = indicators
    return analysis

def analyze_convergence(df_summary: pd.DataFrame) -> dict:
    """
    Analyze model convergence.
    """
    analysis = {
        'converged': False,
        'convergence_ratio': 0.0,
        'recommendations': []
    }
    
    if len(df_summary) < 2:
        return analysis
    
    final_performance = df_summary['mean_iou'].iloc[-1]
    max_performance = df_summary['mean_iou'].max()
    convergence_ratio = final_performance / max_performance
    
    analysis['convergence_ratio'] = convergence_ratio
    
    if convergence_ratio < 0.8:
        analysis['recommendations'].append("Model may not have converged - consider training longer")
    elif convergence_ratio > 0.95:
        analysis['converged'] = True
        analysis['recommendations'].append("Model appears to have converged well")
    else:
        analysis['recommendations'].append("Model shows moderate convergence")
    
    return analysis

def analyze_training_dynamics(df_summary: pd.DataFrame) -> dict:
    """
    Analyze training dynamics and stability.
    """
    analysis = {
        'stability': 'unknown',
        'learning_rate_analysis': {},
        'recommendations': []
    }
    
    if len(df_summary) < 3:
        return analysis
    
    # Analyze learning rate impact
    early_performance = df_summary['mean_iou'].iloc[:len(df_summary)//3].mean()
    mid_performance = df_summary['mean_iou'].iloc[len(df_summary)//3:2*len(df_summary)//3].mean()
    late_performance = df_summary['mean_iou'].iloc[2*len(df_summary)//3:].mean()
    
    analysis['learning_rate_analysis'] = {
        'early': early_performance,
        'mid': mid_performance,
        'late': late_performance
    }
    
    # Stability analysis
    std_performance = df_summary['mean_iou'].std()
    if std_performance < 0.05:
        analysis['stability'] = 'stable'
    elif std_performance < 0.1:
        analysis['stability'] = 'moderate'
    else:
        analysis['stability'] = 'unstable'
    
    # Learning rate recommendations
    if late_performance < mid_performance * 0.9:
        analysis['recommendations'].append("Consider reducing learning rate")
    elif late_performance > mid_performance * 1.1:
        analysis['recommendations'].append("Learning rate might be too conservative")
    
    return analysis

def analyze_model_specific_characteristics(model_dir: str) -> dict:
    """
    Analyze model-specific characteristics based on directory name and configuration.
    """
    analysis = {
        'model_type': 'unknown',
        'learning_rate': 'unknown',
        'frame_count': 'unknown',
        'recommendations': []
    }
    
    model_name = os.path.basename(model_dir)
    
    # Detect model type
    if 'unmasked' in model_name.lower():
        analysis['model_type'] = 'unmasked'
        analysis['recommendations'].extend([
            "Unmasked model detected - consider using masked training data if available",
            "Consider adjusting loss weights for better mask learning",
            "Consider increasing data augmentation"
        ])
    
    # Detect learning rate
    if '0.0001' in model_name:
        analysis['learning_rate'] = '0.0001'
        analysis['recommendations'].extend([
            "Learning rate 0.0001 detected - consider trying 0.00005 for more stable training",
            "Consider using learning rate scheduling"
        ])
    elif '0.00001' in model_name:
        analysis['learning_rate'] = '0.00001'
        analysis['recommendations'].append("Learning rate 0.00001 detected - may be too conservative")
    
    # Detect frame count
    if '15f' in model_name:
        analysis['frame_count'] = '15'
        analysis['recommendations'].extend([
            "15-frame model detected - consider testing with different frame numbers (5, 10, 20)",
            "Consider adjusting temporal consistency loss"
        ])
    
    return analysis

def create_model_performance_report(df_summary: pd.DataFrame, output_dir: str, model_dir: str) -> None:
    """
    Create comprehensive model performance report.
    """
    report_path = os.path.join(output_dir, 'model_performance_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
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
        for metric in ['mean_iou', 'map_50', 'map_75']:
            if metric in df_summary.columns:
                best_idx = df_summary[metric].idxmax()
                best_iter = df_summary.loc[best_idx, 'iteration']
                best_value = df_summary.loc[best_idx, metric]
                metric_name = {
                    'mean_iou': 'Mean IoU',
                    'map_50': 'mAP@0.5',
                    'map_75': 'mAP@0.75'
                }.get(metric, metric)
                f.write(f"{metric_name}: {best_value:.4f} at iteration {best_iter:.0f}\n")
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
        f.write(f"Early performance: {dynamics_analysis['learning_rate_analysis']['early']:.4f}\n")
        f.write(f"Mid performance: {dynamics_analysis['learning_rate_analysis']['mid']:.4f}\n")
        f.write(f"Late performance: {dynamics_analysis['learning_rate_analysis']['late']:.4f}\n")
        
        for rec in dynamics_analysis['recommendations']:
            f.write(f"• {rec}\n")
        f.write("\n")
        
        # Model-specific analysis
        model_analysis = analyze_model_specific_characteristics(model_dir)
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
        f.write("Report generated by analyze_checkpoint_results.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Model performance report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive checkpoint analysis')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing checkpoint evaluations')
    parser.add_argument('--val-json', type=str, default='/data/fishway_ytvis/val.json',
                       help='Path to validation JSON file')
    parser.add_argument('--run-mask-metrics', action='store_true',
                       help='Run mask metrics analysis for checkpoints that don\'t have it')
    parser.add_argument('--skip-mask-metrics', action='store_true',
                       help='Skip mask metrics analysis and only create plots')
    parser.add_argument('--confidence-threshold', type=float, default=0.05,
                       help='Confidence threshold for filtering predictions')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Run in fast mode (skip expensive operations)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip analysis if mask metrics CSV already exists')
    parser.add_argument('--analysis-level', choices=['basic', 'comprehensive'], default='comprehensive',
                       help='Level of analysis to perform')
    
    args = parser.parse_args()
    
    # Find all checkpoint result directories
    checkpoint_dirs = find_checkpoint_results(args.model_dir)
    logger.info(f"Found {len(checkpoint_dirs)} checkpoint result directories")
    
    if not checkpoint_dirs:
        logger.error("No checkpoint result directories found!")
        return
    
    # Run mask metrics analysis if requested
    if args.run_mask_metrics and not args.skip_mask_metrics:
        logger.info("Running mask metrics analysis for checkpoints...")
        successful_analyses = 0
        
        for i, checkpoint_dir in enumerate(checkpoint_dirs):
            logger.info(f"Processing {i+1}/{len(checkpoint_dirs)}: {os.path.basename(checkpoint_dir)}")
            
            # Check if we should skip this checkpoint
            csv_path = os.path.join(checkpoint_dir, "inference", "mask_metrics.csv")
            if args.skip_existing and os.path.exists(csv_path):
                logger.info(f"Skipping {os.path.basename(checkpoint_dir)} - CSV already exists")
                continue
            
            success = run_mask_metrics_analysis(checkpoint_dir, args.val_json, args.confidence_threshold, args.fast_mode)
            if success:
                successful_analyses += 1
        
        logger.info(f"Successfully analyzed {successful_analyses}/{len(checkpoint_dirs)} checkpoints")
    
    # Collect summary data
    logger.info("Collecting summary data...")
    df_summary = collect_summary_data(args.model_dir)
    
    if df_summary.empty:
        logger.error("No summary data found. Make sure mask metrics analysis has been run.")
        return
    
    # Create analysis based on level
    if args.analysis_level == 'comprehensive':
        logger.info("Creating comprehensive analysis...")
        create_comprehensive_analysis(df_summary, args.model_dir)
    else:
        logger.info("Creating basic summary...")
        create_basic_summary_plot(df_summary, args.model_dir)
    
    logger.info("Analysis complete! Check the generated plots and CSV files.")

if __name__ == "__main__":
    main() 
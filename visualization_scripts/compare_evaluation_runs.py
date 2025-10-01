#!/usr/bin/env python3
"""
Script to compare evaluation results across different runs.
This script analyzes the performance of different test configurations
and generates comparison plots and reports.
"""

import os
import json
import yaml
import argparse
import logging
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_evaluation_runs(model_dir: str) -> List[str]:
    """
    Find all evaluation runs in the checkpoint_evaluations directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        List of run directories
    """
    checkpoint_eval_dir = os.path.join(model_dir, "checkpoint_evaluations")
    if not os.path.exists(checkpoint_eval_dir):
        logger.error(f"Checkpoint evaluations directory not found: {checkpoint_eval_dir}")
        return []
    
    runs = []
    for item in os.listdir(checkpoint_eval_dir):
        item_path = os.path.join(checkpoint_eval_dir, item)
        if os.path.isdir(item_path) and item.startswith('run_'):
            runs.append(item_path)
    
    # Sort by run number
    runs.sort(key=lambda x: int(os.path.basename(x).replace('run_', '')))
    return runs

def load_run_config(run_dir: str) -> Dict[str, Any]:
    """
    Load configuration for a specific run.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary containing configuration
    """
    config_file = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_file):
        logger.warning(f"Config file not found: {config_file}")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def extract_config_changes(run_dir: str) -> Dict[str, Any]:
    """
    Extract configuration changes from config_diff.txt.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary of configuration changes
    """
    diff_file = os.path.join(run_dir, "config_diff.txt")
    if not os.path.exists(diff_file):
        return {}
    
    changes = {}
    with open(diff_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ADDED:') or line.startswith('REMOVED:'):
                # Parse the change
                parts = line.split(':', 1)
                if len(parts) == 2:
                    change_type = parts[0].strip()
                    change_value = parts[1].strip()
                    
                    # Extract parameter name and value
                    if ':' in change_value:
                        param_name, param_value = change_value.split(':', 1)
                        param_name = param_name.strip()
                        param_value = param_value.strip()
                        
                        if change_type == 'ADDED':
                            changes[param_name] = param_value
                        elif change_type == 'REMOVED':
                            changes[f"REMOVED_{param_name}"] = param_value
    
    return changes

def load_run_results(run_dir):
    """
    Load evaluation results for a specific run.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary with run results
    """
    results = {
        'bbox_AP': 0,
        'segm_AP': 0,
        'mask_metrics': {}
    }
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint_'):
            checkpoint_dirs.append(item_path)
    
    if not checkpoint_dirs:
        logger.warning(f"No checkpoint directories found in {run_dir}")
        return results
    
    # Load results from the first checkpoint (they should be similar across checkpoints)
    first_checkpoint = sorted(checkpoint_dirs)[0]
    mask_metrics_file = os.path.join(first_checkpoint, 'inference', 'mask_metrics.csv')
    
    if os.path.exists(mask_metrics_file):
        try:
            import pandas as pd
            df = pd.read_csv(mask_metrics_file)
            
            # Extract metrics from the first row (they should be consistent across frames)
            if len(df) > 0:
                first_row = df.iloc[0]
                
                # Extract AP values
                results['bbox_AP'] = first_row.get('mAP@0.5', 0)
                results['segm_AP'] = first_row.get('mAP@0.5', 0)  # Use same value for segmentation
                
                # Extract mask metrics
                results['mask_metrics'] = {
                    'mean_iou': first_row.get('dataset_IoU', 0),
                    'mean_dice': 0,  # Not available in this format
                    'precision': 0,  # Not available in this format
                    'recall': 0      # Not available in this format
                }
                
                logger.debug(f"Loaded results from {mask_metrics_file}")
                logger.debug(f"AP@0.5: {results['segm_AP']:.4f}, Mean IoU: {results['mask_metrics']['mean_iou']:.4f}")
                
        except Exception as e:
            logger.error(f"Error loading mask metrics from {mask_metrics_file}: {e}")
    else:
        logger.warning(f"Mask metrics file not found: {mask_metrics_file}")
    
    return results

def create_config_summary(runs: List[str]) -> pd.DataFrame:
    """
    Create a summary of configurations across all runs.
    
    Args:
        runs: List of run directories
        
    Returns:
        DataFrame with configuration summary
    """
    config_summary = []
    
    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        config = load_run_config(run_dir)
        changes = extract_config_changes(run_dir)
        
        # Extract key configuration parameters
        summary = {
            'run': run_name,
            'config_file': os.path.join(run_name, "config.yaml"),
            'changes_file': os.path.join(run_name, "config_diff.txt"),
        }
        
        # Add configuration parameters
        if config:
            mask_former_test = config.get('MODEL', {}).get('MASK_FORMER', {}).get('TEST', {})
            input_config = config.get('INPUT', {})
            test_aug = config.get('TEST', {}).get('AUG', {})
            
            summary.update({
                'INSTANCE_ON': mask_former_test.get('INSTANCE_ON', 'N/A'),
                'SEMANTIC_ON': mask_former_test.get('SEMANTIC_ON', 'N/A'),
                'MAX_NUM': mask_former_test.get('MAX_NUM', 'N/A'),
                'OBJECT_MASK_THRESHOLD': mask_former_test.get('OBJECT_MASK_THRESHOLD', 'N/A'),
                'OVERLAP_THRESHOLD': mask_former_test.get('OVERLAP_THRESHOLD', 'N/A'),
                'WINDOW_SIZE': mask_former_test.get('WINDOW_SIZE', 'N/A'),
                'MIN_SIZE_TEST': input_config.get('MIN_SIZE_TEST', 'N/A'),
                'MAX_SIZE_TEST': input_config.get('MAX_SIZE_TEST', 'N/A'),
                'TEST_AUG_ENABLED': test_aug.get('ENABLED', 'N/A'),
                'TEST_AUG_FLIP': test_aug.get('FLIP', 'N/A'),
                'TEST_AUG_MIN_SIZES': str(test_aug.get('MIN_SIZES', 'N/A')),
            })
        
        # Add changes summary
        if changes:
            summary['config_changes'] = ', '.join([f"{k}={v}" for k, v in changes.items()])
        else:
            summary['config_changes'] = 'No changes (baseline)'
        
        config_summary.append(summary)
    
    return pd.DataFrame(config_summary)

def create_performance_comparison(runs: List[str]) -> pd.DataFrame:
    """
    Create a performance comparison DataFrame for all runs.
    
    Args:
        runs: List of run directories
        
    Returns:
        DataFrame with performance comparison
    """
    comparison_data = []
    
    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        run_results = load_run_results(run_dir)
        
        # Get the best performing checkpoint for this run
        comparison_data.append({
            'run': run_name,
            'bbox_AP': run_results['bbox_AP'],
            'segm_AP': run_results['segm_AP'],
            'mean_IoU': run_results['mask_metrics'].get('mean_iou', 0),
            'mean_Dice': run_results['mask_metrics'].get('mean_dice', 0),
            'mean_Precision': run_results['mask_metrics'].get('precision', 0),
            'mean_Recall': run_results['mask_metrics'].get('recall', 0)
        })
    
    return pd.DataFrame(comparison_data)

def create_comparison_plots(config_summary: pd.DataFrame, performance_comparison: pd.DataFrame, output_dir: str):
    """
    Create comparison plots for evaluation runs.
    
    Args:
        config_summary: DataFrame with configuration summary
        performance_comparison: DataFrame with performance comparison
        output_dir: Output directory for plots
    """
    if performance_comparison.empty:
        logger.warning("No performance data to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Evaluation Run Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Bounding Box AP comparison
    bars1 = ax1.bar(performance_comparison['run'], performance_comparison['bbox_AP'])
    ax1.set_title('Bounding Box AP Comparison')
    ax1.set_ylabel('AP')
    ax1.set_xlabel('Run')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Segmentation AP comparison
    bars2 = ax2.bar(performance_comparison['run'], performance_comparison['segm_AP'])
    ax2.set_title('Segmentation AP Comparison')
    ax2.set_ylabel('AP')
    ax2.set_xlabel('Run')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 3: Mean IoU comparison
    bars3 = ax3.bar(performance_comparison['run'], performance_comparison['mean_IoU'])
    ax3.set_title('Mean IoU Comparison')
    ax3.set_ylabel('IoU')
    ax3.set_xlabel('Run')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 4: Configuration heatmap (if config data available)
    if not config_summary.empty and len(config_summary.columns) > 1:
        # Create a simple configuration comparison
        config_cols = [col for col in config_summary.columns if col != 'run']
        if config_cols:
            # Convert configuration data to numeric for heatmap
            config_data = config_summary[config_cols].copy()
            
            # Convert boolean and string values to numeric
            for col in config_data.columns:
                config_data[col] = config_data[col].map({
                    True: 1, False: 0, 'true': 1, 'false': 0,
                    'N/A': np.nan, '': np.nan
                }).fillna(config_data[col])
                
                # Try to convert remaining values to numeric
                try:
                    config_data[col] = pd.to_numeric(config_data[col], errors='coerce')
                except:
                    # If conversion fails, create a categorical mapping
                    unique_vals = config_data[col].dropna().unique()
                    if len(unique_vals) > 0:
                        val_map = {val: i for i, val in enumerate(unique_vals)}
                        config_data[col] = config_data[col].map(val_map)
            
            # Transpose for heatmap
            config_data = config_data.T
            
            # Check if we have valid numeric data
            if not config_data.empty and config_data.select_dtypes(include=[np.number]).shape[1] > 0:
                im = ax4.imshow(config_data, cmap='viridis', aspect='auto')
                ax4.set_title('Configuration Comparison')
                ax4.set_ylabel('Parameter')
                ax4.set_xlabel('Run')
                ax4.set_xticks(range(len(config_summary)))
                ax4.set_xticklabels(config_summary['run'], rotation=45)
                ax4.set_yticks(range(len(config_cols)))
                ax4.set_yticklabels(config_cols)
                
                # Add colorbar
                plt.colorbar(im, ax=ax4)
            else:
                ax4.text(0.5, 0.5, 'No valid configuration data for heatmap', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Configuration Comparison')
        else:
            ax4.text(0.5, 0.5, 'No configuration data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Configuration Comparison')
    else:
        ax4.text(0.5, 0.5, 'No configuration data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Configuration Comparison')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance comparison plot saved to: {plot_path}")
    
    # Create a heatmap of performance metrics
    if not performance_comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select numeric columns for heatmap
        numeric_cols = ['bbox_AP', 'segm_AP', 'mean_IoU', 'mean_Dice', 'mean_Precision', 'mean_Recall']
        available_cols = [col for col in numeric_cols if col in performance_comparison.columns]
        
        if available_cols:
            heatmap_data = performance_comparison[available_cols].T
            im = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title('Performance Metrics Heatmap')
            ax.set_xlabel('Run')
            ax.set_ylabel('Metric')
            
            plt.tight_layout()
            
            # Save the heatmap
            heatmap_path = os.path.join(output_dir, 'configuration_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Configuration heatmap saved to: {heatmap_path}")
    
    # Create performance trends plot
    if not performance_comparison.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot multiple metrics on the same graph
        x = range(len(performance_comparison))
        
        if 'segm_AP' in performance_comparison.columns:
            ax.plot(x, performance_comparison['segm_AP'], 'o-', label='Segmentation AP', linewidth=2, markersize=8)
        
        if 'bbox_AP' in performance_comparison.columns:
            ax.plot(x, performance_comparison['bbox_AP'], 's-', label='Bounding Box AP', linewidth=2, markersize=8)
        
        if 'mean_IoU' in performance_comparison.columns:
            ax.plot(x, performance_comparison['mean_IoU'], '^-', label='Mean IoU', linewidth=2, markersize=8)
        
        ax.set_xlabel('Run')
        ax.set_ylabel('Performance Metric')
        ax.set_title('Performance Trends Across Runs')
        ax.set_xticks(x)
        ax.set_xticklabels(performance_comparison['run'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the trends plot
        trends_path = os.path.join(output_dir, 'performance_trends.png')
        plt.savefig(trends_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance trends plot saved to: {trends_path}")

def create_summary_report(config_summary: pd.DataFrame, performance_comparison: pd.DataFrame, output_dir: str):
    """
    Create a summary report comparing all evaluation runs.
    
    Args:
        config_summary: DataFrame with configuration summary
        performance_comparison: DataFrame with performance comparison
        output_dir: Output directory for report
    """
    report_path = os.path.join(output_dir, 'evaluation_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RUN COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runs analyzed: {len(performance_comparison)}\n\n")
        
        # Find best performing run
        if not performance_comparison.empty and 'segm_AP' in performance_comparison.columns:
            best_idx = performance_comparison['segm_AP'].idxmax()
            best_run = performance_comparison.loc[best_idx]
            
            f.write("BEST PERFORMING RUN\n")
            f.write("-" * 40 + "\n")
            f.write(f"Run: {best_run['run']}\n")
            f.write(f"Segmentation AP: {best_run['segm_AP']:.4f}\n")
            f.write(f"Bounding Box AP: {best_run['bbox_AP']:.4f}\n")
            if 'mean_IoU' in best_run:
                f.write(f"Mean IoU: {best_run['mean_IoU']:.4f}\n")
            f.write("\n")
        
        # Configuration summary
        if not config_summary.empty:
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            for _, row in config_summary.iterrows():
                f.write(f"\nRun: {row['run']}\n")
                for col in config_summary.columns:
                    if col != 'run':
                        f.write(f"  {col}: {row[col]}\n")
            f.write("\n")
        
        # Performance comparison table
        if not performance_comparison.empty:
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write("Run\t\tSegm AP\tBBox AP\tMean IoU\n")
            f.write("-" * 40 + "\n")
            
            for _, row in performance_comparison.iterrows():
                segm_ap = row.get('segm_AP', 0)
                bbox_ap = row.get('bbox_AP', 0)
                mean_iou = row.get('mean_IoU', 0)
                f.write(f"{row['run']}\t\t{segm_ap:.4f}\t{bbox_ap:.4f}\t{mean_iou:.4f}\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if not performance_comparison.empty and 'segm_AP' in performance_comparison.columns:
            # Find the best configuration
            best_idx = performance_comparison['segm_AP'].idxmax()
            best_run_name = performance_comparison.loc[best_idx, 'run']
            
            f.write(f"1. Best performing configuration: {best_run_name}\n")
            f.write(f"   - Use this configuration for production inference\n\n")
            
            # Analyze configuration differences
            if not config_summary.empty and len(config_summary) > 1:
                f.write("2. Configuration Analysis:\n")
                
                # Compare with baseline (run_1 if it exists)
                baseline_run = None
                for _, row in config_summary.iterrows():
                    if row['run'] == 'run_1':
                        baseline_run = row
                        break
                
                if baseline_run is not None:
                    f.write("   - Comparing against baseline (run_1):\n")
                    for _, row in config_summary.iterrows():
                        if row['run'] != 'run_1':
                            f.write(f"   - {row['run']} changes:\n")
                            for col in config_summary.columns:
                                if col != 'run' and row[col] != baseline_run[col]:
                                    f.write(f"     * {col}: {baseline_run[col]} → {row[col]}\n")
                            f.write("\n")
        
        f.write("3. Next Steps:\n")
        f.write("   - Review the generated plots for detailed analysis\n")
        f.write("   - Consider running additional configurations if needed\n")
        f.write("   - Validate the best configuration on a separate test set\n")
        f.write("   - Document the optimal configuration for future use\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generated by compare_evaluation_runs.py\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare evaluation runs and analyze configuration impact')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing checkpoint_evaluations')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for comparison results (defaults to model_dir/evaluation_comparison)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "evaluation_comparison")
    
    # Find all evaluation runs
    runs = find_evaluation_runs(args.model_dir)
    if not runs:
        logger.error("No evaluation runs found!")
        return
    
    logger.info(f"Found {len(runs)} evaluation runs: {[os.path.basename(r) for r in runs]}")
    
    # Create configuration summary
    logger.info("Creating configuration summary...")
    config_summary = create_config_summary(runs)
    
    # Create performance comparison
    logger.info("Creating performance comparison...")
    performance_comparison = create_performance_comparison(runs)
    
    if performance_comparison.empty:
        logger.error("No performance data found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save data to CSV
    config_summary.to_csv(os.path.join(args.output_dir, 'configuration_summary.csv'), index=False)
    performance_comparison.to_csv(os.path.join(args.output_dir, 'performance_comparison.csv'), index=False)
    
    # Create plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(config_summary, performance_comparison, args.output_dir)
    
    # Create summary report
    logger.info("Creating summary report...")
    create_summary_report(config_summary, performance_comparison, args.output_dir)
    
    logger.info(f"Comparison complete! Results saved to: {args.output_dir}")
    logger.info("Files created:")
    logger.info(f"  - {args.output_dir}/configuration_summary.csv")
    logger.info(f"  - {args.output_dir}/performance_comparison.csv")
    logger.info(f"  - {args.output_dir}/performance_comparison.png")
    logger.info(f"  - {args.output_dir}/configuration_heatmap.png")
    logger.info(f"  - {args.output_dir}/performance_trends.png")
    logger.info(f"  - {args.output_dir}/evaluation_comparison_report.txt")

if __name__ == "__main__":
    main()

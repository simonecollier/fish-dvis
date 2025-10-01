#!/usr/bin/env python3
"""
Compare two model performances using analysis results from run_analysis_all_checkpoint_results.sh.
This script will:
1. Load analysis results from two model directories
2. Compare performance metrics and training dynamics
3. Analyze configuration differences
4. Generate comprehensive comparison report and plots

Usage:
    python compare_models.py --model1-dir /path/to/model1 --model2-dir /path/to/model2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import yaml
import re
from pathlib import Path

def load_model_analysis_data(model_dir: str) -> dict:
    """
    Load analysis data from a model directory.
    
    Args:
        model_dir: Path to model directory containing analysis results
        
    Returns:
        Dictionary containing analysis data
    """
    data = {
        'summary_csv': None,
        'performance_report': None,
        'config_file': None,
        'model_name': os.path.basename(model_dir)
    }
    
    # Look for comprehensive metrics summary CSV
    csv_path = os.path.join(model_dir, 'comprehensive_metrics_summary.csv')
    if os.path.exists(csv_path):
        data['summary_csv'] = pd.read_csv(csv_path)
    
    # Look for model performance report
    report_path = os.path.join(model_dir, 'model_performance_report.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data['performance_report'] = f.read()
    
    # Look for config file
    config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['config_file'] = yaml.safe_load(f)
    
    return data

def extract_config_differences(config1: dict, config2: dict) -> dict:
    """
    Extract ALL differences between two configuration files.
    This function recursively compares every parameter to ensure comprehensive tracking.
    """
    def flatten_dict(d, parent_key='', sep='.'):
        """Flatten nested dictionary with dot notation."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def compare_values(val1, val2):
        """Compare two values, handling different types."""
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            return list(val1) != list(val2)
        elif isinstance(val1, dict) and isinstance(val2, dict):
            return val1 != val2
        else:
            return val1 != val2
    
    if not config1 or not config2:
        return {'error': 'One or both configs are None'}
    
    # Flatten both configs
    flat_config1 = flatten_dict(config1)
    flat_config2 = flatten_dict(config2)
    
    # Get all unique keys
    all_keys = set(flat_config1.keys()) | set(flat_config2.keys())
    
    differences = {}
    
    for key in sorted(all_keys):
        val1 = flat_config1.get(key, None)
        val2 = flat_config2.get(key, None)
        
        if compare_values(val1, val2):
            differences[key] = {
                'model1': val1,
                'model2': val2,
                'different': True
            }
    
    return differences

def compare_model_performance(df1: pd.DataFrame, df2: pd.DataFrame, model1_name: str, model2_name: str) -> dict:
    """
    Compare performance metrics between two models.
    """
    comparison = {
        'best_performance': {},
        'training_dynamics': {},
        'convergence': {},
        'stability': {}
    }
    
    # Best performance comparison
    for metric in ['mean_iou', 'map_50', 'map_75']:
        if metric in df1.columns and metric in df2.columns:
            best1 = df1[metric].max()
            best2 = df2[metric].max()
            improvement = ((best1 - best2) / best2) * 100 if best2 > 0 else 0
            
            comparison['best_performance'][metric] = {
                'model1': best1,
                'model2': best2,
                'improvement': improvement,
                'better_model': model1_name if best1 > best2 else model2_name
            }
    
    # Training dynamics
    if len(df1) > 0 and len(df2) > 0:
        # Convergence analysis
        final1 = df1['mean_iou'].iloc[-1] if 'mean_iou' in df1.columns else 0
        final2 = df2['mean_iou'].iloc[-1] if 'mean_iou' in df2.columns else 0
        max1 = df1['mean_iou'].max() if 'mean_iou' in df1.columns else 0
        max2 = df2['mean_iou'].max() if 'mean_iou' in df2.columns else 0
        
        conv1 = final1 / max1 if max1 > 0 else 0
        conv2 = final2 / max2 if max2 > 0 else 0
        
        comparison['convergence'] = {
            'model1': conv1,
            'model2': conv2,
            'better_converged': model1_name if conv1 > conv2 else model2_name
        }
        
        # Stability analysis
        std1 = df1['mean_iou'].std() if 'mean_iou' in df1.columns else 0
        std2 = df2['mean_iou'].std() if 'mean_iou' in df2.columns else 0
        
        comparison['stability'] = {
            'model1': std1,
            'model2': std2,
            'more_stable': model1_name if std1 < std2 else model2_name
        }
        
        # Training duration
        duration1 = df1['iteration'].max() - df1['iteration'].min()
        duration2 = df2['iteration'].max() - df2['iteration'].min()
        
        comparison['training_dynamics'] = {
            'duration1': duration1,
            'duration2': duration2,
            'longer_training': model1_name if duration1 > duration2 else model2_name
        }
    
    return comparison

def create_comprehensive_comparison_plots(df1: pd.DataFrame, df2: pd.DataFrame, 
                                        model1_name: str, model2_name: str, output_dir: str) -> None:
    """
    Create comprehensive comparison plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Comparison: {model1_name} vs {model2_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: IoU comparison
    if 'mean_iou' in df1.columns and 'mean_iou' in df2.columns:
        axes[0, 0].plot(df1['iteration'], df1['mean_iou'], 'b-', linewidth=2, label=model1_name)
        axes[0, 0].plot(df2['iteration'], df2['mean_iou'], 'r-', linewidth=2, label=model2_name)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Mean IoU')
        axes[0, 0].set_title('IoU Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot 2: mAP@0.5 comparison
    if 'map_50' in df1.columns and 'map_50' in df2.columns:
        axes[0, 1].plot(df1['iteration'], df1['map_50'], 'b-', linewidth=2, label=model1_name)
        axes[0, 1].plot(df2['iteration'], df2['map_50'], 'r-', linewidth=2, label=model2_name)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('mAP@0.5')
        axes[0, 1].set_title('mAP@0.5 Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot 3: Combined score comparison
    if 'mean_iou' in df1.columns and 'map_50' in df1.columns and 'mean_iou' in df2.columns and 'map_50' in df2.columns:
        df1['combined_score'] = df1['mean_iou'] + df1['map_50']
        df2['combined_score'] = df2['mean_iou'] + df2['map_50']
        
        axes[0, 2].plot(df1['iteration'], df1['combined_score'], 'b-', linewidth=2, label=model1_name)
        axes[0, 2].plot(df2['iteration'], df2['combined_score'], 'r-', linewidth=2, label=model2_name)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Combined Score (IoU + mAP@0.5)')
        axes[0, 2].set_title('Combined Performance Score')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
    
    # Plot 4: Best performance comparison
    models = [model1_name, model2_name]
    best_ious = [df1['mean_iou'].max() if 'mean_iou' in df1.columns else 0, 
                 df2['mean_iou'].max() if 'mean_iou' in df2.columns else 0]
    best_maps = [df1['map_50'].max() if 'map_50' in df1.columns else 0, 
                 df2['map_50'].max() if 'map_50' in df2.columns else 0]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, best_ious, width, label='Best IoU', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, best_maps, width, label='Best mAP@0.5', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Best Performance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Training duration comparison
    duration1 = df1['iteration'].max() - df1['iteration'].min() if len(df1) > 0 else 0
    duration2 = df2['iteration'].max() - df2['iteration'].min() if len(df2) > 0 else 0
    
    axes[1, 1].bar(['Model 1', 'Model 2'], [duration1, duration2], color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Training Duration (iterations)')
    axes[1, 1].set_title('Training Duration Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Stability comparison
    std1 = df1['mean_iou'].std() if 'mean_iou' in df1.columns else 0
    std2 = df2['mean_iou'].std() if 'mean_iou' in df2.columns else 0
    
    axes[1, 2].bar(['Model 1', 'Model 2'], [std1, std2], color=['blue', 'red'], alpha=0.7)
    axes[1, 2].set_ylabel('Standard Deviation (lower = more stable)')
    axes[1, 2].set_title('Training Stability Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive comparison plots saved to: {plot_path}")

def create_comparison_report(data1: dict, data2: dict, comparison: dict, 
                           config_diff: dict, output_dir: str) -> None:
    """
    Create comprehensive comparison report.
    """
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model information
        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model 1: {data1['model_name']}\n")
        f.write(f"Model 2: {data2['model_name']}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration differences
        f.write("CONFIGURATION DIFFERENCES\n")
        f.write("-" * 40 + "\n")
        
        if 'error' in config_diff:
            f.write(f"Error: {config_diff['error']}\n\n")
        elif not config_diff:
            f.write("No configuration differences detected.\n\n")
        else:
            f.write(f"Found {len(config_diff)} parameter differences:\n\n")
            
            # Group differences by category for better readability
            categories = {
                'SOLVER': [],
                'INPUT': [],
                'MODEL': [],
                'DATASETS': [],
                'DATALOADER': [],
                'TEST': [],
                'OTHER': []
            }
            
            for param, diff in config_diff.items():
                if diff['different']:
                    category = param.split('.')[0] if '.' in param else 'OTHER'
                    if category in categories:
                        categories[category].append((param, diff))
                    else:
                        categories['OTHER'].append((param, diff))
            
            # Print differences by category
            for category, params in categories.items():
                if params:
                    f.write(f"{category} PARAMETERS:\n")
                    f.write("-" * 20 + "\n")
                    
                    for param, diff in params:
                        # Format parameter name for display
                        param_display = param.replace('_', ' ').title()
                        if '.' in param:
                            param_display = param.split('.', 1)[1].replace('_', ' ').title()
                        
                        f.write(f"{param_display}:\n")
                        
                        # Handle different value types for better display
                        val1 = diff['model1']
                        val2 = diff['model2']
                        
                        if isinstance(val1, (list, tuple)) or isinstance(val2, (list, tuple)):
                            f.write(f"  • {data1['model_name']}: {val1}\n")
                            f.write(f"  • {data2['model_name']}: {val2}\n")
                        elif isinstance(val1, bool) or isinstance(val2, bool):
                            bool1 = "True" if val1 else "False"
                            bool2 = "True" if val2 else "False"
                            f.write(f"  • {data1['model_name']}: {bool1}\n")
                            f.write(f"  • {data2['model_name']}: {bool2}\n")
                        elif val1 is None:
                            f.write(f"  • {data1['model_name']}: None\n")
                            f.write(f"  • {data2['model_name']}: {val2}\n")
                        elif val2 is None:
                            f.write(f"  • {data1['model_name']}: {val1}\n")
                            f.write(f"  • {data2['model_name']}: None\n")
                        else:
                            f.write(f"  • {data1['model_name']}: {val1}\n")
                            f.write(f"  • {data2['model_name']}: {val2}\n")
                        f.write("\n")
            
            f.write("\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n")
        for metric, perf in comparison['best_performance'].items():
            metric_name = {
                'mean_iou': 'Mean IoU',
                'map_50': 'mAP@0.5',
                'map_75': 'mAP@0.75'
            }.get(metric, metric)
            
            f.write(f"{metric_name}:\n")
            f.write(f"  • {data1['model_name']}: {perf['model1']:.4f}\n")
            f.write(f"  • {data2['model_name']}: {perf['model2']:.4f}\n")
            f.write(f"  • Improvement: {perf['improvement']:.1f}% ({perf['better_model']} is better)\n\n")
        
        # Training dynamics
        f.write("TRAINING DYNAMICS\n")
        f.write("-" * 40 + "\n")
        if comparison['convergence']:
            conv1 = comparison['convergence']['model1']
            conv2 = comparison['convergence']['model2']
            f.write(f"Convergence (final/best ratio):\n")
            f.write(f"  • {data1['model_name']}: {conv1:.2f}\n")
            f.write(f"  • {data2['model_name']}: {conv2:.2f}\n")
            f.write(f"  • Better converged: {comparison['convergence']['better_converged']}\n\n")
        
        if comparison['stability']:
            std1 = comparison['stability']['model1']
            std2 = comparison['stability']['model2']
            f.write(f"Stability (lower std = more stable):\n")
            f.write(f"  • {data1['model_name']}: {std1:.4f}\n")
            f.write(f"  • {data2['model_name']}: {std2:.4f}\n")
            f.write(f"  • More stable: {comparison['stability']['more_stable']}\n\n")
        
        if comparison['training_dynamics']:
            dur1 = comparison['training_dynamics']['duration1']
            dur2 = comparison['training_dynamics']['duration2']
            f.write(f"Training Duration:\n")
            f.write(f"  • {data1['model_name']}: {dur1:.0f} iterations\n")
            f.write(f"  • {data2['model_name']}: {dur2:.0f} iterations\n")
            f.write(f"  • Longer training: {comparison['training_dynamics']['longer_training']}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Best overall model
        better_count = 0
        for metric, perf in comparison['best_performance'].items():
            if perf['better_model'] == data1['model_name']:
                better_count += 1
        
        if better_count > len(comparison['best_performance']) / 2:
            f.write(f"🏆 {data1['model_name']} performs better overall\n")
        elif better_count < len(comparison['best_performance']) / 2:
            f.write(f"🏆 {data2['model_name']} performs better overall\n")
        else:
            f.write("⚖️  Models have different strengths - choose based on priority\n")
        
        # Configuration recommendations
        f.write("\nConfiguration Recommendations:\n")
        
        # Key parameter recommendations
        key_params = {
            'SOLVER.BASE_LR': 'Learning Rate',
            'SOLVER.IMS_PER_BATCH': 'Batch Size',
            'SOLVER.MAX_ITER': 'Training Duration',
            'INPUT.MIN_SIZE_TRAIN': 'Training Resolution',
            'INPUT.MAX_SIZE_TRAIN': 'Max Training Resolution',
            'INPUT.MIN_SIZE_TEST': 'Test Resolution',
            'INPUT.MAX_SIZE_TEST': 'Max Test Resolution'
        }
        
        for param_key, param_name in key_params.items():
            if param_key in config_diff:
                diff = config_diff[param_key]
                val1 = diff['model1']
                val2 = diff['model2']
                
                # Determine which value performed better based on performance comparison
                better_model = None
                if comparison['best_performance']:
                    # Use IoU as primary metric for recommendations
                    if 'mean_iou' in comparison['best_performance']:
                        better_model = comparison['best_performance']['mean_iou']['better_model']
                
                if better_model:
                    better_val = val1 if better_model == data1['model_name'] else val2
                    f.write(f"  💡 {param_name}: {better_val} ({better_model}) performed better\n")
                    f.write(f"  💡 Consider using {better_val} for future training\n")
        
        # Training duration recommendations
        if comparison['training_dynamics']:
            if comparison['training_dynamics']['longer_training'] == data1['model_name']:
                f.write(f"  💡 Longer training duration helped {data1['model_name']}\n")
                f.write(f"  💡 Consider training for more iterations in future\n")
            else:
                f.write(f"  💡 Shorter training was sufficient for {data2['model_name']}\n")
                f.write(f"  💡 Consider early stopping to save time\n")
        
        # Stability recommendations
        if comparison['stability']:
            more_stable = comparison['stability']['more_stable']
            f.write(f"  💡 {more_stable} shows more stable training\n")
            f.write(f"  💡 Consider using similar configuration for production\n")
        
        # Resolution-specific recommendations
        if 'INPUT.MIN_SIZE_TRAIN' in config_diff or 'INPUT.MAX_SIZE_TRAIN' in config_diff:
            f.write(f"  💡 Resolution differences detected - consider impact on performance vs training time\n")
            f.write(f"  💡 Higher resolution may improve accuracy but increase training time\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated by compare_models.py\n")
        f.write("=" * 80 + "\n")
    
    print(f"Comparison report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare two model performances using analysis results')
    parser.add_argument('--model1-dir', type=str, required=True,
                       help='Path to first model directory (containing analysis results)')
    parser.add_argument('--model2-dir', type=str, required=True,
                       help='Path to second model directory (containing analysis results)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Check if model directories exist
    if not os.path.exists(args.model1_dir):
        print(f"Error: Model 1 directory {args.model1_dir} does not exist")
        return
    
    if not os.path.exists(args.model2_dir):
        print(f"Error: Model 2 directory {args.model2_dir} does not exist")
        return
    
    # Load analysis data
    print("Loading analysis data...")
    data1 = load_model_analysis_data(args.model1_dir)
    data2 = load_model_analysis_data(args.model2_dir)
    
    if not data1['summary_csv'] is not None or not data2['summary_csv'] is not None:
        print("Error: Could not find comprehensive_metrics_summary.csv in one or both model directories")
        print("Make sure to run run_analysis_all_checkpoint_results.sh on both models first")
        return
    
    # Compare configurations
    config_diff = extract_config_differences(data1['config_file'], data2['config_file'])
    
    # Compare performance
    comparison = compare_model_performance(data1['summary_csv'], data2['summary_csv'], 
                                        data1['model_name'], data2['model_name'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots and report
    print("Generating comparison plots and report...")
    create_comprehensive_comparison_plots(data1['summary_csv'], data2['summary_csv'],
                                        data1['model_name'], data2['model_name'], args.output_dir)
    
    create_comparison_report(data1, data2, comparison, config_diff, args.output_dir)
    
    print("Model comparison complete!")
    print(f"Results saved to: {args.output_dir}")
    print("  - comprehensive_model_comparison.png (comparison plots)")
    print("  - model_comparison_report.txt (detailed comparison report)")

if __name__ == "__main__":
    main() 
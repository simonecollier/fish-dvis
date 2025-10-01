#!/usr/bin/env python3
"""
Compare a model against the base model for hyperparameter tuning analysis.
This script ensures comprehensive tracking of ALL parameter differences between
the base model and any new model being tested.

Usage:
    python compare_against_base_model.py --new-model-dir /path/to/new_model --base-model-dir /path/to/base_model
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
from compare_models import load_model_analysis_data, extract_config_differences, compare_model_performance

def create_hyperparameter_tuning_report(new_model_data: dict, base_model_data: dict, 
                                      comparison: dict, config_diff: dict, output_dir: str) -> None:
    """
    Create comprehensive hyperparameter tuning report comparing against base model.
    """
    report_path = os.path.join(output_dir, 'hyperparameter_tuning_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model information
        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Base Model: {base_model_data['model_name']}\n")
        f.write(f"New Model: {new_model_data['model_name']}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n")
        
        if comparison['best_performance']:
            f.write("Best Performance Metrics:\n")
            for metric, perf in comparison['best_performance'].items():
                metric_name = {
                    'mean_iou': 'Mean IoU',
                    'map_50': 'mAP@0.5',
                    'map_75': 'mAP@0.75',
                    'mean_boundary_f': 'Boundary F-measure'
                }.get(metric, metric)
                
                improvement = perf['improvement']
                better_model = perf['better_model']
                
                f.write(f"  {metric_name}:\n")
                f.write(f"    • Base Model: {perf['model2']:.4f}\n")
                f.write(f"    • New Model: {perf['model1']:.4f}\n")
                f.write(f"    • Improvement: {improvement:+.1f}% ({better_model} is better)\n")
                
                if better_model == new_model_data['model_name']:
                    f.write(f"    ✅ NEW MODEL IMPROVEMENT\n")
                else:
                    f.write(f"    ⚠️  BASE MODEL BETTER\n")
                f.write("\n")
        else:
            f.write("No performance data available for comparison.\n\n")
        
        # Configuration differences - COMPREHENSIVE
        f.write("CONFIGURATION DIFFERENCES (COMPREHENSIVE)\n")
        f.write("-" * 50 + "\n")
        
        if 'error' in config_diff:
            f.write(f"Error: {config_diff['error']}\n\n")
        elif not config_diff:
            f.write("No configuration differences detected.\n\n")
        else:
            f.write(f"Found {len(config_diff)} parameter differences from base model:\n\n")
            
            # Group differences by category
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
                        base_val = diff['model2']  # Base model is model2
                        new_val = diff['model1']   # New model is model1
                        
                        if isinstance(base_val, (list, tuple)) or isinstance(new_val, (list, tuple)):
                            f.write(f"  • Base Model: {base_val}\n")
                            f.write(f"  • New Model: {new_val}\n")
                        elif isinstance(base_val, bool) or isinstance(new_val, bool):
                            base_bool = "True" if base_val else "False"
                            new_bool = "True" if new_val else "False"
                            f.write(f"  • Base Model: {base_bool}\n")
                            f.write(f"  • New Model: {new_bool}\n")
                        elif base_val is None:
                            f.write(f"  • Base Model: None\n")
                            f.write(f"  • New Model: {new_val}\n")
                        elif new_val is None:
                            f.write(f"  • Base Model: {base_val}\n")
                            f.write(f"  • New Model: None\n")
                        else:
                            f.write(f"  • Base Model: {base_val}\n")
                            f.write(f"  • New Model: {new_val}\n")
                        f.write("\n")
            
            f.write("\n")
        
        # Training dynamics comparison
        f.write("TRAINING DYNAMICS COMPARISON\n")
        f.write("-" * 40 + "\n")
        
        if comparison['convergence']:
            conv_base = comparison['convergence']['model2']
            conv_new = comparison['convergence']['model1']
            f.write(f"Convergence (final/best ratio):\n")
            f.write(f"  • Base Model: {conv_base:.2f}\n")
            f.write(f"  • New Model: {conv_new:.2f}\n")
            if conv_new > conv_base:
                f.write(f"  ✅ New model converges better\n")
            else:
                f.write(f"  ⚠️  Base model converges better\n")
            f.write("\n")
        
        if comparison['stability']:
            std_base = comparison['stability']['model2']
            std_new = comparison['stability']['model1']
            f.write(f"Stability (lower std = more stable):\n")
            f.write(f"  • Base Model: {std_base:.4f}\n")
            f.write(f"  • New Model: {std_new:.4f}\n")
            if std_new < std_base:
                f.write(f"  ✅ New model is more stable\n")
            else:
                f.write(f"  ⚠️  Base model is more stable\n")
            f.write("\n")
        
        if comparison['training_dynamics']:
            dur_base = comparison['training_dynamics']['duration2']
            dur_new = comparison['training_dynamics']['duration1']
            f.write(f"Training Duration:\n")
            f.write(f"  • Base Model: {dur_base:.0f} iterations\n")
            f.write(f"  • New Model: {dur_new:.0f} iterations\n")
            if dur_new > dur_base:
                f.write(f"  ⏱️  New model trained longer\n")
            else:
                f.write(f"  ⏱️  New model trained shorter\n")
            f.write("\n")
        
        # Hyperparameter tuning insights
        f.write("HYPERPARAMETER TUNING INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Overall assessment
        better_count = 0
        total_metrics = len(comparison['best_performance']) if comparison['best_performance'] else 0
        
        for metric, perf in comparison['best_performance'].items():
            if perf['better_model'] == new_model_data['model_name']:
                better_count += 1
        
        if total_metrics > 0:
            improvement_ratio = better_count / total_metrics
            f.write(f"Overall Assessment:\n")
            f.write(f"  • New model performs better on {better_count}/{total_metrics} metrics\n")
            f.write(f"  • Improvement ratio: {improvement_ratio:.1%}\n")
            
            if improvement_ratio > 0.5:
                f.write(f"  🎉 HYPERPARAMETER TUNING SUCCESSFUL\n")
            elif improvement_ratio == 0.5:
                f.write(f"  ⚖️  MIXED RESULTS - Consider specific use case\n")
            else:
                f.write(f"  ⚠️  HYPERPARAMETER TUNING UNSUCCESSFUL\n")
            f.write("\n")
        
        # Key parameter analysis
        f.write("Key Parameter Analysis:\n")
        
        key_params = {
            'SOLVER.BASE_LR': 'Learning Rate',
            'SOLVER.IMS_PER_BATCH': 'Batch Size',
            'SOLVER.MAX_ITER': 'Training Duration',
            'INPUT.MIN_SIZE_TRAIN': 'Training Resolution',
            'INPUT.MAX_SIZE_TRAIN': 'Max Training Resolution',
            'INPUT.MIN_SIZE_TEST': 'Test Resolution',
            'INPUT.MAX_SIZE_TEST': 'Max Test Resolution',
            'INPUT.SAMPLING_FRAME_NUM': 'Frame Sampling',
            'MODEL.MASK_FORMER.TEST.WINDOW_SIZE': 'Inference Window Size'
        }
        
        for param_key, param_name in key_params.items():
            if param_key in config_diff:
                diff = config_diff[param_key]
                base_val = diff['model2']
                new_val = diff['model1']
                
                f.write(f"  {param_name}:\n")
                f.write(f"    • Changed from {base_val} to {new_val}\n")
                
                # Determine if this change was beneficial
                better_model = None
                if comparison['best_performance'] and 'mean_iou' in comparison['best_performance']:
                    better_model = comparison['best_performance']['mean_iou']['better_model']
                
                if better_model == new_model_data['model_name']:
                    f.write(f"    ✅ This change appears beneficial\n")
                elif better_model == base_model_data['model_name']:
                    f.write(f"    ⚠️  This change may be detrimental\n")
                else:
                    f.write(f"    ℹ️  Impact unclear\n")
                f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if total_metrics > 0 and improvement_ratio > 0.5:
            f.write("🎉 SUCCESSFUL HYPERPARAMETER TUNING\n")
            f.write("  • Consider adopting the new configuration\n")
            f.write("  • Document the successful changes for future reference\n")
            f.write("  • Consider further tuning in the same direction\n")
        elif total_metrics > 0 and improvement_ratio < 0.5:
            f.write("⚠️  UNSUCCESSFUL HYPERPARAMETER TUNING\n")
            f.write("  • Revert to base model configuration\n")
            f.write("  • Consider different hyperparameter ranges\n")
            f.write("  • Analyze which specific changes were detrimental\n")
        else:
            f.write("⚖️  INCONCLUSIVE RESULTS\n")
            f.write("  • Need more comprehensive evaluation\n")
            f.write("  • Consider testing on different datasets\n")
            f.write("  • Analyze specific use case requirements\n")
        
        f.write("\n")
        
        # Next steps
        f.write("NEXT STEPS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Review all parameter differences carefully\n")
        f.write("2. Consider the impact of each change individually\n")
        f.write("3. Test the new configuration on validation data\n")
        f.write("4. Document successful changes for reproducibility\n")
        f.write("5. Consider ensemble approaches if both models have strengths\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated by compare_against_base_model.py\n")
        f.write("=" * 80 + "\n")
    
    print(f"Hyperparameter tuning report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare a model against the base model for hyperparameter tuning')
    parser.add_argument('--new-model-dir', type=str, required=True,
                       help='Path to new model directory (containing analysis results)')
    parser.add_argument('--base-model-dir', type=str, required=True,
                       help='Path to base model directory (containing analysis results)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Check if model directories exist
    if not os.path.exists(args.new_model_dir):
        print(f"Error: New model directory {args.new_model_dir} does not exist")
        return
    
    if not os.path.exists(args.base_model_dir):
        print(f"Error: Base model directory {args.base_model_dir} does not exist")
        return
    
    # Load analysis data
    print("Loading analysis data...")
    new_model_data = load_model_analysis_data(args.new_model_dir)
    base_model_data = load_model_analysis_data(args.base_model_dir)
    
    if not new_model_data['summary_csv'] is not None or not base_model_data['summary_csv'] is not None:
        print("Error: Could not find comprehensive_metrics_summary.csv in one or both model directories")
        print("Make sure to run run_analysis_all_checkpoint_results.sh on both models first")
        return
    
    # Compare configurations (comprehensive)
    print("Comparing configurations...")
    config_diff = extract_config_differences(new_model_data['config_file'], base_model_data['config_file'])
    
    # Compare performance (new model is model1, base model is model2)
    print("Comparing performance...")
    comparison = compare_model_performance(new_model_data['summary_csv'], base_model_data['summary_csv'], 
                                        new_model_data['model_name'], base_model_data['model_name'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comprehensive report
    print("Generating hyperparameter tuning report...")
    create_hyperparameter_tuning_report(new_model_data, base_model_data, comparison, config_diff, args.output_dir)
    
    print("Hyperparameter tuning analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("  - hyperparameter_tuning_report.txt (comprehensive analysis)")

if __name__ == "__main__":
    main()

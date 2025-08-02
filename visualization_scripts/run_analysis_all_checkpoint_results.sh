#!/bin/bash

# Comprehensive checkpoint analysis script wrapper
# This script provides a convenient interface to the analyze_checkpoint_results.py script
# Usage: ./run_analysis_all_checkpoint_results.sh <model_dir> [options]

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_dir> [options]"
    echo ""
    echo "Examples:"
    echo "  # Basic analysis of existing results"
    echo "  $0 /path/to/model"
    echo ""
    echo "  # Run mask metrics and then analyze"
    echo "  $0 /path/to/model --run-mask-metrics"
    echo ""
    echo "  # Fast mode for quick analysis"
    echo "  $0 /path/to/model --run-mask-metrics --fast-mode"
    echo ""
    echo "  # Basic summary only"
    echo "  $0 /path/to/model --analysis-level basic"
    echo ""
    echo "  # Skip existing mask metrics"
    echo "  $0 /path/to/model --run-mask-metrics --skip-existing"
    echo ""
    echo "Options:"
    echo "  --run-mask-metrics     Run mask metrics analysis for checkpoints"
    echo "  --skip-mask-metrics    Skip mask metrics analysis"
    echo "  --fast-mode            Run in fast mode (skip expensive operations)"
    echo "  --skip-existing        Skip analysis if mask metrics CSV already exists"
    echo "  --analysis-level       Level of analysis (basic or comprehensive)"
    echo "  --confidence-threshold Confidence threshold for filtering predictions"
    echo "  --val-json            Path to validation JSON file"
    exit 1
fi

MODEL_DIR=$1
shift  # Remove first argument, keep the rest

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

echo "Starting comprehensive checkpoint analysis..."
echo "Model directory: $MODEL_DIR"

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Build command
cmd="python /home/simone/fish-dvis/visualization_scripts/analyze_checkpoint_results.py --model-dir \"$MODEL_DIR\""

# Add additional arguments
for arg in "$@"; do
    cmd="$cmd $arg"
done

echo "Running command: $cmd"

# Run the analysis script
eval $cmd

echo "Analysis complete!"
echo "Check the following files in $MODEL_DIR:"
echo ""
echo "For comprehensive analysis:"
echo "  - performance_comparison.png (comprehensive performance plots)"
echo "  - trend_analysis.png (training progress analysis)"
echo "  - best_checkpoints.png (best performing checkpoints)"
echo "  - comprehensive_metrics_summary.csv (summary table)"
echo "  - model_performance_report.txt (detailed analysis report)"
echo "  - training_loss_curves_by_species.png (training loss by species)"
echo "  - ce_loss_by_species.png (cross-entropy loss by species)"
echo ""
echo "For basic analysis:"
echo "  - mask_metrics_summary.png (basic summary plot)"
echo "  - mask_metrics_summary.csv (basic summary table)"
echo ""
echo "Individual checkpoint results:"
echo "  - checkpoint_evaluations/checkpoint_*/ (individual checkpoint results)"
echo "    - Each checkpoint directory contains:"
echo "      - inference/results.json (raw evaluation results)"
echo "      - inference/mask_metrics.csv (detailed metrics)"
echo "      - inference/confusion_matrix.png (confusion matrix)"
echo "      - inference/AP_per_category.png (AP per category plot)" 
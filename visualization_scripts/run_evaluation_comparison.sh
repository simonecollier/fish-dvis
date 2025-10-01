#!/bin/bash

# Script to compare evaluation runs and analyze configuration impact
# Usage: ./run_evaluation_comparison.sh <model_dir> [options]

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <model_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  <model_dir>    Path to the model directory containing checkpoint_evaluations"
    echo ""
    echo "Optional arguments:"
    echo "  --output-dir <path>    Output directory for comparison results"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Compare all evaluation runs"
    echo "  $0 /path/to/model"
    echo ""
    echo "  # Compare runs with custom output directory"
    echo "  $0 /path/to/model --output-dir /path/to/comparison_results"
    echo ""
    echo "Description:"
    echo "  This script compares all evaluation runs in checkpoint_evaluations/"
    echo "  and analyzes how different test configurations affect model performance."
    echo "  It generates:"
    echo "    - Performance comparison plots"
    echo "    - Configuration heatmap"
    echo "    - Performance trends across checkpoints"
    echo "    - Summary report with recommendations"
    echo "    - CSV files with detailed data"
    exit 1
}

# Parse command line arguments
if [ $# -lt 1 ]; then
    show_usage
fi

# Check for help flag first
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
fi

MODEL_DIR="$1"
shift

# Parse additional arguments
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Check if checkpoint_evaluations exists
CHECKPOINT_EVAL_DIR="$MODEL_DIR/checkpoint_evaluations"
if [ ! -d "$CHECKPOINT_EVAL_DIR" ]; then
    echo "Error: No checkpoint_evaluations directory found in $MODEL_DIR"
    echo "Run evaluation first using run_eval_all_checkpoints.sh"
    exit 1
fi

# Check if there are any runs to compare
RUNS=$(ls "$CHECKPOINT_EVAL_DIR"/run_* 2>/dev/null | wc -l)
if [ "$RUNS" -eq 0 ]; then
    echo "Error: No evaluation runs found in $CHECKPOINT_EVAL_DIR"
    echo "Run evaluations with different configurations first"
    exit 1
fi

echo "Starting evaluation comparison analysis..."
echo "Model directory: $MODEL_DIR"
echo "Found $RUNS evaluation runs to compare"

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Build command
cmd="python /home/simone/fish-dvis/visualization_scripts/compare_evaluation_runs.py --model-dir \"$MODEL_DIR\""

if [ -n "$OUTPUT_DIR" ]; then
    cmd="$cmd --output-dir \"$OUTPUT_DIR\""
fi

echo "Running command: $cmd"

# Run the comparison script
eval $cmd

echo "Evaluation comparison analysis complete!"
echo ""
echo "Results saved to:"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  $OUTPUT_DIR"
else
    echo "  $MODEL_DIR/evaluation_comparison"
fi
echo ""
echo "Generated files:"
echo "  - configuration_summary.csv (configuration comparison table)"
echo "  - performance_comparison.csv (performance comparison table)"
echo "  - performance_comparison.png (performance comparison plots)"
echo "  - configuration_heatmap.png (configuration differences heatmap)"
echo "  - performance_trends.png (performance trends across checkpoints)"
echo "  - evaluation_comparison_report.txt (comprehensive summary report)"
echo ""
echo "Use the report to identify the best performing configuration for your fish classification task!"

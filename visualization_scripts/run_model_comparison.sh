#!/bin/bash

# Model comparison script wrapper
# This script provides a convenient interface to compare two models using their analysis results
# Usage: ./run_model_comparison.sh <model1_dir> <model2_dir> [output_dir]

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model1_dir> <model2_dir> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  # Compare two models (output to current directory)"
    echo "  $0 /path/to/model1 /path/to/model2"
    echo ""
    echo "  # Compare two models with custom output directory"
    echo "  $0 /path/to/model1 /path/to/model2 /path/to/output"
    echo ""
    echo "Prerequisites:"
    echo "  - Both model directories must contain analysis results from run_analysis_all_checkpoint_results.sh"
    echo "  - Look for comprehensive_metrics_summary.csv in each model directory"
    echo ""
    echo "Output files:"
    echo "  - comprehensive_model_comparison.png (comparison plots)"
    echo "  - model_comparison_report.txt (detailed comparison report)"
    exit 1
fi

MODEL1_DIR=$1
MODEL2_DIR=$2
OUTPUT_DIR=${3:-.}

# Check if model directories exist
if [ ! -d "$MODEL1_DIR" ]; then
    echo "Error: Model 1 directory $MODEL1_DIR does not exist"
    exit 1
fi

if [ ! -d "$MODEL2_DIR" ]; then
    echo "Error: Model 2 directory $MODEL2_DIR does not exist"
    exit 1
fi

# Check if analysis results exist
if [ ! -f "$MODEL1_DIR/comprehensive_metrics_summary.csv" ]; then
    echo "Error: comprehensive_metrics_summary.csv not found in $MODEL1_DIR"
    echo "Please run run_analysis_all_checkpoint_results.sh on $MODEL1_DIR first"
    exit 1
fi

if [ ! -f "$MODEL2_DIR/comprehensive_metrics_summary.csv" ]; then
    echo "Error: comprehensive_metrics_summary.csv not found in $MODEL2_DIR"
    echo "Please run run_analysis_all_checkpoint_results.sh on $MODEL2_DIR first"
    exit 1
fi

echo "Starting model comparison..."
echo "Model 1: $MODEL1_DIR"
echo "Model 2: $MODEL2_DIR"
echo "Output directory: $OUTPUT_DIR"

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Run the comparison script
python /home/simone/fish-dvis/visualization_scripts/compare_models.py \
    --model1-dir "$MODEL1_DIR" \
    --model2-dir "$MODEL2_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Model comparison complete!"
echo "Check the following files in $OUTPUT_DIR:"
echo "  - comprehensive_model_comparison.png (comparison plots)"
echo "  - model_comparison_report.txt (detailed comparison report)" 
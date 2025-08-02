#!/bin/bash

# Script to run checkpoint evaluation for a model
# Usage: ./run_eval_all_checkpoints.sh <model_dir> <config_file> [output_dir]

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <model_dir> <config_file> [output_dir]"
    echo "Example: $0 /store/simone/dvis-model-outputs/trained_models/200Q_july25_0.00001 /home/simone/fish-dvis/configs/DAQ_Offline_VitAdapterL_fishway.yaml"
    echo "Example: $0 /store/simone/dvis-model-outputs/trained_models/200Q_july25_0.00001 /home/simone/fish-dvis/configs/DAQ_Offline_VitAdapterL_fishway.yaml /custom/output/dir"
    exit 1
fi

MODEL_DIR=$1
CONFIG_FILE=$2
OUTPUT_DIR=$3

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting checkpoint evaluation..."
echo "Model directory: $MODEL_DIR"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Run the evaluation script
python /home/simone/fish-dvis/training_scripts/evaluate_all_checkpoints.py \
    --model-dir "$MODEL_DIR" \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus 1 \
    --skip-existing

echo "Checkpoint evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Check the following files for results:"
echo "  - $OUTPUT_DIR/checkpoint_comparison.png (comparison plots)"
echo "  - $OUTPUT_DIR/checkpoint_summary.csv (summary table)"
echo "  - $OUTPUT_DIR/checkpoint_*/ (individual checkpoint results)" 
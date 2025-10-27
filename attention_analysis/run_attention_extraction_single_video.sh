#!/bin/bash
# Wrapper script to run attention extraction for a single video using the modified training script

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables exactly like evaluation
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1 to avoid GPU 0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Check if video ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <video_id> [attention_output_dir] [top_n]"
    echo "Example: $0 10"
    echo "Example: $0 10 /path/to/output"
    echo "Example: $0 10 /path/to/output 5"
    exit 1
fi

VIDEO_ID=$1
ATTENTION_OUTPUT_DIR=${2:-"/home/simone/store/simone/attention_maps_single_video"}
TOP_N=${3:-1}

echo "Extracting attention maps with rollout for video ID: $VIDEO_ID"
echo "Output directory: $ATTENTION_OUTPUT_DIR"
echo "Top N predictions: $TOP_N"
echo "Using GPU 1 (CUDA_VISIBLE_DEVICES=1)"

# Create output directory
mkdir -p "$ATTENTION_OUTPUT_DIR"

# Run the modified training script with attention extraction
cd /home/simone/fish-dvis/training_scripts
python train_net_video_attention.py \
    --num-gpus 1 \
    --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_s1_fixed/config.yaml \
    --eval-only \
    --extract-attention \
    --rollout \
    --target-video-id "$VIDEO_ID" \
    --attention-output-dir "$ATTENTION_OUTPUT_DIR" \
    --top-n "$TOP_N" \
    MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_s1_fixed/model_0003635.pth \
    OUTPUT_DIR /tmp/attention_eval_single

echo "Attention extraction with rollout completed!"
echo "Check the output directory: $ATTENTION_OUTPUT_DIR"
echo "The output will contain rolled-out attention maps instead of individual layer maps."

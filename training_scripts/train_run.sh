#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1 
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
export LD_LIBRARY_PATH=/home/simone/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS="ignore::FutureWarning"

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Error: datatype argument is required"
    echo "Usage: $0 {camera|silhouette}"
    exit 1
fi

DATATYPE="$1"

# Validate datatype parameter
if [ "$DATATYPE" != "camera" ] && [ "$DATATYPE" != "silhouette" ]; then
    echo "Error: datatype must be either 'camera' or 'silhouette'"
    echo "Usage: $0 {camera|silhouette}"
    exit 1
fi

# Select config file based on datatype
if [ "$DATATYPE" = "camera" ]; then
    CONFIG_FILE="/home/simone/fish-dvis/configs/DAQ_Fishway_config_camera.yaml"
elif [ "$DATATYPE" = "silhouette" ]; then
    CONFIG_FILE="/home/simone/fish-dvis/configs/DAQ_Fishway_config_silhouette.yaml"
fi

echo "Using datatype: $DATATYPE"
echo "Using config file: $CONFIG_FILE"

# Extract OUTPUT_DIR from config file
OUTPUT_DIR=$(grep "OUTPUT_DIR:" "$CONFIG_FILE" | sed 's/.*OUTPUT_DIR:[[:space:]]*//' | tr -d "'")

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Could not find OUTPUT_DIR in config file: $CONFIG_FILE"
    exit 1
fi

echo "Output directory from config: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Copy config file to output directory
echo "Copying config file to output directory..."
cp "$CONFIG_FILE" "$OUTPUT_DIR/"

# Copy dataset files to output directory
echo "Copying dataset files to output directory..."
cp "/data/fishway_ytvis/train.json" "$OUTPUT_DIR/"
cp "/data/fishway_ytvis/val.json" "$OUTPUT_DIR/"

echo "Files copied to: $OUTPUT_DIR"
echo "Starting training..."

# Run training with specified device and debug settings
echo "Starting training with device $DEVICE..."
python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file "$CONFIG_FILE" \
  --resume \
  MODEL.WEIGHTS /home/simone/checkpoints/model_ytvis21_offline_vitl.pth

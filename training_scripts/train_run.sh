#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
export LD_LIBRARY_PATH=/home/simone/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS="ignore::FutureWarning"

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Error: argument is required"
    echo "Usage: $0 {camera|silhouette|<config_file_path>} [device]"
    echo "  device: GPU device number (0, 1, etc.). Defaults to 0 if not specified."
    exit 1
fi

ARG="$1"
DEVICE="${2:-0}"  # Default to device 0 if not specified

# Validate device is a number
if ! [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Error: Device must be a number (0, 1, etc.)"
    echo "Usage: $0 {camera|silhouette|<config_file_path>} [device]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$DEVICE"
echo "Using GPU device: $DEVICE"

# Check if argument is a file path (contains '/' or ends with '.yaml' or '.yml')
if [[ "$ARG" == *"/"* ]] || [[ "$ARG" == *.yaml ]] || [[ "$ARG" == *.yml ]]; then
    # Argument is a config file path
    if [ ! -f "$ARG" ]; then
        echo "Error: Config file not found: $ARG"
        exit 1
    fi
    CONFIG_FILE="$ARG"
    echo "Using config file: $CONFIG_FILE"
else
    # Argument is a datatype
    DATATYPE="$ARG"
    
    # Validate datatype parameter
    if [ "$DATATYPE" != "camera" ] && [ "$DATATYPE" != "silhouette" ]; then
        echo "Error: datatype must be either 'camera' or 'silhouette'"
        echo "Usage: $0 {camera|silhouette|<config_file_path>}"
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
fi

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

# Copy dataset files to output directory based on config file
echo "Copying dataset files to output directory..."
echo "Analyzing config file for required JSON files..."

# Parse YAML config to extract dataset names
# Use python to parse YAML properly
CONFIG_FILE="$CONFIG_FILE" OUTPUT_DIR="$OUTPUT_DIR" python3 <<PYTHON_SCRIPT
import yaml
import os
import re

config_file = os.environ['CONFIG_FILE']
output_dir = os.environ['OUTPUT_DIR']

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

train_datasets = config.get('DATASETS', {}).get('TRAIN', [])
test_datasets = config.get('DATASETS', {}).get('TEST', [])

# Collect all unique JSON files needed
json_files = set()

# Helper function to extract JSON path from dataset name
def get_json_path_from_dataset(dataset_name):
    # Match stride pattern: _stride{N} at the end
    stride_match = re.search(r'_stride(\d+)$', dataset_name)
    stride_value = int(stride_match.group(1)) if stride_match else None
    stride_suffix = stride_match.group(0) if stride_match else ''
    
    # Match fold pattern: _fold{N} at the end (but check before stride if both exist)
    name_without_stride = dataset_name[:-len(stride_suffix)] if stride_suffix else dataset_name
    fold_match = re.search(r'_fold(\d+)$', name_without_stride)
    fold_value = int(fold_match.group(1)) if fold_match else None
    
    if 'train' in dataset_name:
        json_name = 'train'
    elif 'val' in dataset_name:
        json_name = 'val'
    else:
        return None
    
    # Use fold JSON if fold is specified (takes precedence over stride)
    if fold_value:
        return f"/home/simone/shared-data/fishway_ytvis/{json_name}_fold{fold_value}.json"
    # Use strided JSON if stride is specified
    elif stride_value:
        return f"/data/fishway_ytvis/{json_name}_stride{stride_value}.json"
    else:
        return f"/data/fishway_ytvis/{json_name}.json"

# Extract JSON paths from all datasets
for dataset in train_datasets + test_datasets:
    json_path = get_json_path_from_dataset(dataset)
    if json_path:
        json_files.add(json_path)

# Print and copy JSON files
for json_file in sorted(json_files):
    if os.path.exists(json_file):
        print(f"  Copying: {json_file}")
        # Copy to output directory
        os.system(f"cp '{json_file}' '{output_dir}/'")
    else:
        print(f"  WARNING: JSON file not found: {json_file}")

PYTHON_SCRIPT

echo "Files copied to: $OUTPUT_DIR"
echo "Starting training..."

# Run training with specified device and debug settings
echo "Starting training with device $DEVICE..."
python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file "$CONFIG_FILE" \
  --resume \
  MODEL.WEIGHTS /home/simone/checkpoints/model_ytvis21_offline_vitl.pth

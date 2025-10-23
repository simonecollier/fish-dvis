#!/bin/bash

# Flexible sequential training script
# Usage: ./train_sequential.sh [stride1] [stride2] ... [strideN]
# Example: ./train_sequential.sh 1 2 3
# Example: ./train_sequential.sh 2 4 6
# If no arguments provided, trains all strides 1-6

set -e  # Exit on any error

# Configuration
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Base config file path
BASE_CONFIG="/home/simone/fish-dvis/configs/DAQ_Fishway_config.yaml"

# Default strides if none provided
DEFAULT_STRIDES=(1 2 3 4 5 6)

# Base output directory
BASE_OUTPUT_DIR="/store/simone/dvis-model-outputs/trained_models"

# Pre-trained weights
PRETRAINED_WEIGHTS="/home/simone/checkpoints/model_ytvis21_offline_vitl.pth"

# Parse command line arguments
if [ $# -eq 0 ]; then
    STRIDES=("${DEFAULT_STRIDES[@]}")
    echo "No strides specified, using default: ${STRIDES[*]}"
else
    STRIDES=("$@")
    echo "Training specified strides: ${STRIDES[*]}"
fi

# Validate strides
for stride in "${STRIDES[@]}"; do
    if ! [[ "$stride" =~ ^[0-9]+$ ]] || [ "$stride" -lt 1 ] || [ "$stride" -gt 10 ]; then
        echo "Error: Invalid stride '$stride'. Must be a positive integer between 1 and 10."
        exit 1
    fi
done

# Function to create config for specific stride
create_stride_config() {
    local stride=$1
    local config_path=$2
    
    echo "Creating config for stride $stride..."
    
    # Create a copy of the base config
    cp "$BASE_CONFIG" "$config_path"
    
    # Fix the relative _BASE_ path to be absolute
    local base_dir=$(dirname "$BASE_CONFIG")
    local absolute_base_path="${base_dir}/../DVIS_Plus/DVIS_DAQ/configs/dvis_daq/ytvis21/vit_adapter/DAQ_Offline_VitAdapterL.yaml"
    absolute_base_path=$(realpath "$absolute_base_path")
    
    # Update the _BASE_ path to be absolute
    sed -i "s|_BASE_: .*|_BASE_: ${absolute_base_path}|" "$config_path"
    
    # Update the config for this stride
    sed -i "s/TRAIN: (\"ytvis_fishway_train_stride[0-9]*\",)/TRAIN: (\"ytvis_fishway_train_stride${stride}\",)/" "$config_path"
    sed -i "s/TEST: (\"ytvis_fishway_val_stride[0-9]*\",)/TEST: (\"ytvis_fishway_val_stride${stride}\",)/" "$config_path"
    sed -i "s|OUTPUT_DIR: '.*'|OUTPUT_DIR: '${BASE_OUTPUT_DIR}/model_stride${stride}'|" "$config_path"
    
    echo "Config created: $config_path"
    echo "Base config path: $absolute_base_path"
}

# Function to train a single model
train_model() {
    local stride=$1
    local config_path=$2
    local output_dir=$3
    
    echo ""
    echo "=========================================="
    echo "STARTING TRAINING FOR STRIDE $stride"
    echo "=========================================="
    echo "Config: $config_path"
    echo "Output: $output_dir"
    echo "Start time: $(date)"
    echo "=========================================="
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Copy config file to output directory
    cp "$config_path" "$output_dir/"
    
    # Copy dataset files to output directory
    echo "Copying dataset files to output directory..."
    cp "/data/fishway_ytvis/train_stride${stride}.json" "$output_dir/train.json"
    cp "/data/fishway_ytvis/val_stride${stride}.json" "$output_dir/val.json"
    
    # Start training with error handling
    echo "Starting training for stride $stride..."
    
    # Run training and capture both stdout and stderr
    if CUDA_VISIBLE_DEVICES=0 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
        --num-gpus 1 \
        --config-file "$config_path" \
        --resume \
        MODEL.WEIGHTS "$PRETRAINED_WEIGHTS" 2>&1 | tee "$output_dir/training.log"; then
        
        echo ""
        echo "✅ SUCCESS: Training completed for stride $stride"
        echo "End time: $(date)"
        echo "Output saved to: $output_dir"
        return 0
    else
        echo ""
        echo "❌ ERROR: Training failed for stride $stride"
        echo "End time: $(date)"
        echo "Check logs in: $output_dir/training.log"
        return 1
    fi
}

# Function to log training results
log_results() {
    local log_file="$BASE_OUTPUT_DIR/training_log.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" >> "$log_file"
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local stride=$3
    local percentage=$((current * 100 / total))
    echo "Progress: [$current/$total] ($percentage%) - Currently training stride $stride"
}

# Main execution
main() {
    echo "=========================================="
    echo "SEQUENTIAL TRAINING SCRIPT"
    echo "=========================================="
    echo "Strides to train: ${STRIDES[*]}"
    echo "Base output directory: $BASE_OUTPUT_DIR"
    echo "Pre-trained weights: $PRETRAINED_WEIGHTS"
    echo "Start time: $(date)"
    echo "=========================================="
    
    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # Initialize log file
    log_file="$BASE_OUTPUT_DIR/training_log.txt"
    echo "Training started at $(date)" > "$log_file"
    echo "Strides to train: ${STRIDES[*]}" >> "$log_file"
    
    # Track results
    successful_strides=()
    failed_strides=()
    total_strides=${#STRIDES[@]}
    
    # Train each stride sequentially
    for i in "${!STRIDES[@]}"; do
        stride="${STRIDES[$i]}"
        current=$((i + 1))
        
        show_progress "$current" "$total_strides" "$stride"
        
        config_path="/tmp/config_stride${stride}.yaml"
        output_dir="${BASE_OUTPUT_DIR}/model_stride${stride}"
        
        # Create config for this stride
        create_stride_config "$stride" "$config_path"
        
        # Log start
        log_results "Starting training for stride $stride (${current}/${total_strides})"
        
        # Train model
        if train_model "$stride" "$config_path" "$output_dir"; then
            successful_strides+=("$stride")
            log_results "✅ SUCCESS: Training completed for stride $stride"
        else
            failed_strides+=("$stride")
            log_results "❌ FAILED: Training failed for stride $stride"
            
            # Ask user if they want to continue
            echo ""
            echo "Training failed for stride $stride."
            echo "Remaining strides: ${STRIDES[@]:$current}"
            read -p "Do you want to continue with the next stride? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Stopping training sequence."
                break
            fi
        fi
        
        # Clean up temporary config
        rm -f "$config_path"
        
        # Add delay between trainings (optional)
        if [ "$current" -lt "$total_strides" ]; then
            echo "Waiting 5 seconds before starting next training..."
            sleep 5
        fi
    done
    
    # Final summary
    echo ""
    echo "=========================================="
    echo "TRAINING SEQUENCE COMPLETED"
    echo "=========================================="
    echo "Total time: $(date)"
    echo "Successful strides: ${successful_strides[*]}"
    echo "Failed strides: ${failed_strides[*]}"
    echo "Log file: $log_file"
    echo "=========================================="
    
    # Log final summary
    log_results "Training sequence completed. Successful: ${successful_strides[*]}, Failed: ${failed_strides[*]}"
    
    # Show output directories
    echo ""
    echo "Output directories:"
    for stride in "${successful_strides[@]}"; do
        echo "  - Stride $stride: ${BASE_OUTPUT_DIR}/model_stride${stride}"
    done
    
    # Exit with error code if any training failed
    if [ ${#failed_strides[@]} -gt 0 ]; then
        echo ""
        echo "⚠️  Some trainings failed. Check the logs for details."
        exit 1
    else
        echo ""
        echo "🎉 All trainings completed successfully!"
    fi
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [stride1] [stride2] ... [strideN]"
    echo ""
    echo "Examples:"
    echo "  $0                    # Train all strides 1-6"
    echo "  $0 1 2 3             # Train strides 1, 2, 3"
    echo "  $0 2 4 6             # Train strides 2, 4, 6"
    echo "  $0 1                 # Train only stride 1"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "The script will:"
    echo "  1. Create config files for each stride"
    echo "  2. Train models sequentially"
    echo "  3. Save outputs to separate directories"
    echo "  4. Log all results and handle errors gracefully"
    exit 0
fi

# Run main function
main "$@"

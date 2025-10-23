#!/bin/bash

# Smart parallel training script with advanced GPU management
# This script intelligently manages GPU resources and training schedules
# Usage: ./train_smart_parallel.sh [--strategy=parallel|sequential|mixed] [stride1] [stride2] ... [strideN]

set -e  # Exit on any error

# Configuration
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Base config file path
BASE_CONFIG="/home/simone/fish-dvis/configs/DAQ_Fishway_config.yaml"

# Default strides if none provided
DEFAULT_STRIDES=(1 2 3 4 5 6)

# Base output directory
BASE_OUTPUT_DIR="/store/simone/dvis-model-outputs/trained_models"

# Pre-trained weights
PRETRAINED_WEIGHTS="/home/simone/checkpoints/model_ytvis21_offline_vitl.pth"

# GPU configuration
NUM_GPUS=2
GPU_IDS=(0 1)

# Training strategy: parallel, sequential, or mixed
TRAINING_STRATEGY="parallel"

# Parse command line arguments
STRIDES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy=*)
            TRAINING_STRATEGY="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--strategy=parallel|sequential|mixed] [stride1] [stride2] ... [strideN]"
            echo ""
            echo "Training Strategies:"
            echo "  parallel    - Train all strides simultaneously (default)"
            echo "  sequential  - Train strides one after another"
            echo "  mixed       - Train 2 strides in parallel, then next 2, etc."
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train all strides 1-6 in parallel"
            echo "  $0 --strategy=sequential              # Train all strides sequentially"
            echo "  $0 --strategy=mixed 1 2 3 4 5 6      # Train in batches of 2"
            echo "  $0 1 2 3 4                           # Train strides 1,2,3,4 in parallel"
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                STRIDES+=("$1")
            else
                echo "Error: Invalid argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Use default strides if none provided
if [ ${#STRIDES[@]} -eq 0 ]; then
    STRIDES=("${DEFAULT_STRIDES[@]}")
    echo "No strides specified, using default: ${STRIDES[*]}"
else
    echo "Training specified strides: ${STRIDES[*]}"
fi

# Validate strides
for stride in "${STRIDES[@]}"; do
    if ! [[ "$stride" =~ ^[0-9]+$ ]] || [ "$stride" -lt 1 ] || [ "$stride" -gt 10 ]; then
        echo "Error: Invalid stride '$stride'. Must be a positive integer between 1 and 10."
        exit 1
    fi
done

# Validate strategy
if [[ "$TRAINING_STRATEGY" != "parallel" && "$TRAINING_STRATEGY" != "sequential" && "$TRAINING_STRATEGY" != "mixed" ]]; then
    echo "Error: Invalid strategy '$TRAINING_STRATEGY'. Must be 'parallel', 'sequential', or 'mixed'."
    exit 1
fi

echo "=========================================="
echo "SMART PARALLEL TRAINING SETUP"
echo "=========================================="
echo "Training strategy: $TRAINING_STRATEGY"
echo "Available GPUs: ${GPU_IDS[*]}"
echo "Strides to train: ${STRIDES[*]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Pre-trained weights: $PRETRAINED_WEIGHTS"
echo "=========================================="

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

# Function to train a single model on a specific GPU
train_model_on_gpu() {
    local stride=$1
    local gpu_id=$2
    local config_path=$3
    local output_dir=$4
    
    echo ""
    echo "=========================================="
    echo "STARTING TRAINING FOR STRIDE $stride ON GPU $gpu_id"
    echo "=========================================="
    echo "Config: $config_path"
    echo "Output: $output_dir"
    echo "GPU: $gpu_id"
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
    
    # Start training on specific GPU
    echo "Starting training for stride $stride on GPU $gpu_id..."
    
    # Run training and capture both stdout and stderr
    CUDA_VISIBLE_DEVICES=$gpu_id python /home/simone/fish-dvis/training_scripts/train_net_video.py \
        --num-gpus 1 \
        --config-file "$config_path" \
        --resume \
        MODEL.WEIGHTS "$PRETRAINED_WEIGHTS" 2>&1 | tee "$output_dir/training.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Training completed for stride $stride on GPU $gpu_id"
        echo "End time: $(date)"
        echo "Output saved to: $output_dir"
    else
        echo ""
        echo "❌ ERROR: Training failed for stride $stride on GPU $gpu_id (exit code: $exit_code)"
        echo "End time: $(date)"
        echo "Check logs in: $output_dir/training.log"
    fi
    
    return $exit_code
}

# Function to log training results
log_results() {
    local log_file="$BASE_OUTPUT_DIR/smart_training_log.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" >> "$log_file"
}

# Function to check GPU availability
check_gpu_availability() {
    local gpu_id=$1
    local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "0")
    local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "0")
    local gpu_memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "0")
    
    echo "GPU $gpu_id: Usage=${gpu_usage}%, Memory=${gpu_memory}MB/${gpu_memory_total}MB"
    
    # GPU is available if usage < 10% and memory usage < 80%
    if [ "$gpu_usage" -lt 10 ] && [ "$gpu_memory" -lt $((gpu_memory_total * 80 / 100)) ]; then
        return 0
    else
        return 1
    fi
}

# Function to wait for GPU to be available
wait_for_gpu() {
    local gpu_id=$1
    local max_wait=600  # 10 minutes max wait
    
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if check_gpu_availability $gpu_id; then
            echo "GPU $gpu_id is now available"
            return 0
        fi
        
        echo "GPU $gpu_id is busy, waiting..."
        sleep 15
        wait_time=$((wait_time + 15))
    done
    
    echo "Warning: GPU $gpu_id still busy after $max_wait seconds, proceeding anyway"
    return 0
}

# Function to run parallel training
run_parallel_training() {
    local strides=("$@")
    local successful_strides=()
    local failed_strides=()
    local training_pids=()
    
    echo "Running parallel training for strides: ${strides[*]}"
    
    # Start training for each stride
    for i in "${!strides[@]}"; do
        local stride="${strides[$i]}"
        local gpu_id="${GPU_IDS[$i % ${#GPU_IDS[@]}]}"
        
        local config_path="/tmp/config_stride${stride}.yaml"
        local output_dir="${BASE_OUTPUT_DIR}/model_stride${stride}"
        
        # Create config for this stride
        create_stride_config "$stride" "$config_path"
        
        # Wait for GPU to be available if needed
        if [ $i -ge ${#GPU_IDS[@]} ]; then
            echo "Waiting for GPU $gpu_id to be available..."
            wait_for_gpu $gpu_id
        fi
        
        # Log start
        log_results "Starting parallel training for stride $stride on GPU $gpu_id"
        
        # Start training in background
        echo "Starting training for stride $stride on GPU $gpu_id..."
        train_model_on_gpu "$stride" "$gpu_id" "$config_path" "$output_dir" &
        local pid=$!
        
        training_pids+=($pid)
        echo "Started training for stride $stride on GPU $gpu_id (PID: $pid)"
        
        # Add small delay between starting trainings
        sleep 5
    done
    
    # Wait for all training processes to complete
    echo "Waiting for all parallel training processes to complete..."
    
    for i in "${!training_pids[@]}"; do
        local pid="${training_pids[$i]}"
        local stride="${strides[$i]}"
        local gpu_id="${GPU_IDS[$i % ${#GPU_IDS[@]}]}"
        
        echo "Waiting for stride $stride training (PID: $pid) to complete..."
        
        if wait $pid; then
            successful_strides+=("$stride")
            log_results "✅ SUCCESS: Parallel training completed for stride $stride on GPU $gpu_id"
        else
            failed_strides+=("$stride")
            log_results "❌ FAILED: Parallel training failed for stride $stride on GPU $gpu_id"
        fi
    done
    
    # Clean up temporary configs
    for stride in "${strides[@]}"; do
        rm -f "/tmp/config_stride${stride}.yaml"
    done
    
    echo "Parallel training results:"
    echo "  Successful: ${successful_strides[*]}"
    echo "  Failed: ${failed_strides[*]}"
    
    # Return results via global variables (bash limitation)
    PARALLEL_SUCCESSFUL=("${successful_strides[@]}")
    PARALLEL_FAILED=("${failed_strides[@]}")
}

# Function to run sequential training
run_sequential_training() {
    local strides=("$@")
    local successful_strides=()
    local failed_strides=()
    
    echo "Running sequential training for strides: ${strides[*]}"
    
    for stride in "${strides[@]}"; do
        local gpu_id="${GPU_IDS[0]}"  # Use first GPU for sequential
        local config_path="/tmp/config_stride${stride}.yaml"
        local output_dir="${BASE_OUTPUT_DIR}/model_stride${stride}"
        
        # Create config for this stride
        create_stride_config "$stride" "$config_path"
        
        # Log start
        log_results "Starting sequential training for stride $stride on GPU $gpu_id"
        
        # Train model (blocking)
        if train_model_on_gpu "$stride" "$gpu_id" "$config_path" "$output_dir"; then
            successful_strides+=("$stride")
            log_results "✅ SUCCESS: Sequential training completed for stride $stride on GPU $gpu_id"
        else
            failed_strides+=("$stride")
            log_results "❌ FAILED: Sequential training failed for stride $stride on GPU $gpu_id"
            
            # Ask user if they want to continue
            echo ""
            echo "Training failed for stride $stride."
            read -p "Do you want to continue with the next stride? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Stopping sequential training."
                break
            fi
        fi
        
        # Clean up temporary config
        rm -f "$config_path"
        
        # Add delay between trainings
        if [ "$stride" != "${strides[-1]}" ]; then
            echo "Waiting 10 seconds before starting next training..."
            sleep 10
        fi
    done
    
    echo "Sequential training results:"
    echo "  Successful: ${successful_strides[*]}"
    echo "  Failed: ${failed_strides[*]}"
    
    # Return results via global variables
    SEQUENTIAL_SUCCESSFUL=("${successful_strides[@]}")
    SEQUENTIAL_FAILED=("${failed_strides[@]}")
}

# Function to run mixed training (batches of 2)
run_mixed_training() {
    local strides=("$@")
    local successful_strides=()
    local failed_strides=()
    
    echo "Running mixed training for strides: ${strides[*]}"
    
    # Process strides in batches of 2
    for ((i=0; i<${#strides[@]}; i+=2)); do
        local batch=("${strides[@]:$i:2}")
        echo ""
        echo "Processing batch: ${batch[*]}"
        
        # Run parallel training for this batch
        run_parallel_training "${batch[@]}"
        
        # Collect results
        successful_strides+=("${PARALLEL_SUCCESSFUL[@]}")
        failed_strides+=("${PARALLEL_FAILED[@]}")
        
        # Wait between batches
        if [ $((i + 2)) -lt ${#strides[@]} ]; then
            echo "Waiting 30 seconds before starting next batch..."
            sleep 30
        fi
    done
    
    echo "Mixed training results:"
    echo "  Successful: ${successful_strides[*]}"
    echo "  Failed: ${failed_strides[*]}"
    
    # Return results via global variables
    MIXED_SUCCESSFUL=("${successful_strides[@]}")
    MIXED_FAILED=("${failed_strides[@]}")
}

# Main execution
main() {
    echo "=========================================="
    echo "SMART PARALLEL TRAINING SCRIPT"
    echo "=========================================="
    echo "Training strategy: $TRAINING_STRATEGY"
    echo "Strides to train: ${STRIDES[*]}"
    echo "Available GPUs: ${GPU_IDS[*]}"
    echo "Base output directory: $BASE_OUTPUT_DIR"
    echo "Start time: $(date)"
    echo "=========================================="
    
    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # Initialize log file
    log_file="$BASE_OUTPUT_DIR/smart_training_log.txt"
    echo "Smart training started at $(date)" > "$log_file"
    echo "Strategy: $TRAINING_STRATEGY" >> "$log_file"
    echo "Strides to train: ${STRIDES[*]}" >> "$log_file"
    echo "Available GPUs: ${GPU_IDS[*]}" >> "$log_file"
    
    # Check GPU availability
    echo ""
    echo "Checking GPU availability..."
    for gpu_id in "${GPU_IDS[@]}"; do
        check_gpu_availability $gpu_id
    done
    
    # Run training based on strategy
    case "$TRAINING_STRATEGY" in
        "parallel")
            run_parallel_training "${STRIDES[@]}"
            successful_strides=("${PARALLEL_SUCCESSFUL[@]}")
            failed_strides=("${PARALLEL_FAILED[@]}")
            ;;
        "sequential")
            run_sequential_training "${STRIDES[@]}"
            successful_strides=("${SEQUENTIAL_SUCCESSFUL[@]}")
            failed_strides=("${SEQUENTIAL_FAILED[@]}")
            ;;
        "mixed")
            run_mixed_training "${STRIDES[@]}"
            successful_strides=("${MIXED_SUCCESSFUL[@]}")
            failed_strides=("${MIXED_FAILED[@]}")
            ;;
    esac
    
    # Final summary
    echo ""
    echo "=========================================="
    echo "SMART TRAINING COMPLETED"
    echo "=========================================="
    echo "Strategy used: $TRAINING_STRATEGY"
    echo "End time: $(date)"
    echo "Successful strides: ${successful_strides[*]}"
    echo "Failed strides: ${failed_strides[*]}"
    echo "Log file: $log_file"
    echo "=========================================="
    
    # Log final summary
    log_results "Smart training completed using strategy '$TRAINING_STRATEGY'. Successful: ${successful_strides[*]}, Failed: ${failed_strides[*]}"
    
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

# Run main function
main "$@"

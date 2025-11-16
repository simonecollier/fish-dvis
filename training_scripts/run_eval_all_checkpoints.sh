#!/bin/bash

# Script to run checkpoint evaluation for a model with configurable parameters
# Usage: ./run_eval_all_checkpoints.sh <model_dir> [options]

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <model_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  <model_dir>    Path to the model directory containing config.yaml"
    echo ""
    echo "Optional arguments:"
    echo "  Config settings:"
    echo "    --config-file <path>                  Use specific config file instead of model_dir/config.yaml"
    echo "    --val-json <path>                     Use specific validation JSON file (overrides auto-detection)"
    echo ""
    echo "  Device settings:"
    echo "    --device <number>                     Set CUDA_VISIBLE_DEVICES (default: 0)"
    echo "    --checkpoint-range <start-end>        Evaluate only checkpoints in range (e.g., 1-10, 5-15)"
    echo ""
    echo "  Segmentation settings:"
    echo "    --instance-on/--no-instance-on        Toggle INSTANCE_ON (default: from config)"
    echo "    --semantic-on/--no-semantic-on        Toggle SEMANTIC_ON (default: from config)"
    echo "    --max-num <number>                    Set MAX_NUM (default: from config)"
    echo "    --object-mask-threshold <float>       Set OBJECT_MASK_THRESHOLD (default: from config)"
    echo "    --overlap-threshold <float>           Set OVERLAP_THRESHOLD (default: from config)"
    echo "    --window-size <number>                Set WINDOW_SIZE (default: from config)"
    echo ""
    echo "  Resolution settings:"
    echo "    --min-size-test <number>              Set MIN_SIZE_TEST (default: from config)"
    echo "    --max-size-test <number>              Set MAX_SIZE_TEST (default: from config)"
    echo ""
    echo "  Test augmentation settings:"
    echo "    --test-aug/--no-test-aug              Toggle TEST.AUG.ENABLED (default: from config)"
    echo "    --test-flip/--no-test-flip            Toggle TEST.AUG.FLIP (default: from config)"
    echo "    --test-min-sizes <list>               Set TEST.AUG.MIN_SIZES (comma-separated, default: from config)"
    echo ""
    echo "Examples:"
    echo "  # Test with instance segmentation enabled"
    echo "  $0 /path/to/model --instance-on --max-num 1"
    echo ""
    echo "  # Test with higher resolution"
    echo "  $0 /path/to/model --min-size-test 480 --max-size-test 640"
    echo ""
    echo "  # Test with test-time augmentation"
    echo "  $0 /path/to/model --test-aug --test-min-sizes \"360,480,600\""
    echo ""
    echo "  # Test with different thresholds"
    echo "  $0 /path/to/model --object-mask-threshold 0.9 --overlap-threshold 0.7"
    echo ""
    echo "  # Use custom config file (preserves existing results)"
    echo "  $0 /path/to/model --config-file /path/to/custom_config.yaml"
    echo ""
    echo "  # Test on GPU 1"
    echo "  $0 /path/to/model --device 1"
    echo ""
    echo "  # Test with single GPU"
    echo "  $0 /path/to/model --device 0"
    echo ""
    echo "  # Evaluate checkpoints 1-10 on GPU 0"
    echo "  $0 /path/to/model --checkpoint-range 1-10 --device 0"
    echo ""
    echo "  # Evaluate checkpoints 11-20 on GPU 1 (run simultaneously)"
    echo "  $0 /path/to/model --checkpoint-range 11-20 --device 1"
    echo ""
    echo "  # Use custom validation JSON file"
    echo "  $0 /path/to/model --val-json /path/to/custom_val.json"
    exit 1
}

# Function to compare configs
compare_configs() {
    local config1="$1"
    local config2="$2"
    
    # Generate a simple hash of the config content (excluding comments and whitespace)
    local hash1=$(grep -v '^#' "$config1" | grep -v '^$' | sort | md5sum | cut -d' ' -f1)
    local hash2=$(grep -v '^#' "$config2" | grep -v '^$' | sort | md5sum | cut -d' ' -f1)
    
    if [ "$hash1" = "$hash2" ]; then
        return 0  # Configs match
    else
        return 1  # Configs differ
    fi
}

# Function to find next run number
find_next_run() {
    local base_dir="$1"
    local run_num=1
    
    while [ -d "$base_dir/run_$run_num" ]; do
        ((run_num++))
    done
    
    echo $run_num
}

# Function to create config diff
create_config_diff() {
    local original_config="$1"
    local new_config="$2"
    local diff_file="$3"
    
    echo "# Configuration changes from original:" > "$diff_file"
    echo "# Generated on: $(date)" >> "$diff_file"
    echo "" >> "$diff_file"
    
    # Use diff to show changes, but make it more readable
    diff "$original_config" "$new_config" | grep -E '^[<>]' | sed 's/^< /REMOVED: /' | sed 's/^> /ADDED:   /' >> "$diff_file" 2>/dev/null || true
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

# Initialize config modification variables
declare -A config_changes
config_changes=()

# Initialize device and checkpoint range variables
DEVICE=0
CHECKPOINT_RANGE=""
CUSTOM_CONFIG_FILE=""
CUSTOM_VAL_JSON=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-file)
            CUSTOM_CONFIG_FILE="$2"
            shift 2
            ;;
        --val-json)
            CUSTOM_VAL_JSON="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint-range)
            CHECKPOINT_RANGE="$2"
            shift 2
            ;;
        --instance-on)
            config_changes["INSTANCE_ON"]="true"
            shift
            ;;
        --no-instance-on)
            config_changes["INSTANCE_ON"]="false"
            shift
            ;;
        --semantic-on)
            config_changes["SEMANTIC_ON"]="true"
            shift
            ;;
        --no-semantic-on)
            config_changes["SEMANTIC_ON"]="false"
            shift
            ;;
        --max-num)
            config_changes["MAX_NUM"]="$2"
            shift 2
            ;;
        --object-mask-threshold)
            config_changes["OBJECT_MASK_THRESHOLD"]="$2"
            shift 2
            ;;
        --overlap-threshold)
            config_changes["OVERLAP_THRESHOLD"]="$2"
            shift 2
            ;;
        --window-size)
            config_changes["WINDOW_SIZE"]="$2"
            shift 2
            ;;
        --min-size-test)
            config_changes["MIN_SIZE_TEST"]="$2"
            shift 2
            ;;
        --max-size-test)
            config_changes["MAX_SIZE_TEST"]="$2"
            shift 2
            ;;
        --test-aug)
            config_changes["TEST_AUG_ENABLED"]="true"
            shift
            ;;
        --no-test-aug)
            config_changes["TEST_AUG_ENABLED"]="false"
            shift
            ;;
        --test-flip)
            config_changes["TEST_AUG_FLIP"]="true"
            shift
            ;;
        --no-test-flip)
            config_changes["TEST_AUG_FLIP"]="false"
            shift
            ;;
        --test-min-sizes)
            config_changes["TEST_AUG_MIN_SIZES"]="$2"
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

# Validate device argument
if [ "$DEVICE" -lt 0 ]; then
    echo "Error: Device ID must be non-negative"
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$DEVICE" -ge "$AVAILABLE_GPUS" ]; then
    echo "Error: Device ID $DEVICE is not available. Available GPUs are 0-$((AVAILABLE_GPUS-1))"
    exit 1
fi

# Validate checkpoint range format if provided
if [ -n "$CHECKPOINT_RANGE" ]; then
    if [[ ! "$CHECKPOINT_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
        echo "Error: Checkpoint range must be in format 'start-end' (e.g., 1-10, 5-15)"
        exit 1
    fi
fi

# Determine config file to use
if [ -n "$CUSTOM_CONFIG_FILE" ]; then
    CONFIG_FILE="$CUSTOM_CONFIG_FILE"
    echo "Using custom config file: $CONFIG_FILE"
else
    CONFIG_FILE="$MODEL_DIR/config.yaml"
    echo "Using default config file: $CONFIG_FILE"
fi

OUTPUT_DIR="$MODEL_DIR/checkpoint_evaluations"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

# Clean up config file by removing problematic DEBUG key if it exists
if grep -q "^DEBUG: false$" "$CONFIG_FILE"; then
    echo "Removing problematic DEBUG key from config file..."
    sed -i '/^DEBUG: false$/d' "$CONFIG_FILE"
fi

# Create temporary config file with modifications
TEMP_CONFIG=$(mktemp)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Apply config changes
echo "Applying configuration changes..."

# Apply MASK_FORMER.TEST changes
for key in "${!config_changes[@]}"; do
    case $key in
        INSTANCE_ON|SEMANTIC_ON|MAX_NUM|OBJECT_MASK_THRESHOLD|OVERLAP_THRESHOLD|WINDOW_SIZE)
            value="${config_changes[$key]}"
            # Handle boolean values
            if [[ "$value" == "true" || "$value" == "false" ]]; then
                sed -i "s/^      $key: .*/      $key: $value/" "$TEMP_CONFIG"
            else
                sed -i "s/^      $key: .*/      $key: $value/" "$TEMP_CONFIG"
            fi
            echo "  - $key: $value"
            ;;
    esac
done

# Apply INPUT changes
for key in "${!config_changes[@]}"; do
    case $key in
        MIN_SIZE_TEST|MAX_SIZE_TEST)
            value="${config_changes[$key]}"
            sed -i "s/^  $key: .*/  $key: $value/" "$TEMP_CONFIG"
            echo "  - $key: $value"
            ;;
    esac
done

# Apply TEST.AUG changes
for key in "${!config_changes[@]}"; do
    case $key in
        TEST_AUG_ENABLED)
            value="${config_changes[$key]}"
            sed -i "s/^    ENABLED: .*/    ENABLED: $value/" "$TEMP_CONFIG"
            echo "  - TEST.AUG.ENABLED: $value"
            ;;
        TEST_AUG_FLIP)
            value="${config_changes[$key]}"
            sed -i "s/^    FLIP: .*/    FLIP: $value/" "$TEMP_CONFIG"
            echo "  - TEST.AUG.FLIP: $value"
            ;;
        TEST_AUG_MIN_SIZES)
            value="${config_changes[$key]}"
            # Convert comma-separated list to YAML array format
            yaml_array="[$(echo "$value" | sed 's/,/, /g')]"
            sed -i "/^    MIN_SIZES:/,/^    [A-Z]/ { /^    MIN_SIZES:/!d; }" "$TEMP_CONFIG"
            sed -i "s/^    MIN_SIZES:.*/    MIN_SIZES: $yaml_array/" "$TEMP_CONFIG"
            echo "  - TEST.AUG.MIN_SIZES: $yaml_array"
            ;;
    esac
done

# Initialize flag for reusing existing run
REUSE_EXISTING_RUN=false

# Determine output directory structure
if [ ${#config_changes[@]} -eq 0 ]; then
    # No changes, use simple structure
    FINAL_OUTPUT_DIR="$OUTPUT_DIR"
    RUN_NUM=""
else
    # Check if checkpoint_evaluations already exists
    if [ -d "$OUTPUT_DIR" ]; then
        # Check if it's a simple structure (has checkpoint_* directories directly)
        if ls "$OUTPUT_DIR"/checkpoint_* 1> /dev/null 2>&1; then
            # Simple structure exists, check if config matches
            FIRST_CHECKPOINT=$(ls "$OUTPUT_DIR"/checkpoint_* | head -1)
            FIRST_CONFIG="$FIRST_CHECKPOINT/config.yaml"
            
            if [ -f "$FIRST_CONFIG" ] && compare_configs "$TEMP_CONFIG" "$FIRST_CONFIG"; then
                echo "Configuration matches existing evaluation. Rerun with same config? (y/n)"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    echo "Rerunning evaluation with same configuration..."
                    FINAL_OUTPUT_DIR="$OUTPUT_DIR"
                    RUN_NUM=""
                    REUSE_EXISTING_RUN=true
                else
                    echo "Evaluation cancelled."
                    rm "$TEMP_CONFIG"
                    exit 0
                fi
            else
                # Config differs, create run structure
                RUN_NUM=$(find_next_run "$OUTPUT_DIR")
                echo "Configuration differs from existing evaluation."
                echo "Moving existing evaluations to run_1/ and creating run_$RUN_NUM/"
                
                # Create run_1 directory and move existing evaluations
                mkdir -p "$OUTPUT_DIR/run_1"
                mv "$OUTPUT_DIR"/checkpoint_* "$OUTPUT_DIR/run_1/" 2>/dev/null || true
                
                FINAL_OUTPUT_DIR="$OUTPUT_DIR/run_$RUN_NUM"
            fi
        else
            # Already has run structure, check all runs for matching config
            MATCHING_RUN=""
            REUSE_EXISTING_RUN=false
            for existing_run in "$OUTPUT_DIR"/run_*; do
                if [ -d "$existing_run" ]; then
                    existing_config="$existing_run/config.yaml"
                    if [ -f "$existing_config" ] && compare_configs "$TEMP_CONFIG" "$existing_config"; then
                        MATCHING_RUN=$(basename "$existing_run")
                        REUSE_EXISTING_RUN=true
                        break
                    fi
                fi
            done
            
            if [ -n "$MATCHING_RUN" ]; then
                echo "Configuration matches existing evaluation in $MATCHING_RUN."
                echo "Will reuse $MATCHING_RUN and skip existing checkpoints."
                RUN_NUM=$(echo "$MATCHING_RUN" | sed 's/run_//')
                FINAL_OUTPUT_DIR="$OUTPUT_DIR/$MATCHING_RUN"
            else
                # Config differs from all existing runs, create new run
                RUN_NUM=$(find_next_run "$OUTPUT_DIR")
                echo "Configuration differs from all existing evaluations."
                echo "Creating new run_$RUN_NUM/"
                FINAL_OUTPUT_DIR="$OUTPUT_DIR/run_$RUN_NUM"
            fi
        fi
    else
        # No existing evaluations, use simple structure
        FINAL_OUTPUT_DIR="$OUTPUT_DIR"
        RUN_NUM=""
    fi
fi

# Create output directory
mkdir -p "$FINAL_OUTPUT_DIR"

# Save the modified config
if [ -n "$RUN_NUM" ]; then
    cp "$TEMP_CONFIG" "$FINAL_OUTPUT_DIR/config.yaml"
    create_config_diff "$CONFIG_FILE" "$TEMP_CONFIG" "$FINAL_OUTPUT_DIR/config_diff.txt"
fi

echo "Starting checkpoint evaluation..."
echo "Model directory: $MODEL_DIR"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $FINAL_OUTPUT_DIR"
echo "Using GPU: $DEVICE"
if [ -n "$CHECKPOINT_RANGE" ]; then
    echo "Checkpoint range: $CHECKPOINT_RANGE"
else
    echo "Evaluating all checkpoints"
fi
if [ -n "$RUN_NUM" ]; then
    echo "Run number: $RUN_NUM"
fi

# Show initial GPU memory state
echo "Initial GPU memory state:"
nvidia-smi --id=$DEVICE --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r used total; do
    echo "  GPU $DEVICE: ${used}MB / ${total}MB used"
done

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data

# Handle validation JSON file
if [ -n "$CUSTOM_VAL_JSON" ]; then
    # Use custom val JSON if provided
    if [ ! -f "$CUSTOM_VAL_JSON" ]; then
        echo "Error: Custom validation JSON file not found: $CUSTOM_VAL_JSON"
        exit 1
    fi
    export VAL_JSON_OVERRIDE="$CUSTOM_VAL_JSON"
    echo "Using custom validation JSON: $VAL_JSON_OVERRIDE"
elif [ -f "$CONFIG_FILE" ]; then
    # Auto-detect validation JSON from config and model directory
    # Check for model-local validation JSON files (including strided versions)
    # Parse config to detect stride patterns in dataset names
    # Use python to parse YAML and detect stride patterns
    VAL_JSON_RESULT=$(CONFIG_FILE="$CONFIG_FILE" MODEL_DIR="$MODEL_DIR" python3 <<PYTHON_SCRIPT
import yaml
import os
import re
import sys

config_file = os.environ['CONFIG_FILE']
model_dir = os.environ['MODEL_DIR']

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    test_datasets = config.get('DATASETS', {}).get('TEST', [])
    
    def get_json_path_from_dataset(dataset_name):
        # Match stride pattern: _stride{N} at the end
        stride_match = re.search(r'_stride(\d+)$', dataset_name)
        stride_value = int(stride_match.group(1)) if stride_match else None
        stride_suffix = stride_match.group(0) if stride_match else ''
        
        # Match fold pattern: _fold{N} at the end (but check before stride if both exist)
        name_without_stride = dataset_name[:-len(stride_suffix)] if stride_suffix else dataset_name
        fold_match = re.search(r'_fold(\d+)$', name_without_stride)
        fold_value = int(fold_match.group(1)) if fold_match else None
        
        if 'val' in dataset_name:
            # Use fold JSON if fold is specified (takes precedence over stride)
            if fold_value:
                return f"{model_dir}/val_fold{fold_value}.json"
            # Use strided JSON if stride is specified
            elif stride_value:
                return f"{model_dir}/val_stride{stride_value}.json"
            else:
                return f"{model_dir}/val.json"
        return None
    
    # Check for val JSON files based on dataset names
    expected_json = None
    for dataset in test_datasets:
        json_path = get_json_path_from_dataset(dataset)
        if json_path:
            expected_json = json_path
            if os.path.exists(json_path):
                print(f"FOUND:{json_path}")
                sys.exit(0)
    
    # If we get here, the expected JSON file was not found
    if expected_json:
        print(f"NOT_FOUND:{expected_json}")
        sys.exit(1)
    else:
        print("NO_VAL_DATASET")
        sys.exit(1)
except Exception as e:
    print(f"ERROR:{str(e)}")
    sys.exit(1)

PYTHON_SCRIPT
    )
    
    PYTHON_EXIT_CODE=$?
    
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        if echo "$VAL_JSON_RESULT" | grep -q "^NOT_FOUND:"; then
            EXPECTED_JSON=$(echo "$VAL_JSON_RESULT" | sed 's/^NOT_FOUND://')
            echo "Error: Required validation JSON file not found: $EXPECTED_JSON"
            echo "Please ensure the correct JSON file exists in the model directory."
            echo "If using a fold or stride dataset, make sure the corresponding JSON file (e.g., val_fold4.json or val_stride5.json) was copied during training."
            exit 1
        elif echo "$VAL_JSON_RESULT" | grep -q "^ERROR:"; then
            ERROR_MSG=$(echo "$VAL_JSON_RESULT" | sed 's/^ERROR://')
            echo "Error: Failed to parse config file: $ERROR_MSG"
            exit 1
        else
            echo "Error: Failed to determine required validation JSON file from config."
            exit 1
        fi
    fi
    
    if echo "$VAL_JSON_RESULT" | grep -q "^FOUND:"; then
        VAL_JSON_INFO=$(echo "$VAL_JSON_RESULT" | sed 's/^FOUND://')
        if [ -f "$VAL_JSON_INFO" ]; then
            export VAL_JSON_OVERRIDE="$VAL_JSON_INFO"
            echo "Using model-local validation JSON: $VAL_JSON_OVERRIDE"
        else
            echo "Error: Validation JSON file not found: $VAL_JSON_INFO"
            exit 1
        fi
    else
        echo "Error: Could not determine validation JSON file from config."
        exit 1
    fi
else
    echo "Warning: Config file not found, cannot auto-detect validation JSON."
    echo "Please provide --val-json argument or ensure config file exists."
fi

# Set CUDA_VISIBLE_DEVICES to the specified device
export CUDA_VISIBLE_DEVICES=$DEVICE
echo "Using GPU $DEVICE for evaluation"

export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Clear any cached dataset/evaluation data to prevent interference between models
echo "Clearing cached data to prevent model interference..."
rm -rf /tmp/detectron2_* 2>/dev/null || true
# Note: Not cleaning /tmp/tmp.* to avoid deleting our own temp config file

# Create a unique temp directory for this evaluation to avoid conflicts
export TMPDIR="/tmp/eval_$(basename "$MODEL_DIR")_$$"
mkdir -p "$TMPDIR"
echo "Using isolated temp directory: $TMPDIR"

# Build the evaluation command
EVAL_CMD="python /home/simone/fish-dvis/training_scripts/evaluate_all_checkpoints.py"
EVAL_CMD="$EVAL_CMD --model-dir \"$MODEL_DIR\""
EVAL_CMD="$EVAL_CMD --config-file \"$TEMP_CONFIG\""
EVAL_CMD="$EVAL_CMD --output-dir \"$FINAL_OUTPUT_DIR\""
EVAL_CMD="$EVAL_CMD --device $DEVICE"
EVAL_CMD="$EVAL_CMD --monitor-memory"

# Add checkpoint range if specified
if [ -n "$CHECKPOINT_RANGE" ]; then
    EVAL_CMD="$EVAL_CMD --checkpoint-range \"$CHECKPOINT_RANGE\""
fi

# Add skip-existing if reusing existing run or no config changes
if [ -z "$RUN_NUM" ] || [ "$REUSE_EXISTING_RUN" = true ]; then
    EVAL_CMD="$EVAL_CMD --skip-existing"
fi

# Run the evaluation script
echo "DEBUG: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DEBUG: Running command:"
echo "$EVAL_CMD"

# Run the evaluation and wait for it to complete
eval $EVAL_CMD
EVAL_EXIT_CODE=$?

# Note: Temporary files will be automatically cleaned up by the system
# No manual cleanup needed for /tmp files

# Show completion messages only if evaluation was successful
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Checkpoint evaluation completed!"
    echo "Results saved in: $FINAL_OUTPUT_DIR"
    echo "Check the following files for results:"
    echo "  - $FINAL_OUTPUT_DIR/checkpoint_comparison.png (comparison plots)"
    echo "  - $FINAL_OUTPUT_DIR/checkpoint_summary.csv (summary table)"
    echo "  - $FINAL_OUTPUT_DIR/checkpoint_*/ (individual checkpoint results)"
    if [ -n "$RUN_NUM" ]; then
        echo "  - $FINAL_OUTPUT_DIR/config_diff.txt (configuration changes)"
    fi
else
    echo "Checkpoint evaluation failed with exit code $EVAL_EXIT_CODE"
fi

# Exit with the same code as the evaluation script
exit $EVAL_EXIT_CODE 
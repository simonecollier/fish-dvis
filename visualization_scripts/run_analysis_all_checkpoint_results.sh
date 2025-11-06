#!/bin/bash

# Comprehensive checkpoint analysis script wrapper
# This script provides a convenient interface to the analyze_checkpoint_results.py script
# Usage: ./run_analysis_all_checkpoint_results.sh <model_dir> [options]

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <model_dir> [options]"
    echo ""
    echo "Examples:"
    echo "  # Automatic analysis (computes mask metrics if missing)"
    echo "  $0 /path/to/model"
    echo ""
    echo "  # Force recompute all mask metrics"
    echo "  $0 /path/to/model --run-mask-metrics"
    echo ""
    echo "  # Quick analysis with confidence threshold"
    echo "  $0 /path/to/model --confidence-threshold 0.05"
    echo ""
    echo "  # Parallel processing (much faster for many checkpoints)"
    echo "  $0 /path/to/model --parallel"
    echo ""
    echo "  # Parallel processing with specific number of workers"
    echo "  $0 /path/to/model --parallel --max-workers 6"
    echo ""
    echo "  # Quick analysis + parallel processing"
    echo "  $0 /path/to/model --confidence-threshold 0.05 --parallel"
    echo ""
    echo "  # Basic summary only"
    echo "  $0 /path/to/model --analysis-level basic"
    echo ""
    echo "  # Skip mask metrics computation (use existing data only)"
    echo "  $0 /path/to/model --skip-mask-metrics"
    echo ""
    echo "  # Analyze specific run (for multi-run evaluations)"
    echo "  $0 /path/to/model --run-number 2"
    echo ""
    echo "  # Analyze all runs and create comparison"
    echo "  $0 /path/to/model --compare-runs"
    echo ""
    echo "  # Use custom configuration file"
    echo "  $0 /path/to/model --config-file /path/to/custom/config.yaml"
    echo ""
    echo "Options:"
    echo "  --run-mask-metrics     Force recompute all mask metrics (overwrites existing)"
    echo "  --skip-mask-metrics    Skip mask metrics analysis (use existing data only)"
    echo "  --parallel             Run mask metrics analysis in parallel (much faster)"
    echo "  --max-workers          Maximum number of parallel workers (default: CPU count - 2)"
    echo "  --analysis-level       Level of analysis (basic or comprehensive)"
    echo "  --confidence-threshold Confidence threshold for filtering predictions (default: 0.0 = no threshold)"
    echo "  --val-json            Path to validation JSON file"
    echo "  --config-file         Path to custom configuration file (default: model_dir/config.yaml)"
    echo "  --run-number          Specific run number to analyze (for multi-run structure)"
    echo "  --compare-runs        Compare results across all runs"
    echo "  --list-runs           List all available runs and exit"
    exit 1
}

# Function to detect file structure
detect_structure() {
    local model_dir="$1"
    local checkpoint_eval_dir="$model_dir/checkpoint_evaluations"
    
    if [ ! -d "$checkpoint_eval_dir" ]; then
        echo "none"
        return
    fi
    
    # Check if it's a multi-run structure (has run_* directories)
    if ls "$checkpoint_eval_dir"/run_* 1> /dev/null 2>&1; then
        echo "multi-run"
    # Check if it's a single-run structure (has checkpoint_* directories directly)
    elif ls "$checkpoint_eval_dir"/checkpoint_* 1> /dev/null 2>&1; then
        echo "single-run"
    else
        echo "none"
    fi
}

# Function to list available runs
list_runs() {
    local model_dir="$1"
    local checkpoint_eval_dir="$model_dir/checkpoint_evaluations"
    
    if [ ! -d "$checkpoint_eval_dir" ]; then
        echo "No checkpoint evaluations found in $model_dir"
        return
    fi
    
    local structure=$(detect_structure "$model_dir")
    
    if [ "$structure" = "multi-run" ]; then
        echo "Available runs:"
        for run_dir in "$checkpoint_eval_dir"/run_*; do
            if [ -d "$run_dir" ]; then
                run_name=$(basename "$run_dir")
                checkpoint_count=$(ls "$run_dir"/checkpoint_* 2>/dev/null | wc -l)
                echo "  $run_name: $checkpoint_count checkpoints"
                
                # Show config info if available
                if [ -f "$run_dir/config.yaml" ]; then
                    echo "    Config: $(basename "$run_dir")/config.yaml"
                fi
                if [ -f "$run_dir/config_diff.txt" ]; then
                    echo "    Changes: $(basename "$run_dir")/config_diff.txt"
                fi
            fi
        done
    elif [ "$structure" = "single-run" ]; then
        echo "Single run structure detected:"
        checkpoint_count=$(ls "$checkpoint_eval_dir"/checkpoint_* 2>/dev/null | wc -l)
        echo "  checkpoint_evaluations/: $checkpoint_count checkpoints"
    else
        echo "No evaluation results found"
    fi
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
RUN_NUMBER=""
COMPARE_RUNS=false
LIST_RUNS=false
CONFIG_FILE=""
ADDITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-number)
            RUN_NUMBER="$2"
            shift 2
            ;;
        --compare-runs)
            COMPARE_RUNS=true
            shift
            ;;
        --list-runs)
            LIST_RUNS=true
            shift
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            # Only pass arguments that the Python script understands
            case $1 in
                --run-mask-metrics|--skip-mask-metrics|--parallel|--analysis-level|--confidence-threshold|--val-json|--max-workers)
                    ADDITIONAL_ARGS+=("$1")
                    if [[ $1 == --analysis-level || $1 == --confidence-threshold || $1 == --val-json || $1 == --max-workers ]]; then
                        ADDITIONAL_ARGS+=("$2")
                        shift 2
                    else
                        shift
                    fi
                    ;;
                *)
                    echo "Warning: Unknown argument '$1' will be ignored"
                    shift
                    ;;
            esac
            ;;
    esac
done

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Check if custom config file exists (if provided)
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Custom config file $CONFIG_FILE does not exist"
        exit 1
    fi
    echo "Using custom config file: $CONFIG_FILE"
fi

# Handle list-runs option
if [ "$LIST_RUNS" = true ]; then
    list_runs "$MODEL_DIR"
    exit 0
fi

# Detect file structure
STRUCTURE=$(detect_structure "$MODEL_DIR")

if [ "$STRUCTURE" = "none" ]; then
    echo "Error: No checkpoint evaluations found in $MODEL_DIR"
    echo "Run checkpoint evaluation first using run_eval_all_checkpoints.sh"
    exit 1
fi

echo "Starting comprehensive checkpoint analysis..."
echo "Model directory: $MODEL_DIR"
echo "Structure detected: $STRUCTURE"

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Determine config file to use for val.json detection
if [ -n "$CONFIG_FILE" ]; then
    VAL_JSON_CONFIG_FILE="$CONFIG_FILE"
else
    VAL_JSON_CONFIG_FILE="$MODEL_DIR/config.yaml"
fi

# Check for model-local validation JSON files (including strided versions)
# Parse config to detect stride patterns in dataset names
if [ -f "$VAL_JSON_CONFIG_FILE" ]; then
    # Use python to parse YAML and detect stride patterns
    VAL_JSON_RESULT=$(CONFIG_FILE="$VAL_JSON_CONFIG_FILE" MODEL_DIR="$MODEL_DIR" python3 <<PYTHON_SCRIPT
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
        stride_match = re.search(r'_stride(\d+)$', dataset_name)
        if 'val' in dataset_name:
            if stride_match:
                stride_value = stride_match.group(1)
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
            echo "If using a stride dataset, make sure the corresponding strided JSON file was copied during training."
            exit 1
        elif echo "$VAL_JSON_RESULT" | grep -q "^ERROR:"; then
            ERROR_MSG=$(echo "$VAL_JSON_RESULT" | sed 's/^ERROR://')
            echo "Error: Failed to parse config file: $ERROR_MSG"
            exit 1
        else
            echo "Warning: Could not determine required validation JSON file from config. Will use default detection."
            VAL_JSON_PATH=""
        fi
    else
        if echo "$VAL_JSON_RESULT" | grep -q "^FOUND:"; then
            VAL_JSON_PATH=$(echo "$VAL_JSON_RESULT" | sed 's/^FOUND://')
            if [ -f "$VAL_JSON_PATH" ]; then
                echo "Using model-local validation JSON: $VAL_JSON_PATH"
            else
                echo "Error: Validation JSON file not found: $VAL_JSON_PATH"
                exit 1
            fi
        else
            echo "Warning: Could not determine validation JSON file from config. Will use default detection."
            VAL_JSON_PATH=""
        fi
    fi
else
    echo "Warning: Config file not found: $VAL_JSON_CONFIG_FILE. Will use default detection."
    VAL_JSON_PATH=""
fi

# Handle different structures
if [ "$STRUCTURE" = "multi-run" ]; then
    if [ "$COMPARE_RUNS" = true ]; then
        echo "Comparing results across all runs..."
        # This would require a new script to compare runs
        echo "Cross-run comparison not yet implemented"
        echo "Please analyze individual runs first"
        exit 1
    elif [ -n "$RUN_NUMBER" ]; then
        # Analyze specific run
        RUN_DIR="$MODEL_DIR/checkpoint_evaluations/run_$RUN_NUMBER"
        if [ ! -d "$RUN_DIR" ]; then
            echo "Error: Run $RUN_NUMBER not found"
            list_runs "$MODEL_DIR"
            exit 1
        fi
        echo "Analyzing run $RUN_NUMBER..."
        cmd="python /home/simone/fish-dvis/visualization_scripts/analyze_checkpoint_results.py --model-dir \"$MODEL_DIR\" --evaluation-dir \"$RUN_DIR\""
        if [ -n "$CONFIG_FILE" ]; then
            cmd="$cmd --config-file \"$CONFIG_FILE\""
        fi
        # Only add val-json if user hasn't explicitly provided it
        if [ -n "$VAL_JSON_PATH" ] && ! printf '%s\n' "${ADDITIONAL_ARGS[@]}" | grep -q "^--val-json$"; then
            cmd="$cmd --val-json \"$VAL_JSON_PATH\""
        fi
    else
        # Analyze the most recent run
        LATEST_RUN=$(ls -d "$MODEL_DIR/checkpoint_evaluations"/run_* 2>/dev/null | tail -1)
        if [ -z "$LATEST_RUN" ]; then
            echo "Error: No runs found"
            exit 1
        fi
        RUN_NUM=$(basename "$LATEST_RUN")
        echo "Analyzing latest run: $RUN_NUM"
        cmd="python /home/simone/fish-dvis/visualization_scripts/analyze_checkpoint_results.py --model-dir \"$MODEL_DIR\" --evaluation-dir \"$LATEST_RUN\""
        if [ -n "$CONFIG_FILE" ]; then
            cmd="$cmd --config-file \"$CONFIG_FILE\""
        fi
        # Only add val-json if user hasn't explicitly provided it
        if [ -n "$VAL_JSON_PATH" ] && ! printf '%s\n' "${ADDITIONAL_ARGS[@]}" | grep -q "^--val-json$"; then
            cmd="$cmd --val-json \"$VAL_JSON_PATH\""
        fi
    fi
else
    # Single run structure
    echo "Analyzing single run structure..."
    cmd="python /home/simone/fish-dvis/visualization_scripts/analyze_checkpoint_results.py --model-dir \"$MODEL_DIR\""
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config-file \"$CONFIG_FILE\""
    fi
    # Only add val-json if user hasn't explicitly provided it
    if [ -n "$VAL_JSON_PATH" ] && ! printf '%s\n' "${ADDITIONAL_ARGS[@]}" | grep -q "^--val-json$"; then
        cmd="$cmd --val-json \"$VAL_JSON_PATH\""
    fi
fi

# Add additional arguments
for arg in "${ADDITIONAL_ARGS[@]}"; do
    cmd="$cmd $arg"
done

echo "Running command: $cmd"

# Run the analysis script
eval $cmd

echo "Analysis complete!"

# Show results based on structure
if [ "$STRUCTURE" = "multi-run" ]; then
    if [ -n "$RUN_NUMBER" ]; then
        RUN_DIR="$MODEL_DIR/checkpoint_evaluations/run_$RUN_NUMBER"
    else
        RUN_DIR="$LATEST_RUN"
    fi
    
    echo "Check the following files in $RUN_DIR:"
    echo ""
    echo "For comprehensive analysis:"
    echo "  - performance_comparison.png (comprehensive performance plots)"
    echo "  - coco_comparison.png (COCO vs video metrics comparison)"
    echo "  - comprehensive_comparison.png (all metric types comparison)"
    echo "  - temporal_consistency.png (temporal consistency analysis)"
    echo "  - trend_analysis.png (training progress analysis)"
    echo "  - best_checkpoints.png (best performing checkpoints)"
    echo "  - per_species_performance.png (per-species performance metrics)"
    echo "  - per_category_comparison.png (per-category track vs instance comparison)"
    echo "  - per_species_summary_table.png (per-species summary table)"
    echo "  - per_category_summary_table.png (per-category comprehensive summary table)"
    echo "  - per_species_summary.csv (per-species summary data)"
    echo "  - per_category_summary.csv (per-category summary data)"
    echo "  - tracking_metrics.png (tracking performance metrics)"
    echo "  - tracking_analysis_table.png (tracking metrics analysis table)"
    echo "  - tracking_analysis.csv (tracking metrics analysis data)"
    echo "  - comprehensive_metrics_summary.csv (summary table)"
    echo "  - model_performance_report.txt (detailed analysis report)"
    echo "  - training_loss_curves_by_species.png (training loss by species)"
    echo "  - ce_loss_by_species.png (cross-entropy loss by species)"
    echo ""
    echo "For basic analysis:"
    echo "  - mask_metrics_summary.png (basic summary plot)"
    echo "  - mask_metrics_summary.csv (basic summary table)"
    echo ""
    echo "Configuration:"
    echo "  - config.yaml (config used for this run)"
    echo "  - config_diff.txt (changes from original config)"
    echo ""
    echo "Individual checkpoint results:"
    echo "  - checkpoint_*/ (individual checkpoint results)"
    echo "    - Each checkpoint directory contains:"
    echo "      - inference/results.json (raw evaluation results)"
    echo "      - inference/mask_metrics.csv (combined metrics - backward compatibility)"
    echo "      - inference/mask_metrics_frame.csv (frame-level metrics)"
    echo "      - inference/mask_metrics_video.csv (video-level metrics)"
    echo "      - inference/mask_metrics_category.csv (category-level metrics)"
    echo "      - inference/mask_metrics_dataset.csv (dataset-level metrics)"
    echo "      - inference/confusion_matrix.png (confusion matrix)"
    echo "      - inference/AP_per_category.png (AP per category plot)"
else
    echo "Check the following files in $MODEL_DIR:"
    echo ""
    echo "For comprehensive analysis:"
    echo "  - performance_comparison.png (comprehensive performance plots)"
    echo "  - coco_comparison.png (COCO vs video metrics comparison)"
    echo "  - comprehensive_comparison.png (all metric types comparison)"
    echo "  - temporal_consistency.png (temporal consistency analysis)"
    echo "  - trend_analysis.png (training progress analysis)"
    echo "  - best_checkpoints.png (best performing checkpoints)"
    echo "  - per_species_performance.png (per-species performance metrics)"
    echo "  - per_category_comparison.png (per-category track vs instance comparison)"
    echo "  - per_species_summary_table.png (per-species summary table)"
    echo "  - per_category_summary_table.png (per-category comprehensive summary table)"
    echo "  - per_species_summary.csv (per-species summary data)"
    echo "  - per_category_summary.csv (per-category summary data)"
    echo "  - tracking_metrics.png (tracking performance metrics)"
    echo "  - tracking_analysis_table.png (tracking metrics analysis table)"
    echo "  - tracking_analysis.csv (tracking metrics analysis data)"
    echo "  - area_weighted_comparison.png (area-weighted vs regular mAP@0.5 comparison)"
    echo "  - area_weighted_summary_table.png (area-weighted metrics summary table)"
    echo "  - area_weighted_summary.csv (area-weighted metrics summary data)"
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
    echo "      - inference/mask_metrics.csv (combined metrics - backward compatibility)"
    echo "      - inference/mask_metrics_frame.csv (frame-level metrics)"
    echo "      - inference/mask_metrics_video.csv (video-level metrics)"
    echo "      - inference/mask_metrics_category.csv (category-level metrics)"
    echo "      - inference/mask_metrics_dataset.csv (dataset-level metrics)"
    echo "      - inference/confusion_matrix.png (confusion matrix)"
    echo "      - inference/AP_per_category.png (AP per category plot)"
fi 
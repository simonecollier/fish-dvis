#!/bin/bash
# Wrapper script to run attention extraction for a single video using the modified training script
# with optional visualization functionality

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables exactly like evaluation
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1 to avoid GPU 0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Parse command line arguments
VISUALIZE=false
NO_SHOW=false
FIG_SIZE="12 10"
CMAP="viridis"
VERBOSE=false
NO_SKIP_CONNECTION=false

# Check if video ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <video_id> [attention_output_dir] [top_n] [--visualize] [--no-show] [--figsize W H] [--cmap COLORMAP] [--verbose] [--no-skip-connection]"
    echo "Example: $0 10"
    echo "Example: $0 10 /path/to/output"
    echo "Example: $0 10 /path/to/output 5"
    echo "Example: $0 10 /path/to/output 5 --visualize"
    echo "Example: $0 10 /path/to/output 5 --visualize --no-show --figsize 15 12 --cmap plasma"
    echo "Example: $0 10 /path/to/output 5 --visualize --no-skip-connection"
    echo ""
    echo "Options:"
    echo "  --visualize        Generate heatmap visualizations after extraction"
    echo "  --no-show          Do not display plots (only save) - useful for headless systems"
    echo "  --figsize W H      Set figure size for plots (default: 12 10)"
    echo "  --cmap COLORMAP    Set colormap for heatmaps (default: viridis)"
    echo "  --verbose          Enable verbose logging for visualization"
    echo "  --no-skip-connection  Disable skip connection simulation during rollout (pure matrix multiplication)"
    echo ""
    echo "Available colormaps: viridis, plasma, inferno, magma, hot, cool, RdYlBu, etc."
    exit 1
fi

VIDEO_ID=$1
ATTENTION_OUTPUT_DIR=${2:-"/home/simone/store/simone/attention_maps_single_video"}
TOP_N=${3:-1}

# Parse additional arguments
if [ -n "$2" ]; then
    shift 2
    if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        # If the third argument is a number, shift it too
        shift 1
    fi
else
    shift 1
fi
while [[ $# -gt 0 ]]; do
    case $1 in
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --no-show)
            NO_SHOW=true
            shift
            ;;
        --figsize)
            FIG_SIZE="$2 $3"
            shift 3
            ;;
        --cmap)
            CMAP="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --no-skip-connection)
            NO_SKIP_CONNECTION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Extracting attention maps with rollout for video ID: $VIDEO_ID"
echo "Output directory: $ATTENTION_OUTPUT_DIR"
echo "Top N predictions: $TOP_N"
echo "Using GPU 1 (CUDA_VISIBLE_DEVICES=1)"
if [ "$NO_SKIP_CONNECTION" = true ]; then
    echo "Skip connection simulation: DISABLED (pure matrix multiplication)"
else
    echo "Skip connection simulation: ENABLED (default)"
fi

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
    $([ "$NO_SKIP_CONNECTION" = true ] && echo "--no-skip-connection") \
    MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_s1_fixed/model_0003635.pth \
    OUTPUT_DIR /tmp/attention_eval_single

echo "Attention extraction with rollout completed!"
echo "Check the output directory: $ATTENTION_OUTPUT_DIR"
echo "The output will contain rolled-out attention maps instead of individual layer maps."

# Generate visualizations if requested
if [ "$VISUALIZE" = true ]; then
    echo ""
    echo "Generating attention map visualizations..."
    
    # Find the generated JSON file (check for different versions)
    if [ "$NO_SKIP_CONNECTION" = true ]; then
        ROLLOUT_JSON_FILE="$ATTENTION_OUTPUT_DIR/attention_maps_video_${VIDEO_ID}_top_${TOP_N}_rollout_no_skip.json"
    else
        ROLLOUT_JSON_FILE="$ATTENTION_OUTPUT_DIR/attention_maps_video_${VIDEO_ID}_top_${TOP_N}_rollout.json"
    fi
    REGULAR_JSON_FILE="$ATTENTION_OUTPUT_DIR/attention_maps_video_${VIDEO_ID}_top_${TOP_N}.json"
    
    # Prefer rollout file if it exists, otherwise use regular file
    if [ -f "$ROLLOUT_JSON_FILE" ]; then
        JSON_FILE="$ROLLOUT_JSON_FILE"
        echo "Using rollout attention data: $JSON_FILE"
    elif [ -f "$REGULAR_JSON_FILE" ]; then
        JSON_FILE="$REGULAR_JSON_FILE"
        echo "Using regular attention data: $JSON_FILE"
    else
        JSON_FILE=""
    fi
    
    if [ -f "$JSON_FILE" ]; then
        echo "Found attention data file: $JSON_FILE"
        
        # Set up visualization arguments
        VIZ_ARGS="--output-dir $ATTENTION_OUTPUT_DIR/visualizations"
        
        if [ "$NO_SHOW" = true ]; then
            VIZ_ARGS="$VIZ_ARGS --no-show"
        fi
        
        if [ "$VERBOSE" = true ]; then
            VIZ_ARGS="$VIZ_ARGS --verbose"
        fi
        
        VIZ_ARGS="$VIZ_ARGS --figsize $FIG_SIZE --cmap $CMAP"
        
        # Run visualization
        cd /home/simone/fish-dvis/attention_analysis
        python visualize_attention.py "$JSON_FILE" $VIZ_ARGS
        
        if [ $? -eq 0 ]; then
            echo "Visualization completed successfully!"
            echo "Heatmaps saved to: $ATTENTION_OUTPUT_DIR/visualizations/"
        else
            echo "Error during visualization. Check the logs above."
        fi
    else
        echo "Error: Could not find attention data file: $JSON_FILE"
        echo "Please check if the attention extraction completed successfully."
    fi
fi

echo ""
echo "Process completed!"

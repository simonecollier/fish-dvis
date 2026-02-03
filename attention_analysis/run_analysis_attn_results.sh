#!/bin/bash
# Summarize attention results based on input attention directory
# Usage: ./run_analysis_attn_results.sh <attention_dir>

# Check if attention_dir argument is provided
if [ -z "$1" ]; then
    echo "Error: attention_dir argument is required"
    echo "Usage: $0 <attention_dir>"
    echo "Example: $0 /path/to/attention/directory"
    exit 1
fi

ATTENTION_DIR="$1"

# Check if the directory exists
if [ ! -d "${ATTENTION_DIR}" ]; then
    echo "Error: Directory does not exist: ${ATTENTION_DIR}"
    exit 1
fi

echo "Processing attention directory: ${ATTENTION_DIR}"

# Summarize activation projections and plot
python /home/simone/fish-dvis/attention_analysis/summarize_activation_proj.py "${ATTENTION_DIR}" --plot

# Summarize tracker attention
python /home/simone/fish-dvis/attention_analysis/summarize_tracker_attn.py "${ATTENTION_DIR}"

# Plot decoder attention
python /home/simone/fish-dvis/attention_analysis/plot_decoder_attn.py "${ATTENTION_DIR}" --scale per_frame
python /home/simone/fish-dvis/attention_analysis/plot_decoder_attn.py "${ATTENTION_DIR}" --scale temporal_per_frame

# Create videos of prediction and attention
python /home/simone/fish-dvis/attention_analysis/pred_vids_attn.py "${ATTENTION_DIR}"
python /home/simone/fish-dvis/attention_analysis/pred_vids_attn.py "${ATTENTION_DIR}" --reorder
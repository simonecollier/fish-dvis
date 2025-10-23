#!/bin/bash
# Wrapper script to run attention extraction with proper environment setup
# This mimics the environment setup from train_run.sh

# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:/home/simone/fish-dvis/attention_analysis

# Run the attention extraction script
cd /home/simone/fish-dvis/attention_analysis
python run_attention_extraction.py "$@"

#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Define paths
CONFIG_FILE="/home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml"
MODEL_WEIGHTS="/home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth"
VAL_JSON="/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/fold6/checkpoint_0004443/val_fold6_all_frames.json"
OUTPUT_DIR="/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/attention/fold6_4443_attn_extra"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Copy val-json file to output directory
VAL_JSON_BASENAME=$(basename "${VAL_JSON}")
VAL_JSON_COPY="${OUTPUT_DIR}/${VAL_JSON_BASENAME}"
cp "${VAL_JSON}" "${VAL_JSON_COPY}"
echo "Copied ${VAL_JSON} to ${VAL_JSON_COPY}"

# To enable debug prints, add --debug to the command line
python /home/simone/fish-dvis/attention_analysis/train_net_video_eval_attn.py \
  --num-gpus 1 \
  --config-file "${CONFIG_FILE}" \
  --val-json "${VAL_JSON_COPY}" \
  --eval-only \
  MODEL.WEIGHTS "${MODEL_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}"

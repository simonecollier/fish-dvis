#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
# To enable debug prints, add --debug to the command line
python /home/simone/fish-dvis/attention_analysis/train_net_video_eval_attn.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_ft123/config.yaml  \
  --eval-only \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_ft123/model_0003231.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_ft123/attn_extract_3231

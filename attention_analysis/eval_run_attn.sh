#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
# To enable debug prints, add --debug to the command line
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/attention_analysis/train_net_video_eval_attn.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo/config.yaml  \
  --eval-only \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo/model_0003635.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo/oct30_attn_0003635_eval

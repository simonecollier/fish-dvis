#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/fish-dvis/configs/DAQ_Offline_VitAdapterL_fishway.yaml  \
  --eval-only \
  MODEL.WEIGHTS /home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/model_0006999.pth \
  OUTPUT_DIR /home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped

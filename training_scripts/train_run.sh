#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/fish-dvis/configs/DAQ_Offline_VitAdapterL_fishway.yaml \
  --resume \
  MODEL.WEIGHTS /home/simone/checkpoints/model_ytvis21_offline_vitl.pth

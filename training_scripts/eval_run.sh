#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed1
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed1/val.json
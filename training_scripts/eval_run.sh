#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# # Parse optional --val-json argument
# VAL_JSON_ARG=""
# if [ "$1" == "--val-json" ] && [ -n "$2" ]; then
#   VAL_JSON_ARG="--val-json $2"
#   shift 2
# fi

CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/eval_4443_all_frames

# To enable debug prints, add --debug to the command line
# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold1/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold1_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold1/model_0003433.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold1/eval_3433_all_frames


# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold2/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold2_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold2/model_0005655.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold2/eval_5655_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold3/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold3_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold3/model_0003433.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold3/eval_3433_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold4/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold4_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold4/model_0003635.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold4/eval_3635_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold5/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold5_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold5/model_0005655.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold5/eval_5655_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0006059.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/eval_6059_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold1/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold1_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold1/model_0005655.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold1/eval_5655_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold2/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold2_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold2/model_0005857.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold2/eval_5857_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold3/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold3_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold3/model_0005251.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold3/eval_5251_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold4/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold4_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold4/model_0006059.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold4/eval_6059_all_frames

# CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
#   --num-gpus 1 \
#   --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold5/config.yaml  \
#   --eval-only \
#   --val-json /home/simone/shared-data/fishway_ytvis/val_fold5_all_frames.json \
#   MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold5/model_0005251.pth \
#   OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold5/eval_5251_all_frames
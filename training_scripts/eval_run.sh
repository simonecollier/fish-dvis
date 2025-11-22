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

## Camera Fold 6
# Scrambled 1
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed1
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed1/val.json

# Scrambled 2
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed2.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed2
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed2.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed2/val.json

# Scrambled 3
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed3.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed3
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed3.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed3/val.json

# Scrambled 4
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed4.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed4
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed4.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed4/val.json

# Scrambled 5
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed5.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed5
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed5.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed5/val.json

# Scrambled 6
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed6.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed6
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed6.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed6/val.json

# Scrambled 7
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed7.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed7
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed7.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed7/val.json

# Scrambled 8
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed8.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed8
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed8.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed8/val.json

# Scrambled 9
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed9.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed9
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed9.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed9/val.json

# Scrambled 10
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed10.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed10
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed10.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_all_frames_seed10/val.json


## Silhouette Fold 6
# Scrambled 1
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed1
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed1.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed1/val.json

# Scrambled 2
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed2.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed2
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed2.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed2/val.json

# Scrambled 3
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed3.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed3
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed3.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed3/val.json

# Scrambled 4
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed4.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed4
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed4.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed4/val.json

# Scrambled 5
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed5.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed5
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed5.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed5/val.json

# Scrambled 6
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed6.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed6
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed6.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed6/val.json

# Scrambled 7
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed7.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed7
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed7.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed7/val.json

# Scrambled 8
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed8.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed8
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed8.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed8/val.json

# Scrambled 9
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed9.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed9
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed9.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed9/val.json

# Scrambled 10
CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml  \
  --eval-only \
  --val-json /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed10.json \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed10
# Copy val json to output directory
cp /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_scrambled_seed10.json /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed10/val.json
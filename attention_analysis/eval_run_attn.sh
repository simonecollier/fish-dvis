#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
# To enable debug prints, add --debug to the command line
python /home/simone/fish-dvis/attention_analysis/train_net_video_eval_attn.py \
  --num-gpus 1 \
  --config-file /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold2/config.yaml  \
  --eval-only \
  MODEL.WEIGHTS /home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold2/model_0005655.pth \
  OUTPUT_DIR /home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold2_5655_attn_extra

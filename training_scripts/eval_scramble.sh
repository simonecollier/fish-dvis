#!/bin/bash
source /home/simone/.venv/bin/activate
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Define paths
CONFIG_FILE="/home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/config.yaml"
MODEL_WEIGHTS="/home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_fold6/model_0004443.pth"
BASE_OUTPUT_DIR="/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6"

# Auto-detect directory naming pattern based on BASE_OUTPUT_DIR
# Camera model uses: eval_4443_edit91_seed${seed}
# Silhouette model uses: eval_4443_all_frames_seed${seed}
if [[ "${BASE_OUTPUT_DIR}" == *"silhouette"* ]]; then
  DIR_PREFIX="eval_4443_all_frames_seed"
else
  DIR_PREFIX="eval_4443_edit91_seed"
fi

# Time tracking variables
SCRIPT_START_TIME=$(date +%s)
TOTAL_SEEDS=100
COMPLETED_SEEDS=0
AVG_TIME_PER_SEED=0

# Function to format seconds into human-readable time
format_time() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))
  
  if [ $hours -gt 0 ]; then
    printf "%dh %dm %ds" $hours $minutes $secs
  elif [ $minutes -gt 0 ]; then
    printf "%dm %ds" $minutes $secs
  else
    printf "%ds" $secs
  fi
}

echo "=========================================="
echo "Starting evaluation of ${TOTAL_SEEDS} seeds"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

## Camera Fold 6
# Evaluate all scrambled seeds (1-100)
for seed in {1..100}; do
  SEED_START_TIME=$(date +%s)
  
  echo "=========================================="
  echo "Processing seed ${seed}/${TOTAL_SEEDS}..."
  echo "=========================================="
  
  VAL_JSON="${BASE_OUTPUT_DIR}/${DIR_PREFIX}${seed}/val.json"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DIR_PREFIX}${seed}"
  RESULTS_JSON="${OUTPUT_DIR}/inference/results.json"
  
  # Check if val.json exists
  if [ ! -f "${VAL_JSON}" ]; then
    echo "Warning: ${VAL_JSON} not found. Skipping seed ${seed}."
    continue
  fi
  
  # Check if results.json already exists (evaluation already completed)
  if [ -f "${RESULTS_JSON}" ]; then
    echo "Results already exist at ${RESULTS_JSON}. Skipping seed ${seed}."
    continue
  fi
  
  CUDA_VISIBLE_DEVICES=1 python /home/simone/fish-dvis/training_scripts/train_net_video.py \
    --num-gpus 1 \
    --config-file "${CONFIG_FILE}" \
    --val-json "${VAL_JSON}" \
    --eval-only \
    MODEL.WEIGHTS "${MODEL_WEIGHTS}" \
    OUTPUT_DIR "${OUTPUT_DIR}"
  
  EXIT_CODE=$?
  SEED_END_TIME=$(date +%s)
  SEED_DURATION=$((SEED_END_TIME - SEED_START_TIME))
  COMPLETED_SEEDS=$((COMPLETED_SEEDS + 1))
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed seed ${seed} (took $(format_time $SEED_DURATION))"
  else
    echo "✗ Error processing seed ${seed} (took $(format_time $SEED_DURATION))"
  fi
  
  # Calculate average time per seed and estimate remaining time
  if [ $COMPLETED_SEEDS -eq 1 ]; then
    AVG_TIME_PER_SEED=$SEED_DURATION
    echo "  First seed completed. Estimated time per seed: $(format_time $AVG_TIME_PER_SEED)"
  else
    # Update running average (weighted towards recent times)
    AVG_TIME_PER_SEED=$(( (AVG_TIME_PER_SEED * (COMPLETED_SEEDS - 1) + SEED_DURATION) / COMPLETED_SEEDS ))
  fi
  
  # Calculate elapsed and estimated remaining time
  ELAPSED_TIME=$((SEED_END_TIME - SCRIPT_START_TIME))
  REMAINING_SEEDS=$((TOTAL_SEEDS - COMPLETED_SEEDS))
  ESTIMATED_REMAINING=$((AVG_TIME_PER_SEED * REMAINING_SEEDS))
  ESTIMATED_TOTAL=$((ELAPSED_TIME + ESTIMATED_REMAINING))
  ESTIMATED_END_TIME=$((SCRIPT_START_TIME + ESTIMATED_TOTAL))
  
  echo "  Progress: ${COMPLETED_SEEDS}/${TOTAL_SEEDS} seeds completed"
  echo "  Elapsed time: $(format_time $ELAPSED_TIME)"
  if [ $REMAINING_SEEDS -gt 0 ]; then
    echo "  Estimated remaining: $(format_time $ESTIMATED_REMAINING)"
    echo "  Estimated completion: $(date -d "@${ESTIMATED_END_TIME}" '+%Y-%m-%d %H:%M:%S')"
  fi
  
  echo ""
done

SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo "=========================================="
echo "All evaluations completed!"
echo "Total time: $(format_time $TOTAL_DURATION)"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
## General Usage

### 1. Data Prep
i. Convert coco fishway dataset into ytvis dataset using [`01_convert_coco_to_ytvis.py`](data_scripts/01_convert_coco_to_ytvis.py). Change the hardcoded paths at the bottom of the script.

ii. [`03_create_train_val_jsons.py`](data_scripts/03_create_train_val_jsons.py)

iii. [`04_validate_ytvis.py`](data_scripts/04_validate_ytvis.py)

### 2. Configure Your Experiment
Edit or create a config file in `configs/` to specify model parameters, dataset paths, and training options.

### 3. Train Your Model

#### Core Training Scripts
- **`train_net_video.py`**: Main training script for video segmentation models
- **`train_run.sh`**: Shell script wrapper for training with predefined settings
- **`eval_run.sh`**: Shell script wrapper for evaluation with predefined settings

#### Checkpoint Evaluation Scripts
- **`evaluate_all_checkpoints.py`**: Evaluates all checkpoints in a model directory (evaluation only, no plots)
- **`run_eval_all_checkpoints.sh`**: Shell script wrapper for basic checkpoint evaluation

#### Analysis Scripts
- **`analyze_checkpoint_results.py`**: **NEW** - Combined comprehensive analysis script (recommended) - **MOVED TO visualization_scripts/**
- **`run_analysis_all_checkpoint_results.sh`**: Shell script wrapper for comprehensive analysis - **MOVED TO visualization_scripts/**

### Visualization Scripts (`visualization_scripts/`)

#### Core Visualization Scripts
- **`mask_metrics.py`**: Core script for computing mask metrics (IoU, boundary F-measure, mAP)
- **`output_prediction_videos.py`**: Creates video outputs with prediction overlays
- **`plot_training_loss.py`**: Plots training loss curves by fish species

#### Checkpoint Analysis Scripts
- **`analyze_checkpoint_results.py`**: **NEW** - Combined comprehensive analysis script (recommended)
- **`run_analysis_all_checkpoint_results.sh`**: Shell script wrapper for comprehensive analysis

#### Model Analysis Scripts
- **`compare_models.py`**: Compares two model performances and analyzes configuration impact
- **`run_model_comparison.sh`**: Shell script wrapper for model comparison

### Script Purposes Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_net_video.py` | Main training script | Training new models |
| `evaluate_all_checkpoints.py` | Evaluate all checkpoints (evaluation only) | Model selection |
| `mask_metrics.py` | Compute detailed metrics | Performance analysis |
| `output_prediction_videos.py` | Create video outputs | Results visualization |
| `analyze_checkpoint_results.py` | **NEW** - Combined analysis | **RECOMMENDED** - All checkpoint analysis (in visualization_scripts/) |
| `compare_models.py` | Compare two models | Model selection |

## Getting Started

### 1. Prepare Your Data
Use the scripts in `data_scripts/` to convert and register your fish video dataset.

### 2. Configure Your Experiment
Edit or create a config file in `configs/` to specify model parameters, dataset paths, and training options.

### 3. Train Your Model
```bash
# Basic training
./training_scripts/train_run.sh

# Or with custom settings
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus
python training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file configs/DAQ_Offline_VitAdapterL_fishway.yaml \
  MODEL.WEIGHTS /path/to/checkpoint.pth

# Evaluate a single checkpoint
python training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file configs/DAQ_Offline_VitAdapterL_fishway.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/checkpoint.pth \
  OUTPUT_DIR /path/to/output
```

### 4. Evaluate Your Model
```bash
# Basic evaluation
./training_scripts/eval_run.sh

# Evaluate all checkpoints
./training_scripts/run_eval_all_checkpoints.sh \
  /path/to/model/directory \
  /path/to/config.yaml
```

### 5. Analyze Results
```bash
# Comprehensive analysis (RECOMMENDED)
./visualization_scripts/run_analysis_all_checkpoint_results.sh /path/to/model/directory

# Run mask metrics and then analyze
./visualization_scripts/run_analysis_all_checkpoint_results.sh /path/to/model/directory --run-mask-metrics

# Fast mode for quick analysis
./visualization_scripts/run_analysis_all_checkpoint_results.sh /path/to/model/directory --run-mask-metrics --fast-mode

# Basic summary only
./visualization_scripts/run_analysis_all_checkpoint_results.sh /path/to/model/directory --analysis-level basic

```

### 6. Visualize Predictions
```bash
# Create prediction videos
python visualization_scripts/output_prediction_videos.py \
  --results-json /path/to/results.json \
  --val-json /data/fishway_ytvis/val.json \
  --image-root /data/fishway_ytvis/all_videos \
  --output-dir /path/to/output

# Compute detailed metrics
python visualization_scripts/mask_metrics.py \
  --results-json /path/to/results.json \
  --val-json /data/fishway_ytvis/val.json \
  --csv-path metrics.csv \
  --confidence-threshold 0.01

# Plot training loss curves by fish species
python visualization_scripts/plot_training_loss.py \
  --model-dir /path/to/model/directory

# Compare two models
python visualization_scripts/compare_models.py \
  --model1-dir /path/to/first/model \
  --model2-dir /path/to/second/model

# Or use the shell script wrapper
./visualization_scripts/run_model_comparison.sh /path/to/first/model /path/to/second/model

```

### 7. Compare Models
```bash
# Compare two models using analysis results
./visualization_scripts/run_model_comparison.sh /path/to/model1 /path/to/model2

# Or run the Python script directly
python visualization_scripts/compare_models.py \
  --model1-dir /path/to/model1 \
  --model2-dir /path/to/model2 \
  --output-dir /path/to/comparison/results
```

**Prerequisites**: Both model directories must contain analysis results from `run_analysis_all_checkpoint_results.sh` (look for `comprehensive_metrics_summary.csv`).

**Outputs**:
- `comprehensive_model_comparison.png` (comparison plots)
- `model_comparison_report.txt` (detailed comparison report with configuration differences)

## Individual Script Usage

### Training Scripts
```bash
# Train a model
python training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file configs/DAQ_Offline_VitAdapterL_fishway.yaml \
  MODEL.WEIGHTS /path/to/checkpoint.pth

# Evaluate a single checkpoint
python training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file configs/DAQ_Offline_VitAdapterL_fishway.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/checkpoint.pth \
  OUTPUT_DIR /path/to/output

# Evaluate all checkpoints
python training_scripts/evaluate_all_checkpoints.py \
  --model-dir /path/to/model/directory \
  --config-file /path/to/config.yaml \
  --output-dir /path/to/output \
  --num-gpus 1 \
  --skip-existing
```

### Core Visualization Scripts
```bash
# Plot training loss curves by fish species
python visualization_scripts/plot_training_loss.py \
  --model-dir /path/to/model/directory

# Compute detailed mask metrics
python visualization_scripts/mask_metrics.py \
  --results-json /path/to/results.json \
  --val-json /data/fishway_ytvis/val.json \
  --csv-path metrics.csv \
  --confidence-threshold 0.01 \
  --fast-mode

# Create prediction videos
python visualization_scripts/output_prediction_videos.py \
  --results-json /path/to/results.json \
  --val-json /data/fishway_ytvis/val.json \
  --image-root /data/fishway_ytvis/all_videos \
  --output-dir /path/to/output
```

### Model Comparison Scripts
```bash
# Compare two models using analysis results
python visualization_scripts/compare_models.py \
  --model1-dir /path/to/first/model \
  --model2-dir /path/to/second/model \
  --output-dir /path/to/comparison/results

# Or use the shell script wrapper
./visualization_scripts/run_model_comparison.sh /path/to/first/model /path/to/second/model

# Prerequisites: Both models must have been analyzed with run_analysis_all_checkpoint_results.sh
# Outputs: comprehensive_model_comparison.png and model_comparison_report.txt
```

### Model Analysis Scripts
```bash
# Compare two models
python visualization_scripts/compare_models.py \
  --model1-dir /path/to/first/model \
  --model2-dir /path/to/second/model

# Or use the shell script wrapper
./visualization_scripts/run_model_comparison.sh /path/to/first/model /path/to/second/model

# Comprehensive checkpoint analysis
python visualization_scripts/analyze_checkpoint_results.py \
  --model-dir /path/to/model/directory \
  --run-mask-metrics \
  --analysis-level comprehensive \
  --confidence-threshold 0.01 \
  --fast-mode

# The comprehensive analysis generates:
# - performance_comparison.png (performance plots)
# - trend_analysis.png (training progress)
# - best_checkpoints.png (best checkpoints)
# - model_performance_report.txt (detailed analysis with overfitting/convergence analysis)
# - training_loss_curves_by_species.png (if metrics.json exists)
# - ce_loss_by_species.png (if metrics.json exists)
```

## Common Workflows

### Model Training Workflow
1. Prepare data using `data_scripts/`
2. Configure experiment in `configs/`
3. Train model using `train_net_video.py`
4. Evaluate checkpoints using `run_eval_all_checkpoints.sh`
5. Analyze results using `run_analysis_all_checkpoint_results.sh` (in visualization_scripts/)

### Model Comparison Workflow
1. Train multiple models with different configurations
2. Evaluate all checkpoints for each model
3. Compare models using `compare_models.py`
4. Select best model based on analysis

## Configuration

Key configuration files:
- `configs/DAQ_Offline_VitAdapterL_fishway.yaml`: Main training configuration
- `configs/DAQ_Offline_VitAdapterLfishway_eval.yaml`: Evaluation configuration

## Environment Setup

```bash
# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Activate virtual environment
source /home/simone/.venv/bin/activate
```


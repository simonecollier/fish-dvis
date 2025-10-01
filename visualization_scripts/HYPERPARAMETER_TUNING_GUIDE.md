# Hyperparameter Tuning Guide for Fish-DVIS

This guide explains all the metrics output by the checkpoint analysis scripts and provides best practices for hyperparameter tuning.

## Overview

The analysis scripts output comprehensive metrics that can be used for hyperparameter tuning. The main scripts are:

1. `run_analysis_all_checkpoint_results.sh` - Runs comprehensive analysis on a single model
2. `extract_hyperparameter_metrics.py` - Extracts metrics from a single model for spreadsheet import
3. `batch_extract_hyperparameter_metrics.py` - Processes multiple models and creates a comprehensive spreadsheet

## Complete List of Metrics

### 1. Configuration Parameters (Hyperparameters)

These are extracted from model directory names and config files:

| Metric | Description | Example |
|--------|-------------|---------|
| `model_name` | Name of the model directory | `dvis_lr1e4_bs8_15f` |
| `learning_rate` | Learning rate used for training | `0.0001` |
| `batch_size` | Batch size used for training | `8` |
| `num_frames` | Number of frames used | `15` |
| `model_type` | Type of model (unmasked, masked, dvis, yolo) | `dvis` |
| `optimizer` | Optimizer used (adam, sgd, adamw) | `adam` |
| `scheduler` | Learning rate scheduler | `cosine` |
| `backbone` | Backbone network | `resnet` |

### 2. Performance Metrics (Best Epoch)

These are the key performance metrics at the best checkpoint:

| Metric | Description | Range | Priority |
|--------|-------------|-------|----------|
| `best_mean_iou` | Best Mean Intersection over Union | 0-1 | High |
| `best_map_50` | Best mAP at IoU threshold 0.5 | 0-1 | High |
| `best_map_75` | Best mAP at IoU threshold 0.75 | 0-1 | Medium |
| `best_map_25` | Best mAP at IoU threshold 0.25 | 0-1 | Medium |
| `best_map_10` | Best mAP at IoU threshold 0.1 | 0-1 | Low |
| `best_map_95` | Best mAP at IoU threshold 0.95 | 0-1 | Low |
| `best_mean_boundary_f` | Best Mean Boundary F-measure | 0-1 | Medium |

### 3. Performance Metrics (Final Epoch)

These are the performance metrics at the final checkpoint:

| Metric | Description | Range |
|--------|-------------|-------|
| `final_mean_iou` | Final Mean IoU | 0-1 |
| `final_map_50` | Final mAP@0.5 | 0-1 |
| `final_map_75` | Final mAP@0.75 | 0-1 |
| `final_map_25` | Final mAP@0.25 | 0-1 |
| `final_map_10` | Final mAP@0.1 | 0-1 |
| `final_map_95` | Final mAP@0.95 | 0-1 |
| `final_mean_boundary_f` | Final Mean Boundary F-measure | 0-1 |

### 4. Training Statistics

| Metric | Description | Units |
|--------|-------------|-------|
| `total_iterations` | Total training iterations | iterations |
| `num_checkpoints` | Number of checkpoints evaluated | count |
| `iteration_range` | Range of iterations evaluated | iterations |
| `best_mean_iou_iteration` | Iteration with best IoU | iterations |
| `best_map_50_iteration` | Iteration with best mAP@0.5 | iterations |
| `best_map_75_iteration` | Iteration with best mAP@0.75 | iterations |

### 5. Performance Improvement Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| `map_50_improvement_pct` | Percentage improvement in mAP@0.5 | % |
| `mean_iou_std` | Standard deviation of IoU across checkpoints | - |
| `mean_iou_mean` | Mean IoU across all checkpoints | - |
| `mean_iou_min` | Minimum IoU across checkpoints | - |
| `mean_iou_max` | Maximum IoU across checkpoints | - |
| `map_50_std` | Standard deviation of mAP@0.5 across checkpoints | - |
| `map_50_mean` | Mean mAP@0.5 across all checkpoints | - |
| `map_50_min` | Minimum mAP@0.5 across checkpoints | - |
| `map_50_max` | Maximum mAP@0.5 across checkpoints | - |

### 6. Training Metrics (from metrics.json)

| Metric | Description | Units |
|--------|-------------|-------|
| `final_total_loss` | Final total training loss | - |
| `final_loss_ce` | Final cross-entropy loss | - |
| `final_loss_mask` | Final mask loss | - |
| `final_loss_dice` | Final Dice loss | - |
| `final_loss_bbox` | Final bounding box loss | - |
| `final_loss_giou` | Final GIoU loss | - |
| `final_learning_rate` | Final learning rate | - |
| `total_training_time` | Total training time | seconds |
| `best_total_loss` | Best total loss during training | - |
| `worst_total_loss` | Worst total loss during training | - |
| `loss_std` | Standard deviation of loss | - |
| `loss_mean` | Mean loss during training | - |

### 7. Analysis Metrics (Training Quality)

| Metric | Description | Values |
|--------|-------------|--------|
| `overfitting_detected` | Whether overfitting was detected | True/False |
| `overfitting_severity` | Severity of overfitting | none/mild/moderate/severe |
| `convergence_ratio` | Ratio of final to best performance | 0-1 |
| `converged` | Whether model converged well | True/False |
| `training_stability` | Training stability assessment | stable/moderate/unstable |
| `early_performance` | Performance in early training phase | 0-1 |
| `mid_performance` | Performance in mid training phase | 0-1 |
| `late_performance` | Performance in late training phase | 0-1 |

### 8. Derived Metrics (Efficiency & Rankings)

| Metric | Description | Units |
|--------|-------------|-------|
| `map_50_per_iteration` | mAP@0.5 per iteration (efficiency) | - |
| `iou_per_iteration` | IoU per iteration (efficiency) | - |
| `map_50_rank` | Ranking by mAP@0.5 | rank |
| `iou_rank` | Ranking by IoU | rank |
| `composite_score` | Weighted score (0.6*mAP@0.5 + 0.4*IoU) | 0-1 |
| `composite_rank` | Ranking by composite score | rank |
| `map_50_per_hour` | mAP@0.5 per hour of training | - |

## Standard Practice for Hyperparameter Tuning

### 1. Essential Metrics to Report

For hyperparameter tuning, focus on these key metrics:

**Primary Metrics (Always Report):**
- `best_map_50` - Most important for object detection
- `best_mean_iou` - Most important for segmentation
- `learning_rate` - Key hyperparameter
- `batch_size` - Key hyperparameter
- `num_frames` - Key hyperparameter for video models
- `total_iterations` - Training duration
- `convergence_ratio` - Training quality

**Secondary Metrics (Report if Available):**
- `best_map_75` - Higher precision metric
- `overfitting_detected` - Training stability
- `training_stability` - Training quality
- `map_50_improvement_pct` - Training effectiveness

### 2. Recommended Spreadsheet Structure

Create a spreadsheet with these columns:

```
| Model Name | Learning Rate | Batch Size | Frames | Best mAP@0.5 | Best IoU | Convergence | Overfitting | Training Time | Notes |
|------------|---------------|------------|--------|--------------|----------|-------------|-------------|---------------|-------|
| dvis_lr1e4_bs8_15f | 0.0001 | 8 | 15 | 0.8234 | 0.7567 | 0.95 | No | 12.5h | Baseline |
| dvis_lr5e5_bs8_15f | 0.00005 | 8 | 15 | 0.8156 | 0.7489 | 0.98 | No | 14.2h | Lower LR |
```

### 3. Best Practices

**1. Report Best Epoch Metrics Only**
- Use `best_map_50` and `best_mean_iou` for comparison
- Don't use final epoch metrics as they may be worse due to overfitting

**2. Include Training Quality Metrics**
- `convergence_ratio` > 0.9 indicates good convergence
- `overfitting_detected` = False is preferred
- `training_stability` = "stable" is preferred

**3. Consider Efficiency**
- `map_50_per_iteration` - higher is better
- `total_training_time` - consider time constraints
- `map_50_per_hour` - efficiency metric

**4. Document Configuration Changes**
- Only report hyperparameters that were actually changed
- Include model architecture details if relevant
- Note any data augmentation or preprocessing changes

### 4. Analysis Workflow

1. **Run Analysis**: Use `run_analysis_all_checkpoint_results.sh` on each model
2. **Extract Metrics**: Use `batch_extract_hyperparameter_metrics.py` to create spreadsheet
3. **Analyze Results**: Look for patterns in hyperparameter impact
4. **Identify Best Config**: Consider both performance and training quality
5. **Plan Next Experiments**: Based on analysis results

### 5. Common Patterns to Look For

**Learning Rate Impact:**
- Too high: `overfitting_detected` = True, `training_stability` = "unstable"
- Too low: `convergence_ratio` < 0.8, slow training
- Optimal: `convergence_ratio` > 0.9, `overfitting_detected` = False

**Batch Size Impact:**
- Too small: `training_stability` = "unstable", high variance
- Too large: Memory issues, slower convergence
- Optimal: Stable training, good convergence

**Frame Count Impact:**
- Too few: Lower `best_map_50`, missing temporal information
- Too many: Diminishing returns, longer training time
- Optimal: Balance between performance and efficiency

## Usage Examples

### Iterative Hyperparameter Tuning Workflow

This is the recommended approach for systematic hyperparameter tuning:

```bash
# 1. Start with your first model
./add_model_to_spreadsheet.sh /path/to/model1

# 2. After training your second model, add it to the same spreadsheet
./add_model_to_spreadsheet.sh /path/to/model2

# 3. Continue adding models as you train them
./add_model_to_spreadsheet.sh /path/to/model3
./add_model_to_spreadsheet.sh /path/to/model4

# 4. View your results in the spreadsheet
# The script will show you the top performers after each addition
```

**Benefits of this approach:**
- Automatically runs analysis if needed
- Adds new models to existing spreadsheet
- Shows comparison with previous models
- Provides immediate feedback on performance
- Helps you decide on next hyperparameter changes

### Single Model Analysis
```bash
# Run comprehensive analysis
./run_analysis_all_checkpoint_results.sh /path/to/model --run-mask-metrics

# Extract metrics for spreadsheet (append to existing)
python extract_hyperparameter_metrics.py --model-dir /path/to/model --output-csv results.csv --append

# Update existing model in spreadsheet
python extract_hyperparameter_metrics.py --model-dir /path/to/model --output-csv results.csv --update-existing
```

### Batch Analysis
```bash
# Process all models in a directory (append to existing)
python batch_extract_hyperparameter_metrics.py --models-dir /path/to/models --output-csv all_models.csv --append

# Process specific models (append to existing)
python batch_extract_hyperparameter_metrics.py --model-list model1,model2,model3 --output-csv selected_models.csv --append

# Process models with pattern (append to existing)
python batch_extract_hyperparameter_metrics.py --models-dir /path/to/models --model-pattern "dvis_*" --output-csv dvis_models.csv --append
```

### Advanced Analysis
```bash
# Include all analysis metrics
python batch_extract_hyperparameter_metrics.py --models-dir /path/to/models --include-analysis --add-derived-metrics --append

# Basic metrics only (faster)
python batch_extract_hyperparameter_metrics.py --models-dir /path/to/models --exclude-analysis --append
```

## Iterative Workflow Best Practices

### 1. Start with a Baseline
```bash
# Train your baseline model
# Then add it to the spreadsheet
./add_model_to_spreadsheet.sh /path/to/baseline_model
```

### 2. Systematic Parameter Variation
```bash
# Vary learning rate
./add_model_to_spreadsheet.sh /path/to/model_lr1e4
./add_model_to_spreadsheet.sh /path/to/model_lr5e5
./add_model_to_spreadsheet.sh /path/to/model_lr1e5

# Vary batch size
./add_model_to_spreadsheet.sh /path/to/model_bs4
./add_model_to_spreadsheet.sh /path/to/model_bs8
./add_model_to_spreadsheet.sh /path/to/model_bs16

# Vary frame count
./add_model_to_spreadsheet.sh /path/to/model_5f
./add_model_to_spreadsheet.sh /path/to/model_10f
./add_model_to_spreadsheet.sh /path/to/model_15f
```

### 3. Monitor Progress
After each model, the script will show you:
- Current model performance
- Comparison with existing models
- Current ranking
- Top 3 performers

### 4. Make Informed Decisions
Use the spreadsheet to:
- Identify best performing configurations
- Spot trends in hyperparameter impact
- Avoid overfitting patterns
- Plan next experiments

### 5. Update Existing Models
If you retrain a model with the same name:
```bash
# Update the existing entry
python extract_hyperparameter_metrics.py --model-dir /path/to/model --output-csv results.csv --update-existing
```

## Additional Metrics to Consider

If you want to add more metrics, consider:

1. **Per-category performance** - AP for each fish species
2. **Inference speed** - FPS during evaluation
3. **Memory usage** - GPU memory consumption
4. **Data efficiency** - Performance vs. training data size
5. **Robustness metrics** - Performance on different conditions

These can be added to the extraction scripts as needed for your specific use case.

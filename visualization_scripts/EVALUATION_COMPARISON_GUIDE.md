# Evaluation Comparison System Guide

## Overview

The evaluation comparison system allows you to compare the performance of different test configurations on the same model. This helps you identify the optimal settings for your fish classification task by analyzing how different parameters affect model performance.

## What It Does

The comparison system:

1. **Loads all evaluation runs** from `checkpoint_evaluations/`
2. **Extracts configuration differences** between runs
3. **Compares performance metrics** across different configurations
4. **Generates comprehensive reports** and visualizations
5. **Identifies the best performing configuration**

## Key Features

### 📊 **Performance Metrics Analyzed**
- **Segmentation AP** (Average Precision)
- **Segmentation AP50** (AP at IoU=0.5)
- **Segmentation AP75** (AP at IoU=0.75)
- **Mean IoU** (Intersection over Union)
- **Mean Dice Coefficient**
- **Mean Precision and Recall**

### 🔧 **Configuration Parameters Tracked**
- **INSTANCE_ON** - Instance segmentation toggle
- **SEMANTIC_ON** - Semantic segmentation toggle
- **MAX_NUM** - Maximum objects per frame
- **OBJECT_MASK_THRESHOLD** - Confidence threshold
- **OVERLAP_THRESHOLD** - Non-maximum suppression threshold
- **WINDOW_SIZE** - Inference window size
- **MIN_SIZE_TEST/MAX_SIZE_TEST** - Image resolution
- **TEST_AUG_ENABLED** - Test-time augmentation
- **TEST_AUG_FLIP** - Horizontal flipping
- **TEST_AUG_MIN_SIZES** - Multi-scale testing

### 📈 **Generated Outputs**
- **Performance comparison plots** - Bar charts comparing metrics across runs
- **Configuration heatmap** - Visual representation of parameter differences
- **Performance trends** - Line plots showing progress across checkpoints
- **Summary report** - Text report with recommendations
- **CSV data files** - Detailed data for further analysis

## Usage

### Basic Usage
```bash
# Compare all evaluation runs
./run_evaluation_comparison.sh /path/to/model

# Compare with custom output directory
./run_evaluation_comparison.sh /path/to/model --output-dir /path/to/results
```

### Direct Python Usage
```bash
# Run the comparison script directly
python compare_evaluation_runs.py --model-dir /path/to/model

# With custom output directory
python compare_evaluation_runs.py --model-dir /path/to/model --output-dir /path/to/results
```

## Example Workflow

### Step 1: Run Evaluations with Different Configurations
```bash
# Baseline evaluation
./run_eval_all_checkpoints.sh /path/to/model

# Test with instance segmentation
./run_eval_all_checkpoints.sh /path/to/model --instance-on --max-num 1

# Test with higher resolution
./run_eval_all_checkpoints.sh /path/to/model --min-size-test 480 --max-size-test 640

# Test with test-time augmentation
./run_eval_all_checkpoints.sh /path/to/model --test-aug --test-min-sizes "360,480,600"

# Test with different thresholds
./run_eval_all_checkpoints.sh /path/to/model --object-mask-threshold 0.9 --overlap-threshold 0.7
```

### Step 2: Compare All Runs
```bash
# Run the comparison analysis
./run_evaluation_comparison.sh /path/to/model
```

### Step 3: Analyze Results
Check the generated files in `model_dir/evaluation_comparison/`:
- `evaluation_comparison_report.txt` - Read this first for recommendations
- `performance_comparison.png` - Visual comparison of metrics
- `configuration_heatmap.png` - Parameter differences visualization
- `performance_trends.png` - Training progress comparison

## Output Files Explained

### 📄 **evaluation_comparison_report.txt**
The main report containing:
- **Best performing configuration** identification
- **Configuration summary** for each run
- **Performance comparison table**
- **Recommendations** based on analysis
- **Configuration impact analysis**

### 📊 **performance_comparison.png**
Four-panel plot showing:
- **Segmentation AP** comparison
- **Segmentation AP50** comparison
- **Mean IoU** comparison (if available)
- **Mean Dice** comparison (if available)

### 🔥 **configuration_heatmap.png**
Heatmap showing parameter differences:
- **Rows**: Configuration parameters
- **Columns**: Evaluation runs
- **Colors**: Parameter values (1=Enabled, 0=Disabled, etc.)

### 📈 **performance_trends.png**
Line plots showing performance across checkpoints:
- **X-axis**: Checkpoint numbers
- **Y-axis**: Performance metrics
- **Lines**: Different evaluation runs

### 📋 **CSV Files**
- **configuration_summary.csv** - Detailed configuration comparison
- **performance_comparison.csv** - Performance metrics for each run

## Interpreting Results

### Best Configuration Identification
The system automatically identifies the best performing configuration based on **Segmentation AP**. Look for:
- **Highest AP score** in the performance comparison
- **Consistent performance** across different metrics
- **Stable trends** in the performance plots

### Configuration Impact Analysis
The report analyzes how different parameters affect performance:

#### Instance Segmentation
- **INSTANCE_ON=True** vs **INSTANCE_ON=False**
- Compare average AP scores
- Check if instance segmentation improves fish boundary detection

#### Resolution Impact
- **MIN_SIZE_TEST** and **MAX_SIZE_TEST** values
- Higher resolution often improves performance but increases computation
- Find the sweet spot for your hardware

#### Test-Time Augmentation
- **TEST_AUG_ENABLED=True** with different scales
- **TEST_AUG_FLIP=True** for orientation robustness
- Check if TTA improves performance significantly

#### Threshold Optimization
- **OBJECT_MASK_THRESHOLD** - Confidence filtering
- **OVERLAP_THRESHOLD** - Non-maximum suppression
- Balance between precision and recall

## Recommendations for Fish Classification

### For Single Fish per Video
```bash
# Recommended starting configuration
./run_eval_all_checkpoints.sh /path/to/model \
  --instance-on \
  --max-num 1 \
  --min-size-test 480 \
  --max-size-test 640
```

### For Better Performance (Higher Resolution)
```bash
# Test higher resolutions
./run_eval_all_checkpoints.sh /path/to/model \
  --min-size-test 640 \
  --max-size-test 960
```

### For Robust Classification (with TTA)
```bash
# Test with augmentation
./run_eval_all_checkpoints.sh /path/to/model \
  --instance-on \
  --max-num 1 \
  --test-aug \
  --test-min-sizes "360,480,600"
```

### For Precision vs Recall Balance
```bash
# Test different thresholds
./run_eval_all_checkpoints.sh /path/to/model \
  --object-mask-threshold 0.9 \
  --overlap-threshold 0.7
```

## Troubleshooting

### Common Issues

#### "No evaluation runs found"
- Make sure you've run evaluations with different configurations
- Check that `checkpoint_evaluations/` contains `run_*` directories

#### "No performance data found"
- Ensure evaluations completed successfully
- Check that `results.json` files exist in checkpoint directories

#### Missing mask metrics
- Run mask metrics analysis first: `--run-mask-metrics`
- Some metrics (IoU, Dice) require mask analysis

### Performance Considerations
- **Higher resolution** = more GPU memory and slower inference
- **Test-time augmentation** = 2-10x slower but often better performance
- **Instance segmentation** = more computationally expensive than semantic only

## Advanced Usage

### Custom Analysis
You can modify the Python script to:
- Add new performance metrics
- Change the best configuration criteria
- Customize plot styles and layouts
- Add statistical significance testing

### Integration with Other Tools
- Use CSV files for further analysis in Excel/R/Python
- Import plots into presentations or papers
- Automate configuration optimization

## Example Results Interpretation

### Good Configuration
- **High AP scores** (>0.8 for fish classification)
- **Consistent performance** across AP, AP50, AP75
- **Good IoU/Dice scores** (>0.7)
- **Stable trends** across checkpoints

### Areas for Improvement
- **Low AP scores** (<0.6) - Try higher resolution or different thresholds
- **Inconsistent metrics** - Check for overfitting or data issues
- **Unstable trends** - May need more training or different learning rate

### Configuration Recommendations
Based on typical fish classification results:
1. **Instance segmentation** usually helps for single fish
2. **Higher resolution** (480-640) often improves performance
3. **Test-time augmentation** provides 1-3% improvement
4. **Threshold tuning** can balance precision/recall

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated report for insights
3. Examine the CSV files for detailed data
4. Use the visualization scripts for individual run analysis

The evaluation comparison system helps you systematically optimize your test configuration for the best fish classification performance! 🐟

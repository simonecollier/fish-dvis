## General Usage

### Data Prep
1. Convert coco fishway dataset into ytvis dataset using [`01_convert_coco_to_ytvis.py`](data_scripts/01_convert_coco_to_ytvis.py). (Change the hardcoded paths and variables at the bottom of the script).
2. Split data into training and validation sets using [`03_create_train_val_jsons.py`](data_scripts/03_create_train_val_jsons.py). (Change the hardcoded paths and variables at the bottom of the script).
3. Validate the datasets using [`04_validate_ytvis.py`](data_scripts/04_validate_ytvis.py). (Change the hardcoded paths and variables at the bottom of the script).
4. Create masked versions of the images using the wrapper script [`run_mask.sh`](data_scripts/run_mask.sh).

### Train Model
1. Edit config file [`DAQ_Fishway_config_camera.yaml`](configs/DAQ_Fishway_config_camera.yaml) for the camera model or [`DAQ_Fishway_config_silhouette.yaml`](configs/DAQ_Fishway_config_silhouette.yaml) for the silhouette model.
2. Train either model using wrapper script [`train_run.sh`](training_scripts/train_run.sh). (Specify camera or silhouette in command line as well as GPU id.)

### Evaluate Model
1. Run inference on validation set using wrapper script [`eval_run.sh`](training_scripts/eval_run.sh).
2. Analyze results of evaluation to get performance metrics using wrapper script [`run_analysis_all_checkpoint_results.sh`](visualization_scripts/run_analysis_all_checkpoint_results.sh).

### Model Interpretation Analysis
#### Frame Scrambling
1. Scramble frames of validation videos using [`scramble_val.py`](data_scripts/scramble_val.py) or [`batch_scramble_folds.py`](data_scripts/batch_scramble_folds.py) if multiple scrambling permutations are desired.
2. Re-evaluate model on scrambled validation set using above method and compare performance to unscrambled evaluation.
3. Decompose the mAP score into its components to determine what cause the performance drop on the scrambled videos using [`component_isolation_tests.py`](temporal_analysis/component_isolation_tests.py).

#### Extract Internal Model Weights
1. Extract internal model weights for spatial attention and frame importance by reevaluating the model on the validation set and hooking into specific components using wrapper script [`eval_run_attn.sh](attention_analysis/eval_run_attn.sh).
2. Create summary json of frame importance values using [`summarize_activation_proj.py`](attention_analysis/summarize_activation_proj.py).
3. Examine differences in frame importance distributions for camera vs silhouette evaluations or scrambled vs unscrambled using [`jensen_shannon_test.py`](temporal_analysis/jensen_shannon_test.py)
4. Create summary json of tracker attention across object queries using [`summarize_tracker_attn.py`](attention_analysis/summarize_tracker_attn.py)
5. Create images of spatial attention on each video frame using [`plot_decoder_attn.py`](attention_analysis/plot_decoder_attn.py)
6. Create videos of spatial attention using [`pred_vids_attn.py`](attention_analysis/pred_vids_attn.py)

### Psersonal Environment Setup

```bash
# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Activate virtual environment
source /home/simone/.venv/bin/activate
```


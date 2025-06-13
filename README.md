# fish-dvis

This repository is designed for training video instance segmentation models on custom video datasets, specifically for classifying fish species in underwater videos. It builds upon the [DVIS_Plus](https://github.com/zhang-tao-whu/DVIS_Plus) repository, leveraging its advanced video segmentation capabilities and extending them for the task of fish species classification.

## Overview

- **Base Model:** The core segmentation and tracking functionality is provided by the DVIS_Plus repo (included as a subdirectory).
- **Custom Training:** This repo provides scripts and utilities for preparing your own video data, configuring experiments, and training models to classify fish species in videos.

## Directory Structure

- **DVIS_Plus/**  
  Contains the original DVIS_Plus codebase.

- **data_scripts/**  
  Scripts and utilities for preparing and processing your video dataset.  
  Example contents:
  - `fishway_convert_ytvis_train.py`, `save_ytvis_json.py`, `create_valid_json.py`: Convert and prepare your fish video data into the required format.
  - `register_datasets.py`, `ytvis_loader.py`: Register and load datasets for training and evaluation.
  - `fishway_metadata.csv`: Metadata about your fish video dataset.
  - `delete_pycache.py`: Utility to clean up Python cache files.

- **visualization_scripts/**  
  Scripts for visualizing model predictions and dataset samples.  
  Example:
  - `ytvis_video_pred.py`: Visualize video predictions from the trained model.

- **configs/**  
  Configuration files for training experiments.  
  Example:
  - `DAQ_Offline_VitAdapterL_fishway.yaml`: Example config for training on the fishway dataset.

- **training_scripts/**  
  Training scripts for running experiments.  
  Example:
  - `train_net_video.py`: Main script for training a video segmentation model on your custom fish dataset.

## Getting Started

1. **Prepare your data:**  
   Use the scripts in `data_scripts/` to convert and register your fish video dataset.

2. **Configure your experiment:**  
   Edit or create a config file in `configs/` to specify model parameters, dataset paths, and training options.

3. **Train your model:**  
   Run the training script in `training_scripts/` (see below for an example command).

4. **Visualize results:**  
   Use the scripts in `visualization_scripts/` to inspect model predictions.

## Example Training Command

```bash
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus python /home/simone/fish-dvis/training_scripts/train_net_video.py \
  --num-gpus 1 \
  --config-file /home/simone/fish-dvis/configs/DAQ_Offline_VitAdapterL_fishway.yaml  \
  --eval-only \
  MODEL.WEIGHTS /home/simone/checkpoints/model_ytvis21_offline_vitl.pth \
  OUTPUT_DIR /home/simone/dvis-model-outputs/model_runxx
```

## Acknowledgements

- This project is based on [DVIS_Plus](https://github.com/zhang-tao-whu/DVIS_Plus) by Zhang Tao et al.
- Please refer to the DVIS_Plus repository for details on the underlying model and its original usage. 
# `data_scripts/`

Dataset preparation utilities for Fishway video data in YTVIS-style JSON format: conversion from COCO-style annotations, split creation, k-fold generation, and JSON transforms (trim/complete/scramble/subset/stride).

Many scripts contain hardcoded paths and are intended to be edited per dataset location.

## Scripts

- `01_convert_coco_to_ytvis.py`: Converts COCO-style annotations into a YTVIS JSON (and associated image layout) for Fishway videos. Uses `fishway_metadata.csv` and expects a specific labeled-data directory layout.

- `fishway_coco2ytvis.py`: Original script for COCO to YTVIS converter for the Fishway dataset (similar intent to `01_convert_coco_to_ytvis.py`).

- `03_create_train_val_jsons.py`: Creates `train.json` and `val.json` splits from a combined YTVIS JSON (e.g. `all_videos.json`).

- `04_validate_ytvis.py`: Validates a YTVIS JSON (basic consistency checks, RLE/length sanity) and prints summary stats.

- `create_k_folds.py`: Creates stratified K-fold splits from `train.json` + `val.json`, producing `train_foldX.json` and `val_foldX.json`.

- `analyze_folds.py`: Prints per-fold / per-category distributions for fold JSONs (useful sanity checks before training).

- `trim_val_json.py`: Trims validation videos to a max length (e.g. centered around annotated frames), writing a new val JSON.

- `complete_val.py`: Takes a trimmed val JSON and restores full frame sequences/annotations using the combined source JSON (e.g. `all_videos.json`), writing a new “all frames” JSON.

- `scramble_val.py`: Scrambles frame ordering in a val JSON (for temporal-order ablation experiments), writing a new scrambled JSON.

- `subset_val.py`: Creates a subset val JSON containing only specific `video_id`s.

- `create_strided_jsons.py`: Creates strided versions of train/val JSONs for experiments that sample frames with stride N.

- `create_mask_video.py`: Generates mask renderings/videos from YTVIS annotations (optionally with degradations like blur/noise), writing to an output directory.

- `check_video_fps.py`: Utility to print FPS/duration information for a list of videos.

- `batch_scramble_folds.py`: Batch utility for creating scrambled versions across folds/seeds (see in-file configuration/paths).

- `run_mask.sh`: Wrapper to run mask/attention-related processing (calls into the analysis/eval scripts; paths are configured inside).

- `ytvis_loader.py`: YTVIS dataset loader/registration helpers (typically imported, not run directly).

## Notes

- If a script has no CLI arguments or mentions “hardcoded paths”, edit the variables near the top / in `__main__` to match your data locations.
- Inputs/outputs are generally YTVIS JSON files plus derived JSON variants for folds/stride/scramble experiments.

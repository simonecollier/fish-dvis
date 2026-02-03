# `training_scripts/`

Training and evaluation scripts for the DVIS model. Most scripts expect a configured environment (e.g. `DETECTRON2_DATASETS`, `PYTHONPATH` pointing at `DVIS_Plus/DVIS_DAQ`) and several paths are hardcoded inside the scripts.

## Scripts

- `train_net_video.py`: Main training/eval entrypoint (Detectron2-style). Typically run with `--config-file ...`, optionally `--eval-only`, plus `MODEL.WEIGHTS ...` and/or `OUTPUT_DIR ...` overrides.

- `train_run.sh`: Wrapper to launch training using a datatype (`camera`/`silhouette`) or an explicit config path, and an optional GPU id.

- `train_sequential.sh`: Train stride variants sequentially (defaults to strides 1–6 if none provided). Creates per-stride configs and trains one after another.

- `eval_run.sh`: Evaluate a single checkpoint on a specific validation JSON (currently set up for a scrambled val JSON) by calling `train_net_video.py --eval-only`. Copies `val.json` into the output directory.

- `eval_scramble.sh`: Batch-evaluate many scrambled seeds (1–100). For each seed, reads `val.json` from an existing per-seed folder and writes `inference/results.json` to that same folder if missing.

- `evaluate_all_checkpoints.py`: Evaluate all `model_*.pth` checkpoints in a model directory, saving each checkpoint’s results into a separate subdirectory. Intended to be followed by analysis in `visualization_scripts/`.

- `run_eval_all_checkpoints.sh`: Wrapper around `evaluate_all_checkpoints.py` that supports optional config overrides (thresholds, resolution, test-time augmentation), GPU selection, and checkpoint ranges.

- `gpu_queue.sh`: Simple “run commands sequentially” queue runner for one GPU. Can read from a queue file that you append to while it runs.

## Notes

- Many `.sh` scripts `source /home/simone/.venv/bin/activate` and set `DETECTRON2_DATASETS=/data` and `PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ`.
- Several scripts have hardcoded paths (configs, weights, output dirs). If you copy these scripts to new runs, update those paths near the top of the file.

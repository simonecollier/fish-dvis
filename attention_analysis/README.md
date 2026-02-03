# `attention_analysis/`

Attention extraction and visualization pipeline for DVIS-style models. This folder contains:
- an evaluation entrypoint that records attention maps
- plotting utilities for decoder/temporal attention
- scripts to summarize attention-derived signals (activation projection weights, tracker attention)
- utilities to render videos with attention overlays

## Scripts

- `train_net_video_eval_attn.py`: Evaluation entrypoint that runs inference and captures attention maps via hooks during forward passes (writes attention artifacts into `OUTPUT_DIR`).

- `attention_extractor.py`: Hook/utilities module used to register attention capture points (imported by the eval/analysis scripts).

- `eval_run_attn.sh`: Wrapper to run evaluation with attention extraction (sets environment variables, calls `train_net_video_eval_attn.py`). Paths are configured inside the script.

- `run_analysis_attn_results.sh`: Orchestrates the standard attention analysis workflow for one run directory.
  - Usage: `./run_analysis_attn_results.sh <attention_dir>`
  - Runs: `summarize_activation_proj.py`, `summarize_tracker_attn.py`, `plot_decoder_attn.py`, `pred_vids_attn.py`

- `summarize_activation_proj.py`: Summarizes activation projection weights (frame-importance-style signals) and optionally plots them. Writes JSON summaries and plots under the run’s `inference/` directory.

- `summarize_tracker_attn.py`: Summarizes tracker attention (e.g. top-attended queries/frames) and writes JSON summaries (and optional plots) under `inference/`.

- `plot_decoder_attn.py`: Visualizes decoder cross-attention maps for top predictions (supports different scaling modes). Writes plots under `inference/`.

- `plot_temporal_attn.py`: Visualizes temporal attention matrices and rollout-style views. Writes plots under `inference/`.

- `temporal_rollout.py`: Computes temporal attention rollout matrices by multiplying normalized attention matrices across refiner layers; writes rollout outputs under `attention_maps/rolled_out/`.

- `pred_vids_attn.py`: Renders videos that combine predictions and attention visualizations; writes outputs under `inference/`.

- `temporal_ytvis_eval.py`: Temporal/YTVIS evaluation helper (see in-file usage/config).

## Notes

- The common contract for a “run directory” is: evaluation results under `inference/` plus attention artifacts under `attention_maps/` (exact filenames vary by script).
- Many scripts assume Detectron2 + DVIS-DAQ are available and set `PYTHONPATH` accordingly.

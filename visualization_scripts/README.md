# `visualization_scripts/`

Post-processing, analysis, and visualization utilities for model outputs (typically `inference/results.json` plus a `val.json`). This folder includes:
- checkpoint/run aggregation and reports
- metric computation (mask AP, confusion matrices, extra/boundary metrics)
- plots (AP bar charts, training loss, scramble comparisons)
- video/GIF generation for qualitative inspection

## Scripts

### Checkpoint/run analysis

- `run_analysis_all_checkpoint_results.sh`: Wrapper to analyze an evaluated model directory (supports parallel metric computation, run comparison, and different analysis levels).

- `analyze_checkpoint_results.py`: Analyzes a model directory’s checkpoint evaluations and produces summaries/plots (used via `run_analysis_all_checkpoint_results.sh`).

- `analyze_single_checkpoint.py`: Detailed model evaluation analysis for one checkpoint directory.

### Metric computation

- `mask_metrics.py`: Computes mask metrics (dataset/category/frame breakdowns) using DVIS-DAQ evaluation; also supports confusion matrix output. Used by other scripts.

- `extra_mask_metrics.py`: Computes additional/extended mask metrics (intended as a companion to `mask_metrics.py`).

- `dvis_daq_eval.py`: Script used by `mask_metrics.py` to use DVIS-DAQ's evaluation.

- `confusion_mat_plot.py`: Helpers for computing/plotting confusion matrices (used by metric/analysis scripts).

### Plots and comparisons

- `plot_AP_barchart.py`: Builds AP/AP50/AP75 bar charts from metrics CSVs (optionally with confusion matrices).

- `plot_scrambled_results.py`: Compares “scrambled” vs “original” evaluation results and generates comparison plots.

- `plot_ranked_score_distributions.py`: Plots ranked score distributions (often to compare original vs scrambled evaluations).

- `plot_training_loss.py`: Plots training loss curves from training logs/metrics (as emitted during training).

- `combine_fold_csvs.py`: Combines per-fold CSV outputs and produces aggregated summaries.

- `summarize_scrambles.py`: Aggregates metrics across many scramble seeds and produces summary CSVs.

- `create_summary_table.py`: Builds a weighted summary table (e.g., from an Excel file) grouped by `video_id`.

- `create_top_frame_csv.py`: Creates a CSV of “top frames” (e.g. from activation projection summaries).

- `top_scoring_prediction_per_video.py`: Finds top-scoring predictions per video and reports/exports a CSV.

### Video/GIF utilities

- `output_prediction_videos.py`: Renders MP4s with predicted masks/labels overlaid on frames.

- `visualize_stride_videos.py`: Generates qualitative videos for stride datasets (overlaying predictions/labels).

- `convert_images_to_video.py`: Converts a folder of images into a video with FPS matched to the corresponding original video; writes outputs to a fixed output directory (see docstring).

- `create_side_by_side_gif.py`: Produces side-by-side GIF comparisons from two videos or frame directories.

- `make_sorted_vids.py`: Creates videos from images sorted by category/species (paths typically configured inside the script).

## Notes

- Many scripts expect Detectron2/DVIS-DAQ deps to be importable; set `PYTHONPATH` appropriately (often `.../DVIS_Plus/DVIS_DAQ`).
- If you’re unsure about arguments, run `python <script>.py --help` (when implemented) or check the module docstring at the top of the file.

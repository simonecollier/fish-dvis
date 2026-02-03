# `temporal_analysis/`

Small, focused experiments and diagnostics for temporal behavior/robustness analyses (e.g., “scrambled vs normal” comparisons and distributional/statistical tests).

## Scripts

- `component_isolation_tests.py`: Diagnostic experiment suite to isolate which component (e.g., classification vs segmentation vs calibration) contributes to performance differences between “normal” and “scrambled” runs. Paths are configured near the top of the file.

- `jensen_shannon_test.py`: Computes Jensen–Shannon divergence (with a permutation test) between activation vectors from two model outputs and generates plots (histograms/bar plots). 

## Notes

- These scripts are intended to be run after you have produced model outputs (e.g. `inference/results.json` plus any saved activations/aux outputs) and usually require editing the configured paths.

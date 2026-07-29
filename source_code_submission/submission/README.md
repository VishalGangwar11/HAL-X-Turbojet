# Source Code — Turbojet Digital Twin

This folder contains the complete, working pipeline for the Physics-Informed Digital Twin
submitted for the HAL x IIT Indore Aerothon 2026 (Challenge: Physics-Informed Digital Twin
for Real-Time Four-Stage Turbojet Health Monitoring).

Scripts are organized in the order they were actually developed and should be run in,
so the folder itself documents the engineering process, not just the final answer.

## Setup

```
pip install pandas numpy scikit-learn filterpy scipy --break-system-packages
```

Place `train.csv`, `test.csv`, `ground_truth.csv`, and `turbojet_complete_dataset.csv`
(from the official dataset) in the same working directory before running any script.

## Folder-by-folder guide

### 01_data_exploration
Initial profiling of the dataset: engine/cycle counts, sensor statistics, and
visualizations of raw sensor trends vs. ground-truth health trends. Establishes that
true health follows a smooth, near-deterministic decay while raw sensors are noisy
per-cycle due to varying flight conditions.

### 02_thermodynamic_model
Standalone 0-D Brayton-cycle thermodynamic model (compressor → combustor → turbine)
with three free health parameters (eta_c, eta_comb, eta_t) replacing fixed design
constants. This is the physics backbone used throughout the rest of the pipeline.

### 03_calibration
Calibrates the model's design parameters (pressure ratios, air mass flow) against
real sensor data via regression, since these cannot be assumed as fixed constants
without introducing large model-plant mismatch.
- `03_diagnose_forward_model.py` is a diagnostic script that revealed a critical bug:
  an initial naive air-flow assumption caused combustor temperature predictions of
  several thousand Kelvin. `04_fit_air_flow_corrected.py` fixes this by inverting the
  combustor energy balance against true health values and refitting air flow
  properly (R² = 0.985 after the fix, vs. physically implausible values before).

### 04_ukf_estimation
Builds and tunes the Unscented Kalman Filter that estimates the 3-dimensional hidden
health state (eta_c, eta_comb, eta_t) from all six sensor channels, cycle by cycle.
Measurement noise (R) is set from actual residual variance (not guessed); process
noise (Q) is tuned per-state via grid search against MAE across all 10 engines.

### 05_residual_correction
Gradient-boosted correction layer that learns the residual gap between the UKF's
physics-based estimate and true health, evaluated via leave-one-engine-out
cross-validation to avoid overfitting given the small number of engines available.

### 06_validation_and_comparison
The most important folder for judging rigor. Contains:
- Held-out test set validation (data never touched during any tuning step)
- A pure black-box ML baseline (no physics at all) for honest comparison
- Data-efficiency and extrapolation stress tests
- Inference latency benchmarking
- Uncertainty calibration check (does the UKF's reported confidence interval
  actually contain the true value at the expected rate?)

All results in this folder are reported as found, including cases where the
hybrid approach did not outperform simpler alternatives — see the Technical
Report, Section 9, for full discussion.

### 07_final_pipeline
End-to-end script that runs the full pipeline (UKF + residual correction).
computes the unified health index (Mahalanobis distance from a healthy
baseline), predicts thrust/TSFC, and exports `ukf_results_corrected.csv`,
which is the data file consumed by the interactive dashboard
(`dashboard/index.html`).

## Live Dashboard

https://vishalgangwar11.github.io/HAL-X-Turbojet/

## Key Results Summary

See the Technical Report (`Technical_Report.pdf`) for full detail. Headline numbers:

| Subsystem | Held-out test MAE (UKF+Residual) | vs. naive baseline |
|---|---|---|
| Compressor | 0.0468 | ~28% better |
| Turbine | 0.0189 | ~53% better |

Thrust prediction R² = 0.92, TSFC prediction R² = 0.88.

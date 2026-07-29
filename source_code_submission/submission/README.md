# Source Code — Turbojet Digital Twin

This folder contains the complete end-to-end workflow for the physics-informed digital twin
submitted for the HAL × IIT Indore Aerothon 2026 challenge: real-time four-stage turbojet
health monitoring from limited sensor data.

The implementation follows the full machine-learning and physics-based pipeline used in the
project: data exploration, thermodynamic modeling, parameter calibration, UKF-based state
estimation, residual correction, validation, and final dashboard export.

## Project goal

Estimate hidden engine health states (compressor, combustor, and turbine efficiency) and use
those estimates to generate a unified health index, predict thrust and TSFC, and visualize the
result in an interactive dashboard.

## Setup

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn filterpy scipy --break-system-packages
```

Place the following files in the same working directory before running any script:

- `train.csv`
- `test.csv`
- `ground_truth.csv`
- `turbojet_complete_dataset.csv`

## Pipeline overview

### 01_data_exploration
Explores the dataset structure, engine-cycle distributions, and sensor behavior. It establishes
that true health evolves smoothly over time while raw sensors are noisy and strongly affected by
varying flight conditions.

### 02_thermodynamic_model
Builds a standalone 0-D Brayton-cycle model for the compressor → combustor → turbine path.
The model replaces fixed design constants with three free health parameters:

- `eta_c` for compressor efficiency
- `eta_comb` for combustor efficiency
- `eta_t` for turbine efficiency

These parameters are the core latent variables estimated by the rest of the pipeline.

### 03_calibration
Calibrates the thermodynamic model's pressure ratios and air-mass-flow assumptions against real
sensor data. The calibration scripts identify and correct model-plant mismatch, including a key
correction to the air-flow formulation that makes the forward model physically plausible.

### 04_ukf_estimation
Uses an Unscented Kalman Filter to estimate the hidden health states from six sensor channels
for each engine cycle. The UKF provides recursive estimates and uncertainty estimates for the
three latent health variables.

### 05_residual_correction
Applies a gradient-boosted regression layer to learn the residual gap between the physics-based
UKF estimate and the true health values. This step improves prediction quality while preserving
an interpretable model structure.

### 06_validation_and_comparison
Evaluates the model rigorously with:

- held-out validation on engines never used during tuning
- a pure black-box machine-learning baseline with no physics
- data-efficiency and extrapolation stress tests
- inference latency benchmarking
- uncertainty calibration checks

This folder is important because it compares the hybrid physics + ML approach against simpler
alternatives in a fair, apples-to-apples manner.

### 07_final_pipeline
Runs the full end-to-end workflow and produces:

- `ukf_results_full.csv` with UKF-based health estimates and predictions
- `ukf_results_corrected.csv` with corrected outputs used by the dashboard
- thrust and TSFC predictions derived from the estimated efficiencies
- a unified health index based on Mahalanobis distance from a healthy baseline

## Machine-learning components

The project combines physics-informed estimation with supervised learning:

- Physics model: thermodynamic state evolution
- State estimator: Unscented Kalman Filter
- Residual learner: gradient-boosted regression
- Baseline comparator: black-box gradient boosting on raw sensor features

## Results summary

The workflow produces a strong hybrid estimate of engine health and performance. In the
submitted analysis, the hybrid method substantially improves health estimation over the naive
baseline, with the strongest gains observed for turbine health. The final pipeline also predicts
thrust and TSFC with high accuracy and exports the results for visualization.

## Live Dashboard

https://vishalgangwar11.github.io/HAL-X-Turbojet/

## Files of interest

- `01_data_exploration/01_profile_dataset.py`
- `02_thermodynamic_model/01_thermo_model.py`
- `03_calibration/01_fit_pressure_ratios.py`
- `03_calibration/04_fit_air_flow_corrected.py`
- `04_ukf_estimation/01_ukf_final.py`
- `05_residual_correction/01_residual_correction_compressor.py`
- `05_residual_correction/02_residual_correction_turbine.py`
- `06_validation_and_comparison/02_blackbox_ml_baseline.py`
- `07_final_pipeline/01_health_index_and_thrust_tsfc.py`
- `dashboard/index.html`

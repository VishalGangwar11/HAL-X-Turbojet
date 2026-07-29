# Turbojet Digital Twin — Health Monitoring

**HAL × IIT Indore Aerothon 2026** — Physics-Informed Digital Twin for Real-Time Four-Stage Turbojet Health Monitoring

🔗 **[Live Dashboard](https://vishalgangwar11.github.io/HAL-X-Turbojet/)**

## Overview

This repository contains a complete physics-informed digital twin for real-time health monitoring
of a four-stage turbojet engine. The project combines a calibrated 0-D thermodynamic model,
an Unscented Kalman Filter (UKF) for hidden-state estimation, and a gradient-boosted residual
correction layer to estimate compressor, combustor, and turbine health using only limited sensor
measurements.

## What the project does

The pipeline estimates hidden engine health states from flight-condition-dependent measurements
such as pressures, temperatures, RPM, and fuel flow. It then uses those estimates to:

- generate a unified health index
- predict thrust and TSFC
- export results for an interactive dashboard
- compare the hybrid physics + ML approach against black-box baselines

## Core methodology

1. **Physics model** — a Brayton-cycle thermodynamic model of the compressor → combustor → turbine path.
2. **Calibration** — regression-based fitting of pressure ratios and air-mass-flow assumptions to real sensor data.
3. **UKF estimation** — recursive estimation of three latent health variables from six sensor channels each cycle.
4. **Residual correction** — gradient-boosted learning of the residual gap between UKF output and true health.
5. **Validation** — held-out engine evaluation, black-box baseline comparison, extrapolation testing, and latency analysis.

## Repository structure

- [source_code_submission/submission](source_code_submission/submission) — complete implementation scripts and pipeline steps
- [source_code_submission/submission/dashboard](source_code_submission/submission/dashboard) — interactive dashboard assets
- [docs](docs) — GitHub Pages landing page assets

## Setup

Install the required packages:

```bash
pip install pandas numpy scikit-learn filterpy scipy --break-system-packages
```

Place the required dataset files in the working directory before running the scripts:

- `train.csv`
- `test.csv`
- `ground_truth.csv`
- `turbojet_complete_dataset.csv`

## Key results

The hybrid pipeline provides strong health estimates and performance predictions. The project reports
substantial improvements over a naive baseline, especially for turbine health, and exports the
results used by the dashboard.

## Team

- Vishal Gangwar — Govind Ballabh Pant Institute of Engineering and Technology
- Mohit Bohra — Govind Ballabh Pant Institute of Engineering and Technology

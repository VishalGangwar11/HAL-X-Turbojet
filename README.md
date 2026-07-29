# Turbojet Digital Twin — Health Monitoring

**HAL × IIT Indore Aerothon 2026** — Physics-Informed Digital Twin for Real-Time Four-Stage Turbojet Health Monitoring

🔗 **[Live Dashboard](https://vishalgangwar11.github.io/HAL-X-Turbojet/)**

## Overview

This project estimates the hidden health state (compressor, combustor, turbine efficiency) of a single-spool four-stage turbojet engine in real time, using only limited sensor measurements (pressures, temperatures, RPM, fuel flow). It combines a calibrated 0-D thermodynamic cycle model, an Unscented Kalman Filter (UKF) for state estimation, and a gradient-boosted residual correction layer to close the gap between pure physics and real degradation behavior.

## Architecture

1. **Physics model** — Brayton-cycle thermodynamic equations (compressor → combustor → turbine), with compressor/turbine pressure ratios and air mass flow calibrated from data via regression against real sensor readings.
2. **Unscented Kalman Filter** — treats compressor/combustor/turbine efficiency as a 3-dimensional hidden state, updated recursively from six sensor channels each cycle. Provides native uncertainty quantification via its covariance matrix.
3. **Residual correction (Gradient Boosting)** — learns the gap between the UKF's physics-based estimate and true health, conditioned on flight condition. Improves turbine MAE by ~53% and compressor MAE by ~68% on held-out test data.
4. **Unified health index** — Mahalanobis distance from a healthy-baseline reference, combining all three subsystem estimates into one scalar.
5. **Interactive dashboard** — real-time gauges, confidence-banded trend charts, thrust/TSFC prediction, and Remaining Useful Life estimation.

## Key Results

| Subsystem | Naive Baseline (MAE) | UKF Alone (MAE) | UKF + Residual Correction (MAE) |
|---|---|---|---|
| Compressor | 0.0615 | 0.0571 | 0.0195 |
| Combustor | 0.0219 | 0.0147 | — |
| Turbine | 0.0440 | 0.0517 | 0.0236 |

- Thrust prediction: R² = 0.92
- TSFC prediction: R² = 0.88
- Held-out test set validation confirms these gains generalize to unseen engines (turbine MAE improved 53% vs. naive baseline on data never used in training or tuning)
- Inference latency: ~1.5 ms/cycle (real-time capable on standard CPU)

## Honest Limitations

- On this clean synthetic benchmark, a pure black-box ML model (no physics) matches or slightly exceeds the hybrid approach on raw accuracy. The hybrid's advantages are interpretability, principled uncertainty quantification tied to the estimation process itself, and a physics-grounded architecture that stays meaningful outside the training distribution — not raw MAE on this specific dataset.
- Uncertainty calibration varies by subsystem: turbine is reasonably calibrated, compressor is overconfident (band too narrow), combustor is underconfident (band too wide). Documented rather than hidden.

## Files

- `index.html` — self-contained interactive dashboard (data embedded, no server required)
- See technical report for full methodology, calibration process, and evaluation details

## Team

Vishal Gangwar — Govind Ballabh Pant Institute of Engineerimg and Technology 
Mohit Bohra - Govind Ballabh Pant Institute of Engineering and Technology 

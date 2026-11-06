"""
Day 3 (final tune) - UKF with R matrix set from real measured residual
variances instead of guesses.
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import r2_score

GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
CP_AIR = 1005.0
CP_GAS = 1148.0
LHV_FUEL = 43e6
R_AIR = 287.0

train = pd.read_csv("train.csv")
gt = pd.read_csv("ground_truth.csv")
df = train.merge(gt, on=['EngineID', 'Cycle']).sort_values(['EngineID', 'Cycle']).reset_index(drop=True)


def fx(state, dt, **kwargs):
    return state


def hx(state, Pamb, Tamb, RPM, FuelFlow, Mach=0.4, Altitude=6000):
    eta_c, eta_comb, eta_t = state

    PR_c_assumed = (0.0001286487 * RPM + 2.624160 * Mach
                    + 0.0000033082 * Altitude - 3.931887)
    PR_c_assumed = max(PR_c_assumed, 1.05)

    rho_amb = Pamb / (R_AIR * Tamb)
    air_flow = 0.0006162388 * RPM - 0.424435 * rho_amb - 8.280180
    air_flow = max(air_flow, 1.0)

    P2 = Pamb * PR_c_assumed
    T2_ideal = Tamb * (PR_c_assumed) ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    T2 = Tamb + (T2_ideal - Tamb) / max(eta_c, 1e-3)

    PRESSURE_LOSS = 0.96
    P3 = P2 * PRESSURE_LOSS
    total_flow = air_flow + FuelFlow
    T3 = (air_flow * CP_AIR * T2 + eta_comb * FuelFlow * LHV_FUEL) / (total_flow * CP_GAS + 1e-9)

    PR_t_assumed = -0.0000118963 * RPM + 1.334673
    PR_t_assumed = min(max(PR_t_assumed, 0.1), 0.99)

    P4 = P3 * PR_t_assumed
    T4_ideal = T3 * (PR_t_assumed) ** ((GAMMA_GAS - 1) / GAMMA_GAS)
    T4 = T3 - eta_t * (T3 - T4_ideal)

    return np.array([P2, T2, P3, T3, P4, T4])
import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import mean_absolute_error, r2_score

def run_triggered_ukf_for_engine(engine_df, Q_scale=0.001, gamma_threshold=0.0):
    """
    Runs the Physics-Informed UKF with an Uncertainty-Triggered Telemetry Gatekeeper.
    If gamma_threshold == 0.0, it transmits 100% of data (Baseline UKF).
    As gamma_threshold increases, only high-uncertainty/anomalous cycles are transmitted.
    """
    dim_x = 3
    dim_z = 6
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1.0, fx=fx, hx=hx, points=points)
    
    ukf.x = np.array([0.9, 0.98, 0.9])
    ukf.P *= 0.01
    ukf.Q = np.eye(dim_x) * Q_scale
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    transmitted_flags = []
    gamma_scores = []

    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        hx_args = dict(Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'],
                       RPM=row['RPM_rev_min'], FuelFlow=row['FuelFlow_kg_s'],
                       Mach=row['Mach'], Altitude=row['Altitude_m'])

        # Step 1: On-board & Ground Prior Prediction
        ukf.predict()

        # Step 2: Compute predicted measurement z_pred and innovation covariance S on-board
        # Propagate current sigma points through thermodynamic model hx()
        sigmas_h = np.array([hx(s, **hx_args) for s in ukf.sigmas_f])
        z_pred = np.dot(ukf.Wm, sigmas_h)
        
        # Innovation (Residual) vector r_k
        y_res = z - z_pred
        
        # Innovation Covariance Matrix S_k
        S = ukf.R.copy()
        for i in range(len(ukf.Wc)):
            diff = (sigmas_h[i] - z_pred).reshape(-1, 1)
            S += ukf.Wc[i] * (diff @ diff.T)

        # Step 3: Uncertainty Trigger Score (Normalized Innovation Squared / Mahalanobis)
        S_inv = np.linalg.inv(S)
        gamma_k = float(y_res.T @ S_inv @ y_res)
        gamma_scores.append(gamma_k)

        # Step 4: Selective Telemetry Gatekeeper Decision
        if gamma_k >= gamma_threshold:
            # High novelty/uncertainty -> Transmit telemetry & update Ground Twin
            ukf.update(z, **hx_args)
            transmitted_flags.append(1)
        else:
            # Nominal flight -> Suppress telemetry; Ground Twin coasts on prediction
            transmitted_flags.append(0)

        estimates.append(ukf.x.copy())

    tx_rate = np.mean(transmitted_flags) * 100.0
    return np.array(estimates), tx_rate, np.array(gamma_scores), np.array(transmitted_flags)


"""
Day 3 (final tune) + Conference Paper Extension:
Physics-Informed UKF with Uncertainty-Triggered Telemetry, 
DCT-Lasso Compressive Sensing, and Adaptive Covariance Tuning (Sage-Husa).
"""

import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.fftpack import dct
from sklearn.linear_model import Lasso

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


def run_ukf_for_engine(engine_df, Q_scale=0.0005):
    dim_x = 3
    dim_z = 6
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.x = np.array([0.9, 0.98, 0.9])
    ukf.P *= 0.01
    ukf.Q = np.eye(dim_x) * Q_scale
    # Calibrated from actual residual variances (Day 3 tuning step)
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    for _, row in engine_df.iterrows():
        z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
        ukf.predict()
        ukf.update(z, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'],
                   RPM=row['RPM_rev_min'], FuelFlow=row['FuelFlow_kg_s'],
                   Mach=row['Mach'], Altitude=row['Altitude_m'])
        estimates.append(ukf.x.copy())
    return np.array(estimates)


# ==============================================================================
# NEW MODULES FOR CONFERENCE PAPER (PASTE RIGHT BELOW run_ukf_for_engine)
# ==============================================================================

def compress_and_reconstruct_window(residual_window, M_samples, seed=42):
    """
    Stage 2: Compresses a (W, 6) residual matrix down to (M, 6) on-board,
    and reconstructs it back to (W, 6) on the ground using DCT + L1 Lasso.
    """
    W, num_sensors = residual_window.shape
    if M_samples >= W:
        return residual_window.copy()

    rng = np.random.default_rng(seed)
    Phi = rng.normal(0, 1.0 / np.sqrt(M_samples), size=(M_samples, W))
    Psi = dct(np.eye(W), norm='ortho', axis=0)
    Theta_mat = Phi @ Psi

    reconstructed_window = np.zeros_like(residual_window)

    for s in range(num_sensors):
        r_vec = residual_window[:, s]
        scale = np.std(r_vec) + 1e-6
        r_norm = r_vec / scale

        # On-Board Compression: W -> M_samples
        y_cs = Phi @ r_norm

        # On-Ground Reconstruction: M_samples -> W via Lasso
        lasso = Lasso(alpha=0.01, max_iter=2000, tol=1e-3)
        lasso.fit(Theta_mat, y_cs)
        r_rec = (Psi @ lasso.coef_) * scale
        reconstructed_window[:, s] = r_rec

    return reconstructed_window


def run_adaptive_cs_triggered_ukf(engine_df, Q_init_diag=(1e-4, 1e-3, 5e-4), 
                                  gamma_threshold=0.25, W=20, M_cs=8, 
                                  forgetting_factor=0.90, adaptive_Q=True):
    """
    Full Three-Stage Proposed Framework:
    - Stage 1: On-Board Uncertainty Trigger (Mahalanobis NIS gatekeeper)
    - Stage 2: DCT-Lasso Compressive Sensing (W -> M_cs compression)
    - Stage 3: Innovation-Based Adaptive Covariance Tuning (Sage-Husa Q_k update)
    """
    dim_x, dim_z = 3, 6
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.5, beta=2.0, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1.0, fx=fx, hx=hx, points=points)
    
    ukf.x = np.array([0.9, 0.98, 0.9])
    ukf.P = np.eye(dim_x) * 0.01
    ukf.Q = np.diag(Q_init_diag)
    ukf.R = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])

    estimates = []
    covariances = []
    total_samples_sent = 0
    total_raw_samples = len(engine_df) * dim_z

    for start_idx in range(0, len(engine_df), W):
        window_df = engine_df.iloc[start_idx : start_idx + W]
        w_len = len(window_df)
        
        z_pred_list, res_list, hx_args_list, gamma_list = [], [], [], []

        # 1. ON-BOARD BUFFER: Evaluate uncertainty across window W
        for _, row in window_df.iterrows():
            z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
            hx_args = dict(Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'],
                           RPM=row['RPM_rev_min'], FuelFlow=row['FuelFlow_kg_s'],
                           Mach=row['Mach'], Altitude=row['Altitude_m'])
            
            z_pred = hx(ukf.x, **hx_args)
            y_res = z - z_pred
            
            sigmas_f = points.sigma_points(ukf.x, ukf.P)
            sigmas_h = np.array([hx(s, **hx_args) for s in sigmas_f])
            S = ukf.R.copy()
            for i in range(len(ukf.Wc)):
                diff = (sigmas_h[i] - z_pred).reshape(-1, 1)
                S += ukf.Wc[i] * (diff @ diff.T)

            gamma_k = float(y_res.T @ np.linalg.inv(S) @ y_res)

            z_pred_list.append(z_pred)
            res_list.append(y_res)
            hx_args_list.append(hx_args)
            gamma_list.append(gamma_k)

        # 2. STAGE 1 (UNCERTAINTY TRIGGER) + STAGE 2 (COMPRESSIVE SENSING)
        if np.mean(gamma_list) >= gamma_threshold:
            M_eff = min(M_cs, w_len)
            res_window = np.array(res_list)
            rec_res_window = compress_and_reconstruct_window(res_window, M_samples=M_eff)
            total_samples_sent += (M_eff * dim_z)

            for i in range(w_len):
                ukf.predict()
                z_reconstructed = z_pred_list[i] + rec_res_window[i]
                ukf.update(z_reconstructed, **hx_args_list[i])

                # 3. STAGE 3: ADAPTIVE COVARIANCE TUNING (Sage-Husa Q_k Update)
                if adaptive_Q:
                    state_innovation = (ukf.K @ ukf.y).reshape(-1, 1)
                    Q_innovation = np.diag(np.diag(state_innovation @ state_innovation.T))
                    ukf.Q = (forgetting_factor * ukf.Q) + ((1.0 - forgetting_factor) * Q_innovation)
                    ukf.Q = np.clip(ukf.Q, 1e-7, 1e-2)

                estimates.append(ukf.x.copy())
                covariances.append(np.diag(ukf.P).copy())
        else:
            # Nominal window -> Coast on prior prediction (0 bytes transmitted)
            for i in range(w_len):
                ukf.predict()
                estimates.append(ukf.x.copy())
                covariances.append(np.diag(ukf.P).copy())

    compression_ratio = total_raw_samples / max(total_samples_sent, 1)
    bw_saved_pct = (1.0 - (total_samples_sent / total_raw_samples)) * 100.0
    return np.array(estimates), np.array(covariances), bw_saved_pct, compression_ratio


def compute_picp_95(true_vals, est_vals, var_vals):
    """Computes 95% Prediction Interval Coverage Probability (PICP)."""
    std_vals = np.sqrt(np.maximum(var_vals, 1e-12))
    lower = est_vals - 1.96 * std_vals
    upper = est_vals + 1.96 * std_vals
    covered = (true_vals >= lower) & (true_vals <= upper)
    return np.mean(covered) * 100.0


# ==============================================================================
# UNIFIED EXPERIMENT BLOCK (REPLACES THE OLD if __name__ == "__main__": BLOCK)
# ==============================================================================

if __name__ == "__main__":
    eng1 = df[df['EngineID'] == 1].sort_values('Cycle').reset_index(drop=True)

    print("\n================ EXPERIMENT 1: FIXED Q vs ADAPTIVE Q (100% Telemetry) ================")
    for mode, is_adap in [("Baseline UKF (Fixed Q)", False), ("Proposed UKF (Adaptive Q)", True)]:
        est, cov, bw, cr = run_adaptive_cs_triggered_ukf(
            eng1, gamma_threshold=0.0, W=20, M_cs=20, adaptive_Q=is_adap
        )
        mae_c = mean_absolute_error(eng1['CompressorHealth'], est[:, 0])
        mae_comb = mean_absolute_error(eng1['CombustorHealth'], est[:, 1])
        mae_t = mean_absolute_error(eng1['TurbineHealth'], est[:, 2])
        picp_c = compute_picp_95(eng1['CompressorHealth'].to_numpy(), est[:, 0], cov[:, 0])
        picp_t = compute_picp_95(eng1['TurbineHealth'].to_numpy(), est[:, 2], cov[:, 2])
        print(f"{mode:26s} | Comp MAE: {mae_c:.4f} (95% PICP: {picp_c:5.1f}%) | "
              f"Comb MAE: {mae_comb:.4f} | Turb MAE: {mae_t:.4f} (95% PICP: {picp_t:5.1f}%)")

    print("\n================ EXPERIMENT 2: FULL PIPELINE (TRIGGER + CS + ADAPTIVE Q) ================")
    print("Gamma_Th | CS (M/W) | BW Saved (%) | Comp Ratio | Comp MAE | Comb MAE | Turb MAE")
    print("-" * 86)
    for th in [0.0, 0.05, 0.10, 0.25, 0.50]:
        for M_cs in [20, 10, 5]:
            est, cov, bw_saved, cr = run_adaptive_cs_triggered_ukf(
                eng1, gamma_threshold=th, W=20, M_cs=M_cs, adaptive_Q=True
            )
            mae_c = mean_absolute_error(eng1['CompressorHealth'], est[:, 0])
            mae_comb = mean_absolute_error(eng1['CombustorHealth'], est[:, 1])
            mae_t = mean_absolute_error(eng1['TurbineHealth'], est[:, 2])
            print(f"{th:8.2f} | {M_cs:2d}/20    | {bw_saved:11.1f}% | {cr:9.1f}x | "
                  f"{mae_c:.4f}   | {mae_comb:.4f}   | {mae_t:.4f}")

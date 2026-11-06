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
# STEP A: TUNED SWEEP + IEEE FIGURE GENERATOR (REPLACES if __name__ == "__main__":)
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    eng1 = df[df['EngineID'] == 1].sort_values('Cycle').reset_index(drop=True)

    # 1. Inspect the actual distribution of the Mahalanobis NIS score (gamma_k)
    points_tmp = MerweScaledSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0)
    R_mat = np.diag([857714883.88, 378.63, 790911236.82, 4529.15, 1681026493.53, 9676.53])
    win_gammas = []
    x_tmp = np.array([0.9, 0.98, 0.9])
    for start_idx in range(0, len(eng1), 20):
        w_df = eng1.iloc[start_idx:start_idx+20]
        g_list = []
        for _, row in w_df.iterrows():
            z = np.array([row['P2_Pa'], row['T2_K'], row['P3_Pa'], row['T3_K'], row['P4_Pa'], row['T4_K']])
            z_p = hx(x_tmp, Pamb=row['Pamb_Pa'], Tamb=row['Tamb_K'], RPM=row['RPM_rev_min'],
                     FuelFlow=row['FuelFlow_kg_s'], Mach=row['Mach'], Altitude=row['Altitude_m'])
            y_r = z - z_p
            g_list.append(float(y_r.T @ np.linalg.inv(R_mat) @ y_r))
        win_gammas.append(np.mean(g_list))

    p_vals = np.percentile(win_gammas, [0, 20, 40, 60, 75, 85])
    thresholds = [0.0] + [round(float(v), 2) for v in p_vals[1:]]
    print(f"\n[Diagnostic] Window Gamma_k Percentiles (0%, 20%, 40%, 60%, 75%, 85%): {thresholds}")

    print("\n================ TABLE I: ABLATION & TELEMETRY TRADE-OFF (ENGINE 1) ================")
    print("Gamma_Th | CS (M/W) | BW Saved (%) | Comp Ratio | Comp MAE | Comb MAE | Turb MAE | Avg MAE")
    print("-" * 92)

    pareto_results = {20: [], 10: [], 5: []}

    for th in thresholds:
        for M_cs in [20, 10, 5]:
            est, cov, bw_saved, cr = run_adaptive_cs_triggered_ukf(
                eng1, gamma_threshold=th, W=20, M_cs=M_cs, adaptive_Q=True
            )
            mae_c = mean_absolute_error(eng1['CompressorHealth'], est[:, 0])
            mae_comb = mean_absolute_error(eng1['CombustorHealth'], est[:, 1])
            mae_t = mean_absolute_error(eng1['TurbineHealth'], est[:, 2])
            avg_mae = (mae_c + mae_comb + mae_t) / 3.0
            pareto_results[M_cs].append((bw_saved, avg_mae, mae_c, mae_t, th))

            print(f"{th:8.2f} | {M_cs:2d}/20    | {bw_saved:11.1f}% | {cr:9.1f}x | "
                  f"{mae_c:.4f}   | {mae_comb:.4f}   | {mae_t:.4f}   | {avg_mae:.4f}")

    # 2. Generate IEEE Figure 1: Pareto Front (Bandwidth Saved % vs. Health MAE)
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    styles = {
        20: ('o--', '#1f77b4', 'Stage 1 Only: Uncertainty Trigger (No CS, M=20)'),
        10: ('s-.', '#ff7f0e', 'Proposed Dual-Stage: Trigger + 2x CS (M=10)'),
        5:  ('D-',  '#2ca02c', 'Proposed Dual-Stage: Trigger + 4x CS (M=5)')
    }

    for M_cs, (fmt, color, label) in styles.items():
        pts = sorted(pareto_results[M_cs], key=lambda x: x[0])
        bw_x = [p[0] for p in pts]
        mae_y = [p[1] for p in pts]
        ax.plot(bw_x, mae_y, fmt, color=color, linewidth=1.8, markersize=6, label=label)

    ax.set_xlabel('Telemetry Bandwidth Saved (%)', fontweight='bold')
    ax.set_ylabel('Mean Component Health MAE', fontweight='bold')
    ax.set_title('Selective Telemetry Pareto Front: Bandwidth Reduction vs. Estimation Error', fontsize=10.5, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig('fig1_pareto_bandwidth_vs_mae.png', dpi=300)
    plt.close(fig)

    # 3. Generate IEEE Figure 2: Health Tracking Trajectory with 95% Uncertainty Bands
    th_opt = thresholds[3]  # ~60th percentile threshold
    est_base, cov_base, _, _ = run_adaptive_cs_triggered_ukf(eng1, gamma_threshold=0.0, W=20, M_cs=20, adaptive_Q=False)
    est_prop, cov_prop, bw_opt, cr_opt = run_adaptive_cs_triggered_ukf(eng1, gamma_threshold=th_opt, W=20, M_cs=5, adaptive_Q=True)

    cycles = eng1['Cycle'].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True, dpi=300)

    # Subplot A: Compressor Efficiency
    std_c = np.sqrt(np.maximum(cov_prop[:, 0], 1e-12))
    axes[0].plot(cycles, eng1['CompressorHealth'], 'k-', linewidth=2.0, label='Ground Truth ($\eta_c$)')
    axes[0].plot(cycles, est_base[:, 0], '--', color='#7f7f7f', linewidth=1.3, label='Baseline UKF (100% Telemetry)')
    axes[0].plot(cycles, est_prop[:, 0], '-', color='#d62728', linewidth=1.6,
                 label=f'Proposed Trigger+CS ({bw_opt:.1f}% BW Saved, {cr_opt:.1f}x CR)')
    axes[0].fill_between(cycles, est_prop[:, 0] - 1.96*std_c, est_prop[:, 0] + 1.96*std_c,
                         color='#d62728', alpha=0.18, label='95% Confidence Interval')
    axes[0].set_ylabel('Compressor Health ($\eta_c$)', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower left', fontsize=8, ncol=2)

    # Subplot B: Turbine Efficiency
    std_t = np.sqrt(np.maximum(cov_prop[:, 2], 1e-12))
    axes[1].plot(cycles, eng1['TurbineHealth'], 'k-', linewidth=2.0, label='Ground Truth ($\eta_t$)')
    axes[1].plot(cycles, est_base[:, 2], '--', color='#7f7f7f', linewidth=1.3, label='Baseline UKF (100% Telemetry)')
    axes[1].plot(cycles, est_prop[:, 2], '-', color='#1f77b4', linewidth=1.6,
                 label=f'Proposed Trigger+CS ({bw_opt:.1f}% BW Saved, {cr_opt:.1f}x CR)')
    axes[1].fill_between(cycles, est_prop[:, 2] - 1.96*std_t, est_prop[:, 2] + 1.96*std_t,
                         color='#1f77b4', alpha=0.18, label='95% Confidence Interval')
    axes[1].set_xlabel('Engine Flight Cycles', fontweight='bold')
    axes[1].set_ylabel('Turbine Health ($\eta_t$)', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left', fontsize=8, ncol=2)

    fig.suptitle('On-Ground Digital Twin State Reconstruction Under Selective Compressed Telemetry', fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig('fig2_health_tracking_uq.png', dpi=300)
    plt.close(fig)

    print("\n[Success] Saved IEEE figures:")
    print("  1. fig1_pareto_bandwidth_vs_mae.png")
    print("  2. fig2_health_tracking_uq.png")

import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
rows = []
for eng in range(1, 8):
    for cyc in range(1, 21):
        base = 0.99 - (eng - 1) * 0.003 - (cyc - 1) * 0.0012
        comp = max(0.85, min(1.0, base + rng.normal(0, 0.008)))
        comb = max(0.88, min(1.0, 0.995 - (eng - 1) * 0.002 - (cyc - 1) * 0.0008 + rng.normal(0, 0.005)))
        turb = max(0.86, min(1.0, 0.985 - (eng - 1) * 0.0025 - (cyc - 1) * 0.0011 + rng.normal(0, 0.006)))
        pamb = 101325 + 2000 * rng.normal(0, 1)
        tamb = 288 + 2 * rng.normal(0, 1)
        rpm = 120000 + 6000 * eng + 300 * (cyc - 1) + rng.normal(0, 1200)
        fuel = 0.8 + 0.03 * (cyc - 1) + 0.01 * eng + rng.normal(0, 0.02)
        mach = 0.35 + 0.01 * eng + 0.003 * (cyc - 1) + rng.normal(0, 0.01)
        alt = 6000 + 100 * eng + 20 * (cyc - 1) + rng.normal(0, 100)
        p2 = pamb * (1.05 + 0.002 * (cyc - 1) + 0.0005 * eng + rng.normal(0, 100))
        t2 = tamb * (1.0 + (p2 / pamb - 1) ** 0.286) + rng.normal(0, 3)
        p3 = p2 * 0.96 + 1000 * rng.normal(0, 1)
        t3 = (0.85 * 1005 * t2 + comp * 1e7 * fuel) / (0.9 * 1148 + 1e-9) + rng.normal(0, 15)
        p4 = p3 * (0.8 + 0.003 * (cyc - 1) + 0.001 * eng) + rng.normal(0, 200)
        t4 = t3 - (t3 * (1 - ((0.82 + 0.002 * (cyc - 1) + 0.001 * eng)) ** 0.24)) + rng.normal(0, 10)
        thrust = 60000 + 1200 * (eng - 1) + 900 * (cyc - 1) + rng.normal(0, 500)
        tsfc = 0.014 + 0.0005 * eng + 0.0001 * (cyc - 1) + rng.normal(0, 0.0004)
        rows.append({
            'EngineID': eng,
            'Cycle': cyc,
            'Pamb_Pa': round(pamb, 2),
            'Tamb_K': round(tamb, 2),
            'RPM_rev_min': round(rpm, 2),
            'FuelFlow_kg_s': round(fuel, 6),
            'Mach': round(mach, 4),
            'Altitude_m': round(alt, 2),
            'P2_Pa': round(p2, 2),
            'T2_K': round(t2, 2),
            'P3_Pa': round(p3, 2),
            'T3_K': round(t3, 2),
            'P4_Pa': round(p4, 2),
            'T4_K': round(t4, 2),
            'Thrust_N': round(thrust, 2),
            'TSFC_g_N_s': round(tsfc, 6),
            'CompressorHealth': round(comp, 6),
            'CombustorHealth': round(comb, 6),
            'TurbineHealth': round(turb, 6),
            'OverallHealth': round((comp + comb + turb) / 3, 6),
        })

df = pd.DataFrame(rows)
df.to_csv('synthetic_random_dataset.csv', index=False)
print(df.head().to_string(index=False))
print('saved', df.shape[0], 'rows')

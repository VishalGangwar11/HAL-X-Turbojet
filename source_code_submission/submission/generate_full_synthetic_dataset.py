import pandas as pd
import numpy as np

rng = np.random.default_rng(7)
rows = []
for eng in range(1, 11):
    for cyc in range(1, 21):
        base = 0.99 - (eng - 1) * 0.0028 - (cyc - 1) * 0.0011
        comp = max(0.84, min(1.0, base + rng.normal(0, 0.0075)))
        comb = max(0.86, min(1.0, 0.994 - (eng - 1) * 0.0018 - (cyc - 1) * 0.0009 + rng.normal(0, 0.0048)))
        turb = max(0.85, min(1.0, 0.986 - (eng - 1) * 0.0022 - (cyc - 1) * 0.0010 + rng.normal(0, 0.0055)))
        pamb = 101325 + 1800 * rng.normal(0, 1)
        tamb = 288 + 2.2 * rng.normal(0, 1)
        rpm = 118000 + 5500 * eng + 280 * (cyc - 1) + rng.normal(0, 1100)
        fuel = 0.78 + 0.028 * (cyc - 1) + 0.009 * eng + rng.normal(0, 0.018)
        mach = 0.34 + 0.009 * eng + 0.0026 * (cyc - 1) + rng.normal(0, 0.009)
        alt = 5900 + 95 * eng + 18 * (cyc - 1) + rng.normal(0, 90)
        p2 = pamb * (1.046 + 0.0018 * (cyc - 1) + 0.00045 * eng + rng.normal(0, 95))
        t2 = tamb + 25 + 0.6 * (cyc - 1) + rng.normal(0, 2.2)
        p3 = p2 * 0.96 + 950 * rng.normal(0, 1)
        t3 = 1100 + 8 * (cyc - 1) + 4 * eng + rng.normal(0, 12)
        p4 = p3 * (0.79 + 0.0028 * (cyc - 1) + 0.0009 * eng) + rng.normal(0, 190)
        t4 = t3 - 40 - 1.2 * (cyc - 1) + rng.normal(0, 8)
        thrust = 60000 + 1100 * (eng - 1) + 850 * (cyc - 1) + rng.normal(0, 480)
        tsfc = 0.0138 + 0.00045 * eng + 0.00009 * (cyc - 1) + rng.normal(0, 0.00038)
        rows.append({
            'EngineID': eng,
            'Cycle': cyc,
            'Pamb_Pa': round(float(pamb), 2),
            'Tamb_K': round(float(tamb), 2),
            'RPM_rev_min': round(float(rpm), 2),
            'FuelFlow_kg_s': round(float(fuel), 6),
            'Mach': round(float(mach), 4),
            'Altitude_m': round(float(alt), 2),
            'P2_Pa': round(float(p2), 2),
            'T2_K': round(float(t2), 2),
            'P3_Pa': round(float(p3), 2),
            'T3_K': round(float(t3), 2),
            'P4_Pa': round(float(p4), 2),
            'T4_K': round(float(t4), 2),
            'Thrust_N': round(float(thrust), 2),
            'TSFC_g_N_s': round(float(tsfc), 6),
            'CompressorHealth': round(float(comp), 6),
            'CombustorHealth': round(float(comb), 6),
            'TurbineHealth': round(float(turb), 6),
            'OverallHealth': round(float((comp + comb + turb) / 3), 6),
        })

df = pd.DataFrame(rows)
df.to_csv('synthetic_random_dataset.csv', index=False)
print('rows', len(df))
print(df[['EngineID', 'Cycle', 'CompressorHealth', 'CombustorHealth', 'TurbineHealth']].head(10).to_string(index=False))
print('engines', sorted(df['EngineID'].unique().tolist()))

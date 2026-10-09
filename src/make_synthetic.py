"""Generate SYNTHETIC inspection data in the DOHMH raw schema.

Only for smoke-testing the pipeline when the real dataset is not available.
Results on this data say nothing about real-world performance.
"""
import argparse

import numpy as np
import pandas as pd

BOROS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BORO_P = [0.32, 0.27, 0.24, 0.12, 0.05]
BORO_EFF = {"Manhattan": 0.0, "Brooklyn": 0.1, "Queens": 0.05, "Bronx": 0.25, "Staten Island": -0.1}
CUISINES = {
    "American": 0.0, "Chinese": 0.2, "Pizza": 0.0, "Coffee/Tea": -0.3, "Latin American": 0.2,
    "Mexican": 0.15, "Italian": 0.0, "Japanese": 0.05, "Bakery Products/Desserts": -0.2,
    "Caribbean": 0.25, "Indian": 0.15, "Sandwiches": -0.1, "Chicken": 0.1, "Other": 0.0,
}
CODES = ["02G", "04L", "04N", "06C", "06D", "08A", "10F", "10B", "04A", "05D", "09C", "16C"]
INIT = "Cycle Inspection / Initial Inspection"
REINSP = "Cycle Inspection / Re-inspection"


def make_synthetic(n_restaurants=4000, seed=42):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2017-01-01"), pd.Timestamp("2025-06-30")
    cuisines, cuis_names = list(CUISINES.values()), list(CUISINES.keys())
    rows = []
    for cid in range(n_restaurants):
        camis = 40000000 + cid
        boro = rng.choice(BOROS, p=BORO_P)
        ci = rng.integers(0, len(cuis_names))
        q = rng.normal(0, 1) + BORO_EFF[boro] + cuisines[ci]  # latent restaurant "hygiene quality"
        date = start + pd.Timedelta(days=int(rng.integers(0, 400)))
        reinsp, n_insp, t = False, int(rng.integers(2, 9)), 0
        while t < n_insp and date <= end:
            qi = q - (0.5 if reinsp else 0.0) + rng.normal(0, 0.35)
            nv = int(rng.poisson(np.exp(1.0 + 0.5 * qi)))
            pts = rng.choice([2, 3, 5, 7, 8, 10, 12], size=nv, p=[.15, .2, .2, .15, .15, .1, .05])
            crit = rng.random(nv) < 0.55
            score = int(pts.sum())
            base = {
                "CAMIS": camis, "BORO": boro, "CUISINE DESCRIPTION": cuis_names[ci],
                "INSPECTION DATE": date.strftime("%m/%d/%Y"),
                "INSPECTION TYPE": REINSP if reinsp else INIT, "SCORE": score,
            }
            if nv == 0:
                rows.append({**base, "CRITICAL FLAG": "Not Applicable", "VIOLATION CODE": np.nan})
            for i in range(nv):
                rows.append({**base, "CRITICAL FLAG": "Critical" if crit[i] else "Not Critical",
                             "VIOLATION CODE": rng.choice(CODES)})
            if score >= 14 and rng.random() < 0.7:
                date += pd.Timedelta(days=int(rng.integers(20, 41)))
                reinsp = True
            else:
                date += pd.Timedelta(days=int(rng.integers(300, 451)))
                reinsp = False
            t += 1
    df = pd.DataFrame(rows)
    # messy rows like the real file: not-yet-inspected placeholders and missing scores
    junk = df.sample(frac=0.002, random_state=seed).copy()
    junk["INSPECTION DATE"] = "01/01/1900"
    junk["SCORE"] = np.nan
    df.loc[df.sample(frac=0.003, random_state=seed + 1).index, "SCORE"] = np.nan
    return pd.concat([df, junk], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_inspections.csv")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    d = make_synthetic(a.n, a.seed)
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d):,} synthetic rows to {a.out}")

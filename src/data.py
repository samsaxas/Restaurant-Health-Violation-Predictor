"""Load NYC DOHMH inspection data and build a leakage-safe, inspection-level table.

Raw data has one row per violation. We aggregate to one row per (restaurant, inspection date)
and build features ONLY from a restaurant's *previous* inspections, so nothing about the
inspection being predicted leaks into its own features.
"""
import numpy as np
import pandas as pd

HIGH_RISK_SCORE = 28  # NYC scoring: 28+ points is the grade-C range (higher score = worse)
TARGET = "high_risk"

RAW_COLS = [
    "CAMIS", "BORO", "CUISINE DESCRIPTION", "INSPECTION DATE", "INSPECTION TYPE",
    "CRITICAL FLAG", "SCORE", "VIOLATION CODE",
]
NUM_FEATURES = [
    "n_prev_inspections", "days_since_prev", "prev_score", "prev_n_violations",
    "prev_n_critical", "hist_mean_score", "hist_max_score", "hist_high_risk_rate",
]
CAT_FEATURES = ["boro", "cuisine", "inspection_type"]


def load_raw(path):
    """Read only the columns we need (the full CSV is large)."""
    return pd.read_csv(path, usecols=lambda c: c in RAW_COLS, low_memory=False)


def _grp(df):
    return df.groupby("CAMIS", sort=False)


def build_inspection_table(raw):
    """Violation-level rows -> one row per inspection with history features and target."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["INSPECTION DATE"], format="%m/%d/%Y", errors="coerce")
    # DOHMH uses 01/01/1900 for "not yet inspected"; drop those and unparsable dates
    df = df[df["date"].notna() & (df["date"].dt.year > 1900) & df["CAMIS"].notna()]
    df["is_violation"] = df["VIOLATION CODE"].notna().astype(int)
    df["is_critical"] = (df["CRITICAL FLAG"] == "Critical").astype(int)

    insp = (
        df.groupby(["CAMIS", "date"], as_index=False)
        .agg(
            score=("SCORE", "max"),
            n_violations=("is_violation", "sum"),
            n_critical=("is_critical", "sum"),
            boro=("BORO", "first"),
            cuisine=("CUISINE DESCRIPTION", "first"),
            inspection_type=("INSPECTION TYPE", "first"),
        )
    )
    insp = insp[insp["score"].notna()].sort_values(["CAMIS", "date"]).reset_index(drop=True)
    insp[TARGET] = (insp["score"] >= HIGH_RISK_SCORE).astype(int)

    # ---- history features: everything is shifted so the current inspection is excluded ----
    insp["n_prev_inspections"] = _grp(insp).cumcount()
    insp["days_since_prev"] = _grp(insp)["date"].diff().dt.days
    insp["prev_score"] = _grp(insp)["score"].shift(1)
    insp["prev_n_violations"] = _grp(insp)["n_violations"].shift(1)
    insp["prev_n_critical"] = _grp(insp)["n_critical"].shift(1)

    n_prev = insp["n_prev_inspections"].replace(0, np.nan)
    insp["hist_mean_score"] = (_grp(insp)["score"].cumsum() - insp["score"]) / n_prev
    insp["hist_high_risk_rate"] = (_grp(insp)[TARGET].cumsum() - insp[TARGET]) / n_prev
    insp["_cummax"] = _grp(insp)["score"].cummax()
    insp["hist_max_score"] = _grp(insp)["_cummax"].shift(1)

    insp = insp[insp["n_prev_inspections"] >= 1].drop(columns="_cummax")
    return insp.reset_index(drop=True)


def time_split(table, test_frac=0.2):
    """Chronological split: train on the past, test on the most recent inspections."""
    cutoff = table["date"].quantile(1 - test_frac)
    train = table[table["date"] < cutoff]
    test = table[table["date"] >= cutoff]
    return train, test, cutoff

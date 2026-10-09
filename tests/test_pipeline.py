import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data import CAT_FEATURES, NUM_FEATURES, TARGET, build_inspection_table, time_split
from src.make_synthetic import make_synthetic
from src.model import RiskModel
from src.report import generate_report
from src.smote import smote_resample

FEATURES = NUM_FEATURES + CAT_FEATURES


def _raw(rows):
    cols = ["CAMIS", "BORO", "CUISINE DESCRIPTION", "INSPECTION DATE", "INSPECTION TYPE",
            "CRITICAL FLAG", "SCORE", "VIOLATION CODE"]
    return pd.DataFrame(rows, columns=cols)


def test_history_features_exclude_current_inspection():
    # one restaurant, three inspections scored 10, 30, 5
    raw = _raw([
        (1, "Queens", "Pizza", "01/10/2022", "Initial", "Critical", 10, "04L"),
        (1, "Queens", "Pizza", "01/10/2023", "Initial", "Critical", 30, "04L"),
        (1, "Queens", "Pizza", "01/10/2023", "Initial", "Not Critical", 30, "08A"),
        (1, "Queens", "Pizza", "01/10/2024", "Initial", "Not Critical", 5, "10F"),
    ])
    t = build_inspection_table(raw)
    assert len(t) == 2  # first inspection has no history and is dropped
    second, third = t.iloc[0], t.iloc[1]
    assert second["prev_score"] == 10 and second["hist_mean_score"] == 10
    assert second["hist_max_score"] == 10 and second["hist_high_risk_rate"] == 0
    assert second[TARGET] == 1 and second["n_violations"] == 2
    assert third["prev_score"] == 30 and third["hist_mean_score"] == 20
    assert third["hist_max_score"] == 30 and third["hist_high_risk_rate"] == 0.5
    assert third[TARGET] == 0


def test_cleaning_drops_placeholder_dates_and_missing_scores():
    raw = _raw([
        (1, "Queens", "Pizza", "01/01/1900", "Initial", None, np.nan, None),
        (1, "Queens", "Pizza", "01/10/2022", "Initial", "Critical", 10, "04L"),
        (1, "Queens", "Pizza", "01/10/2023", "Initial", "Critical", np.nan, "04L"),
        (1, "Queens", "Pizza", "05/10/2023", "Initial", "Critical", 12, "04L"),
    ])
    t = build_inspection_table(raw)
    assert len(t) == 1 and t.iloc[0]["prev_score"] == 10


def test_time_split_has_no_overlap():
    t = build_inspection_table(make_synthetic(300, seed=1))
    train, test, cutoff = time_split(t)
    assert train["date"].max() < cutoff <= test["date"].min()


def test_smote_balances_classes_and_stays_in_range():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(3, 1, (20, 3))])
    y = np.array([0] * 200 + [1] * 20)
    Xr, yr = smote_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum() == 200
    X_min = X[y == 1]
    new = Xr[len(X):]
    assert (new >= X_min.min(0) - 1e-9).all() and (new <= X_min.max(0) + 1e-9).all()


def test_model_outputs_valid_probabilities():
    t = build_inspection_table(make_synthetic(400, seed=2))
    train, test, _ = time_split(t)
    m = RiskModel(RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=-1), True, 0)
    m.fit(train[FEATURES], train[TARGET])
    p = m.predict_proba(test[FEATURES])
    assert p.shape == (len(test),) and ((p >= 0) & (p <= 1)).all()
    assert m.smote_backend is not None


def test_report_template_fallback_without_api_key(monkeypatch=None):
    import os
    os.environ.pop("GEMINI_API_KEY", None)
    row = {"boro": "Bronx", "cuisine": "Pizza", "prev_score": 30, "hist_mean_score": 22.5,
           "prev_n_critical": 3, "days_since_prev": 200, "n_prev_inspections": 4}
    text, backend = generate_report(row, 0.72, ["prev_score", "hist_mean_score"])
    assert backend == "template" and "72%" in text and "HIGH" in text

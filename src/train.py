"""Train and evaluate the risk models; write metrics, figures, and sample reports.

Usage:
    python -m src.train --data data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv --label "NYC DOHMH (real)"
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .data import (CAT_FEATURES, HIGH_RISK_SCORE, NUM_FEATURES, TARGET,
                   build_inspection_table, load_raw, time_split)
from .model import RiskModel
from .report import generate_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURES = NUM_FEATURES + CAT_FEATURES
MAIN_MODEL = "Random Forest + SMOTE"


def make_models(seed):
    return {
        "Logistic Regression + SMOTE": RiskModel(LogisticRegression(max_iter=1000), True, seed),
        MAIN_MODEL: RiskModel(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed), True, seed),
        "Random Forest (no SMOTE, class_weight=balanced)": RiskModel(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, class_weight="balanced_subsample",
            random_state=seed), False, seed),
    }


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred low", "Pred high"])
    ax.set_yticks([0, 1], ["Actual low", "Actual high"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(names, imps, path, top=12):
    order = np.argsort(imps)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], [imps[i] for i in order], color="#2b6cb0")
    ax.set_title(f"{MAIN_MODEL}: top {top} feature importances", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="unlabelled run")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reports", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)

    raw = load_raw(a.data)
    table = build_inspection_table(raw)
    train, test, cutoff = time_split(table)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    results, fitted = {}, {}
    for name, model in make_models(a.seed).items():
        model.fit(Xtr, ytr)
        results[name] = evaluate(model, Xte, yte)
        fitted[name] = model
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[name].items()
                                      if k in ("precision", "recall", "f1", "roc_auc", "pr_auc")))

    main_model = fitted[MAIN_MODEL]
    names = main_model.feature_names
    imps = main_model.estimator.feature_importances_
    imp_df = pd.DataFrame({"feature": names, "importance": imps}).sort_values("importance", ascending=False)
    imp_df.to_csv(out / "feature_importance.csv", index=False)
    plot_importance(names, imps, out / "feature_importance.png")
    plot_confusion(results[MAIN_MODEL], out / "confusion_matrix.png", f"{MAIN_MODEL} (threshold 0.5)")
    joblib.dump(main_model, Path(a.models_dir) / "risk_model.joblib")

    metrics = {
        "label": a.label, "seed": a.seed, "smote_backend": main_model.smote_backend,
        "n_raw_rows": int(len(raw)), "n_inspections_modelled": int(len(table)),
        "n_restaurants": int(table["CAMIS"].nunique()),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "split_cutoff_date": str(cutoff.date()), "high_risk_score_threshold": HIGH_RISK_SCORE,
        "train_prevalence": float(ytr.mean()), "test_prevalence": float(yte.mean()),
        "majority_baseline_accuracy": float(1 - yte.mean()),
        "models": results, "top_features": imp_df.head(8).to_dict("records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # sample hazard reports for the highest-risk test inspections
    proba = main_model.predict_proba(Xte)
    top_feats = list(imp_df["feature"].head(3))
    lines = [f"# Sample hazard reports ({a.label})\n"]
    for idx in np.argsort(proba)[::-1][:a.reports]:
        row = test.iloc[idx]
        text, backend = generate_report(row, float(proba[idx]), top_feats)
        lines.append(f"**Restaurant {int(row['CAMIS'])}** (actual score {row['score']:.0f}, report backend: {backend})\n\n{text}\n")
    (out / "sample_reports.md").write_text("\n".join(lines))
    print(f"saved metrics + figures to {out}/")


if __name__ == "__main__":
    main()

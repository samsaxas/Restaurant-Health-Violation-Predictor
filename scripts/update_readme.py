"""Fill a README results block from results/metrics.json.

    python scripts/update_readme.py --block real      --metrics results/metrics.json
    python scripts/update_readme.py --block synthetic --metrics results/synthetic/metrics.json
"""
import argparse
import json
import re
from pathlib import Path


def render(m):
    rows = ["| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for name, r in m["models"].items():
        rows.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                    f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} |")
    main = m["models"]["Random Forest + SMOTE"]
    feats = ", ".join(f"`{f['feature']}` ({f['importance']:.2f})" for f in m["top_features"][:5])
    return "\n".join([
        f"- **Data:** {m['label']} | raw rows: {m['n_raw_rows']:,} | modelled inspections: "
        f"{m['n_inspections_modelled']:,} across {m['n_restaurants']:,} restaurants",
        f"- **Split:** chronological, train before {m['split_cutoff_date']} "
        f"({m['n_train']:,} inspections), test on the latest ({m['n_test']:,})",
        f"- **High-risk rate in test set:** {m['test_prevalence']:.1%} "
        f"(a model that always predicts 'low risk' would score {m['majority_baseline_accuracy']:.1%} accuracy)",
        f"- **SMOTE backend:** {m['smote_backend']} | seed {m['seed']} | threshold 0.5", "",
        *rows, "",
        f"Main model (Random Forest + SMOTE) confusion matrix on the test set: "
        f"TN={main['tn']:,}, FP={main['fp']:,}, FN={main['fn']:,}, TP={main['tp']:,}.", "",
        f"Top features: {feats}.",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["real", "synthetic"], required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    body = render(json.loads(Path(a.metrics).read_text()))
    tag = a.block.upper()
    path = Path(a.readme)
    text = path.read_text()
    pat = re.compile(rf"(<!-- RESULTS_{tag}:START -->)(.*?)(<!-- RESULTS_{tag}:END -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"markers for {tag} not found in {a.readme}")
    path.write_text(pat.sub(lambda mo: f"{mo.group(1)}\n{body}\n{mo.group(3)}", text))
    print(f"updated {a.block} results block in {a.readme}")


if __name__ == "__main__":
    main()

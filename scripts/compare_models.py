"""Session-grouped CV comparison of the four reranking strategies.

    python -m pip install -r requirements-experiments.txt
    python scripts/compare_models.py

Every fold trains on the training sessions' captured candidate pools and is then
scored by the REAL evaluator on the held-out sessions with the trained reranker
plugged into ``HybridRetriever`` – so the headline numbers are Hit@10 / MRR /
MTTC, not classification accuracy. Writes docs/model_experiments.md and
results_rerank_cv.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cv import METHODS, run

_LABEL = {
    "manual": "Manual weights",
    "logistic": "Logistic regression",
    "forest": "Random forest",
    "lambdamart": "LightGBM LambdaRank",
}


def _fmt(stat: dict) -> str:
    if stat["mean"] != stat["mean"]:  # nan
        return "–"
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def render_markdown(summary: dict, full_fit: dict, importances: dict, n_splits: int) -> str:
    L: list[str] = []
    L.append("# Reranking model comparison\n")
    L.append(
        f"{n_splits}-fold cross-validation, folds split by `session_id` (never by row, so "
        "no turn from one session leaks across train/test). Each fold trains on the "
        "training sessions' captured candidate pools — every positive + the 24 hardest "
        "negatives by fused score + 8 random negatives per turn — and is then scored by "
        "the **real `evaluator.local_evaluator`** on the held-out sessions with the "
        "trained model plugged into `HybridRetriever.reranker`.\n"
    )
    L.append(
        "All four rerankers consume the identical 36-feature vector "
        "(`shopping_copilot/features.py`). `manual` is the shipped hand-tuned score "
        "expressed over those features; it reproduces the built-in result "
        "(Hit@10 0.9450 / MRR 0.5172 / tech 0.7745) exactly.\n"
    )

    L.append("## Cross-validated ranking quality\n")
    L.append("| Method | Hit@10 (μ±σ) | MRR (μ±σ) | MTTC (μ±σ) | ROC-AUC | PR-AUC | train s | infer ms/turn |")
    L.append("| --- | --- | --- | --- | --- | --- | ---: | ---: |")
    for m in METHODS:
        s = summary[m]
        L.append(
            f"| {_LABEL[m]} | {_fmt(s['hit_rate_at_10'])} | {_fmt(s['mrr'])} | {_fmt(s['mttc'])} | "
            f"{_fmt(s['roc_auc'])} | {_fmt(s['pr_auc'])} | "
            f"{s['train_seconds']['mean']:.2f} | {s['infer_ms_per_turn']['mean']:.2f} |"
        )
    L.append("")

    L.append("### Per-fold MRR\n")
    L.append("| Method | " + " | ".join(f"fold {i}" for i in range(n_splits)) + " |")
    L.append("| --- | " + " | ".join("---" for _ in range(n_splits)) + " |")
    for m in METHODS:
        L.append(f"| {_LABEL[m]} | " + " | ".join(f"{v:.4f}" for v in summary[m]["mrr"]["per_fold"]) + " |")
    L.append("")

    L.append("### Per-scenario Hit@10 / MRR (CV mean)\n")
    scen_names = sorted(next(iter(summary.values()))["scenario"])
    L.append("| Method | " + " | ".join(scen_names) + " |")
    L.append("| --- | " + " | ".join("---" for _ in scen_names) + " |")
    for m in METHODS:
        cells = [
            f"{summary[m]['scenario'][sc]['hit_rate_at_10']['mean']:.3f} / "
            f"{summary[m]['scenario'][sc]['mrr']['mean']:.3f}"
            for sc in scen_names
        ]
        L.append(f"| {_LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")

    if full_fit:
        L.append("## Full-fit reference (trained on all 200, scored on all 200)\n")
        L.append("Optimistically biased for the learned models — shown only to size the "
                 "train/CV gap (overfitting signal).\n")
        L.append("| Method | Hit@10 | MRR | MTTC | TechnicalScore |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for m in METHODS:
            f = full_fit[m]
            L.append(f"| {_LABEL[m]} | {f['hit_rate_at_10']:.4f} | {f['mrr']:.4f} | {f['mttc']:.4f} | {f['technical_score']:.4f} |")
        L.append("")

    if importances:
        L.append("## Feature importance (top 8, full-fit)\n")
        for m in ("logistic", "forest", "lambdamart"):
            if importances.get(m):
                L.append(f"- **{_LABEL[m]}**: " + ", ".join(f"`{n}` ({v:.3f})" for n, v in importances[m]))
        L.append("")

    if full_fit:
        L.append("## Overfitting signal (full-fit MRR − CV MRR)\n")
        L.append("| Method | CV MRR | Full-fit MRR | gap |")
        L.append("| --- | ---: | ---: | ---: |")
        for m in METHODS:
            cv = summary[m]["mrr"]["mean"]
            ff = full_fit[m]["mrr"]
            L.append(f"| {_LABEL[m]} | {cv:.4f} | {ff:.4f} | {ff - cv:+.4f} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare four reranking methods under session-grouped CV.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="results_rerank_cv.json")
    parser.add_argument("--md-out", default="docs/model_experiments.md")
    parser.add_argument("--checkpoint", default="results_rerank_cv.checkpoint.json")
    parser.add_argument("--skip-full-fit", action="store_true")
    args = parser.parse_args()

    comparison = run(
        args.catalog, args.dataset, n_splits=args.splits, seed=args.seed,
        checkpoint_path=args.checkpoint, skip_full_fit=args.skip_full_fit,
    )
    summary = comparison.summary()

    Path(args.json_out).write_text(
        json.dumps(
            {
                "summary": summary,
                "full_fit": comparison.full_fit,
                "importances": comparison.importances,
                "folds": [vars(f) for f in comparison.folds],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md = render_markdown(summary, comparison.full_fit, comparison.importances, args.splits)
    Path(args.md_out).write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
